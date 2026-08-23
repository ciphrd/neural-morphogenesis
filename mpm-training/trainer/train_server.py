"""FastAPI server that runs evolve.py's training loop in the background
and hands each generation's stats + winning weights to any connected
browser over a websocket — same overall shape as envnca/train_server.py
and trainer/backend/train_server.py, ported to this project's own
evolve.py/training_sim.py.

Unlike envnca's frontend (which replays the winning rollout itself,
entirely client-side, on WebGPU), mpm-training's own viewer is still
unbuilt (../viewer/README.md's own staging is explicitly blocked on,
among other things, this exact server's message schema) — so for now
this server *also* renders each generation's winning rollout server-side
(debug_images.py) and serves those PNGs directly: the "collect renders
of the best of each generation so the frontend can parse these" fallback
envnca/train_server.py already uses for its own debug snapshots, just
promoted here from a debug aid to the primary way progress is shown. The
`weights` broadcast on every message is still included (see
latest_generation_message below) so a future client-side replay isn't
blocked on anything this server would need to change.

Usage:
    python train_server.py --target circle --population 16 --generations 100 --port 8003
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agents_gpu import AgentsGPU
from debug_images import save_grown_image, save_raster_image
from device import pick_device
from environment_gpu import EnvironmentGPU
from evolve import (
    CHECKPOINTS_DIR,
    RASTER_EXTENT,
    build_arg_parser,
    get_weights,
    rollout,
    run_generation,
    set_weights,
)
from mpm_core import MpmCore
from parallel_workers import build_pool
from raster import build_target_distance_field, build_target_raster, training_raster_distance
from simulation_settings import (
    ANGULAR_DAMPING,
    CHEM_CHANNELS,
    CHIRALITY,
    COMMUNICATION_SPEED,
    DAMPING_LOSS_FRACTION,
    DECAY,
    DEPOSIT_DISTANCE,
    DEPOSIT_RATE,
    DEPOSIT_SIGMA,
    DIVISION_COOLDOWN,
    ELASTIC_STRAIN_SCALE,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    FIELD_N,
    FRICTION,
    GROWTH_DURATION_MACRO_STEPS,
    GROWTH_MAX,
    GROWTH_THRESHOLD,
    HIDDEN_DIM,
    MASS_RAMP_MACRO_STEPS,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    NEURAL_UPDATES_PER_MACRO,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MAX_ACCEL,
    MAX_ANGULAR_ACCEL,
    MAX_ANGULAR_VELOCITY,
    MAX_ENV_WRITE,
    MAX_STRAFE,
    MPM_ENABLED,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
    SPLIT_DISPLACEMENT,
)
from targets import TARGETS_DIR, load_target
from update_rule import UpdateRule

parser = build_arg_parser()
parser.add_argument("--port", type=int, default=8003)

# `args`/`wgpu_device`/`target`/`target_raster`/`target_distance_field`
# are set by _setup() below, called only under `if __name__ ==
# "__main__":` at the bottom of this file — NOT computed directly here
# at plain module level, despite every route handler (and
# _training_loop_body()) referencing them as ordinary module globals
# (Python resolves those at CALL time, not at function-definition time,
# so this is safe as long as _setup() runs before the server actually
# starts taking requests).
#
# This matters for a real, confirmed reason, not just tidiness: this
# file is also the "main module" multiprocessing's own 'spawn' start
# method (parallel_workers.py's own module docstring explains why spawn,
# not fork) RE-EXECUTES in every worker process build_pool() creates —
# spawn needs to re-run the entry script to reconstruct enough state to
# unpickle tasks sent to it, a standard, documented Python multiprocessing
# behavior, not a bug in this design. Every one of these used to be
# computed at plain module level, which meant EVERY worker process
# wastefully redid all of it — including picking its own throwaway wgpu
# device it never actually uses (workers get their own real device from
# parallel_workers.py's own _worker_init instead, entirely separately).
# Confirmed via a real, reproduced bug before this fix: `--workers N`
# was observed picking 2N wgpu devices instead of N — one from this
# module's own top-level code re-running in each spawned child, one from
# that worker's own proper _worker_init.
args = None
wgpu_device = None
target = None
target_raster = None
target_distance_field = None


def _setup() -> None:
    global args, wgpu_device, target, target_raster, target_distance_field
    args = parser.parse_args()

    if not 1 <= args.elites <= args.population:
        raise SystemExit("--elites must be between 1 and --population")
    if args.growth_steps is not None and not 0 <= args.growth_steps <= args.macro_steps:
        raise SystemExit("--growth-steps must be between 0 and --macro-steps")

    wgpu_device = pick_device()

    # Fixed for this server's lifetime (no target-switching endpoint).
    target = load_target(args.target)
    # Fixed for this server's lifetime too — precomputed once rather than
    # recomputing the same thing on every rollout's own fitness-scoring
    # call AND on every _save_generation_images() debug-raster build (see
    # that function's own docstring). Passed to build_pool() below (baked
    # into every worker's own globals — see parallel_workers.py) for the
    # first use, and used directly, here in the main process, for the
    # second.
    target_raster = build_target_raster(
        target.points, args.raster_resolution, RASTER_EXTENT, args.raster_sigma, half_size=target.texel_size() / 2.0
    )
    target_distance_field = build_target_distance_field(target_raster)


# Every generation's own message (stats + weights) is appended here as it
# happens, so a browser tab that connects mid-run — or reconnects after a
# reload, or after this server process itself restarts — isn't starting
# blind; see /history and _training_loop_body()'s append below. Generation-
# specific data ONLY (generation/best/mean/worst/allTimeBest/seed/weights)
# — everything fixed for the whole run lives in SETTINGS_PATH instead (see
# that path's own comment for why the two were split apart).
HISTORY_PATH = CHECKPOINTS_DIR / "history.jsonl"
MAX_HISTORY = 500

# Every simulation/search setting that's fixed for this run's entire
# lifetime (target, particles, channels, decay, population, ...) — written
# ONCE, near the very start of _training_loop_body(), before generation 0
# has even started evaluating. Deliberately split out of HISTORY_PATH's
# own per-generation records (which used to carry a full copy of all of
# this on EVERY single message — real, avoidable duplication once weights
# are already the dominant payload size) so a browser tab connecting
# mid-run — or, more importantly, one connecting during the (potentially
# long, population x workers) window BEFORE generation 0 has finished
# evaluating — can still learn what this run is even configured with via
# GET /settings, without waiting on a generation that may still be
# minutes away. That in turn is what lets the frontend build a channels/
# hiddenDim-correct GpuSimulation and start rendering a live rollout under
# RANDOM weights immediately (gpu/agents.ts's own randomWeights(), the
# same generator "Randomize weights" already uses) rather than showing
# blank placeholders until real weights exist.
SETTINGS_PATH = CHECKPOINTS_DIR / "settings.json"

# Where a previous run's history.jsonl/settings.json/best.npy/
# best_meta.json get moved before this run starts overwriting those same
# fixed filenames.
RUNS_DIR = CHECKPOINTS_DIR / "runs"

# End-of-generation debug renders (see _save_generation_images()), served
# to the frontend via GET /runs/{run_id}/images/{filename} below
# (run_id="current" reads straight from here; anything else reads an
# archived run's own copy) — archived by _archive_previous_run() the same
# way history.jsonl/best.npy are. Created eagerly so it reliably exists
# from the moment this module is imported, before any generation has run.
IMAGES_DIR = CHECKPOINTS_DIR / "generation_images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _archive_previous_run() -> None:
    """Moves the previous run's history.jsonl/best_meta.json/weight
    checkpoints (plus IMAGES_DIR, if it has anything in it) into a
    timestamped RUNS_DIR subdirectory before this run starts overwriting
    them — same reasoning as envnca/train_server.py's own version of this
    function. Only archives if there's actually something to keep
    (history.jsonl is the signal; a run that never got as far as its
    first generation has nothing worth archiving).

    Every plain file directly under CHECKPOINTS_DIR gets archived —
    globbed, not a hardcoded filename list."""
    if not HISTORY_PATH.is_file() or HISTORY_PATH.stat().st_size == 0:
        return

    meta_path = CHECKPOINTS_DIR / "best_meta.json"
    label = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            label = f"{meta.get('target', 'unknown')}_gen{meta.get('generation', '?')}"
        except (json.JSONDecodeError, OSError):
            label = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = RUNS_DIR / (f"{timestamp}_{label}" if label else timestamp)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for path in CHECKPOINTS_DIR.iterdir():
        if path.is_file():
            shutil.move(str(path), str(archive_dir / path.name))

    if IMAGES_DIR.is_dir() and any(IMAGES_DIR.iterdir()):
        shutil.move(str(IMAGES_DIR), str(archive_dir / "generation_images"))
        # Recreated immediately — _save_generation_images() expects this
        # directory to already exist the next time it's called.
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train_server] archived previous run to {archive_dir}")


def _save_generation_images(
    generation: int, winner_weights: np.ndarray, winner_seed: int, core: MpmCore, agents: AgentsGPU, environment: EnvironmentGPU
) -> None:
    """Three PNGs per generation — see debug_images.py's own module
    docstring for what each one is and why: `..._grown.png` (raw,
    un-aligned positions), `..._target.png` (the target's own raster,
    fixed all run), and `..._agents.png` (this winner's own positions,
    rotated to whichever pose raster.py's own rotation search actually
    scored it under — literally the same raster training picked this
    candidate on, meant to sit next to `..._target.png` for a direct
    visual check). Re-runs the winner's rollout (same weights + seed
    run_generation already scored it with, so this reproduces the
    identical final snapshot — see rollout()'s own docstring on
    reproducibility) since fitnesses from the population loop don't
    carry final positions along with them."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    _, positions = rollout(
        winner_weights,
        target,
        target_raster,
        target_distance_field,
        args,
        winner_seed,
        core,
        agents,
        environment,
        return_positions=True,
    )
    _, agent_raster = training_raster_distance(
        positions,
        target.points,
        target_raster,
        target_distance_field,
        args.raster_resolution,
        RASTER_EXTENT,
        args.raster_sigma,
        outside_weight=args.outside_weight,
        track_best_raster=True,
    )

    prefix = f"gen_{generation:05d}"
    save_grown_image(positions, target.points, IMAGES_DIR / f"{prefix}_grown.png")
    save_raster_image(target_raster, IMAGES_DIR / f"{prefix}_target.png")
    if agent_raster is not None:
        save_raster_image(agent_raster, IMAGES_DIR / f"{prefix}_agents.png")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    asyncio.create_task(training_loop())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connections: set[WebSocket] = set()
latest_generation_message: Optional[dict] = None
# This run's own fixed settings (see SETTINGS_PATH's own comment) — set
# once, near the top of _training_loop_body(), well before generation 0
# has finished (unlike latest_generation_message above, which stays None
# until it has). GET /settings below serves this directly.
settings: Optional[dict] = None


async def broadcast(message: dict) -> None:
    dead = set()
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    connections.difference_update(dead)


async def training_loop() -> None:
    """Thin wrapper so a crash anywhere in the run is loud and visible
    instead of vanishing into "Task exception was never retrieved" —
    asyncio.create_task() (see lifespan() above) never awaits this task,
    so nothing else would ever surface an uncaught exception here."""
    try:
        await _training_loop_body()
    except Exception:
        print("[train_server] training_loop crashed — training has stopped:")
        traceback.print_exc()


async def _training_loop_body() -> None:
    global latest_generation_message, settings

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # One MpmCore/AgentsGPU/EnvironmentGPU (wgpu pipeline compilation is
    # real, avoidable overhead — see evolve.py's own module docstring),
    # used ONLY for _save_generation_images()'s own single-candidate
    # winner replay below — the hot per-generation path runs on `pool`
    # instead (see parallel_workers.py's own module docstring for why a
    # persistent multi-process pool replaced a single reused triple
    # there). `update_rule` is a CPU-only scratch nn.Module used ONLY for
    # random weight initialization (below) and checkpoint JSON export
    # (below) — never a live forward pass, see training_sim.py's own
    # module docstring.
    core = MpmCore(wgpu_device)
    environment = EnvironmentGPU(wgpu_device, CHEM_CHANNELS, FIELD_N, FIELD_N, DECAY, DEPOSIT_RATE)
    agents = AgentsGPU(
        wgpu_device,
        core,
        environment,
        CHEM_CHANNELS,
        HIDDEN_DIM,
        MAX_ACCEL,
        MAX_STRAFE,
        MAX_ENV_WRITE,
        MAX_ANGULAR_ACCEL,
        ANGULAR_DAMPING,
        MAX_ANGULAR_VELOCITY,
        CHIRALITY,
        DEPOSIT_DISTANCE,
        args.particles,
        SPLIT_DISPLACEMENT,
        DIVISION_COOLDOWN,
        FRICTION,
        DEPOSIT_SIGMA,
        1.0,
        args.spawn_x,
        args.spawn_y,
    )
    num_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, args.population)
    # log_device=False — _setup() already logged the "[device] adapter:
    # ..." confirmation once, above, for this process's own wgpu_device;
    # see build_pool()'s own docstring for why it would otherwise repeat
    # that exact line a second time.
    pool = build_pool(num_workers, args.particles, target, target_raster, target_distance_field, args, log_device=False)
    update_rule = UpdateRule(CHEM_CHANNELS)
    population = [get_weights(UpdateRule(CHEM_CHANNELS)) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    _archive_previous_run()
    # Fresh log for this run — generation numbers restart at 0 each
    # invocation, so appending onto a previous run's log would collide
    # rather than continue a meaningful timeline. Whatever was here before
    # is now safely under RUNS_DIR, not discarded.
    HISTORY_PATH.write_text("")

    # This run's own fixed settings — see SETTINGS_PATH's own comment for
    # why this is written once, here, well before generation 0 has
    # finished (rather than folded into every latest_generation_message
    # broadcast the way it used to be). Everything a replay needs to
    # reproduce ANY generation's rollout EXCEPT that generation's own
    # weights/seed, which arrive separately per generation instead.
    settings = {
        "target": args.target,
        "particles": args.particles,
        "macroSteps": args.macro_steps,
        "growthSteps": args.growth_steps,
        "substepsPerMacro": args.substeps_per_macro,
        "gravity": args.gravity,
        "spawnX": args.spawn_x,
        "spawnY": args.spawn_y,
        "spawnHalfWidth": args.spawn_half_width,
        "channels": CHEM_CHANNELS,
        "fieldN": FIELD_N,
        "morphologyBlurSigma": MORPHOLOGY_BLUR_SIGMA,
        "morphologyDensityReference": MORPHOLOGY_DENSITY_REFERENCE,
        "neuralUpdatesPerMacro": NEURAL_UPDATES_PER_MACRO,
        "communicationSpeed": COMMUNICATION_SPEED,
        "elasticStrainScale": ELASTIC_STRAIN_SCALE,
        "elasticStrainInputsEnabled": ELASTIC_STRAIN_INPUTS_ENABLED,
        "hiddenDim": HIDDEN_DIM,
        "decay": DECAY,
        "depositRate": DEPOSIT_RATE,
        "maxAccel": MAX_ACCEL,
        "maxStrafe": MAX_STRAFE,
        "maxEnvWrite": MAX_ENV_WRITE,
        "maxAngularAccel": MAX_ANGULAR_ACCEL,
        "angularDamping": ANGULAR_DAMPING,
        "maxAngularVelocity": MAX_ANGULAR_VELOCITY,
        "depositDistance": DEPOSIT_DISTANCE,
        "depositSigma": DEPOSIT_SIGMA,
        "splitDisplacement": SPLIT_DISPLACEMENT,
        "divisionCooldown": DIVISION_COOLDOWN,
        "friction": FRICTION,
        "massRampMacroSteps": MASS_RAMP_MACRO_STEPS,
        "growthDuration": GROWTH_DURATION_MACRO_STEPS,
        "growthMax": GROWTH_MAX,
        "growthThreshold": GROWTH_THRESHOLD,
        # simulation_settings.py's own MPM_ENABLED (that constant's own
        # comment has the full "why" — a real testing/debug mode that
        # also skips MPM physics in the actual worker-pool population
        # evaluation driving fitness/selection, not just this broadcast
        # value) — sets the frontend's own starting toggle state (see
        # viewer/src/gpu/types.ts's own RunSettings.mpmEnabled), still
        # live-flippable there afterward regardless of this constant.
        "mpmEnabled": MPM_ENABLED,
        "chirality": CHIRALITY,
        "damping": DAMPING_LOSS_FRACTION,
        "materialE": MATERIAL_E,
        "materialNu": MATERIAL_NU,
        "materialHardening": MATERIAL_HARDENING,
        "materialElasticity": MATERIAL_ELASTICITY,
        "splatRadius": SPLAT_RADIUS,
        "repulsionStrength": REPULSION_STRENGTH,
        "repulsionMaxDelta": REPULSION_MAX_DELTA,
        "population": args.population,
        "elites": args.elites,
        "mutationSigma": args.mutation_sigma,
        "rasterResolution": args.raster_resolution,
        "rasterSigma": args.raster_sigma,
        "outsideWeight": args.outside_weight,
        "runSeed": args.seed,
        "totalGenerations": args.generations,
        "checkpointEvery": args.checkpoint_every,
    }
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))

    best_fitness = float("inf")
    best_weights = population[0]

    for generation in range(args.generations):
        # Off the event loop thread — run_generation blocks for the whole
        # generation (waiting on pool.map() across every worker process,
        # see parallel_workers.py's own module docstring), and doing that
        # directly on the event loop thread would stall websocket message
        # flushing for as long as it takes.
        population, fitnesses, winner_seed = await asyncio.to_thread(run_generation, population, args, rng, pool)

        winner_weights = population[0]
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_weights = winner_weights.copy()

        # Also off the event loop thread — re-runs the winner's rollout
        # once more (see _save_generation_images()'s own docstring) plus
        # PNG encoding/disk I/O, neither of which should stall websocket
        # message flushing either.
        await asyncio.to_thread(_save_generation_images, generation, winner_weights, winner_seed, core, agents, environment)

        finite = [f for f in fitnesses if np.isfinite(f)]
        print(
            f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(finite) if finite else float('inf'):.4f}  "
            f"worst {fitnesses[-1]:.4f}  (all-time best {best_fitness:.4f})"
        )

        # update_rule's currently-loaded weights are whatever the last
        # candidate run_generation evaluated used — load the winner's
        # before exporting.
        set_weights(update_rule, winner_weights)

        latest_generation_message = {
            "type": "generation",
            "generation": generation,
            "best": fitnesses[0],
            "mean": float(np.mean(finite)) if finite else float("inf"),
            "worst": fitnesses[-1],
            "allTimeBest": best_fitness,
            # This generation's winning rollout's own seed (what a replay
            # would re-seed its jitter with), NOT the top-level run seed
            # the whole training invocation was started with — that one
            # lives in `settings` instead (runSeed), fixed for the run.
            "seed": winner_seed,
            "weights": update_rule.export_weights(),
            # Everything else a replay needs (particles/channels/decay/
            # target/population/...) is fixed for the whole run and lives
            # in `settings`/SETTINGS_PATH instead — see that global's own
            # comment for why this split exists (this message used to
            # carry a full, redundant copy of all of it on every single
            # generation). The frontend merges {...settings, ...thisMessage}
            # into one SimulationConfig (see net/trainingSocket.ts's own
            # applyGeneration()) — settings is fetched once, not resent
            # here.
            #
            # No image URL fields here either, same reasoning as always:
            # this generation's server-rendered debug renders (see
            # _save_generation_images() above) always live at
            # /runs/{run_id}/images/gen_{N}_{grown,aligned}.png, but
            # embedding a URL in the message itself would have to
            # hardcode run_id="current", which goes stale the moment this
            # run gets archived under a different id (see
            # _archive_previous_run()) — every other run_id-scoped fact
            # this message could carry has the same problem. The frontend
            # builds these URLs itself from whichever run_id it's
            # currently viewing (net/images.ts's generationImageUrl()),
            # same as envnca/frontend's own net/images.ts already does.
        }
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(latest_generation_message) + "\n")
        await broadcast(latest_generation_message)

        if (generation + 1) % args.checkpoint_every == 0 or generation == args.generations - 1:
            np.save(CHECKPOINTS_DIR / "best.npy", best_weights)
            set_weights(update_rule, best_weights)
            (CHECKPOINTS_DIR / "best_weights.json").write_text(json.dumps(update_rule.export_weights()))
            (CHECKPOINTS_DIR / "best_meta.json").write_text(
                json.dumps(
                    {
                        "generation": generation,
                        "fitness": best_fitness,
                        "target": args.target,
                        "particles": args.particles,
                        "macro_steps": args.macro_steps,
                        "growth_steps": args.growth_steps,
                        "substeps_per_macro": args.substeps_per_macro,
                        "gravity": args.gravity,
                        "spawn_x": args.spawn_x,
                        "spawn_y": args.spawn_y,
                        "spawn_half_width": args.spawn_half_width,
                        "channels": CHEM_CHANNELS,
                        "field_n": FIELD_N,
                        "population": args.population,
                        "elites": args.elites,
                        "mutation_sigma": args.mutation_sigma,
                        "seed": args.seed,
                        "winner_seed": winner_seed,
                        # simulation_settings.py's own values this run
                        # actually simulated under — see evolve.py's own
                        # checkpoint metadata for why these ride along
                        # even though every message already carries them.
                        "decay": DECAY,
                        "deposit_rate": DEPOSIT_RATE,
                        "max_accel": MAX_ACCEL,
                        "max_strafe": MAX_STRAFE,
                        "max_env_write": MAX_ENV_WRITE,
                        "max_angular_accel": MAX_ANGULAR_ACCEL,
                        "angular_damping": ANGULAR_DAMPING,
                        "max_angular_velocity": MAX_ANGULAR_VELOCITY,
                        "deposit_distance": DEPOSIT_DISTANCE,
                        "deposit_sigma": DEPOSIT_SIGMA,
                        "split_displacement": SPLIT_DISPLACEMENT,
                        "division_cooldown": DIVISION_COOLDOWN,
                        "friction": FRICTION,
                        "mass_ramp_macro_steps": MASS_RAMP_MACRO_STEPS,
                        "growth_duration_macro_steps": GROWTH_DURATION_MACRO_STEPS,
                        "morphology_blur_sigma": MORPHOLOGY_BLUR_SIGMA,
                        "morphology_density_reference": MORPHOLOGY_DENSITY_REFERENCE,
                        "neural_updates_per_macro": NEURAL_UPDATES_PER_MACRO,
                        "communication_speed": COMMUNICATION_SPEED,
                        "elastic_strain_scale": ELASTIC_STRAIN_SCALE,
                        "elastic_strain_inputs_enabled": ELASTIC_STRAIN_INPUTS_ENABLED,
                        "growth_max": GROWTH_MAX,
                        "growth_threshold": GROWTH_THRESHOLD,
                        "chirality": CHIRALITY,
                        "damping": DAMPING_LOSS_FRACTION,
                        "material_e": MATERIAL_E,
                        "material_nu": MATERIAL_NU,
                        "material_hardening": MATERIAL_HARDENING,
                        "material_elasticity": MATERIAL_ELASTICITY,
                        "splat_radius": SPLAT_RADIUS,
                        "repulsion_strength": REPULSION_STRENGTH,
                        "repulsion_max_delta": REPULSION_MAX_DELTA,
                    },
                    indent=2,
                )
            )

    pool.shutdown()
    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


@app.get("/history")
def history() -> dict:
    """Every generation persisted so far — the frontend fetches this once
    on mount to backfill its chart/gallery state before the live
    websocket picks up from wherever the run currently is. Capped to
    MAX_HISTORY so the response stays bounded regardless of how long the
    on-disk log has grown."""
    if not HISTORY_PATH.is_file():
        return {"generations": []}
    lines = [line for line in HISTORY_PATH.read_text().splitlines() if line]
    return {"generations": [json.loads(line) for line in lines[-MAX_HISTORY:]]}


@app.get("/settings")
def get_settings() -> dict:
    """This run's own fixed settings — see the `settings` global's own
    comment. 404 only in the narrow startup window before
    _training_loop_body() has reached its own settings assignment (well
    before generation 0 finishes evaluating) — a browser tab connecting
    that early should treat this the same as any other transient
    connection hiccup and retry, not as "no run is configured"."""
    if settings is None:
        raise HTTPException(503, "training hasn't started yet")
    return settings


@app.get("/runs/{run_id}/settings")
def run_settings(run_id: str) -> dict:
    """Same shape as /settings, for one specific run — "current" is just
    /settings itself; anything else reads that archived run's own copy
    of settings.json (moved, not copied, by _archive_previous_run(),
    same as history.jsonl). 404 (not the transient 503 /settings itself
    can return) for an archived run with no settings.json at all — a run
    archived before this endpoint existed, not a startup race, so
    retrying wouldn't help; the frontend falls back to whatever that
    generation's own history record still carries inline for such a
    run — see net/trainingSocket.ts's own applyGeneration()."""
    if run_id == "current":
        return get_settings()
    run_dir = _run_dir_for_id(run_id)
    if run_dir is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    settings_path = run_dir / "settings.json"
    if not settings_path.is_file():
        raise HTTPException(404, f"run '{run_id}' has no settings.json (archived before this existed)")
    return json.loads(settings_path.read_text())


@app.get("/target/points")
def target_points() -> dict:
    """This server only ever has the one target it was launched with."""
    return {"points": target.points.tolist()}


@app.get("/targets/{name}/points")
def named_target_points(name: str) -> dict:
    """Like /target/points, but for *any* target by name — needed
    because an archived run (see /runs below) may have been trained
    against a different --target than this server's own (args.target),
    so the frontend's "load run" picker can't always rely on the fixed
    /target/points response when browsing history."""
    path = TARGETS_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"unknown target '{name}'")
    loaded = load_target(name)
    return {"points": loaded.points.tolist()}


def _find_latest_preview_prefix(images_dir: Path) -> Optional[str]:
    """Zero-padded generation prefix (e.g. "gen_00042") of the highest-
    generation-numbered debug image SET in `images_dir`, found by
    scanning actual files rather than trusting a run's best_meta.json —
    _save_generation_images() runs every generation, but best_meta.json
    only updates at --checkpoint-every boundaries, so a run stopped
    between checkpoints can have images saved past whatever generation
    number the metadata last reported. Filenames are zero-padded, so
    lexicographic sort is numeric sort.

    Returns the shared PREFIX, not one specific file, because
    _save_generation_images() always writes its "grown"/"agents"/
    "target" trio together, every generation (see that function's own
    docstring) — run_preview() and run_target_preview() below both derive
    their own filename from this same prefix, so a run's best-result
    thumbnail and its target thumbnail are always the exact same
    generation's own pair, not two independently-"latest" images that
    could theoretically disagree (they can't in practice, since target
    is fixed for a whole run and every generation resaves the identical
    raster, but deriving both from one shared prefix is the same
    single-source-of-truth reasoning regardless)."""
    if not images_dir.is_dir():
        return None
    for pattern in ("gen_*_agents.png", "gen_*_grown.png"):
        candidates = sorted(images_dir.glob(pattern))
        if candidates:
            # "gen_00042_agents.png" -> "gen_00042"
            return candidates[-1].name.rsplit("_", 1)[0]
    return None


def _run_dir_for_id(run_id: str) -> Optional[Path]:
    """Resolves an archived run id (an archive directory's own name — see
    _archive_previous_run()) to its path, rejecting anything that isn't
    literally a direct child of RUNS_DIR. run_id arrives as a URL path
    segment from the browser; this is the only thing standing between it
    and path traversal (a run_id of e.g. "../../etc")."""
    if not run_id or "/" in run_id or "\\" in run_id:
        return None
    candidate = RUNS_DIR / run_id
    if candidate.is_dir() and candidate.parent == RUNS_DIR:
        return candidate
    return None


@app.get("/runs")
def list_runs() -> dict:
    """Every archived run (checkpoints/runs/*) plus the current one (if
    it's at least gotten as far as writing its own settings — see the
    `settings` global's own comment), newest first — what the frontend's
    "load run" picker shows. Each entry is enough to render a list item
    (label, target, generation, fitness, preview thumbnail URL) without
    fetching that run's full history.

    Gated on `settings`, not `latest_generation_message` (this used to
    wait on the first COMPLETED generation before "Current run" appeared
    at all) — generation/bestFitness fall back to None until a generation
    actually has finished, same as they would for a run that genuinely
    has zero generations so far; the frontend already renders that as
    "—", same as it does today for these two fields on a freshly started
    archived-run-less server."""
    runs = []
    if settings is not None:
        runs.append(
            {
                "id": "current",
                "isLive": True,
                "label": "Current run",
                "target": settings["target"],
                "generation": latest_generation_message["generation"] if latest_generation_message else None,
                "bestFitness": latest_generation_message["allTimeBest"] if latest_generation_message else None,
                "previewUrl": "/runs/current/preview.png",
                "targetPreviewUrl": "/runs/current/target-preview.png",
            }
        )

    if RUNS_DIR.is_dir():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "best_meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            runs.append(
                {
                    "id": run_dir.name,
                    "isLive": False,
                    "label": run_dir.name,
                    "target": meta.get("target"),
                    "generation": meta.get("generation"),
                    "bestFitness": meta.get("fitness"),
                    "previewUrl": f"/runs/{run_dir.name}/preview.png",
                    "targetPreviewUrl": f"/runs/{run_dir.name}/target-preview.png",
                }
            )

    return {"runs": runs}


@app.get("/runs/{run_id}/history")
def run_history(run_id: str) -> dict:
    """Same shape as /history, for one specific run — "current" is just
    /history itself (the live, in-progress run); anything else reads
    that archived run's own copy of checkpoints/history.jsonl (moved, not
    copied, by _archive_previous_run())."""
    if run_id == "current":
        return history()
    run_dir = _run_dir_for_id(run_id)
    if run_dir is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    history_path = run_dir / "history.jsonl"
    if not history_path.is_file():
        return {"generations": []}
    lines = [line for line in history_path.read_text().splitlines() if line]
    return {"generations": [json.loads(line) for line in lines[-MAX_HISTORY:]]}


def _images_dir_for_run(run_id: str) -> Path:
    """Shared by run_preview() and run_image() below — "current" is the
    live, in-progress run's own IMAGES_DIR; anything else must resolve to
    an actual archived run (raises 404 otherwise, via _run_dir_for_id's
    path-traversal-safe validation)."""
    if run_id == "current":
        return IMAGES_DIR
    run_dir = _run_dir_for_id(run_id)
    if run_dir is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    return run_dir / "generation_images"


@app.get("/runs/{run_id}/preview.png")
def run_preview(run_id: str) -> FileResponse:
    """This run's best-result thumbnail — the winning rollout's own
    best-rotation raster (falling back to the raw grown positions — see
    _find_latest_preview_prefix()'s own docstring)."""
    images_dir = _images_dir_for_run(run_id)
    prefix = _find_latest_preview_prefix(images_dir)
    if prefix is None:
        raise HTTPException(404, "no preview image available yet")
    path = images_dir / f"{prefix}_agents.png"
    if not path.is_file():
        path = images_dir / f"{prefix}_grown.png"
    return FileResponse(path)


@app.get("/runs/{run_id}/target-preview.png")
def run_target_preview(run_id: str) -> FileResponse:
    """This run's target thumbnail — the same target raster
    run_preview()'s own best-result thumbnail was actually scored
    against (same generation prefix — see
    _find_latest_preview_prefix()'s own docstring), so the two sit
    side by side as a genuine, literally-comparable pair, not two
    independently-picked images."""
    images_dir = _images_dir_for_run(run_id)
    prefix = _find_latest_preview_prefix(images_dir)
    if prefix is None:
        raise HTTPException(404, "no target preview image available yet")
    path = images_dir / f"{prefix}_target.png"
    if not path.is_file():
        raise HTTPException(404, "no target preview image available yet")
    return FileResponse(path)


@app.get("/runs/{run_id}/images/{filename}")
def run_image(run_id: str, filename: str) -> FileResponse:
    """A specific gen_{N:05d}_{grown,aligned}.png from a specific run —
    the frontend's per-generation gallery builds this URL. Serves
    whichever run is currently being *viewed*, live or archived, not
    always the live one specifically."""
    if "/" in filename or "\\" in filename:
        raise HTTPException(404)
    path = _images_dir_for_run(run_id) / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connections.add(websocket)
    if latest_generation_message is not None:
        await websocket.send_json(latest_generation_message)

    try:
        while True:
            # No meaningful messages expected from the client — training
            # drives itself. This just blocks until the client disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connections.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    _setup()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

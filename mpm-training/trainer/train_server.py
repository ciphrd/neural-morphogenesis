"""FastAPI training server: stream generation results and serve run settings, archives, and diagnostic images."""
from __future__ import annotations
from config import CONFIG
from evolve import checkpoint_metadata

import asyncio
import json
import os
import shutil
import traceback
from time import perf_counter
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from chemical_channels import profiles_to_wire
from debug_images import save_grown_image, save_raster_image, save_polar_images
from density import DENSITY_MODEL_VERSION
from device import pick_device
from evolve import (
    CAPTURE_OFFSETS,
    CHECKPOINTS_DIR,
    RASTER_EXTENT,
    build_arg_parser,
    finalize_density_configuration,
    finalize_policy_configuration,
    get_weights,
    initialize_search,
    shape_settings,
    report_shape_capacity,
    run_generation,
    set_weights,
    validate_fitness_configuration,
)
from mpm_core import PARTICLE_MASS, VOL
from parallel_workers import build_pool
from rollout_snapshot import RolloutSnapshot
from history_index import compact_history, generation_record
from policy_parameters import mutation_scales, policy_hidden_dim
from raster import build_target_distance_field
from domain_fitness import FITNESS_MODEL_VERSION, target_mask
from simulation_settings import (
    CHEM_CHANNELS,
    CHEMICAL_CHANNEL_PROFILES,
    CHEMICAL_GRADIENT_INPUT_SCALE,
    CHEMICAL_VALUE_INPUT_MULTIPLIER,
    COMMUNICATION_SPEED,
    DAMPING_LOSS_FRACTION,
    DECAY,
    DEPOSIT_RATE,
    NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
    ELASTIC_STRAIN_SCALE,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    FIELD_N,
    FRICTION,
    GROWTH_DURATION_MACRO_STEPS,
    GROWTH_COMPRESSION_FEEDBACK,
    GROWTH_COMPRESSION_START,
    GROWTH_COMPRESSION_STOP,
    GROWTH_ANISOTROPY_AUTHORITY,
    GROWTH_MODEL_VERSION,
    INTERNAL_STATE_SPEED,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    NEURAL_UPDATES_PER_MACRO,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MAX_ENV_WRITE,
    MPM_ENABLED,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
    SAMPLE_SPACING,
)
from targets import TargetShape, available_targets, load_target
from update_rule import UpdateRule

parser = build_arg_parser()
parser.add_argument("--port", type=int, default=CONFIG["server"]["port"])
parser.add_argument(
    "--serve-only",
    action="store_true",
    help="serve the saved current run and archives without starting training or initializing a GPU",
)

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

    if args.serve_only:
        _restore_current_run()
        return

    finalize_policy_configuration(args)

    if args.seeds_per_candidate < 1:
        raise SystemExit("--seeds-per-candidate must be at least 1")
    if not 1 <= args.initial_particles <= args.particles // 2:
        raise SystemExit("--initial-particles seed cells must fit floor(--particles/2)")
    validate_fitness_configuration(args)
    if args.growth_steps is not None and not 0 <= args.growth_steps <= args.macro_steps:
        raise SystemExit("--growth-steps must be between 0 and --macro-steps")
    finalize_density_configuration(args)

    wgpu_device = pick_device()

    # Fixed for this server's lifetime (no target-switching endpoint).
    target = load_target(args.target)
    report_shape_capacity(args, target)
    # Fixed target fields are shared with workers and preview rendering.
    target_raster = target_mask(target, args.raster_resolution)
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

def _save_generation_images(generation: int, snapshot: RolloutSnapshot) -> dict[str, object] | None:
    """Save the worker's selected scoring pose without replay or rescoring."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = snapshot.evaluation
    positions = snapshot.positions
    agent_raster, breakdown = evaluation.raster, evaluation.breakdown

    prefix = f"gen_{generation:05d}"
    save_grown_image(positions, target.overlay_points(args.raster_resolution), IMAGES_DIR / f"{prefix}_grown.png")
    target_rgb = evaluation.target_color_raster
    if target_rgb is None and evaluation.target_raster is None:
        target_rgb = target.color_raster(args.raster_resolution)
    comparison_mask = evaluation.target_raster if evaluation.target_raster is not None else target_raster
    save_raster_image(target_rgb if target_rgb is not None else comparison_mask, IMAGES_DIR / f"{prefix}_target.png")
    if agent_raster is not None:
        save_raster_image(evaluation.color_raster if evaluation.color_raster is not None else agent_raster, IMAGES_DIR / f"{prefix}_agents.png")
    if target_rgb is not None and evaluation.color_raster is not None:
        save_raster_image(np.abs(evaluation.color_raster-target_rgb), IMAGES_DIR / f"{prefix}_diff.png")
    polar_info = save_polar_images(evaluation.polar, IMAGES_DIR, prefix) if evaluation.polar is not None else None
    if breakdown is None:
        return None
    return {
        "polar": polar_info,
        "total": breakdown.total,
        "coverage": breakdown.coverage,
        "spill": breakdown.spill,
        "boundary": breakdown.boundary,
        "crowding": breakdown.crowding,
        "color": breakdown.color,
        "angle": breakdown.angle,
        "rollout": snapshot.diagnostics,
    }

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not args.serve_only:
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

def _restore_current_run() -> None:
    """Restore read-only API state without mutating checkpoints or touching the GPU."""
    global settings, latest_generation_message, target

    if not SETTINGS_PATH.is_file():
        print("[train_server] serve-only mode: no saved current run; serving archives")
        return

    try:
        settings = json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"cannot read saved run settings from {SETTINGS_PATH}: {error}") from error

    embedded_target = settings.get("shapeTarget")
    if embedded_target and "mask" in embedded_target:
        target = TargetShape.from_wire(embedded_target)
    else:
        target_name = settings.get("target")
        if target_name not in available_targets():
            raise SystemExit(f"saved run refers to unavailable target {target_name!r}")
        target = load_target(target_name)

    latest_generation_message = compact_history(HISTORY_PATH)["latestGeneration"]
    generation = latest_generation_message["generation"] if latest_generation_message else "none"
    print(f"[train_server] serve-only mode: restored current run (latest generation: {generation})")

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
    policy_hidden = policy_hidden_dim(args.policy_architecture)

    num_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, args.population)
    # log_device=False — _setup() already logged the "[device] adapter:
    # ..." confirmation once, above, for this process's own wgpu_device;
    # see build_pool()'s own docstring for why it would otherwise repeat
    # that exact line a second time.
    population, optimizer = initialize_search(args, rng)
    pool = build_pool(num_workers, args.particle_capacity, target, target_raster, target_distance_field, args, log_device=False)
    update_rule = UpdateRule(CHEM_CHANNELS, args.policy_architecture)

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
        "initialParticleCount": args.initial_particles,
        "initialCondition": args.initial_condition,
        "initialConditionStrength": args.initial_condition_strength,
        "initialConditionChannel": args.initial_condition_channel,
        "densityModelVersion": DENSITY_MODEL_VERSION,
        "trainingDensityMultipliers": args.particle_densities,
        "densityAggregation": args.density_aggregation,
        "particleCapacity": args.particle_capacity,
        "particleMass": PARTICLE_MASS,
        "particleVolume": VOL,
        "chemicalValueInputMultiplier": CHEMICAL_VALUE_INPUT_MULTIPLIER,
        "chemicalGradientInputScale": CHEMICAL_GRADIENT_INPUT_SCALE,
        "macroSteps": args.macro_steps,
        "growthSteps": args.growth_steps,
        "substepsPerMacro": args.substeps_per_macro,
        "gravity": args.gravity,
        "spawnX": args.spawn_x,
        "spawnY": args.spawn_y,
        "channels": CHEM_CHANNELS,
        "baseResolution": FIELD_N,
        "chemicalChannelProfiles": profiles_to_wire(CHEMICAL_CHANNEL_PROFILES),
        "morphologyBlurSigma": MORPHOLOGY_BLUR_SIGMA,
        "morphologyDensityReference": MORPHOLOGY_DENSITY_REFERENCE,
        "neuralUpdatesPerMacro": NEURAL_UPDATES_PER_MACRO,
        "communicationSpeed": COMMUNICATION_SPEED,
        "internalStateSpeed": INTERNAL_STATE_SPEED,
        "elasticStrainScale": ELASTIC_STRAIN_SCALE,
        "elasticStrainInputsEnabled": ELASTIC_STRAIN_INPUTS_ENABLED,
        "hiddenDim": policy_hidden,
        "hiddenLayers": args.hidden_layers,
        "cellMemory": args.cell_memory,
        "policyArchitecture": args.policy_architecture,
        "chemicalCommunicationArchitecture": args.chemical_communication_architecture,
        "decay": DECAY,
        "depositRate": DEPOSIT_RATE,
        "normalizeDepositsByLocalDensity": NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
        "maxEnvWrite": MAX_ENV_WRITE,
        "sampleSpacing": SAMPLE_SPACING,
        "friction": FRICTION,
        "growthDuration": GROWTH_DURATION_MACRO_STEPS,
        "growthCompressionStart": GROWTH_COMPRESSION_START,
        "growthCompressionStop": GROWTH_COMPRESSION_STOP,
        "growthCompressionFeedback": GROWTH_COMPRESSION_FEEDBACK,
        "growthModelVersion": GROWTH_MODEL_VERSION,
        "domainGeometry": "triangle-vertices",
        **shape_settings(args, target),
        "growthAnisotropy": GROWTH_ANISOTROPY_AUTHORITY,
        # simulation_settings.py's own MPM_ENABLED (that constant's own
        # comment has the full "why" — a real testing/debug mode that
        # also skips MPM physics in the actual worker-pool population
        # evaluation driving fitness/selection, not just this broadcast
        # value) — sets the frontend's own starting toggle state (see
        # viewer/src/gpu/types.ts's own RunSettings.mpmEnabled), still
        # live-flippable there afterward regardless of this constant.
        "mpmEnabled": MPM_ENABLED,
        "damping": DAMPING_LOSS_FRACTION,
        "materialE": MATERIAL_E,
        "materialNu": MATERIAL_NU,
        "materialHardening": MATERIAL_HARDENING,
        "materialElasticity": MATERIAL_ELASTICITY,
        "splatRadius": SPLAT_RADIUS,
        "repulsionStrength": REPULSION_STRENGTH,
        "repulsionMaxDelta": REPULSION_MAX_DELTA,
        "population": args.population,
        "optimizer": args.optimizer,
        "cmaCovariance": args.cma_covariance if args.optimizer == "cma-es" else None,
        "seedsPerCandidate": args.seeds_per_candidate,
        "elites": args.elites if args.optimizer == "ga" else 0,
        "referenceCandidates": 1 if args.optimizer == "cma-es" else 0,
        "mutationSigma": args.mutation_sigma,
        "mutationFactors": list(getattr(args, "mutation_factors", [1.])),
        "initialWeights": str(args.initial_weights) if getattr(args, "initial_weights", None) else None,
        "fitnessFunction": getattr(args, "fitness_function", "multiscale"),
        "fitnessAlignment": "polar" if getattr(args, "fitness_function", "multiscale") == "polar" else ("svg" if target.svg_source is not None else getattr(args, "fitness_alignment", "raster")),
        "deterministicReference": getattr(args, "deterministic_reference", False),
        "rasterResolution": args.raster_resolution,
        "outsideWeight": args.outside_weight,
        "fitnessColorWeight": args.fitness_color_weight,
        "fitnessCoverageWeight": args.fitness_coverage_weight,
        "fitnessSpillWeight": args.fitness_spill_weight,
        "fitnessBoundaryWeight": args.fitness_boundary_weight,
        "fitnessCrowdingWeight": args.fitness_crowding_weight,
        "fitnessTemporalAggregation": "min",
        "fitnessCaptureFractions": [1-offset for offset in CAPTURE_OFFSETS],
        "fitnessModelVersion": FITNESS_MODEL_VERSION,
        "polarFitness": dict(CONFIG["polarFitness"]),
        "runSeed": args.seed,
        "totalGenerations": args.generations,
        "checkpointEvery": args.checkpoint_every,
    }
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))

    best_fitness = float("inf")
    best_weights = population[0]
    best_winner_seed = args.seed
    best_winner_density = args.particle_densities[0]
    best_density_fitnesses: dict[str, float] = {}
    best_evaluation_seeds: list[int] = []

    for generation in range(args.generations):
        generation_started = perf_counter()
        # Off the event loop thread — run_generation blocks for the whole
        # generation (waiting on pool.map() across every worker process,
        # see parallel_workers.py's own module docstring), and doing that
        # directly on the event loop thread would stall websocket message
        # flushing for as long as it takes.
        result = await asyncio.to_thread(
            run_generation, population, args, rng, pool, return_snapshot=True, optimizer=optimizer
        )
        population, fitnesses, winner_seed, winner_density, evaluation_seeds, density_fitnesses = result[:6]
        snapshot = result.snapshot

        winner_weights = result.winner_weights
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_weights = winner_weights.copy()
            best_winner_seed = winner_seed
            best_winner_density = winner_density
            best_density_fitnesses = dict(density_fitnesses)
            best_evaluation_seeds = list(evaluation_seeds)

        # Only PNG encoding and disk I/O remain; reuse the worker's scored state.
        preview_started = perf_counter()
        selected_snapshot_fitness = await asyncio.to_thread(
            _save_generation_images, generation, snapshot,
        )

        preview_seconds = perf_counter()-preview_started
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
            "particleDensityMultiplier": winner_density,
            "densityFitnesses": density_fitnesses,
            # Terms for the representative rollout's minimum-loss pose.
            # Candidate selection also aggregates across seeds/densities.
            "selectedSnapshotFitness": selected_snapshot_fitness,
            # Legacy wire alias retained for older consumers.
            "finalSnapshotFitness": selected_snapshot_fitness,
            # Shared by every candidate in this generation. The batch rotates
            # on the next generation; `seed` above is the winning candidate's
            # worst member, used for the single-rollout browser replay.
            "evaluationSeeds": evaluation_seeds,
            "optimizerState": optimizer.diagnostics() if optimizer is not None else None,
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
        checkpoint_started = perf_counter()
        if (generation + 1) % args.checkpoint_every == 0 or generation == args.generations - 1:
            np.save(CHECKPOINTS_DIR / "best.npy", best_weights)
            set_weights(update_rule, best_weights)
            (CHECKPOINTS_DIR / "best_weights.json").write_text(json.dumps(update_rule.export_weights()))
            (CHECKPOINTS_DIR / "best_meta.json").write_text(
                json.dumps(
                    checkpoint_metadata(args, target, generation, best_fitness, best_winner_seed, best_winner_density, best_density_fitnesses, best_evaluation_seeds, optimizer),
                    indent=2,
                )
            )

        checkpoint_seconds = perf_counter()-checkpoint_started
        elapsed = perf_counter()-generation_started
        timing = dict(snapshot.generation_timings)
        timing.update({"seconds": elapsed, "previewSeconds": preview_seconds,
                       "checkpointSeconds": checkpoint_seconds,
                       "otherSeconds": max(0., elapsed-timing["poolSeconds"]-timing["selectionSeconds"]-preview_seconds-checkpoint_seconds)})
        latest_generation_message["timing"] = timing
        print(f"[timing] generation {generation}: {elapsed:.2f}s; pool {timing['poolSeconds']:.2f}s; "
              f"preview {preview_seconds:.3f}s; longest rollout {timing['rollouts']['maxSeconds']:.2f}s")
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(latest_generation_message) + "\n")
        await broadcast(latest_generation_message)

    pool.shutdown()
    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")

def _history_payload(path: Path) -> dict:
    """Keep all compact timing records, but cap expensive weight snapshots."""
    generations = deque(maxlen=MAX_HISTORY)
    timings = {}
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrent append can leave an incomplete last line.
                generations.append(record)
                if record.get("timing") is not None:
                    timings[record["generation"]] = {"generation": record["generation"], "timing": record["timing"]}
    return {"generations": list(generations), "timings": [timings[key] for key in sorted(timings)]}

@app.get("/history")
def history(compact: bool = True) -> dict:
    return compact_history(HISTORY_PATH) if compact else _history_payload(HISTORY_PATH)

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
    """Read the complete current-schema settings for a saved run."""
    if run_id == "current":
        return get_settings()
    run_dir = _run_dir_for_id(run_id)
    if run_dir is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    settings_path = run_dir / "settings.json"
    if not settings_path.is_file():
        raise HTTPException(404, f"run '{run_id}' has no settings.json")
    settings = json.loads(settings_path.read_text())
    if settings["growthModelVersion"] != GROWTH_MODEL_VERSION:
        raise HTTPException(422, "Run schema does not match the current simulation")
    return settings

@app.get("/target/points")
def target_points() -> dict:
    """This server only ever has the one target it was launched with."""
    if target is None:
        raise HTTPException(503, "no saved current run is available")
    resolution = settings.get("rasterResolution", args.raster_resolution) if settings else args.raster_resolution
    return {"points": target.overlay_points(resolution).tolist()}

@app.get("/targets/{name}/points")
def named_target_points(name: str) -> dict:
    """Like /target/points, but for *any* target by name — needed
    because an archived run (see /runs below) may have been trained
    against a different --target than this server's own (args.target),
    so the frontend's "load run" picker can't always rely on the fixed
    /target/points response when browsing history."""
    if name not in available_targets():
        raise HTTPException(404, f"unknown target '{name}'")
    loaded = load_target(name)
    return {"points": loaded.overlay_points(args.raster_resolution).tolist()}

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
def run_history(run_id: str, compact: bool = True) -> dict:
    """Same shape as /history, for one specific run — "current" is just
    /history itself (the live, in-progress run); anything else reads
    that archived run's own copy of checkpoints/history.jsonl (moved, not
    copied, by _archive_previous_run())."""
    if run_id == "current":
        return history(compact)
    run_dir = _run_dir_for_id(run_id)
    if run_dir is None:
        raise HTTPException(404, f"unknown run '{run_id}'")
    history_path = run_dir / "history.jsonl"
    return compact_history(history_path) if compact else _history_payload(history_path)

@app.get("/runs/{run_id}/generations/{generation}")
def run_generation_record(run_id: str, generation: int) -> dict:
    run_dir = CHECKPOINTS_DIR if run_id == "current" else _run_dir_for_id(run_id)
    if run_dir is None:
        raise HTTPException(404, "unknown run")
    record = generation_record(run_dir / "history.jsonl", generation)
    if record is None:
        raise HTTPException(404, "unknown generation")
    return record


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
    uvicorn.run(app, host=CONFIG["server"]["bindHost"], port=args.port)

"""FastAPI server that runs evolve.py's training loop in the background
and hands each generation's winning weights to any connected browser over
a websocket. Same overall shape as trainer/backend/train_server.py, ported
to this project's own evolve.py (see that module for why its
run_generation() takes `device` instead of an `executor` — envnca trains
sequentially on one shared GPU, no CPU worker pool).

The frontend does the actual rollout itself, entirely client-side, on
WebGPU (envnca/frontend/src/gpu/) — this server's only job is the real,
authoritative (headless) evolutionary search, same as evolve.py's CLI.
Separate process/port from any other tool in this project.

Usage:
    python train_server.py --target circle --population 24 --generations 100 --port 8002
"""

from __future__ import annotations

import asyncio
import json
import shutil
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from debug_images import save_agents_image, save_raster_image
from device import pick_device
from environment import DECAY
from evolve import (
    CHECKPOINTS_DIR,
    build_arg_parser,
    get_weights,
    load_target,
    raster_extent,
    rollout,
    run_generation,
    set_weights,
)
from raster import build_target_distance_field, build_target_raster, training_raster_distance
from simulation import EDGE_MARGIN
from update_rule import HIDDEN_DIM, MAX_ACCEL, MAX_SPEED, MAX_STRAFE, UpdateRule

parser = build_arg_parser()
parser.add_argument("--port", type=int, default=8002)
args = parser.parse_args()

if not 1 <= args.elites <= args.population:
    raise SystemExit("--elites must be between 1 and --population")

device = pick_device()
print(f"device: {device}")

# Fixed for this server's lifetime (no target-switching endpoint) —
# loaded once at module level so both the training loop and the
# /target/points endpoint below share the exact same instance.
target = load_target(args.target, args.grid_size)
RASTER_EXTENT = raster_extent(args.grid_size)
# Also fixed for the whole run — see raster.build_target_raster()'s own
# docstring for why this is computed once here rather than on every
# rollout's training_raster_distance() call.
target_raster = build_target_raster(
    target.points,
    args.raster_resolution,
    RASTER_EXTENT,
    args.raster_sigma,
    half_size=target.texel_size(args.grid_size) / 2.0,
)
# Also fixed for the whole run — see raster.build_target_distance_field()'s
# own docstring for what this feeds (the outside-shape penalty).
target_distance_field = build_target_distance_field(target_raster)

# Every generation's full message (stats + weights) is appended here as
# it happens, so a browser tab that connects mid-run — or reconnects
# after a reload, or after this server process itself restarts — isn't
# starting blind; see /history and training_loop()'s append below.
HISTORY_PATH = CHECKPOINTS_DIR / "history.jsonl"
MAX_HISTORY = 500

# Where a previous run's history.jsonl/best.npy/best_meta.json get moved
# before this run starts overwriting those same fixed filenames.
RUNS_DIR = CHECKPOINTS_DIR / "runs"

# End-of-generation debug snapshots (see _save_generation_images()),
# served directly to the frontend via the /images static mount below —
# archived by _archive_previous_run() the same way history.jsonl/
# best.npy are, so a fresh run doesn't inherit a previous run's images
# under generation numbers that collide with its own. Created eagerly
# (not lazily inside _save_generation_images(), which also does this)
# because StaticFiles requires the directory to already exist at mount
# time, which happens below at import time, before any generation has
# actually run.
IMAGES_DIR = CHECKPOINTS_DIR / "generation_images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _archive_previous_run() -> None:
    """Moves the previous run's history.jsonl/best.npy/best_meta.json
    (plus IMAGES_DIR, if it has anything in it) into a timestamped
    RUNS_DIR subdirectory before this run starts overwriting them — same
    reasoning as trainer/backend/train_server.py's own version of this
    function, extended to cover IMAGES_DIR too so a fresh run doesn't
    silently inherit a previous run's PNGs under generation numbers that
    collide with its own. Only archives if there's actually something to
    keep (history.jsonl is the signal; a run that never got as far as
    its first generation has nothing worth archiving)."""
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

    for name in ("history.jsonl", "best.npy", "best_meta.json"):
        src = CHECKPOINTS_DIR / name
        if src.is_file():
            shutil.move(str(src), str(archive_dir / name))

    if IMAGES_DIR.is_dir() and any(IMAGES_DIR.iterdir()):
        shutil.move(str(IMAGES_DIR), str(archive_dir / "generation_images"))
        # /images is mounted (at import time) against this exact path —
        # StaticFiles resolves files from it lazily on each request, not
        # a directory handle captured at mount time, so recreating the
        # path here is enough for the mount to keep working once
        # _save_generation_images() starts writing into it again.
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train_server] archived previous run to {archive_dir}")


def _save_generation_images(
    generation: int, winner_weights: np.ndarray, winner_seed: int, update_rule: UpdateRule
) -> None:
    """Three PNGs per generation, for visually sanity-checking the
    raster fitness (see raster.py) against what actually happened:
    - `..._target.png` — the target's own raster (fixed all run, saved
      again every generation anyway, for a self-contained per-generation
      folder — see IMAGES_DIR's own comment).
    - `..._agents.png` — the winning rollout's raw, un-rotated replay
      positions (agents in white, target in red) in native grid-pixel
      space, the same drawing convention the deleted render.py used —
      "what actually happened, pose included."
    - `..._raster.png` — that same rollout's positions, but rotation-
      aligned to the target's pose and rasterized — the actual raster
      training's fitness function compared against target_raster, as
      opposed to the raw (unaligned) positions in the agents image.

    Re-runs the winner's rollout (same weights + seed run_generation
    already scored it with, so this reproduces the identical positions —
    see rollout()'s own docstring on reproducibility) since fitnesses
    from the population loop don't carry the final positions along with
    them — the same "replay the winner once more" pattern the WebGPU
    frontend itself already relies on for visualization."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    _, positions = rollout(
        winner_weights,
        target,
        target_raster,
        target_distance_field,
        args,
        winner_seed,
        device,
        update_rule,
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
    save_raster_image(target_raster, IMAGES_DIR / f"{prefix}_target.png")
    save_agents_image(positions, target.points, args.grid_size, IMAGES_DIR / f"{prefix}_agents.png")
    if agent_raster is not None:
        save_raster_image(agent_raster, IMAGES_DIR / f"{prefix}_raster.png")


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
# Serves gen_{N:05d}_{target,agents,raster}.png directly by filename —
# see _save_generation_images() for what's actually in here. Plain
# <img>/SVG <image> tags don't need CORS to render a cross-origin image
# (that only gates *programmatic* access to the response, e.g. reading
# pixels back out via canvas), so this doesn't need any extra handling
# beyond the blanket CORSMiddleware already applied above.
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

connections: set[WebSocket] = set()
latest_generation_message: Optional[dict] = None


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
    global latest_generation_message

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    update_rule = UpdateRule(num_channels=args.channels).to(device)
    population = [get_weights(UpdateRule(num_channels=args.channels)) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    _archive_previous_run()
    # Fresh log for this run — generation numbers restart at 0 each
    # invocation, so appending onto a previous run's log would collide
    # rather than continue a meaningful timeline. Whatever was here
    # before is now safely under RUNS_DIR, not discarded.
    HISTORY_PATH.write_text("")
    best_fitness = float("inf")
    best_weights = population[0]

    for generation in range(args.generations):
        # Still off the event loop thread even though this runs
        # sequentially (no worker pool here, unlike trainer's server) —
        # run_generation blocks this thread for the whole generation, and
        # doing that directly on the event loop thread would stall
        # websocket message flushing for as long as it takes.
        population, fitnesses, winner_seed = await asyncio.to_thread(
            run_generation,
            population,
            target,
            target_raster,
            target_distance_field,
            args,
            rng,
            device,
            update_rule,
        )

        winner_weights = population[0]
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_weights = winner_weights.copy()

        # Also off the event loop thread — this re-runs the winner's
        # rollout once more (see _save_generation_images()'s own
        # docstring for why) plus PNG encoding/disk I/O, neither of
        # which should stall websocket message flushing either.
        await asyncio.to_thread(_save_generation_images, generation, winner_weights, winner_seed, update_rule)

        finite = [f for f in fitnesses if np.isfinite(f)]
        print(
            f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(finite) if finite else float('inf'):.4f}  "
            f"worst {fitnesses[-1]:.4f}  (all-time best {best_fitness:.4f})"
        )

        # update_rule's currently-loaded weights are whatever the last
        # candidate run_generation evaluated used — load the winner's
        # before exporting.
        set_weights(update_rule, winner_weights, device)

        latest_generation_message = {
            "type": "generation",
            "generation": generation,
            "best": fitnesses[0],
            "mean": float(np.mean(finite)) if finite else float("inf"),
            "worst": fitnesses[-1],
            "allTimeBest": best_fitness,
            # This generation's winning rollout's own seed (SimulationConfig's
            # `seed` — what the frontend re-seeds its replay jitter with),
            # NOT the top-level run seed the whole training invocation was
            # started with — that one's runSeed, below. Two different
            # things that happen to both be "a seed."
            "seed": winner_seed,
            "weights": update_rule.export_weights(),
            "gridWidth": args.grid_size,
            "gridHeight": args.grid_size,
            "channels": args.channels,
            "agentCount": args.agents,
            "spawnSpread": args.spawn_spread,
            "steps": args.steps,
            "decay": DECAY,
            "maxSpeed": MAX_SPEED,
            "maxAccel": MAX_ACCEL,
            "maxStrafe": MAX_STRAFE,
            "edgeMargin": EDGE_MARGIN,
            "hiddenDim": HIDDEN_DIM,
            # Everything above is simulation config — what a WebGPU replay
            # needs to reproduce this generation's winner (SimulationConfig).
            # Everything below is training-specific — describes the search
            # itself, not any one rollout, and doesn't change generation to
            # generation, but is sent on every message anyway (same
            # redundant-but-self-contained convention as the simulation
            # fields above) so a tab that connects mid-run doesn't need a
            # separate fetch to know what the run was actually configured
            # with.
            "target": args.target,
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
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(latest_generation_message) + "\n")
        await broadcast(latest_generation_message)

        if (generation + 1) % args.checkpoint_every == 0 or generation == args.generations - 1:
            np.save(CHECKPOINTS_DIR / "best.npy", best_weights)
            (CHECKPOINTS_DIR / "best_meta.json").write_text(
                json.dumps(
                    {
                        "generation": generation,
                        "fitness": best_fitness,
                        "target": args.target,
                        "agents": args.agents,
                        "steps": args.steps,
                        "grid_size": args.grid_size,
                        "channels": args.channels,
                        "spawn_spread": args.spawn_spread,
                        "population": args.population,
                        "elites": args.elites,
                        "mutation_sigma": args.mutation_sigma,
                        "raster_resolution": args.raster_resolution,
                        "raster_sigma": args.raster_sigma,
                        "outside_weight": args.outside_weight,
                        "seed": args.seed,
                        "winner_seed": winner_seed,
                    },
                    indent=2,
                )
            )

    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


@app.get("/history")
def history() -> dict:
    """Every generation persisted so far — the frontend fetches this once
    on mount to backfill its chart/replay state before the live websocket
    picks up from wherever the run currently is. Capped to MAX_HISTORY so
    the response stays bounded regardless of how long the on-disk log has
    grown."""
    if not HISTORY_PATH.is_file():
        return {"generations": []}
    lines = [line for line in HISTORY_PATH.read_text().splitlines() if line]
    return {"generations": [json.loads(line) for line in lines[-MAX_HISTORY:]]}


@app.get("/target/points")
def target_points() -> dict:
    """This server only ever has the one target it was launched with."""
    return {"points": target.points.tolist()}


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

    uvicorn.run(app, host="0.0.0.0", port=args.port)

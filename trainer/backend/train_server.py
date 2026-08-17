"""FastAPI server that runs evolve.py's training loop in the background
and hands each generation's winning weights to any connected browser
over a websocket. See trainer/README.md's "Training" section for
context on the training approach itself.

Unlike an earlier version of this file, this server never runs a
rollout itself for visualization — the frontend does that entirely
client-side (trainer/frontend/src/sim/), replaying the winning weights
with its own from-scratch reimplementation of sensing, the update rule,
and physics, animated at full requestAnimationFrame speed with no
per-frame network dependency. This server's only job is the actual
(headless, authoritative) evolutionary search, same as evolve.py's CLI.

Separate process and port from main.py's interactive server —
training runs independently of whether the interactive tool is open,
and can never affect main.py's state.

Usage:
    python train_server.py --target circle --population 24 --generations 100 --port 8001
"""

from __future__ import annotations

import asyncio
import json
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from cell_state import INITIAL_ENERGY
from evolve import (
    CHECKPOINTS_DIR,
    _init_worker,
    build_arg_parser,
    get_weights,
    load_target,
    run_generation,
    set_weights,
)
from physics import (
    CLEANUP_CONVERGENCE_TOL,
    CLEANUP_ITERATIONS,
    CONTACT_DISTANCE,
    ENABLE_COLLISION,
    RADIUS,
    SETTLE_CONVERGENCE_TOL,
    SETTLE_ITERATIONS,
    SETTLE_STIFFNESS,
    TENSION_RANGE,
    TENSION_STIFFNESS,
)
from update_rule import (
    CHEMICAL_CLIP,
    ENERGY_INJECTION,
    ENERGY_INJECTION_NOISE,
    MAX_ACCEL,
    MAX_ENERGY,
    MAX_NODES,
    MAX_SPEED,
    MIN_SPLIT_ENERGY,
    SENSING_SIGMA,
    UpdateRule,
)

# Sent once per generation message (see PHYSICS_CONFIG usage below)
# rather than re-read from the physics/update_rule modules on every
# message — these are module-level constants, not runtime state, so
# there's nothing to gain from re-fetching them every generation, and
# building the dict once makes it obvious at a glance that this is meant
# to be the frontend's *only* source for these values (sim/physics.ts no
# longer hardcodes its own copies — see that file's own comment).
PHYSICS_CONFIG = {
    "radius": RADIUS,
    "contactDistance": CONTACT_DISTANCE,
    "tensionRange": TENSION_RANGE,
    "tensionStiffness": TENSION_STIFFNESS,
    "settleStiffness": SETTLE_STIFFNESS,
    "settleIterations": SETTLE_ITERATIONS,
    "cleanupIterations": CLEANUP_ITERATIONS,
    "settleConvergenceTol": SETTLE_CONVERGENCE_TOL,
    "cleanupConvergenceTol": CLEANUP_CONVERGENCE_TOL,
    "collisionEnabled": ENABLE_COLLISION,
}

parser = build_arg_parser()
parser.add_argument("--port", type=int, default=8001)
args = parser.parse_args()

if not 1 <= args.elites <= args.population:
    raise SystemExit("--elites must be between 1 and --population")

# Fixed for this server's lifetime (no target-switching endpoint, unlike
# main.py) — loaded once at module level so both the training loop and
# the /target/points endpoint below share the exact same instance.
target = load_target(args.target)

# Every generation's full message (stats + weights) is appended here as
# it happens, so a browser tab that connects mid-run — or reconnects
# after a reload, or after this server process itself restarts — isn't
# starting blind; see /history and training_loop()'s append below.
# Capped the same as the frontend's own retention (trainingSocket.ts)
# when *served*, even though the on-disk log itself is left to grow for
# the life of one run.
HISTORY_PATH = CHECKPOINTS_DIR / "history.jsonl"
MAX_HISTORY = 500

# Where a previous run's history.jsonl/best.npy/best_meta.json get moved
# before this run starts overwriting those same fixed filenames — see
# _archive_previous_run()'s own docstring for why this exists.
RUNS_DIR = CHECKPOINTS_DIR / "runs"


def _archive_previous_run() -> None:
    """Moves the previous run's history.jsonl/best.npy/best_meta.json
    into a timestamped RUNS_DIR subdirectory before this run starts
    overwriting them. Without this, every restart of train_server.py
    silently discarded whatever the last run had produced —
    training_loop() always truncates HISTORY_PATH fresh (generation
    numbers restart at 0 each invocation, so appending onto a previous
    run's log would collide rather than continue a meaningful timeline),
    and best.npy/best_meta.json share one fixed filename with no per-run
    distinction at all. Only archives if there's actually something to
    keep — a missing or empty history.jsonl (the very first run ever, or
    a previous run that crashed before its first generation) has nothing
    worth preserving."""
    if not HISTORY_PATH.is_file() or HISTORY_PATH.stat().st_size == 0:
        return

    # best_meta.json (if the previous run got far enough to write one)
    # carries its target/generation, which makes a far more useful
    # directory name than a bare timestamp — falls back to just the
    # timestamp if that metadata isn't there for some reason.
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

    print(f"[train_server] archived previous run to {archive_dir}")


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
    so nothing else would ever surface an uncaught exception here. Per-
    candidate failures shouldn't reach this point at all now (see
    evolve.py's rollout()/_rollout_worker() guards) — this is a backstop
    for anything else in the loop itself (checkpointing, broadcasting,
    ...), which is rarer but not impossible."""
    try:
        await _training_loop_body()
    except Exception:
        print("[train_server] training_loop crashed — training has stopped:")
        traceback.print_exc()


async def _training_loop_body() -> None:
    global latest_generation_message

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = UpdateRule()
    population = [get_weights(UpdateRule()) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    _archive_previous_run()
    # Fresh log for this run — generation numbers restart at 0 each
    # invocation, so appending onto a previous run's log would collide
    # rather than continue a meaningful timeline. Whatever was here
    # before is now safely under RUNS_DIR, not discarded.
    HISTORY_PATH.write_text("")
    best_fitness = float("inf")
    best_weights = population[0]

    # Population members are independent rollouts — evaluate them across
    # worker processes (one per core by default) rather than one at a
    # time. The executor is created once and reused for the whole
    # training run, not per-generation, so it isn't paying process-spawn
    # cost every iteration.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
        for generation in range(args.generations):
            # Still off the event loop thread even though the real
            # parallelism now happens in worker processes: run_generation
            # blocks this thread on executor.map()'s results, and doing
            # that from the event loop thread directly would stall
            # websocket message flushing for the whole generation.
            population, fitnesses = await asyncio.to_thread(
                run_generation, population, target, args, rng, model, executor
            )

            winner_weights = population[0]
            if fitnesses[0] < best_fitness:
                best_fitness = fitnesses[0]
                best_weights = winner_weights.copy()

            print(
                f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(fitnesses):.4f}  "
                f"worst {fitnesses[-1]:.4f}  (all-time best {best_fitness:.4f})"
            )

            # model's currently-loaded weights are whatever the last candidate
            # run_generation evaluated used — load the winner's before exporting.
            set_weights(model, winner_weights)

            latest_generation_message = {
                "type": "generation",
                "generation": generation,
                "best": fitnesses[0],
                "mean": float(np.mean(fitnesses)),
                "worst": fitnesses[-1],
                "allTimeBest": best_fitness,
                "weights": model.export_weights(),
                "steps": args.steps,
                "maxNodes": MAX_NODES,
                "sensingSigma": SENSING_SIGMA,
                "initialEnergy": INITIAL_ENERGY,
                "minSplitEnergy": MIN_SPLIT_ENERGY,
                "maxEnergy": MAX_ENERGY,
                "energyInjection": ENERGY_INJECTION,
                "energyInjectionNoise": ENERGY_INJECTION_NOISE,
                "chemicalClip": CHEMICAL_CLIP,
                "maxAccel": MAX_ACCEL,
                "maxSpeed": MAX_SPEED,
                "physics": PHYSICS_CONFIG,
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
                            "steps": args.steps,
                            "population": args.population,
                            "elites": args.elites,
                            "mutation_sigma": args.mutation_sigma,
                            "damage_prob": args.damage_prob,
                            "damage_fraction": args.damage_fraction,
                            "seed": args.seed,
                        },
                        indent=2,
                    )
                )

    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


@app.get("/history")
def history() -> dict:
    """Every generation persisted so far (training_loop's append to
    HISTORY_PATH) — the frontend fetches this once on mount to backfill
    its chart/replay state before the live websocket picks up from
    wherever the run currently is, so a fresh or reloaded tab shows the
    whole run, not just what's happened since it connected. Capped to
    the same MAX_HISTORY the frontend itself retains, so the response
    stays bounded regardless of how long the on-disk log has grown."""
    if not HISTORY_PATH.is_file():
        return {"generations": []}
    lines = [line for line in HISTORY_PATH.read_text().splitlines() if line]
    return {"generations": [json.loads(line) for line in lines[-MAX_HISTORY:]]}


@app.get("/target/points")
def target_points() -> dict:
    """Same shape as main.py's endpoint of the same name, so the frontend
    can reuse GraphRenderer's targetPoints overlay unchanged. This
    server only ever has the one target it was launched with."""
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

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
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
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
from update_rule import (
    ENERGY_INJECTION,
    ENERGY_INJECTION_NOISE,
    MAX_ENERGY,
    MAX_NODES,
    MIN_SPLIT_ENERGY,
    SENSING_SIGMA,
    UpdateRule,
)

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
    # Fresh log for this run — generation numbers restart at 0 each
    # invocation, so appending onto a previous run's log would collide
    # rather than continue a meaningful timeline.
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

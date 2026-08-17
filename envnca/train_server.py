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
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from constants import DECAY, HIDDEN_DIM, MAX_ACCEL, MAX_SPEED, MAX_STRAFE
from debug_images import save_agents_image, save_raster_image
from device import pick_device
from evolve import (
    CHECKPOINTS_DIR,
    TARGETS_DIR,
    build_arg_parser,
    get_weights,
    load_target,
    raster_extent,
    rollout,
    run_generation,
    set_weights,
)
from raster import build_target_distance_field, build_target_raster, training_raster_distance
from target import TargetShape
from update_rule import UpdateRule

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
# served to the frontend via GET /runs/{run_id}/images/{filename} below
# (run_id="current" reads straight from here; anything else reads an
# archived run's own copy) — archived by _archive_previous_run() the
# same way history.jsonl/best.npy are, so a fresh run doesn't inherit a
# previous run's images under generation numbers that collide with its
# own. Created eagerly (not lazily inside _save_generation_images(),
# which also does this) just so it reliably exists from the moment this
# module is imported, before any generation has actually run.
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
        # Recreated immediately — _save_generation_images() expects this
        # directory to already exist the next time it's called.
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
                        # Not CLI args (nothing above this line is) — the
                        # constants.py values this run actually simulated
                        # under. Every generation in history.jsonl already
                        # carries these too (see latest_generation_message
                        # above), so a full replay never depended on this
                        # file for them; recorded here as well purely so
                        # this "metadata" file is a complete, standalone
                        # description of the run on its own, without
                        # requiring a history.jsonl read to answer "what
                        # was this run's MAX_SPEED".
                        "decay": DECAY,
                        "hidden_dim": HIDDEN_DIM,
                        "max_speed": MAX_SPEED,
                        "max_accel": MAX_ACCEL,
                        "max_strafe": MAX_STRAFE,
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


@app.get("/targets/{name}/points")
def named_target_points(name: str, grid_size: int) -> dict:
    """Like /target/points, but for *any* target by name at any grid
    size — needed because an archived run (see /runs below) may have
    been trained against a different --target than this server's own
    (args.target), so the frontend's "Load run" picker can't always rely
    on the fixed /target/points response when browsing history."""
    path = TARGETS_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"unknown target '{name}'")
    loaded = TargetShape.from_export(json.loads(path.read_text()), grid_size)
    return {"points": loaded.points.tolist()}


def _find_latest_preview(images_dir: Path) -> Optional[Path]:
    """Highest-generation-numbered raster (falling back to agents) image
    in `images_dir`, found by scanning actual files rather than trusting
    a run's best_meta.json — _save_generation_images() runs every
    generation, but best_meta.json only updates at --checkpoint-every
    boundaries, so a run stopped between checkpoints can have images
    saved past whatever generation number the metadata last reported.
    Filenames are zero-padded (gen_00042_....png), so lexicographic sort
    is numeric sort."""
    if not images_dir.is_dir():
        return None
    for pattern in ("gen_*_raster.png", "gen_*_agents.png"):
        candidates = sorted(images_dir.glob(pattern))
        if candidates:
            return candidates[-1]
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
    training has produced at least one generation so far), newest first
    — what the frontend's "Load run" picker shows. Each entry is enough
    to render a list item (label, target, generation, fitness, preview
    thumbnail URL) without fetching that run's full history."""
    runs = []
    if latest_generation_message is not None:
        runs.append(
            {
                "id": "current",
                "isLive": True,
                "label": "Current run",
                "target": latest_generation_message["target"],
                "generation": latest_generation_message["generation"],
                "bestFitness": latest_generation_message["allTimeBest"],
                "previewUrl": "/runs/current/preview.png",
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
                }
            )

    return {"runs": runs}


@app.get("/runs/{run_id}/history")
def run_history(run_id: str) -> dict:
    """Same shape as /history, for one specific run — "current" is just
    /history itself (the live, in-progress run); anything else reads
    that archived run's own copy of checkpoints/history.jsonl (moved,
    not copied, by _archive_previous_run() — see its own docstring, so
    this is the exact same file the live run itself would have served
    from at the point it got archived)."""
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
    path = _find_latest_preview(_images_dir_for_run(run_id))
    if path is None:
        raise HTTPException(404, "no preview image available yet")
    return FileResponse(path)


@app.get("/runs/{run_id}/images/{filename}")
def run_image(run_id: str, filename: str) -> FileResponse:
    """A specific gen_{N:05d}_{target,agents,raster}.png from a specific
    run — net/images.ts's generationImageUrl() builds this URL. Replaces
    an earlier flat StaticFiles mount at /images (which only ever served
    the live run's own IMAGES_DIR) now that the Snapshot panel and the
    fitness chart's hover tooltip need to show images from whichever run
    is currently being *viewed*, live or archived, not always the live
    one specifically."""
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

    uvicorn.run(app, host="0.0.0.0", port=args.port)

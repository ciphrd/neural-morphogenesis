"""FastAPI server that runs train_gd.py's gradient-descent training loop
in the background and hands each episode's weights to any connected
browser over a websocket — the gradient-descent counterpart to
train_server.py, letting the exact same frontend (envnca/frontend/)
watch/replay a GD run exactly the way it already watches an ES one.
Same message shape, same REST surface (/history, /target/points,
/targets/{name}/points, /runs, /runs/{id}/history, /runs/{id}/preview.png,
/runs/{id}/images/{filename}, /ws), same checkpoints/ directory
(history.jsonl, generation_images/, runs/) — launch this instead of
train_server.py on the same port and the frontend can't tell the
difference; no frontend change was needed for this at all.

"Generation" in the frontend's own vocabulary maps to "episode" here —
see train_gd.py's own module docstring for what an episode actually is
(one seed-to-`--steps` rollout, split into `--bptt-steps` truncated-BPTT
windows, one optimizer.step() per window). Each episode gets exactly one
broadcast message, the same "one message per unit of training progress"
granularity evolve.py's own generations have. best/mean/worst are
computed across that episode's own *windows* instead of a population
(GD trains one set of weights directly, no population to score) — a
window that ends up far from the target after backpropagating still
counts toward mean/worst, the same way a bad ES candidate still counts
toward its generation's.

Usage:
    python train_server_gd.py --target circle --epochs 200 --steps 200 --bptt-steps 20 --port 8002
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

from constants import (
    DECAY,
    HIDDEN_DIM,
    MAX_ACCEL,
    MAX_ENV_WRITE,
    MAX_SPEED,
    MAX_STRAFE,
    REPULSION_SIGMA,
    REPULSION_STRENGTH,
)
from debug_images import save_agents_image, save_raster_image
from device import pick_device
from environment import Environment
from evolve import CHECKPOINTS_DIR, TARGETS_DIR, load_target, raster_extent
from raster import build_target_distance_field, build_target_raster, training_raster_distance
from raster_torch import target_rasters_to_torch, training_raster_distance_torch
from simulation import Simulation
from target import TargetShape
from train_gd import build_arg_parser
from update_rule import UpdateRule

parser = build_arg_parser()
parser.add_argument("--port", type=int, default=8002)
args = parser.parse_args()

if args.bptt_steps < 1:
    raise SystemExit("--bptt-steps must be >= 1")

device = pick_device()
print(f"device: {device}")

# Fixed for this server's lifetime (no target-switching endpoint) —
# loaded once at module level so both the training loop and the
# /target/points endpoint below share the exact same instance. Same
# precompute-once reasoning as train_server.py's own target_raster/
# target_distance_field (target points never change episode to episode);
# the torch versions are what the differentiable loss actually uses,
# converted once via raster_torch.target_rasters_to_torch — see that
# function's own docstring for why the numpy originals still get built
# first (build_target_raster()'s hard-set-to-1.0 target texels never
# need gradients, so there's no reason to reimplement that in torch).
target = load_target(args.target, args.grid_size)
RASTER_EXTENT = raster_extent(args.grid_size)
target_raster = build_target_raster(
    target.points,
    args.raster_resolution,
    RASTER_EXTENT,
    args.raster_sigma,
    half_size=target.texel_size(args.grid_size) / 2.0,
)
target_distance_field = build_target_distance_field(target_raster)
target_raster_t, target_distance_field_t = target_rasters_to_torch(target_raster, target_distance_field, device)
target_points_t = torch.tensor(target.points, dtype=torch.float32, device=device)

# Deliberately the *same* paths train_server.py uses (both import
# CHECKPOINTS_DIR from evolve.py) — switching between running this
# server and train_server.py on the same port is meant to feel like
# switching --target on either one: the previous run gets archived, a
# fresh history/image set starts, and the frontend's run picker shows
# both training methods' runs in one unified list. See
# _archive_previous_run()'s own docstring for the file-format
# implication of sharing this directory.
HISTORY_PATH = CHECKPOINTS_DIR / "history.jsonl"
MAX_HISTORY = 500
RUNS_DIR = CHECKPOINTS_DIR / "runs"
IMAGES_DIR = CHECKPOINTS_DIR / "generation_images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _archive_previous_run() -> None:
    """Identical in spirit and outcome to train_server.py's own version
    of this function (see that module's docstring for why this exists at
    all) — kept as its own copy here rather than a shared import,
    matching this project's established convention for the ES/GD split
    elsewhere (raster.py/raster_torch.py, evolve.py/train_gd.py: parallel
    sibling modules, not a shared base). The file-selection logic itself
    is training-method-agnostic — every plain file directly under
    CHECKPOINTS_DIR gets archived (globbed, not a hardcoded filename
    list) — so this works correctly whether the run being superseded was
    this server's own GD checkpoint (best_weights.pt) or an ES one
    (best.npy) left behind by train_server.py running earlier against
    this same directory."""
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
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train_server_gd] archived previous run to {archive_dir}")


def _save_episode_images(episode: int, positions_np: np.ndarray) -> None:
    """Same three PNGs train_server.py's own _save_generation_images()
    produces (see that function's docstring for what each one is), but
    no re-run needed here the way the ES version needs one: this
    episode's own final rollout positions are already sitting in memory
    by the time this is called (the truncated-BPTT loop in _run_episode()
    below never throws them away), so there's nothing to reconstruct."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    _, agent_raster = training_raster_distance(
        positions_np,
        target.points,
        target_raster,
        target_distance_field,
        args.raster_resolution,
        RASTER_EXTENT,
        args.raster_sigma,
        outside_weight=args.outside_weight,
        track_best_raster=True,
    )
    prefix = f"gen_{episode:05d}"
    save_raster_image(target_raster, IMAGES_DIR / f"{prefix}_target.png")
    save_agents_image(positions_np, target.points, args.grid_size, IMAGES_DIR / f"{prefix}_agents.png")
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
    """Thin wrapper so a crash anywhere in the run is loud and visible —
    same reasoning as train_server.py's own version."""
    try:
        await _training_loop_body()
    except Exception:
        print("[train_server_gd] training_loop crashed — training has stopped:")
        traceback.print_exc()


def _run_episode(
    update_rule: UpdateRule, optimizer: torch.optim.Optimizer, epoch: int
) -> tuple[float, float, float, int, np.ndarray, float, float, float]:
    """Runs one full episode — fresh Environment/AgentState, `args.steps`
    total simulation steps split into `args.bptt_steps`-sized truncated-
    BPTT windows, one loss + backward + optimizer.step() per window, the
    graph severed (state detached) between windows — see train_gd.py's
    own module docstring for the full reasoning. Synchronous/blocking,
    called via asyncio.to_thread by _training_loop_body() below, same
    reasoning train_server.py's own run_generation call has for that.

    Returns (best, mean, worst, seed, final_positions, grad_norm,
    hidden_sat_frac, max_input_abs): best/mean/worst computed across this
    episode's own window losses (see this module's own docstring for why
    there's no population to compute them across instead); `seed` is
    this episode's own agent-jitter seed — reported for the same reason
    evolve.py's winner_seed is, so a browser replay can reproduce this
    exact rollout's initial jitter; `final_positions` is the last
    window's final agent positions (numpy), already at hand with nothing
    to re-run for _save_episode_images(); the last three are a
    diagnostic snapshot from this episode's *last* window's *last*
    forward pass (see UpdateRule.record_diagnostics's own docstring for
    what they're checking — whether GD's plateau is caused by a
    saturated hidden layer, a failure mode ES is structurally immune
    to)."""
    seed = args.seed + epoch + 1
    rng = torch.Generator().manual_seed(seed)
    env = Environment(height=args.grid_size, width=args.grid_size, channels=args.channels, device=device)
    sim = Simulation(env, update_rule, device, population=args.agents, spawn_spread=args.spawn_spread, rng=rng)

    step = 0
    window_losses: list[float] = []
    final_positions = np.zeros((0, 2), dtype=np.float32)
    grad_norm_value = float("nan")
    hidden_sat_frac = float("nan")
    max_input_abs = float("nan")

    while step < args.steps:
        window_end = min(step + args.bptt_steps, args.steps)
        for _ in range(window_end - step):
            sim.step()
        step = window_end

        positions = sim.agents.positions
        if positions.shape[0] == 0 or not torch.isfinite(positions).all():
            # A diverged episode should end with a terrible score, not
            # crash the run — same backstop evolve.py's own rollout() has.
            window_losses.append(float("inf"))
            break

        loss = training_raster_distance_torch(
            positions,
            target_points_t,
            target_raster_t,
            target_distance_field_t,
            args.raster_resolution,
            RASTER_EXTENT,
            args.raster_sigma,
            outside_weight=args.outside_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(update_rule.parameters(), args.grad_clip)
        if torch.isfinite(grad_norm):
            optimizer.step()
        grad_norm_value = float(grad_norm.item())
        if update_rule.last_hidden is not None:
            hidden_sat_frac = float((update_rule.last_hidden.abs() > 0.99).float().mean().item())
            max_input_abs = float(update_rule.last_input.abs().max().item())

        # Sever the graph before the next window — see train_gd.py's
        # own module docstring for why.
        sim.agents.positions = sim.agents.positions.detach()
        sim.agents.velocity = sim.agents.velocity.detach()
        sim.env.grid = sim.env.grid.detach()

        window_losses.append(float(loss.item()))
        final_positions = positions.detach().cpu().numpy()

    finite = [w for w in window_losses if np.isfinite(w)]
    best = min(finite) if finite else float("inf")
    mean = float(np.mean(finite)) if finite else float("inf")
    worst = max(window_losses) if window_losses else float("inf")
    return best, mean, worst, seed, final_positions, grad_norm_value, hidden_sat_frac, max_input_abs


async def _training_loop_body() -> None:
    global latest_generation_message

    torch.manual_seed(args.seed)
    update_rule = UpdateRule(num_channels=args.channels).to(device)
    optimizer = torch.optim.Adam(update_rule.parameters(), lr=args.lr)
    # See UpdateRule's own docstring on record_diagnostics for what this
    # is checking and why — logged per episode below.
    update_rule.record_diagnostics = True

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    _archive_previous_run()
    # Fresh log for this run, same convention as train_server.py's own.
    HISTORY_PATH.write_text("")
    all_time_best = float("inf")

    for epoch in range(args.epochs):
        best, mean, worst, seed, final_positions, grad_norm_value, hidden_sat_frac, max_input_abs = (
            await asyncio.to_thread(_run_episode, update_rule, optimizer, epoch)
        )

        if best < all_time_best:
            all_time_best = best
            torch.save(update_rule.state_dict(), CHECKPOINTS_DIR / "best_weights.pt")

        # Also off the event loop thread — PNG encoding/disk I/O
        # shouldn't stall websocket message flushing either.
        await asyncio.to_thread(_save_episode_images, epoch, final_positions)

        print(
            f"epoch {epoch:4d}  best {best:.4f}  mean {mean:.4f}  worst {worst:.4f}  "
            f"(all-time best {all_time_best:.4f})  grad_norm={grad_norm_value:.4g}  "
            f"hidden_sat={hidden_sat_frac:.1%}  max|input|={max_input_abs:.4g}"
        )

        latest_generation_message = {
            "type": "generation",
            "generation": epoch,
            "best": best,
            "mean": mean,
            "worst": worst,
            "allTimeBest": all_time_best,
            # This episode's own agent-jitter seed (SimulationConfig's
            # `seed` — what the frontend re-seeds its replay jitter
            # with), not the top-level run seed the whole training
            # invocation started with — that one's runSeed, below.
            "seed": seed,
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
            "maxEnvWrite": MAX_ENV_WRITE,
            "repulsionSigma": REPULSION_SIGMA,
            "repulsionStrength": REPULSION_STRENGTH,
            "hiddenDim": HIDDEN_DIM,
            # Everything above is simulation config — what a WebGPU
            # replay needs to reproduce this episode's rollout
            # (SimulationConfig). Everything below is training-specific.
            "target": args.target,
            "method": "gd",
            "bpttSteps": args.bptt_steps,
            "lr": args.lr,
            "gradClip": args.grad_clip,
            "rasterResolution": args.raster_resolution,
            "rasterSigma": args.raster_sigma,
            "outsideWeight": args.outside_weight,
            "runSeed": args.seed,
            "totalGenerations": args.epochs,
            "checkpointEvery": args.checkpoint_every,
        }
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(latest_generation_message) + "\n")
        await broadcast(latest_generation_message)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save(update_rule.state_dict(), CHECKPOINTS_DIR / "latest_weights.pt")
            (CHECKPOINTS_DIR / "best_meta.json").write_text(
                json.dumps(
                    {
                        "generation": epoch,
                        "fitness": all_time_best,
                        "target": args.target,
                        "method": "gd",
                        "agents": args.agents,
                        "steps": args.steps,
                        "bptt_steps": args.bptt_steps,
                        "grid_size": args.grid_size,
                        "channels": args.channels,
                        "spawn_spread": args.spawn_spread,
                        "lr": args.lr,
                        "grad_clip": args.grad_clip,
                        "raster_resolution": args.raster_resolution,
                        "raster_sigma": args.raster_sigma,
                        "outside_weight": args.outside_weight,
                        "seed": args.seed,
                        "decay": DECAY,
                        "hidden_dim": HIDDEN_DIM,
                        "max_speed": MAX_SPEED,
                        "max_accel": MAX_ACCEL,
                        "max_strafe": MAX_STRAFE,
                        "max_env_write": MAX_ENV_WRITE,
                        "repulsion_sigma": REPULSION_SIGMA,
                        "repulsion_strength": REPULSION_STRENGTH,
                    },
                    indent=2,
                )
            )

    print(f"done. all-time best: {all_time_best:.4f}.")


@app.get("/history")
def history() -> dict:
    """Every episode persisted so far — same role as train_server.py's
    own /history."""
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
    size — same reasoning as train_server.py's own version (an archived
    run in the shared /runs list may have been trained against a
    different --target than this server's own)."""
    path = TARGETS_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"unknown target '{name}'")
    loaded = TargetShape.from_export(json.loads(path.read_text()), grid_size)
    return {"points": loaded.points.tolist()}


def _find_latest_preview(images_dir: Path) -> Optional[Path]:
    """Same logic as train_server.py's own version — see that function's
    docstring."""
    if not images_dir.is_dir():
        return None
    for pattern in ("gen_*_raster.png", "gen_*_agents.png"):
        candidates = sorted(images_dir.glob(pattern))
        if candidates:
            return candidates[-1]
    return None


def _run_dir_for_id(run_id: str) -> Optional[Path]:
    """Same path-traversal-safe resolution as train_server.py's own
    version."""
    if not run_id or "/" in run_id or "\\" in run_id:
        return None
    candidate = RUNS_DIR / run_id
    if candidate.is_dir() and candidate.parent == RUNS_DIR:
        return candidate
    return None


@app.get("/runs")
def list_runs() -> dict:
    """Every archived run (checkpoints/runs/* — shared with
    train_server.py, so this includes ES runs too if any were trained
    into this same directory) plus the current one — same shape as
    train_server.py's own /runs, so RunPicker.tsx doesn't need to know
    which server produced which entry."""
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
    """Same shape as /history, for one specific run — same logic as
    train_server.py's own version."""
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
    """Same logic as train_server.py's own version."""
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
    run — net/images.ts's generationImageUrl() builds this URL, same as
    it does against train_server.py."""
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

"""Gradient-descent (truncated BPTT) counterpart to evolve.py's
evolutionary training loop — same target-matching goal (raster.py's
shape-matching fitness, now via raster_torch.py's differentiable port),
but instead of scoring a population of randomly mutated weight-sets and
keeping the best, this trains *one* set of UpdateRule weights directly:
roll the simulation forward with gradients enabled (simulation.py's
step() no longer forces @torch.no_grad() — see that method's own
docstring), score the rollout with
raster_torch.training_raster_distance_torch, and step an Adam optimizer
on the resulting loss.

Doesn't touch evolve.py/train_server.py's ES path or its checkpoint
files at all — writes to its own checkpoints/gd/ subdirectory (a plain
torch state_dict, not evolve.py's flattened-numpy-vector format), so
both approaches can run side by side — e.g. an ES run already training
via train_server.py on its own port — without colliding on the same
files. (This project's checkpoints/ directory has bitten exactly that
collision before: an ES server actively writing history.jsonl/best.npy
while an unrelated CLI invocation overwrote the same paths. This module
was deliberately kept out of that shared namespace from the start.)

Truncated BPTT, not full-episode backprop: an unrolled rollout over
hundreds of steps on a few-hundred-pixel grid would need autograd to
retain every step's conv2d/deposit/MLP activations simultaneously until
backward() — a lot of memory for a real grid/step count (see the
differentiability design discussion this followed). Each "episode" (one
seed-to-`--steps` rollout, mirroring evolve.py's own rollout()) is split
into `--bptt-steps`-sized windows: step forward with grad enabled,
score + backprop + optimizer.step() once at the end of each window, then
`.detach()` the agent/grid state (values unchanged, autograd history
severed) before continuing into the next window from that same state.
This bounds the backward graph to one window's worth of steps regardless
of how long the episode runs, at the cost of the gradient never seeing
further back than one window — the standard truncated-BPTT tradeoff.

Usage:
    python train_gd.py --target circle --epochs 200 --steps 200 --bptt-steps 20
"""

from __future__ import annotations

import argparse
import json

import torch

from device import pick_device
from environment import Environment
from evolve import CHECKPOINTS_DIR, load_target, raster_extent
from raster import build_target_distance_field, build_target_raster
from raster_torch import target_rasters_to_torch, training_raster_distance_torch
from simulation import Simulation
from update_rule import UpdateRule

# Deliberately its own subdirectory, disjoint from every filename
# evolve.py/train_server.py write to (best.npy, best_meta.json,
# history.jsonl, generation_images/) — see module docstring.
GD_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "gd"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="circle", help="target name in envnca/targets/ (without .json)")
    parser.add_argument("--epochs", type=int, default=200, help="number of fresh-reset episodes to train on")
    parser.add_argument("--agents", type=int, default=200, help="agent count per rollout")
    parser.add_argument("--steps", type=int, default=200, help="total simulation steps per episode")
    parser.add_argument(
        "--bptt-steps",
        type=int,
        default=20,
        help="truncated-BPTT window size, in steps — see module docstring",
    )
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--spawn-spread", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="max gradient norm (torch.nn.utils.clip_grad_norm_) — BPTT through tanh over many steps is prone "
        "to exploding gradients; a non-finite total norm skips that window's optimizer step entirely rather "
        "than poisoning the weights with a NaN/Inf update",
    )
    parser.add_argument(
        "--raster-resolution",
        type=int,
        default=128,
        help="side length of the square lattice target/agent point clouds are splatted onto for fitness — see raster.py",
    )
    parser.add_argument(
        "--raster-sigma",
        type=float,
        default=1.5,
        help="Gaussian splat width, in raster pixels (not grid pixels) — see raster.rasterize_points",
    )
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=1.0,
        help="weight of the distance-transform penalty for agents landing outside the target's footprint "
        "(0 disables it) — see raster.outside_shape_penalty",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.bptt_steps < 1:
        raise SystemExit("--bptt-steps must be >= 1")

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    target = load_target(args.target, args.grid_size)
    extent = raster_extent(args.grid_size)
    # Fixed for the whole run — precomputed once, same reasoning as
    # evolve.py's own target_raster/target_distance_field (target points
    # never change generation/epoch to generation/epoch). Still built via
    # raster.py's numpy implementation (target points are fixed constants
    # that never need gradients) and converted once — see
    # raster_torch.target_rasters_to_torch's own docstring.
    target_raster_np = build_target_raster(
        target.points,
        args.raster_resolution,
        extent,
        args.raster_sigma,
        half_size=target.texel_size(args.grid_size) / 2.0,
    )
    target_dist_np = build_target_distance_field(target_raster_np)
    target_raster_t, target_dist_t = target_rasters_to_torch(target_raster_np, target_dist_np, device)
    target_points_t = torch.tensor(target.points, dtype=torch.float32, device=device)

    update_rule = UpdateRule(num_channels=args.channels).to(device)
    optimizer = torch.optim.Adam(update_rule.parameters(), lr=args.lr)
    # See UpdateRule's own docstring on record_diagnostics for what this
    # is checking and why — logged per episode below (grad_norm/
    # hidden_sat/max|input|).
    update_rule.record_diagnostics = True

    GD_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = GD_CHECKPOINTS_DIR / "history.jsonl"
    # Fresh log for this run, same convention as train_server.py's own
    # HISTORY_PATH — a plain CLI invocation, unlike that server, doesn't
    # try to preserve/archive a previous run's log first (nothing else is
    # reading this path concurrently the way a browser tab watches
    # train_server.py's).
    history_path.write_text("")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Derived, not reused verbatim, so every episode's agent jitter
        # differs while the whole run stays reproducible given --seed —
        # same reasoning as evolve.py's per-candidate seeds.
        rng = torch.Generator().manual_seed(args.seed + epoch + 1)
        env = Environment(height=args.grid_size, width=args.grid_size, channels=args.channels, device=device)
        sim = Simulation(
            env, update_rule, device, population=args.agents, spawn_spread=args.spawn_spread, rng=rng
        )

        step = 0
        window_loss = float("inf")
        diverged = False
        # Diagnostic snapshot from the *last* window's *last* forward
        # pass (record_diagnostics overwrites these every call, not an
        # accumulated history) — printed alongside the loss below to
        # check whether training is plateauing because the hidden layer
        # has saturated (see UpdateRule's own docstring).
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
                # A diverged candidate should end the episode with a
                # terrible score, not crash the run — same backstop
                # evolve.py's own rollout() has.
                diverged = True
                break

            loss = training_raster_distance_torch(
                positions,
                target_points_t,
                target_raster_t,
                target_dist_t,
                args.raster_resolution,
                extent,
                args.raster_sigma,
                outside_weight=args.outside_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(update_rule.parameters(), args.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print(f"epoch {epoch:4d} step {step:4d}  non-finite gradient norm, skipping this window's update")
            grad_norm_value = float(grad_norm.item())
            if update_rule.last_hidden is not None:
                hidden_sat_frac = float((update_rule.last_hidden.abs() > 0.99).float().mean().item())
                max_input_abs = float(update_rule.last_input.abs().max().item())

            # Sever the autograd graph before the next window — values
            # are unchanged, only the backward history is cut. Without
            # this, the *next* window's backward() would walk back
            # through every step of every earlier window in this
            # episode too, defeating the whole point of truncating.
            sim.agents.positions = sim.agents.positions.detach()
            sim.agents.velocity = sim.agents.velocity.detach()
            sim.env.grid = sim.env.grid.detach()

            window_loss = float(loss.item())

        final_loss = float("inf") if diverged else window_loss

        if final_loss < best_loss:
            best_loss = final_loss
            torch.save(update_rule.state_dict(), GD_CHECKPOINTS_DIR / "best.pt")

        print(
            f"epoch {epoch:4d}  final-window loss {final_loss:.4f}  (best {best_loss:.4f})  "
            f"grad_norm={grad_norm_value:.4g}  hidden_sat={hidden_sat_frac:.1%}  max|input|={max_input_abs:.4g}"
        )

        with history_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "loss": final_loss, "best_loss": best_loss}) + "\n")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save(update_rule.state_dict(), GD_CHECKPOINTS_DIR / "latest.pt")
            (GD_CHECKPOINTS_DIR / "meta.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "target": args.target,
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
                    },
                    indent=2,
                )
            )

    print(f"done. best final-window loss: {best_loss:.4f}. checkpoint saved to {GD_CHECKPOINTS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()

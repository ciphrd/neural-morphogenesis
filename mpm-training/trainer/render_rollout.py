"""One-off visual sanity check for evolve.py's checkpoints — not part of
the eventual trainer/server (see ../viewer/README.md's own staging).
Loads checkpoints/best.npy + best_meta.json, replays that rollout, and
rasterizes the *aligned* final particle positions (via alignment.py's
own Chamfer-based best_alignment — the SAME rotation-search algorithm
evolve.py's own _score_fitness() currently scores/selects checkpoints
with, see that module's own module docstring for the current Chamfer/
raster back-and-forth; if fitness is ever switched back to raster.py's
own rotation search instead, this file's own alignment would once again
be a reasonable-but-not-necessarily-exact stand-in, not the literal pose
that scored the checkpoint) overlaid on the target point cloud — showing
the raw, un-transformed positions instead would make a perfectly-scoring
but rotated/translated result look wrong by eye, since fitness is
pose-invariant but a flat image isn't. Mirrors render_check.py's own
role for feasibility_check.py.

Usage:
    python render_rollout.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from agents_gpu import AgentsGPU
from alignment import best_alignment
from debug_images import rasterize
from device import pick_device
from environment_gpu import EnvironmentGPU
from evolve import CHECKPOINTS_DIR
from mpm_core import MpmCore
from simulation_settings import (
    ANGULAR_DAMPING,
    CHEM_CHANNELS,
    CHIRALITY,
    DAMPING_LOSS_FRACTION,
    DECAY,
    DEPOSIT_DISTANCE,
    DEPOSIT_RATE,
    DEPOSIT_SIGMA,
    DIVISION_COOLDOWN,
    FIELD_N,
    FRICTION,
    HIDDEN_DIM,
    MATERIAL_E,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MAX_ACCEL,
    MAX_ANGULAR_ACCEL,
    MAX_ANGULAR_VELOCITY,
    MAX_ENV_WRITE,
    MAX_STRAFE,
    SPLIT_DISPLACEMENT,
)
from targets import load_target
from training_sim import TrainingRollout


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "render_out"
    out_dir.mkdir(exist_ok=True)

    meta = json.loads((CHECKPOINTS_DIR / "best_meta.json").read_text())
    weights = np.load(CHECKPOINTS_DIR / "best.npy")

    wgpu_device = pick_device()

    core = MpmCore(wgpu_device)
    core.set_material(MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, elasticity=0.0)
    core.set_damping(DAMPING_LOSS_FRACTION, meta["substeps_per_macro"])

    environment = EnvironmentGPU(wgpu_device, CHEM_CHANNELS, FIELD_N, FIELD_N, DECAY, meta.get("deposit_rate", DEPOSIT_RATE))
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
        # Falls back to the current constant/value for older checkpoints
        # saved before "chirality"/"deposit_distance"/"split_displacement"/
        # "division_cooldown"/"friction" rode along in best_meta.json —
        # "particles" itself has ALWAYS been recorded, but meant "starting
        # count" on any checkpoint trained before growth existed; using it
        # as the growth cap here regardless is still correct (evolve.py's
        # own module docstring: a policy that never learns to use the
        # growth channel just stays at 1 particle forever either way, same
        # as this used to be the ONLY option for an old, pre-growth
        # checkpoint).
        meta.get("chirality", CHIRALITY),
        meta.get("deposit_distance", DEPOSIT_DISTANCE),
        meta["particles"],
        meta.get("split_displacement", SPLIT_DISPLACEMENT),
        meta.get("division_cooldown", DIVISION_COOLDOWN),
        meta.get("friction", FRICTION),
        meta.get("deposit_sigma", DEPOSIT_SIGMA),
        meta["spawn_x"],
        meta["spawn_y"],
    )
    agents.load_weights(weights)

    target = load_target(meta["target"])

    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(meta["spawn_x"], meta["spawn_y"]),
        spawn_half_width=meta["spawn_half_width"],
        gravity=meta["gravity"],
        seed=meta["seed"],
    )

    for i in range(meta["macro_steps"]):
        sim.macro_step(meta["substeps_per_macro"])
        if i % 4 == 0 or i == meta["macro_steps"] - 1:
            pos = sim.positions()
            _, aligned = best_alignment(pos, target.points)
            img = rasterize(aligned, target.points)
            path = out_dir / f"rollout_{i:03d}.png"
            img.save(path)
            print(f"wrote {path}")

    print(f"\nDone (target={meta['target']!r}, checkpoint fitness={meta['fitness']:.4f}) — gray=target, white=grown")
    return 0


if __name__ == "__main__":
    sys.exit(main())

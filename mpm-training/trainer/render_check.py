"""One-off visual sanity check — not part of the eventual trainer/server.

Runs MpmCore headlessly on a blocks.ts-style falling blob (same scene
shape as feasibility_check.py's check_settle) and rasterizes particle
positions to a PNG at a few points in time, so the physics can be
checked by eye without a websocket server or browser viewer.

Usage:
    python render_check.py [out_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from device import pick_device
from mpm_core import MpmCore
from simulation_settings import MATERIAL_E, MATERIAL_HARDENING, MATERIAL_NU, SUBSTEPS_PER_DAMPING_FRAME

IMG_SIZE = 512


def seed_blob(count: int, center: tuple[float, float], half_width: float, rng: np.random.Generator):
    positions = center + rng.uniform(-half_width, half_width, size=(count, 2))
    velocities = np.zeros((count, 2), dtype=np.float32)
    F = np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (count, 1))
    C = np.zeros((count, 4), dtype=np.float32)
    Jp = np.ones((count,), dtype=np.float32)
    return positions.astype(np.float32), velocities, F, C, Jp


def rasterize(positions: np.ndarray, size: int = IMG_SIZE) -> Image.Image:
    img = np.zeros((size, size), dtype=np.uint8)
    # sim space is [0, 1]^2, y-up; image space is y-down.
    xs = np.clip((positions[:, 0] * size).astype(np.int32), 0, size - 1)
    ys = np.clip(((1.0 - positions[:, 1]) * size).astype(np.int32), 0, size - 1)
    img[ys, xs] = 255
    # thicken points a bit so they're visible at this particle count
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(ys + dy, 0, size - 1)
            xx = np.clip(xs + dx, 0, size - 1)
            img[yy, xx] = 255
    return Image.fromarray(img, mode="L")


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "render_out"
    out_dir.mkdir(exist_ok=True)

    device = pick_device()
    core = MpmCore(device)
    core.set_gravity(200.0)
    core.set_material(MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, elasticity=0.0)

    rng = np.random.default_rng(seed=1)
    positions, velocities, F, C, Jp = seed_blob(2000, (0.5, 0.8), 0.08, rng)
    core.load_scene(positions, velocities, F, C, Jp)

    batch = SUBSTEPS_PER_DAMPING_FRAME * 25  # 200 substeps/batch
    frame_batches = [0, 4, 8, 20]  # steps 0, 800, 1600, 4000

    frame = 0
    for target in frame_batches:
        while frame < target:
            core.step(batch)
            frame += 1
        pos = core.read_positions()
        img = rasterize(pos)
        path = out_dir / f"frame_{frame * batch:05d}.png"
        img.save(path)
        print(f"wrote {path}  (mean_y={pos[:,1].mean():.4f})")

    print(f"\nDone — inspect PNGs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

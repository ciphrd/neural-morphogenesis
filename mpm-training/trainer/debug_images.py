"""PNG snapshots of a rollout's grown particles vs. its target — the
"collect renders of the best of each generation" piece train_server.py
needs so a frontend can show generation-over-generation progress without
having to reimplement MLS-MPM physics client-side (../viewer/README.md's
own staging is still blocked on that).

Three images per generation (see train_server.py's own call site):
- `..._grown.png` — the winning rollout's raw, un-aligned final
  positions overlaid on the target, in MpmCore's own [0,1]^2 domain —
  "what actually happened, pose included." Point-cloud dot rendering
  (rasterize()/save_grown_image() below), not a raster heatmap.
- `..._target.png` — the target's own raster (raster.build_target_raster,
  fixed for the whole run), via save_raster_image() below.
- `..._agents.png` — the winning rollout's positions, rotated to
  whichever pose raster.training_raster_distance()'s own rotation search
  actually scored this candidate under (track_best_raster=True — see
  that function's own docstring), rasterized via the *same*
  save_raster_image() — meant to sit next to `..._target.png` for a
  direct, literally-aligned, pixel-for-pixel visual comparison. This
  supersedes an earlier `..._aligned.png` (alignment.py's own, separate
  Chamfer-based rotation search) — that rotation wasn't guaranteed to
  match the one raster-distance scoring actually picked, so it was a
  plausible-looking but not-necessarily-accurate stand-in; this one, by
  construction, IS the pose training scored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 512
TARGET_COLOR = (90, 90, 90)
GROWN_COLOR = (255, 255, 255)


def rasterize(grown: np.ndarray, target: np.ndarray, size: int = IMG_SIZE) -> Image.Image:
    """Both point clouds share MpmCore's own [0,1]^2 domain convention
    (y-up, gravity pulls toward y=0) — flipped to image space (y-down)
    for display only, not touched anywhere else in this project."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    def splat(points: np.ndarray, color: tuple[int, int, int]) -> None:
        if points.shape[0] == 0:
            return
        xs = np.clip((points[:, 0] * size).astype(np.int32), 0, size - 1)
        ys = np.clip(((1.0 - points[:, 1]) * size).astype(np.int32), 0, size - 1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = np.clip(ys + dy, 0, size - 1)
                xx = np.clip(xs + dx, 0, size - 1)
                img[yy, xx] = color

    splat(target, TARGET_COLOR)
    splat(grown, GROWN_COLOR)  # drawn second/on top, so overlap reads as "covered"
    return Image.fromarray(img, mode="RGB")


def save_grown_image(positions: np.ndarray, target_points: np.ndarray, path: Path) -> None:
    rasterize(positions, target_points).save(path)


def save_raster_image(raster: np.ndarray, path: Path) -> None:
    """Grayscale heatmap of a [0,1]-ish raster (raster.rasterize_points/
    rasterize_points_sum — the sum-combine variant can exceed 1.0 under
    piled-up particles, see that function's own docstring, hence the
    clip) — used for both the target's own raster and the winning
    rollout's own best-rotation agent raster, so the two are directly,
    visually comparable side by side.

    Row-flipped before saving (`raster[::-1]`), unlike envnca/
    debug_images.py's own save_raster_image(): raster.py's own
    rasterize_points()/rasterize_points_sum() map row index directly
    from domain-y (row 0 = y=0), but this project's domain is y-*up*
    (gravity pulls toward y=0 — see rasterize()'s own comment above),
    so row 0 unflipped would put the domain's bottom at the image's top,
    upside-down relative to save_grown_image()'s own explicit y-flip.
    Flipping here keeps every image this module produces in the same
    visual orientation."""
    img = np.clip(raster[::-1], 0.0, 1.0)
    Image.fromarray((img * 255.0).astype(np.uint8), mode="L").save(path)

"""PIL-based PNG snapshots for train_server.py's end-of-generation debug
output — kept separate from raster.py (dependency-free besides numpy)
since Pillow is only needed on this optional, debug-only path.

Three images per generation (see train_server.py's own call site for
why exactly these three): the target's own raster, the winning
rollout's raw (un-rotated) replay positions in native grid-pixel space —
the same drawing convention the deleted render.py used — and the
rotation-aligned agent raster training actually scored against the
target raster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Matches the deleted render.py's own constants.
BACKGROUND_GRAY = 127
TARGET_COLOR = (220, 60, 60)
AGENT_COLOR = (255, 255, 255)
DOT_RADIUS = 1  # pixels either side of center — a 3x3 block, same as render.py's agents


def _draw_dots(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = canvas.shape[:2]
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        y0, y1 = max(0, yi - DOT_RADIUS), min(h, yi + DOT_RADIUS + 1)
        x0, x1 = max(0, xi - DOT_RADIUS), min(w, xi + DOT_RADIUS + 1)
        if y0 < y1 and x0 < x1:
            canvas[y0:y1, x0:x1] = color


def save_agents_image(positions: np.ndarray, target_points: np.ndarray, grid_size: int, path: Path) -> None:
    """Raw (un-rotated) agent + target positions in native grid-pixel
    space, drawn exactly like the deleted render.py's render_frame() did
    (minus the chemical-field background, not readily available as a
    cheap debug snapshot — flat gray stands in for it) — this is the
    "what actually happened, pose and all" view, as opposed to
    save_raster_image()'s rotation-aligned raster."""
    canvas = np.full((grid_size, grid_size, 3), BACKGROUND_GRAY, dtype=np.uint8)
    _draw_dots(canvas, target_points, TARGET_COLOR)
    _draw_dots(canvas, positions, AGENT_COLOR)
    Image.fromarray(canvas, mode="RGB").save(path)


def save_raster_image(raster: np.ndarray, path: Path) -> None:
    """Grayscale heatmap of a [0,1] raster (see raster.rasterize_points)
    — used for both the target's own raster and an aligned agent raster,
    so the two are directly, visually comparable side by side. Raster
    row 0 is already the grid's own row 0 (environment.py's (C,H,W)
    convention, y increasing downward) — no flip needed, same
    orientation save_agents_image draws in."""
    img = np.clip(raster, 0.0, 1.0)
    Image.fromarray((img * 255.0).astype(np.uint8), mode="L").save(path)

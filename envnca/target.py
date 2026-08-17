"""Target shape loaded from the same pixel-export format
trainer/backend/target.py uses ({nx, ny, pixels: [{x,y}, ...]}) — see
envnca/targets/*.json, which are literally the same files.

Represented as a point cloud of occupied-pixel centers, but in *grid
pixel space* (the same coordinate system Simulation.agents.positions
already live in, 0..width / 0..height) rather than trainer's "graph
space" scaled by a physics constant (REST_LENGTH) — this project has no
physics pass, so there's no "one graph edge length" to anchor a scale
to. Anchored to the grid's own size instead: recentered on the grid's
center (where the agent population is seeded — see simulation.py) and
scaled so the shape's longer axis spans a fixed *fraction* of the grid,
regardless of the export's own resolution (a 64x64 export of the same
drawing as a 32x32 one comes out the same physical size, not twice as
big, exactly like trainer's own resolution-independence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fraction of the grid's size the target shape's longer axis should span.
# Agents start jittered within a few pixels of the grid's center (see
# agent_state.py's seed()) and have a bounded per-step speed (MAX_SPEED,
# MAX_STRAFE in update_rule.py) — too large a span and a short rollout
# can't plausibly reach the far edges of the shape at all; too small and
# the whole rollout plays out in a tiny, cramped corner of a 512x512
# grid. 0.6 leaves margin on every side while still giving agents real
# distance to cover.
TARGET_SPAN_FRACTION = 0.6


@dataclass
class TargetShape:
    points: np.ndarray  # (N, 2), grid pixel coords
    resolution: tuple[int, int]

    def texel_size(self, grid_size: int) -> float:
        """Width, in grid-pixel units, of one texel of the *original*
        pixel-art export at this target's own resolution — i.e. the
        same `scale` from_export() used to place `points`, recomputed
        rather than stored, so there's one source of truth for it. Used
        by raster.py to size each target point's rasterized footprint
        so it renders as a filled block matching its actual coverage,
        not a point-sized dot — a target texel represents a whole
        square *area* of the original drawing, unlike an agent, which
        really is just a point."""
        span = max(self.resolution)
        return (grid_size * TARGET_SPAN_FRACTION) / span

    @classmethod
    def from_export(cls, data: dict, grid_size: int) -> "TargetShape":
        nx, ny = data["nx"], data["ny"]
        pixels = data["pixels"]
        if not pixels:
            return cls(points=np.zeros((0, 2)), resolution=(nx, ny))

        coords = np.array([[p["x"], p["y"]] for p in pixels], dtype=np.float64)
        centers = coords + 0.5  # pixel centers, not corners
        centroid = centers.mean(axis=0)

        # sculpture's export grid is usually square but isn't guaranteed
        # to be — take the longer axis so a non-square export still
        # scales sanely against TARGET_SPAN_FRACTION.
        span = max(nx, ny)
        scale = (grid_size * TARGET_SPAN_FRACTION) / span

        points = (centers - centroid) * scale + grid_size / 2.0
        return cls(points=points, resolution=(nx, ny))

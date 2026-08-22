"""Target shapes loaded from sculpture's pixel-export format
(`{nx, ny, pixels: [{x,y}, ...]}`) — the same format trainer/backend/
target.py and envnca/target.py consume, files living in ./targets/
(*.json). Earlier revisions of this module generated targets
procedurally; the trainer now trains against whatever's actually in that
folder instead.

Represented as a point cloud of occupied-pixel centers, recentered on
the target's own centroid and scaled so its longer axis spans a fixed
fraction of MpmCore's own [0,1]^2 domain — mirrors envnca/target.py's
own TargetShape almost exactly (same recenter+scale idea), just anchored
to this project's domain instead of a pixel-space grid. Where exactly
the target ends up doesn't actually matter for training: evolve.py
scores candidates via alignment.training_alignment_distance, which
re-centers the grown blob onto the target's own centroid and searches
rotation, so only the target's *shape* (relative point positions) is
ever compared — this module's centroid/scale choice just keeps target
point clouds a sane, comparable size across files of different
resolution/pixel-count, same reasoning envnca/target.py's own docstring
gives.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

TARGETS_DIR = Path(__file__).parent / "targets"

# Fraction of MpmCore's own [0,1)^2 toroidal domain the target shape's
# longer axis should span. Not the full [0,1) — particles have real
# physical extent, and alignment.py's own rotation search needs the
# grown blob to have somewhere to translate/rotate into without
# immediately wrapping around on itself — 0.5 leaves real margin.
TARGET_SPAN_FRACTION = 0.5


@dataclass
class TargetShape:
    points: np.ndarray  # (N, 2), MpmCore domain coords
    resolution: tuple[int, int]

    def texel_size(self) -> float:
        """Width, in MpmCore [0,1]^2 domain units, of one texel of the
        *original* pixel-art export at this target's own resolution —
        the same `scale` from_export() used to place `points`,
        recomputed rather than stored, so there's one source of truth
        for it. Used by raster.py to size each target point's
        rasterized footprint so it renders as a filled block matching
        its actual coverage, not a point-sized dot — a target texel
        represents a whole square *area* of the original drawing,
        unlike a particle, which really is just a point. Mirrors
        envnca/target.py's own texel_size(grid_size) with grid_size
        fixed at 1.0 (this project's domain span), since it's baked
        into TARGET_SPAN_FRACTION here rather than passed in."""
        span = max(self.resolution)
        return TARGET_SPAN_FRACTION / span

    @classmethod
    def from_export(cls, data: dict) -> "TargetShape":
        nx, ny = data["nx"], data["ny"]
        pixels = data["pixels"]
        if not pixels:
            return cls(points=np.zeros((0, 2), dtype=np.float32), resolution=(nx, ny))

        coords = np.array([[p["x"], p["y"]] for p in pixels], dtype=np.float64)
        centers = coords + 0.5  # pixel centers, not corners
        centroid = centers.mean(axis=0)

        span = max(nx, ny)
        scale = TARGET_SPAN_FRACTION / span

        points = (centers - centroid) * scale + 0.5  # centered in the domain
        return cls(points=points.astype(np.float32), resolution=(nx, ny))


def available_targets() -> list[str]:
    return sorted(p.stem for p in TARGETS_DIR.glob("*.json"))


def load_target(name: str) -> TargetShape:
    path = TARGETS_DIR / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"unknown target {name!r} — choices: {available_targets()} (looked for {path})")
    return TargetShape.from_export(json.loads(path.read_text()))

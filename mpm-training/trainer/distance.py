"""Symmetric nearest-neighbor (Chamfer) distance between two point
clouds. Deliberately the simplest thing that works: no rotation/
translation search (alignment.py's own training_alignment_distance, in
this same directory, adds that), no Gaussian-splat rasterization
(raster.py) — just raw nearest-neighbor distance, both directions, via
scipy's cKDTree. Lower is better; symmetric so neither "grew somewhere
the target doesn't cover" nor "target has uncovered regions" scores well
by accident.

For a while this (via alignment.py's own wrapper) was evolve.py's own
genetic-selection fitness signal — raster.py's own Gaussian-splat raster
distance is what actually scores/selects candidates now (see evolve.py's
own module docstring for why it came back); this module is kept fully
implemented, not dead code, since render_rollout.py still uses
alignment.py's own best_alignment() for its own quick visual checks.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    return float(d_ab.mean() + d_ba.mean())

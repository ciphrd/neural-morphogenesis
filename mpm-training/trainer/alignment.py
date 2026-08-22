"""Rotation+translation-invariant Chamfer distance — ported from
envnca/alignment.py / trainer/backend/alignment.py's own
training_alignment_distance: nothing pins the grown particle blob to the
target's exact pose (there's no anchoring during physics, and the
target's own orientation is an arbitrary artifact of however its pixel
export was drawn), so a pose-aware fitness needs to reward getting the
*shape* right, independent of pose — the same "lowest distance after
transform" convention every other evolve.py in this repo trains against,
whichever underlying point-distance it's built on.

training_alignment_distance IS evolve.py's own live fitness function
again (see that module's own module docstring for the Chamfer/raster
back-and-forth — raster.py's own Gaussian-splat raster distance held
that role for a while instead, and stays fully wired for train_server.py's
own debug-image rendering, just not for scoring). best_alignment() (the
rotated-points version this wraps, not just the scalar distance) also
does its own separate work regardless of which strategy is live:
render_rollout.py's own quick visual sanity checks call it directly to
pose-align a checkpoint's replay for display. Only training_alignment_distance/
best_alignment are ported (the cheap coarse-rotation-grid version) —
best_fit_distance's expensive 12-restart Nelder-Mead search is for
one-off reporting, not needed here yet.
"""
from __future__ import annotations

import numpy as np

from distance import chamfer_distance

# Coarse, cheap rotation search — cost matters more than precision here,
# since this runs on every candidate of every generation. Same default as
# envnca/alignment.py's own TRAIN_NUM_ANGLES.
TRAIN_NUM_ANGLES = 16


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def best_alignment(
    points: np.ndarray, target_points: np.ndarray, num_angles: int = TRAIN_NUM_ANGLES
) -> tuple[float, np.ndarray]:
    """Chamfer distance minimized over rotation about the target's
    centroid, with translation fixed by centroid-matching (`points` is
    first re-centered on its own centroid, then re-placed at the
    target's). Returns (distance, transformed_points) — the shared
    search training_alignment_distance (training's own hot path, scalar
    only) and render_rollout.py (wants the actual aligned points to draw)
    both build on."""
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        return float("inf"), points

    centered = points - points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    best_dist = float("inf")
    best_points = points
    for i in range(num_angles):
        theta = 2.0 * np.pi * i / num_angles
        rotated = centered @ _rotation_matrix(theta).T + target_centroid
        dist = chamfer_distance(rotated, target_points)
        if dist < best_dist:
            best_dist = dist
            best_points = rotated
    return best_dist, best_points


def training_alignment_distance(
    points: np.ndarray, target_points: np.ndarray, num_angles: int = TRAIN_NUM_ANGLES
) -> float:
    """Scalar-only convenience wrapper around best_alignment() — training
    only ever wants the one number, not the transformed points."""
    return best_alignment(points, target_points, num_angles)[0]

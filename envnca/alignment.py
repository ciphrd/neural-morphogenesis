"""Rigid (rotation + translation) alignment search — copied essentially
verbatim from trainer/backend/alignment.py, generic over what `points`
represents (that project's node positions, this one's agent positions).

Finds the pose of `points` that minimizes Chamfer distance to
`target_points`, so the reported distance reflects best-possible fit
rather than being an accident of whatever orientation/position agents
happen to have drifted to — nothing pins agent motion to the target's
pose, and the target's own orientation is an arbitrary artifact of
however it was drawn, so fitness (training and reporting alike) should
score shape, not pose.

Two variants, same underlying idea, different cost budgets:
- best_fit_distance: precise (12-restart Nelder-Mead) search, for one-off
  reporting.
- training_alignment_distance: coarse rotation grid search with analytic
  (centroid-matching) translation, cheap enough to run as the fitness
  function for every candidate of every generation — see evolve.py's
  rollout().
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from distance import chamfer_distance

# Evenly-spaced starting angles for the search. Chamfer distance as a
# function of rotation is not convex (nearest-neighbor assignment changes
# discontinuously), so a single gradient descent from one guess can land
# in a bad local minimum — multiple restarts around the circle make that
# much less likely without needing a real global optimizer.
NUM_RESTARTS = 12


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _transform(points: np.ndarray, theta: float, translation: np.ndarray) -> np.ndarray:
    return points @ _rotation_matrix(theta).T + translation


def best_fit_distance(points: np.ndarray, target_points: np.ndarray) -> dict:
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        metrics = chamfer_distance(points, target_points)
        metrics["rotation"] = 0.0
        metrics["translation"] = [0.0, 0.0]
        return metrics

    points_centroid = points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    def objective(params: np.ndarray) -> float:
        theta, dx, dy = params
        transformed = _transform(points, theta, np.array([dx, dy]))
        return chamfer_distance(transformed, target_points)["chamfer"]

    best = None
    for i in range(NUM_RESTARTS):
        theta0 = 2.0 * np.pi * i / NUM_RESTARTS
        rotated_centroid = _rotation_matrix(theta0) @ points_centroid
        t0 = target_centroid - rotated_centroid

        result = minimize(
            objective,
            x0=np.array([theta0, t0[0], t0[1]]),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 150},
        )
        if best is None or result.fun < best.fun:
            best = result

    theta, dx, dy = best.x
    transformed = _transform(points, theta, np.array([dx, dy]))
    metrics = chamfer_distance(transformed, target_points)
    metrics["rotation"] = float(theta)
    metrics["translation"] = [float(dx), float(dy)]
    return metrics


# Coarse, cheap rotation+translation alignment for use as a *training*
# fitness signal — unlike best_fit_distance above (a precise but
# expensive search meant for one-off reporting), this runs on every
# candidate of every generation, so cost matters more than precision.
TRAIN_NUM_ANGLES = 16


def training_alignment_distance(
    points: np.ndarray, target_points: np.ndarray, num_angles: int = TRAIN_NUM_ANGLES
) -> float:
    """Chamfer distance minimized over rotation about the target's
    centroid, with translation fixed by centroid-matching. Returns just
    the scalar (not the points_to_target/target_to_points breakdown) —
    training only ever wants the one number."""
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        return float("inf")

    centered = points - points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    best = float("inf")
    for i in range(num_angles):
        theta = 2.0 * np.pi * i / num_angles
        rotated = centered @ _rotation_matrix(theta).T + target_centroid
        dist = chamfer_distance(rotated, target_points)["chamfer"]
        if dist < best:
            best = dist
    return best

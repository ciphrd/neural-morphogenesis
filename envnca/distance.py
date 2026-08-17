"""Generic point-cloud-to-target-shape distance — copied essentially
verbatim from trainer/backend/distance.py, since this is dependency-free
(numpy + scipy only) and has nothing project-specific to invert; envnca
is its own standalone project (see README.md), so this is duplicated
rather than imported across the two."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(points: np.ndarray, target_points: np.ndarray) -> dict:
    """Symmetric Chamfer distance between two point sets.

    - points_to_target: mean distance from each point in `points` to its
      nearest neighbor in `target_points` — penalizes agents that ended up
      somewhere the target doesn't cover.
    - target_to_points: mean distance from each point in `target_points`
      to its nearest neighbor in `points` — penalizes parts of the target
      no agent reached.
    - chamfer: the sum of both, a single scalar summarizing overall fit.

    Both directions matter and aren't redundant: a single point sitting
    exactly at the target's centroid scores perfectly on
    points_to_target while covering almost none of target_to_points.
    """
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        return {
            "points_to_target": float("inf"),
            "target_to_points": float("inf"),
            "chamfer": float("inf"),
        }

    target_tree = cKDTree(target_points)
    points_tree = cKDTree(points)

    dist_points_to_target, _ = target_tree.query(points)
    dist_target_to_points, _ = points_tree.query(target_points)

    points_to_target = float(np.mean(dist_points_to_target))
    target_to_points = float(np.mean(dist_target_to_points))

    return {
        "points_to_target": points_to_target,
        "target_to_points": target_to_points,
        "chamfer": points_to_target + target_to_points,
    }

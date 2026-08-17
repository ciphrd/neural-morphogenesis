"""Differentiable torch port of distance.py's chamfer_distance and
alignment.py's training_alignment_distance — same purpose as
physics_torch.py: prototype pieces for the differentiable-training
effort, not wired into any live path. distance.py/alignment.py remain
authoritative for evolve.py's actual fitness function.

Only one substantive change from the numpy originals, and it isn't in
the math — nearest-neighbor matching via a dense torch.cdist + min
reduction is *exactly* what scipy's cKDTree.query computes too (same
nearest neighbor, same distance), it's just traceable: gradient flows
back through whichever point turned out to be the nearest neighbor,
since `.min()` on a tensor is an ordinary differentiable reduction (zero
gradient to the losing candidates, full gradient to the winner) — no
relaxation or approximation needed here, unlike physics_torch.py's
pairwise cutoffs or the split-gate this will eventually need. cKDTree
itself just isn't a torch op, so it can't participate in a backward
pass; that's the only thing being replaced.

training_alignment_distance_torch keeps alignment.py's coarse grid
search over rotation exactly as-is (still a discrete, non-differentiable
choice of *which* angle wins — same treatment as physics_torch.py's
pairwise proximity cutoffs: a topology/selection decision, not a value
being learned) but reuses torch's own `.min()` over the resulting
per-angle distances, which — same reasoning as above — automatically
backprops through whichever angle turned out best without needing to
special-case the argmin.
"""

from __future__ import annotations

import torch

from alignment import TRAIN_NUM_ANGLES


def chamfer_distance_torch(points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance — mirrors distance.py's chamfer_distance,
    but returns just the scalar `chamfer` (points_to_target +
    target_to_points), since that's the only piece any differentiable
    caller needs; the breakdown is still there via chamfer_breakdown_torch
    below for anything that wants it (e.g. matching main.py's /target/distance
    endpoint shape later)."""
    return chamfer_breakdown_torch(points, target_points)["chamfer"]


def chamfer_breakdown_torch(points: torch.Tensor, target_points: torch.Tensor) -> dict[str, torch.Tensor]:
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        inf = torch.tensor(float("inf"))
        return {"points_to_target": inf, "target_to_points": inf, "chamfer": inf}

    pairwise = torch.cdist(points, target_points)  # (P, T)
    dist_points_to_target = pairwise.min(dim=1).values  # (P,) — nearest target per point
    dist_target_to_points = pairwise.min(dim=0).values  # (T,) — nearest point per target

    points_to_target = dist_points_to_target.mean()
    target_to_points = dist_target_to_points.mean()
    return {
        "points_to_target": points_to_target,
        "target_to_points": target_to_points,
        "chamfer": points_to_target + target_to_points,
    }


def _rotation_matrix_torch(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _transform_torch(points: torch.Tensor, theta: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return points @ _rotation_matrix_torch(theta).T + translation


def training_alignment_distance_torch(
    points: torch.Tensor, target_points: torch.Tensor, num_angles: int = TRAIN_NUM_ANGLES
) -> torch.Tensor:
    """Chamfer distance minimized over a coarse rotation grid about the
    target's centroid, translation fixed by centroid-matching — mirrors
    alignment.py's training_alignment_distance exactly (same angle
    count, same centroid-matching translation), differentiable w.r.t.
    `points` through whichever angle in the grid wins."""
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        return torch.tensor(float("inf"))

    centered = points - points.mean(dim=0)
    target_centroid = target_points.mean(dim=0)

    distances = []
    for i in range(num_angles):
        theta = torch.tensor(2.0 * torch.pi * i / num_angles, dtype=points.dtype)
        rotated = centered @ _rotation_matrix_torch(theta).T + target_centroid
        distances.append(chamfer_distance_torch(rotated, target_points))

    return torch.stack(distances).min()

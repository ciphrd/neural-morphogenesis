"""Minimal position-based dynamics: iterative constraint projection, no
velocities. Two constraints — edges pull toward a rest length, all
sufficiently-close pairs push apart — relaxed for a fixed number of Gauss-
Seidel sweeps. Good enough for a click-driven scaffold; O(n^2) repulsion is
fine at the node counts this is used at."""

from __future__ import annotations

import numpy as np

REST_LENGTH = 1.0
MIN_DISTANCE = 0.85
ITERATIONS = 30


def relax(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    pinned: set[int],
) -> np.ndarray:
    pos = positions.copy()
    n = pos.shape[0]
    if n == 0:
        return pos

    for _ in range(ITERATIONS):
        for i, j in edges:
            _project_distance(pos, i, j, REST_LENGTH, pinned)

        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[j] - pos[i]
                dist = np.linalg.norm(delta)
                if dist < MIN_DISTANCE:
                    _project_distance(pos, i, j, MIN_DISTANCE, pinned, only_push_apart=True)

    return pos


def _project_distance(
    pos: np.ndarray,
    i: int,
    j: int,
    target: float,
    pinned: set[int],
    only_push_apart: bool = False,
) -> None:
    delta = pos[j] - pos[i]
    dist = np.linalg.norm(delta)
    if dist < 1e-8:
        # coincident points: nudge apart along a random direction to break
        # the singularity rather than dividing by ~0
        delta = np.random.normal(size=3)
        dist = np.linalg.norm(delta)
    direction = delta / dist

    diff = dist - target
    if only_push_apart and diff >= 0:
        return

    i_free = i not in pinned
    j_free = j not in pinned
    if not i_free and not j_free:
        return
    share = 0.5 if (i_free and j_free) else 1.0

    correction = direction * diff * share
    if i_free:
        pos[i] += correction
    if j_free:
        pos[j] -= correction

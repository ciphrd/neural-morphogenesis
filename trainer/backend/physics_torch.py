"""Differentiable torch port of physics.py's relax() — prototype for the
differentiable-training effort (see trainer/README.md / the conversation
that led here). Not wired into any live path yet: physics.py remains the
authoritative implementation for evolve.py/main.py/train_server.py. This
file exists to answer one question before anything else gets built on
top of it — does gradient survive backprop through a ~100+400-iteration
Jacobi solve without vanishing/exploding, and does the dense torch
version actually match physics.py's numeric output at convergence?

Two changes from physics.py, both required for autograd/batching, not
just style:
- Dense (N, N) pairwise ops instead of a KD-tree/grid neighbor search.
  physics.py's cKDTree.query_pairs is a discrete, non-differentiable
  index computation — there's no gradient to ask for "which pairs were
  found." At the scale this needs to work at (MAX_NODES = 400, see
  update_rule.py), a dense (400, 400) distance matrix is cheap enough
  that pruning to a sparse candidate set isn't worth the differentiability
  cost. Pairs that are out of range still enter the computation, they
  just contribute a zeroed-out term via `active`, exactly the same
  physical result as never having found them.
- A FIXED iteration count, not a data-dependent convergence check.
  physics.py's `if max_delta < tol: break` is a per-input, data-dependent
  Python branch — harmless for a single eager numpy call, but it means
  gradient only flows through however many iterations *actually ran* for
  that particular input, which varies candidate to candidate and blocks
  batching a whole population through one traced/compiled graph. Running
  every iteration unconditionally (wasted compute on already-settled
  inputs) is the tradeoff differentiability asks for here.

`alive` (defaults to "everyone alive") is accepted now, unused beyond
masking pairs and freezing dead-slot corrections to zero, purely so this
file's signature is already compatible with the fixed-capacity graph
representation (graph_torch.py) it'll be wired into next — see that
file's own docstring once it exists.
"""

from __future__ import annotations

from typing import Optional

import torch

from physics import (
    CLEANUP_ITERATIONS,
    CONTACT_DISTANCE,
    SETTLE_ITERATIONS,
    SETTLE_STIFFNESS,
    TENSION_RANGE,
    TENSION_STIFFNESS,
)

# Added inside the sqrt of every pairwise distance, not applied as a
# post-hoc torch.where swap — physics.py's `np.where(dist < 1e-9, 1.0,
# dist)` is fine for a forward-only numpy call, but torch.where computes
# (and backprops through) *both* branches, so a branch that would divide
# by ~0 still poisons the gradient with NaN/Inf even though its value is
# discarded. Adding eps before the sqrt keeps dist, direction, and their
# gradients all finite everywhere, including exactly-coincident points.
DIST_EPS = 1e-9


def _tension_compatibility(id_vectors: torch.Tensor) -> torch.Tensor:
    """(N, N) pairwise cosine similarity of id vectors, clipped to
    [0, 1] — mirrors physics.py's _tension_compatibility exactly."""
    norms = id_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    unit = id_vectors / norms
    cosine = unit @ unit.T
    return cosine.clamp(0.0, 1.0)


def _pairwise_diff_dist(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """diff[i, j] = pos[j] - pos[i] (matches physics.py's per-pair
    `diff = pos[j] - pos[i]` convention exactly); dist[i, j] = ||diff[i, j]||,
    both (N, N[, 2])."""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist = torch.sqrt((diff * diff).sum(-1) + DIST_EPS)
    return diff, dist


def _dense_corrections(
    pos: torch.Tensor,
    target: torch.Tensor,
    stiffness: torch.Tensor,
    active: torch.Tensor,
    free_mask: torch.Tensor,
) -> torch.Tensor:
    """Jacobi-averaged correction from every active ordered pair — dense
    equivalent of physics.py's _pair_corrections. `target`/`stiffness`
    are (N, N); `active` is a (N, N) bool (already excludes the
    diagonal and any dead-slot pairs).

    Antisymmetry does the "equal and opposite" bookkeeping for free: for
    an active pair (i, j), contrib[i, j] = direction(i->j) * magnitude
    and contrib[j, i] = direction(j->i) * magnitude = -contrib[i, j]
    (direction flips sign, magnitude doesn't, since dist/target/stiffness
    are symmetric) — exactly physics.py's `correction[i] += contrib;
    correction[j] -= contrib`, just falling out of summing every ordered
    pair rather than looping unordered ones and touching both endpoints
    by hand."""
    diff, dist = _pairwise_diff_dist(pos)
    direction = diff / dist.unsqueeze(-1)
    magnitude = (dist - target) * stiffness
    contrib = direction * magnitude.unsqueeze(-1)  # (N, N, 2)

    active_f = active.to(pos.dtype)
    contrib = contrib * active_f.unsqueeze(-1)
    count = active_f.sum(dim=1).clamp_min(1.0)  # (N,)

    correction = contrib.sum(dim=1) / count.unsqueeze(-1)  # (N, 2)
    correction = correction * free_mask.to(pos.dtype).unsqueeze(-1)
    return correction


def relax_torch(
    positions: torch.Tensor,
    pinned: torch.Tensor,
    id_vectors: torch.Tensor,
    alive: Optional[torch.Tensor] = None,
    settle_iterations: int = SETTLE_ITERATIONS,
    cleanup_iterations: int = CLEANUP_ITERATIONS,
) -> torch.Tensor:
    """Differentiable equivalent of physics.py's relax(). `positions` is
    (N, 2), `pinned` is a (N,) bool (True = never moves, same meaning as
    physics.py's `pinned` set), `id_vectors` is (N, D). `alive` (N,) in
    [0, 1] — omit for "everyone alive" (the common case while this isn't
    yet wired to a fixed-capacity graph); a dead slot is excluded from
    every pair and never moves, same treatment as a pinned node, so it
    doesn't need special-casing beyond that.

    Runs the *full* settle_iterations + cleanup_iterations unconditionally
    — see this module's docstring for why that's a deliberate tradeoff,
    not an oversight."""
    n = positions.shape[0]
    if alive is None:
        alive = torch.ones(n, dtype=positions.dtype, device=positions.device)

    free_mask = (~pinned) & (alive > 0.5)
    diag_mask = ~torch.eye(n, dtype=torch.bool, device=positions.device)
    alive_pair = (alive.unsqueeze(0) > 0.5) & (alive.unsqueeze(1) > 0.5)

    compatibility = _tension_compatibility(id_vectors)

    pos = positions
    for _ in range(settle_iterations):
        _, dist = _pairwise_diff_dist(pos)
        in_range = dist <= TENSION_RANGE
        active = in_range & diag_mask & alive_pair
        collision = dist < CONTACT_DISTANCE
        stiffness = torch.where(
            collision, torch.full_like(dist, SETTLE_STIFFNESS), TENSION_STIFFNESS * compatibility
        )
        target = torch.full_like(dist, CONTACT_DISTANCE)
        correction = _dense_corrections(pos, target, stiffness, active, free_mask)
        pos = pos + correction

    for _ in range(cleanup_iterations):
        _, dist = _pairwise_diff_dist(pos)
        active = (dist < CONTACT_DISTANCE) & diag_mask & alive_pair
        stiffness = torch.ones_like(dist)
        target = torch.full_like(dist, CONTACT_DISTANCE)
        correction = _dense_corrections(pos, target, stiffness, active, free_mask)
        pos = pos + correction

    return pos

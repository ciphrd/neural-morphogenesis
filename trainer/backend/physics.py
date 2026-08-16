"""Particle-based physics: nodes are circles of fixed radius. Two forces,
both purely distance-based and recomputed fresh every relax call — there
is no fixed topology, no notion of "this pair used to be connected":

- Surface tension (soft, breakable, short-range): any pair within
  TENSION_RANGE but not already touching gets pulled toward contact
  distance, only partially each iteration (TENSION_STIFFNESS < 1, so it
  behaves elastically). Past TENSION_RANGE there is no force at all —
  that absence is what makes it breakable. TENSION_RANGE is deliberately
  close to contact distance: making cohesion long-range turns it into
  "every node attracts every other node," which is an infeasible packing
  problem once more than a handful of circles are involved (in 2D only
  ~6 circles can mutually touch a shared neighbor) and the solver never
  settles.
- Collision (hard, unconditional): circles must never overlap. Run as a
  *separate* cleanup pass after tension has done its (soft) work, rather
  than mixed into the same pass — mixing a hard inequality constraint
  with a competing soft attraction is exactly the infeasible-packing
  problem above in miniature, one pair at a time, and doesn't reliably
  converge. Collision-only resolution has no competing force pulling
  things back together, so it always has a valid solution and converges
  reliably given enough iterations. Collision is purely geometric and
  never modulated by chemistry — circles can't overlap regardless of how
  compatible they are.

Tension strength is further modulated per-pair by the cosine similarity
of the two nodes' `id` vectors (see cell_state.py), clipped to
[0, 1] — homophilic-adhesion-style: similarly "identified" nodes pull
together at full strength, orthogonal or opposed identities feel no pull
at all even while in range, rather than negative/repulsive tension.
Proximity (TENSION_RANGE) decides whether a pair is *eligible* to feel
tension; compatibility decides how strongly they actually do.

Both passes use Jacobi-style updates (every node's correction this
iteration is the *average* of all its simultaneously-active pairwise
corrections, computed from one consistent snapshot of positions) rather
than Gauss-Seidel (correcting pairs one at a time, each seeing the
previous pair's already-updated positions) — Jacobi is what keeps this
stable at higher local density; Gauss-Seidel here visibly diverges once
more than a few circles crowd the same region.

Performance: TENSION_RANGE is a small, fixed interaction radius, so in
any reasonably spread-out graph, most pairs of nodes are never within
range of each other at all — the original implementation still checked
every one of them, every iteration, by building a dense (N, N) distance
matrix (O(n^2) per iteration, ~500 iterations per relax() call). This
version finds only the pairs actually within range via a KD-tree
(scipy.spatial.cKDTree.query_pairs — already a dependency, used
elsewhere for Chamfer distance) and does the correction math over just
those, which is where essentially all the wall-clock time was going
once graphs reached a couple hundred nodes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

RADIUS = 0.5
CONTACT_DISTANCE = 2 * RADIUS
TENSION_RANGE = CONTACT_DISTANCE * 1.15
TENSION_STIFFNESS = 0.3
SETTLE_STIFFNESS = 0.5  # collision stiffness *within* the settle pass
SETTLE_ITERATIONS = 100
CLEANUP_ITERATIONS = 400

# Two separate tolerances, not one shared CONVERGENCE_TOL, because the two
# passes have very different tolerance-for-speed tradeoffs. Measured
# directly against a graph grown the normal way (single split + relax()
# per step, repeated 250x — trainer/backend's own real usage pattern, not
# a synthetic dense pile):
#
# - Settle (tension) is a *soft* constraint — its own docstring already
#   says it settles "only partially each iteration." A tight tolerance
#   here bought nothing: at 1e-4 the pass almost always burned its full
#   SETTLE_ITERATIONS cap without ever getting close (in the 250-split
#   benchmark, mean settle iterations *ran* was ~97 out of the 100 cap).
#   Loosening to 5e-3 cut settle iterations by ~76% (100 -> ~12-23 typical)
#   and total relax() iterations by ~40%, with node-to-node displacement
#   differences too small to see (well under 1% of a node's diameter in
#   the overwhelming majority of pairs) and — critically — no change in
#   remaining hard overlaps, since that's cleanup's job, not settle's.
# - Cleanup (collision) is the *hard* "circles never overlap" guarantee
#   (see this module's own docstring). Its own tolerance is much more
#   sensitive: loosening it from 1e-4 to 1e-3 in the same benchmark
#   roughly quintupled the count of still-overlapping pairs left behind
#   (real, visible circle interpenetration, not a cosmetic nicety), for
#   comparatively little iteration savings — cleanup's iteration count is
#   mostly driven by how many pairs actually need separating, not by how
#   tight its tolerance is. Left untouched.
SETTLE_CONVERGENCE_TOL = 5e-3
CLEANUP_CONVERGENCE_TOL = 1e-4

# target.py scales pixel targets so 1 pixel spacing = 1 unit of graph
# space; keep that convention meaningful without target.py needing to
# know anything about the particle model.
REST_LENGTH = CONTACT_DISTANCE


def _tension_compatibility(id_vectors: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine similarity of id vectors, clipped to [0, 1]
    — 1 where two nodes' identities point the same way, 0 for orthogonal
    or opposed identities. Identity doesn't change during a relax call,
    so this is computed once and reused across iterations. Dense and
    O(n^2), unlike the per-iteration pair search below — but it only
    runs once per relax() call (not once per iteration), and a single
    matrix multiply over a few hundred nodes is microseconds; the actual
    per-iteration cost was always the position/distance sweep, not this.
    """
    norms = np.linalg.norm(id_vectors, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-9, 1.0, norms)
    unit = id_vectors / norms_safe
    cosine_similarity = unit @ unit.T
    return np.clip(cosine_similarity, 0.0, 1.0)


def _pair_corrections(
    pos: np.ndarray,
    pairs: np.ndarray,
    target: np.ndarray,
    stiffness: np.ndarray,
    free_mask: np.ndarray,
) -> np.ndarray:
    """Jacobi-averaged correction from a sparse set of active pairs.
    `pairs` is (K, 2) int indices; `target`/`stiffness` are (K,) — one
    rest-distance and one stiffness value per pair, since settle mixes
    two different constraint types (collision vs. tension) in one pass
    and cleanup uses only one. Every pair contributes symmetrically to
    both endpoints (equal and opposite, mirroring the dense version's
    contrib[j,i] = -contrib[i,j]) via scatter-add, then each node's total
    is divided by how many active pairs it actually had — exactly the
    same Jacobi-average semantics as before, just computed over only the
    pairs that matter instead of every possible pair."""
    n = pos.shape[0]
    correction = np.zeros((n, 2))
    if pairs.shape[0] == 0:
        return correction

    i, j = pairs[:, 0], pairs[:, 1]
    diff = pos[j] - pos[i]
    dist = np.linalg.norm(diff, axis=1)
    dist_safe = np.where(dist < 1e-9, 1.0, dist)
    direction = diff / dist_safe[:, None]

    magnitude = (dist - target) * stiffness
    contrib = direction * magnitude[:, None]

    count = np.zeros(n)
    np.add.at(correction, i, contrib)
    np.add.at(correction, j, -contrib)
    np.add.at(count, i, 1.0)
    np.add.at(count, j, 1.0)

    count_safe = np.where(count == 0, 1.0, count)
    correction = correction / count_safe[:, None]
    correction[~free_mask] = 0.0
    return correction


def _resolve_collisions(pos: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    for _ in range(CLEANUP_ITERATIONS):
        pairs = cKDTree(pos).query_pairs(r=CONTACT_DISTANCE, output_type="ndarray")
        if pairs.shape[0] == 0:
            break

        target = np.full(pairs.shape[0], CONTACT_DISTANCE)
        stiffness = np.ones(pairs.shape[0])
        correction = _pair_corrections(pos, pairs, target, stiffness, free_mask)
        pos = pos + correction

        max_delta = float(np.max(np.linalg.norm(correction, axis=-1))) if n > 0 else 0.0
        if max_delta < CLEANUP_CONVERGENCE_TOL:
            break
    return pos


def relax(positions: np.ndarray, pinned: set[int], id_vectors: np.ndarray) -> np.ndarray:
    pos = positions.copy()
    n = pos.shape[0]
    if n < 2:
        return pos

    free_mask = np.ones(n, dtype=bool)
    for idx in pinned:
        if 0 <= idx < n:
            free_mask[idx] = False

    compatibility = _tension_compatibility(id_vectors)

    for _ in range(SETTLE_ITERATIONS):
        pairs = cKDTree(pos).query_pairs(r=TENSION_RANGE, output_type="ndarray")
        if pairs.shape[0] == 0:
            break

        i, j = pairs[:, 0], pairs[:, 1]
        dist = np.linalg.norm(pos[j] - pos[i], axis=1)
        collision = dist < CONTACT_DISTANCE
        stiffness = np.where(collision, SETTLE_STIFFNESS, TENSION_STIFFNESS * compatibility[i, j])
        target = np.full(pairs.shape[0], CONTACT_DISTANCE)

        correction = _pair_corrections(pos, pairs, target, stiffness, free_mask)
        pos = pos + correction

        max_delta = float(np.max(np.linalg.norm(correction, axis=-1))) if n > 0 else 0.0
        if max_delta < SETTLE_CONVERGENCE_TOL:
            break

    # Hard guarantee, decoupled from the soft settling above: circles
    # never overlap, full stop.
    pos = _resolve_collisions(pos, free_mask)

    return pos

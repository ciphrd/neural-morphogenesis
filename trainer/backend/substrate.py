"""Scalar fields defined implicitly by node positions, each node
contributing a Gaussian "bump" of influence. Two flavors sharing the same
closed-form Gaussian-sum machinery:

- field_and_gradient: the density field — every source contributes
  equally, evaluated across NUM_LAYERS bandwidths (a fine-to-coarse
  pyramid). Generic sensory infrastructure; nothing consumes it yet.
- weighted_field_and_gradient: each source contributes a *value* (e.g.
  one of its chemical channels) rather than a unit weight, at a single
  bandwidth. This is what the update rule's "Sense" step uses — the
  gradient of a chemical channel's local field, not of node density.

Dimension-agnostic (works for 2D node positions today, would work
unchanged for 3D).
"""

from __future__ import annotations

import numpy as np

NUM_LAYERS = 4
BASE_SIGMA = 1.0  # kernel width (in graph-space units) of the finest layer
SCALE_FACTOR = 2.0  # each subsequent layer's sigma multiplies by this


def layer_sigmas() -> np.ndarray:
    return BASE_SIGMA * (SCALE_FACTOR ** np.arange(NUM_LAYERS))


def _gaussian_kernel(
    query_points: np.ndarray, source_points: np.ndarray, sigmas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """diff: (Q, S, D); kernel: (Q, S, len(sigmas)) — exp(-sq_dist / 2*sigma^2)
    for each requested bandwidth."""
    diff = query_points[:, None, :] - source_points[None, :, :]  # (Q, S, D)
    sq_dist = np.sum(diff**2, axis=-1)  # (Q, S)
    sigma2 = (sigmas**2)[None, None, :]  # (1, 1, K)
    kernel = np.exp(-sq_dist[:, :, None] / (2.0 * sigma2))  # (Q, S, K)
    return diff, kernel


def field_and_gradient(
    query_points: np.ndarray, source_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    query_points: (Q, D) — points to evaluate the substrate at (e.g. the
        graph's own node positions, to sense the local field at each node)
    source_points: (S, D) — node positions defining the field (the
        "emitters"); usually the same array as query_points

    Returns:
        values: (Q, NUM_LAYERS) — field value per query point per layer
        gradients: (Q, NUM_LAYERS, D) — gradient vector per query point
            per layer, pointing toward higher field value
    """
    d = query_points.shape[1] if query_points.ndim == 2 else 2
    if query_points.shape[0] == 0 or source_points.shape[0] == 0:
        return (
            np.zeros((query_points.shape[0], NUM_LAYERS)),
            np.zeros((query_points.shape[0], NUM_LAYERS, d)),
        )

    sigmas = layer_sigmas()  # (L,)
    diff, kernel = _gaussian_kernel(query_points, source_points, sigmas)  # (Q,S,D), (Q,S,L)

    values = kernel.sum(axis=1)  # (Q, L)

    # d/dx exp(-|x-p|^2 / 2*sigma^2) = -(x-p)/sigma^2 * exp(...), summed
    # over sources. Note this is exact and zero-cost to differentiate —
    # no autodiff or finite differences needed for a sum of Gaussians.
    sigma2 = (sigmas**2)[None, None, :]
    weight = kernel / sigma2  # (Q, S, L)
    gradients = -np.einsum("qsl,qsd->qld", weight, diff)  # (Q, L, D)

    return values, gradients


def weighted_field_and_gradient(
    query_points: np.ndarray,
    source_points: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Like field_and_gradient, but each source contributes `weights[s, k]`
    (not a uniform 1) to output channel k, at a single bandwidth `sigma`
    rather than a multi-scale stack. This is an *unnormalized* weighted
    sum — sources' contributions add, the way diffusible concentrations
    actually superpose, rather than averaging down as more sources
    appear nearby.

    query_points: (Q, D)
    source_points: (S, D)
    weights: (S, K) — per-source value for each of K channels (e.g. each
        node's chemical vector, K = NUM_CHEMICAL_CHANNELS)
    sigma: scalar kernel bandwidth

    Returns:
        values: (Q, K)
        gradients: (Q, K, D) — pointing toward higher field value
    """
    d = query_points.shape[1] if query_points.ndim == 2 else 2
    k = weights.shape[1] if weights.ndim == 2 else 0
    if query_points.shape[0] == 0 or source_points.shape[0] == 0:
        return (
            np.zeros((query_points.shape[0], k)),
            np.zeros((query_points.shape[0], k, d)),
        )

    diff, kernel = _gaussian_kernel(query_points, source_points, np.array([sigma]))
    kernel = kernel[:, :, 0]  # (Q, S) — single bandwidth, drop the length-1 axis

    values = kernel @ weights  # (Q, S) @ (S, K) -> (Q, K)

    # d/dx sum_s w_sk * exp(-|x-p_s|^2/2sigma^2)
    #   = sum_s w_sk * (-(x-p_s)/sigma^2) * exp(...)
    weighted_kernel = kernel[:, :, None] * weights[None, :, :] / (sigma**2)  # (Q, S, K)
    gradients = -np.einsum("qsk,qsd->qkd", weighted_kernel, diff)  # (Q, K, D)

    return values, gradients

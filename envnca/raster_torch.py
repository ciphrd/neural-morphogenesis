"""Differentiable torch port of raster.py's Gaussian-splat rasterization
and training_raster_distance — same purpose/status as
trainer/backend/distance_torch.py: prototype pieces for the
differentiable-training effort, not wired into any live path. raster.py
remains authoritative for evolve.py's actual (ES) fitness function.

Two substantive changes from the numpy original, both required for
gradients to flow rather than get silently dropped or produce a wrong
zero:

1. rasterize_points_torch() drops the numpy version's "force nearest
   pixel to exactly 1.0" hard override. That line assigns a constant
   regardless of the point's exact sub-pixel position, which zeroes out
   the local gradient exactly where the signal matters most (right at a
   point's own home pixel) — it doesn't just fail to help gradients, it
   actively discards them there. Dropping it changes almost nothing
   numerically: the plain Gaussian kernel already evaluates to ~0.9-1.0
   at the nearest pixel for any sigma this project actually uses
   (sigma=1.5 default -> worst case, a point sitting exactly on a pixel
   boundary, still scores exp(-0.5^2/(2*1.5^2)) ~= 0.945), so this is a
   differentiability fix, not a behavior change. Kept as-is (still
   hard-set) for the *target* raster, which never needs gradients —
   target points are fixed constants — so there's no reason to touch
   build_target_raster()'s own numpy implementation; see
   target_rasters_to_torch() below for how that stays numpy-computed and
   just gets converted once.

2. np.maximum.at's scatter-max becomes torch's own scatter_reduce(...,
   reduce="amax") — an ordinary autograd-aware reduction (gradient flows
   to whichever point actually won the max at each pixel, zero to the
   rest), same "no relaxation needed" situation distance_torch.py's own
   docstring describes for torch.cdist().min() replacing cKDTree.

Everything else (the coarse rotation-search grid, centroid-matching
translation, MSE coverage term, distance-transform outside-shape
penalty) is an unmodified value-for-value port — see raster.py's own
docstring for the reasoning behind each of those, which doesn't change
here. Rotation search itself stays a discrete, non-differentiable choice
of *which* angle wins (same treatment distance_torch.py's own rotation
search gets) — torch.stack(scores).min() automatically backprops through
whichever angle turned out best without needing to special-case argmin.

Agent rasterization now uses *sum*-combine (rasterize_points_sum_torch),
not max — mirrors raster.py's own rasterize_points_sum(), added after a
trained policy was observed collapsing every agent onto a single point:
max-combine made the coverage metric blind to density (piling agents on
one pixel couldn't inflate its score above what one point there already
gave), which let that collapse hide as "one point's worth" of coverage
instead of scoring like the disaster it actually is. rasterize_points_torch
(max-combine) is kept around too, factored through a shared
_rasterize_points_torch() helper — see each public function's own
docstring for when each combine mode is the right one.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from raster import TRAIN_NUM_ANGLES


def _rasterize_points_torch(
    points: torch.Tensor,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
    reduce: str,
) -> torch.Tensor:
    """Shared windowed-kernel-scatter implementation behind
    rasterize_points_torch (reduce="amax") and rasterize_points_sum_torch
    (reduce="sum") below — identical either way except for how
    overlapping points' contributions combine at a shared pixel. See
    each public wrapper's own docstring for when each combine mode is
    the right one. No half_size parameter (unlike raster.rasterize_points)
    — see module docstring for why only the target's own
    build_target_raster() ever needs a nonzero footprint, and that one
    stays numpy-only."""
    device = points.device
    dtype = points.dtype
    n = points.shape[0]
    if n == 0:
        return torch.zeros((resolution, resolution), dtype=dtype, device=device)

    xmin, xmax, ymin, ymax = extent
    scale_x = (resolution - 1) / (xmax - xmin)
    scale_y = (resolution - 1) / (ymax - ymin)
    cx = (points[:, 0] - xmin) * scale_x  # (N,)
    cy = (points[:, 1] - ymin) * scale_y  # (N,)

    # Integer window center — like environment.py's own bilinear corner
    # indices, this is a discrete choice of *which* pixels a point's
    # kernel can possibly touch, not a value being learned; only the
    # kernel *weights* computed from cx/cy below need to stay traceable.
    # round()/long() already carry no gradient on their own, but
    # detaching explicitly documents that this branch is deliberately
    # constant w.r.t. points, not an oversight.
    ix = torch.round(cx).detach().long()
    iy = torch.round(cy).detach().long()

    radius = max(1, math.ceil(3.0 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=device)
    w = offsets.shape[0]

    row_grid = iy[:, None] + offsets[None, :]  # (N, W) int64
    col_grid = ix[:, None] + offsets[None, :]  # (N, W) int64
    dy = row_grid.to(dtype) - cy[:, None]  # (N, W) — traceable through cy
    dx = col_grid.to(dtype) - cx[:, None]  # (N, W) — traceable through cx

    kernel = torch.exp(-(dy[:, :, None] ** 2 + dx[:, None, :] ** 2) / (2.0 * sigma * sigma))  # (N, W, W)

    row_idx = row_grid[:, :, None].expand(n, w, w)
    col_idx = col_grid[:, None, :].expand(n, w, w)
    valid = (row_idx >= 0) & (row_idx < resolution) & (col_idx >= 0) & (col_idx < resolution)

    flat_idx = row_idx.clamp(0, resolution - 1) * resolution + col_idx.clamp(0, resolution - 1)
    flat_idx = flat_idx.reshape(-1)
    flat_kernel = kernel.reshape(-1)
    # An out-of-bounds window cell's clamped index would otherwise land
    # on a real pixel and contaminate it — route those to one disposable
    # "sink" slot past the raster's own resolution*resolution extent
    # instead, then slice it off below.
    sink = resolution * resolution
    flat_idx = torch.where(valid.reshape(-1), flat_idx, torch.full_like(flat_idx, sink))

    # include_self=True: the reduction starts from this all-zero buffer
    # as one of the candidates being combined at each pixel, exactly
    # matching the numpy originals' own `raster = np.zeros(...);
    # np.maximum.at(...)` / `np.add.at(...)`.
    raster_flat = torch.zeros(sink + 1, dtype=dtype, device=device)
    raster_flat = raster_flat.scatter_reduce(0, flat_idx, flat_kernel, reduce=reduce, include_self=True)
    return raster_flat[:sink].view(resolution, resolution)


def rasterize_points_torch(
    points: torch.Tensor,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
) -> torch.Tensor:
    """Differentiable counterpart to raster.rasterize_points() — max-
    combine, the same "coverage without letting concentration inflate a
    pixel's score" property that function's own docstring describes.
    training_raster_distance_torch below uses rasterize_points_sum_torch
    instead for scoring live agent positions (see that function's own
    docstring for why) — this one's kept for anything that genuinely
    wants max-combine semantics."""
    return _rasterize_points_torch(points, resolution, extent, sigma, reduce="amax")


def rasterize_points_sum_torch(
    points: torch.Tensor,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
) -> torch.Tensor:
    """Differentiable counterpart to raster.rasterize_points_sum() — see
    that function's own docstring for why agent rasterization uses sum,
    not max, combine: max-combine made the coverage metric blind to
    density, which let every agent collapsing onto a single point score
    as "one point's worth" of coverage instead of the disaster it
    actually is. torch's scatter_reduce(reduce="sum") is an ordinary
    autograd-aware reduction (gradient flows to *every* contributing
    point at an overlapping pixel, not just a winner) — no relaxation
    needed, same situation distance_torch.py's own docstring describes
    for torch.cdist().min() replacing cKDTree."""
    return _rasterize_points_torch(points, resolution, extent, sigma, reduce="sum")


def raster_distance_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mirrors raster.raster_distance — mean squared error between two
    same-shaped [0,1] rasters."""
    diff = a - b
    return (diff * diff).mean()


def _bilinear_sample_torch(
    field: torch.Tensor, points: torch.Tensor, extent: tuple[float, float, float, float]
) -> torch.Tensor:
    """Differentiable counterpart to raster._bilinear_sample — same
    floor/fractional-weight trick environment.py's own _corners/
    _sample_grid already uses for sensing/deposit, applied here to a
    fixed (non-trainable) constant field instead of the live grid."""
    resolution = field.shape[0]
    xmin, xmax, ymin, ymax = extent
    scale_x = (resolution - 1) / (xmax - xmin)
    scale_y = (resolution - 1) / (ymax - ymin)
    cx = ((points[:, 0] - xmin) * scale_x).clamp(0.0, resolution - 1)
    cy = ((points[:, 1] - ymin) * scale_y).clamp(0.0, resolution - 1)

    x0 = torch.floor(cx)
    y0 = torch.floor(cy)
    x1 = torch.clamp(x0 + 1, max=resolution - 1)
    y1 = torch.clamp(y0 + 1, max=resolution - 1)
    fx = cx - x0
    fy = cy - y0

    x0i, x1i, y0i, y1i = x0.detach().long(), x1.detach().long(), y0.detach().long(), y1.detach().long()
    top = field[y0i, x0i] * (1.0 - fx) + field[y0i, x1i] * fx
    bottom = field[y1i, x0i] * (1.0 - fx) + field[y1i, x1i] * fx
    return top * (1.0 - fy) + bottom * fy


def outside_shape_penalty_torch(
    points: torch.Tensor, target_distance_field: torch.Tensor, extent: tuple[float, float, float, float]
) -> torch.Tensor:
    """Mirrors raster.outside_shape_penalty exactly (same extent-fraction
    normalization — see that function's own docstring for why it's
    normalized by the grid's physical span rather than raster
    resolution)."""
    if points.shape[0] == 0:
        return torch.zeros((), dtype=target_distance_field.dtype, device=target_distance_field.device)
    resolution = target_distance_field.shape[0]
    xmin, xmax, ymin, ymax = extent
    raster_px_per_extent_unit = (resolution - 1) / (xmax - xmin)
    d_raster = _bilinear_sample_torch(target_distance_field, points, extent)
    d_extent_fraction = d_raster / raster_px_per_extent_unit / (xmax - xmin)
    return (d_extent_fraction * d_extent_fraction).mean()


def _rotation_matrix_torch(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def training_raster_distance_torch(
    points: torch.Tensor,
    target_points: torch.Tensor,
    target_raster: torch.Tensor,
    target_distance_field: torch.Tensor,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
    outside_weight: float = 1.0,
    num_angles: int = TRAIN_NUM_ANGLES,
) -> torch.Tensor:
    """Differentiable counterpart to raster.training_raster_distance —
    `points` is the only argument expected to require grad (a live
    rollout's agent positions); target_points/target_raster/
    target_distance_field are fixed constants (see
    target_rasters_to_torch() below for how the latter two get here)."""
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=points.device)

    centered = points - points.mean(dim=0)
    target_centroid = target_points.mean(dim=0)

    scores = []
    for i in range(num_angles):
        theta = torch.tensor(2.0 * math.pi * i / num_angles, dtype=points.dtype, device=points.device)
        rotated = centered @ _rotation_matrix_torch(theta).T + target_centroid
        candidate_raster = rasterize_points_sum_torch(rotated, resolution, extent, sigma)
        coverage = raster_distance_torch(target_raster, candidate_raster)
        penalty = outside_shape_penalty_torch(rotated, target_distance_field, extent)
        scores.append(coverage + outside_weight * penalty)

    return torch.stack(scores).min()


def target_rasters_to_torch(
    target_raster: np.ndarray, target_distance_field: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-time conversion of raster.py's numpy-precomputed target
    constants (build_target_raster()/build_target_distance_field(),
    computed once per run from the target's own fixed points — see
    those functions' docstrings) into torch tensors on `device`. These
    never need gradients themselves; they just need to be torch tensors
    so training_raster_distance_torch's ops, which mix them with
    `points` (a genuine autograd leaf), stay on one tensor library."""
    return (
        torch.from_numpy(target_raster).float().to(device),
        torch.from_numpy(target_distance_field).float().to(device),
    )

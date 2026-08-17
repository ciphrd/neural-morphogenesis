"""Gaussian-splat rasterization + a smooth raster distance — a
resolution-independent, gradient-friendlier alternative to
alignment.py's Chamfer-distance fitness. Designed in conversation around
a specific concern: raw Hamming (or IoU/Dice) distance on a hard 0/1
lattice gives population-based random search almost nothing to climb —
a near-miss scores the same as a far-miss unless a point lands in the
*exact* same cell as a target point, and that problem gets *worse*, not
better, at higher raster resolution (exact-cell hits become rarer).
Splatting each point as a small Gaussian "blob" (unnormalized — peak
exactly 1 at the point itself, not a probability density) and comparing
two rasters via mean squared error keeps a smooth, graded gradient while
staying raster-based and fast — evolve.py wires this in as a drop-in
replacement for alignment.training_alignment_distance.

That MSE term alone only weakly discriminates *how far* outside the
target shape a stray point landed (a flat, diluted per-pixel penalty —
see build_target_distance_field()'s docstring). outside_shape_penalty(),
scored against a precomputed Euclidean distance transform of the
target's footprint, adds an explicit, unbounded, quadratically-growing
penalty for that instead — training_raster_distance() combines both.

Performance note: this runs in evolve.py's hot training path (every
candidate x every rotation-search angle x every generation), so
rasterize_points() below is fully vectorized (no Python-level loop over
points) via broadcasting + np.maximum.at's scatter-max — a naive
per-point Python loop was measured to be the dominant cost otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

# Matches alignment.py's TRAIN_NUM_ANGLES — same reasoning (a coarse,
# cheap rotation grid search is enough for a per-candidate training
# signal; alignment.py's own best_fit_distance is the precise variant
# for one-off reporting, which this module doesn't need an equivalent
# of since nothing currently reports raster distance outside training).
TRAIN_NUM_ANGLES = 16


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rasterize_points(
    points: np.ndarray,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
    half_size: float = 0.0,
) -> np.ndarray:
    """Splats `points` (N,2) onto a `resolution` x `resolution` grid
    covering `extent` = (xmin, xmax, ymin, ymax), in whatever coordinate
    space `points` are already in (envnca's grid-pixel space, same for
    both target points and agent positions). Each point contributes an
    *unnormalized* kernel — peak exactly 1, not a probability density
    that integrates to 1 — and overlapping points combine via max, not
    sum, so a pixel's brightness always means "how close is the nearest
    point (or its footprint)," never "how many points landed here."
    That's what makes this a fair stand-in for a binary occupancy mask:
    piling agents onto one spot can't inflate a cell's value above 1, so
    the metric can't be gamed by clustering instead of actually covering
    the target shape (the same "coverage, not concentration" property
    alignment.py's Chamfer distance gets for free from being symmetric
    — see that module's own docstring).

    `half_size` (in the same units as `extent`/`points`, default 0) is
    the radius of a flat-topped square each point's kernel is 1.0 across
    — not just at the point itself — before the Gaussian falloff (still
    controlled by `sigma`) takes over outside that square. A target
    point represents a whole texel-sized *area* of the original pixel-
    art drawing (see target.py's texel_size()), not a single location,
    so build_target_raster() passes a nonzero half_size to render solid
    filled blocks matching that actual footprint — a plain peaked
    Gaussian (half_size=0, this function's default, matching a true
    point) would only ever light up a small dot near each texel's
    center and leave most of its real area dim. Agents, by contrast,
    really are points (no inherent footprint), so
    training_raster_distance()'s own rasterize_points() calls leave this
    at its default.

    Regardless of `half_size`, the pixel closest to each point is always
    forced to exactly 1.0 afterward. Without that, a point sitting near
    the boundary between two pixels could leave *both* somewhat dim even
    directly under the point, which would make raster resolution and
    Gaussian width fight each other — needing a wider sigma just to
    guarantee visible coverage at finer resolutions, defeating the point
    of choosing a finer resolution in the first place. (For half_size>0
    this is already implied by the flat top, but forcing it explicitly
    costs nothing and keeps the point=footprint behavior consistent.)"""
    raster = np.zeros((resolution, resolution), dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return raster

    xmin, xmax, ymin, ymax = extent
    scale_x = (resolution - 1) / (xmax - xmin)
    scale_y = (resolution - 1) / (ymax - ymin)
    cx = (points[:, 0] - xmin) * scale_x  # (N,) continuous raster-space coords
    cy = (points[:, 1] - ymin) * scale_y  # (N,)

    ix = np.round(cx).astype(np.int64)  # (N,) nearest-pixel index
    iy = np.round(cy).astype(np.int64)

    # extent/resolution are square in every caller (grid_size x
    # grid_size, resolution x resolution), so scale_x == scale_y in
    # practice — used interchangeably below to convert half_size into
    # raster-pixel units.
    half_size_px = half_size * scale_x

    # Only touch each point's local window, not the whole grid — this is
    # what keeps this O(N * (half_size+sigma)^2) instead of
    # O(N * resolution^2).
    radius = max(1, int(np.ceil(half_size_px + 3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)  # (W,)
    w = offsets.shape[0]

    row_grid = iy[:, None] + offsets[None, :]  # (N, W) absolute row per point/window-row
    col_grid = ix[:, None] + offsets[None, :]  # (N, W) absolute col per point/window-col
    dy = row_grid - cy[:, None]  # (N, W) distance from the point's true (sub-pixel) row
    dx = col_grid - cx[:, None]  # (N, W)

    # Distance from the flat-topped square's own edge (0 while inside
    # it), not from the point itself — collapses to plain |dy|/|dx| (a
    # normal peaked Gaussian) when half_size_px is 0.
    dy_outside = np.maximum(0.0, np.abs(dy) - half_size_px)
    dx_outside = np.maximum(0.0, np.abs(dx) - half_size_px)

    # (N, W, W): kernel[n, i, j] is this point's weight at
    # (row_grid[n, i], col_grid[n, j]) — exactly 1.0 for any pixel
    # within the half_size_px square, Gaussian-decaying beyond it.
    kernel = np.exp(-(dy_outside[:, :, None] ** 2 + dx_outside[:, None, :] ** 2) / (2.0 * sigma * sigma))
    row_idx = np.broadcast_to(row_grid[:, :, None], (n, w, w))
    col_idx = np.broadcast_to(col_grid[:, None, :], (n, w, w))

    valid = (row_idx >= 0) & (row_idx < resolution) & (col_idx >= 0) & (col_idx < resolution)
    # np.maximum.at, not plain fancy-index assignment: overlapping
    # points' windows routinely hit the same pixel, and only .at()
    # correctly max-accumulates over repeated indices within one call —
    # `raster[idx] = np.maximum(raster[idx], values)` silently drops all
    # but one contribution per duplicated index.
    np.maximum.at(raster, (row_idx[valid], col_idx[valid]), kernel[valid])

    in_bounds = (iy >= 0) & (iy < resolution) & (ix >= 0) & (ix < resolution)
    raster[iy[in_bounds], ix[in_bounds]] = 1.0

    return raster


def raster_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two same-shaped [0,1] rasters —
    resolution-independent (doesn't scale with pixel count), unlike a
    raw sum of squared differences."""
    diff = a - b
    return float(np.mean(diff * diff))


def build_target_raster(
    target_points: np.ndarray,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
    half_size: float = 0.0,
) -> np.ndarray:
    """Precompute once per training run — target_points never change
    generation to generation, so recomputing this on every
    training_raster_distance() call (population x generations x
    rotation-search angles) would be pure waste. Callers hold onto the
    result and pass it into training_raster_distance() below.

    `half_size` should normally be `target.texel_size(grid_size) / 2`
    (see target.py) — see rasterize_points()'s own docstring for why a
    target point needs a nonzero footprint and an agent point doesn't."""
    return rasterize_points(target_points, resolution, extent, sigma, half_size=half_size)


def build_target_distance_field(target_raster: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Precompute once per training run, alongside target_raster (the
    target is fixed for the whole run — see build_target_raster()'s own
    docstring for why this matters).

    Returns, for every pixel of the lattice, the Euclidean distance (in
    raster pixels) to the nearest pixel that's actually *inside* the
    target's footprint (`target_raster > threshold`) — 0 for pixels
    already inside it, growing the further outside you go. This is the
    field outside_shape_penalty() samples: raster_distance()'s plain
    per-pixel MSE treats "one pixel past the edge" and "clear across the
    grid" almost the same (each contributes at most 1 to that pixel's
    own squared error, capped and undiscriminated by how far off it
    actually is), and that flat penalty gets further diluted by however
    many thousands of correctly-agreeing background pixels sit in the
    mean alongside it. A distance transform fixes both problems at once:
    the penalty for straying outside the shape grows quadratically with
    distance (see outside_shape_penalty), and it's evaluated once per
    *agent* rather than diluted across the whole lattice's pixel count."""
    inside = target_raster > threshold
    if not inside.any():
        # Degenerate target (nothing rasterized) — no defined "outside"
        # to measure distance from; contribute no penalty rather than
        # raising or fabricating an arbitrary distance.
        return np.zeros_like(target_raster)
    return distance_transform_edt(~inside)


def _bilinear_sample(field: np.ndarray, points: np.ndarray, extent: tuple[float, float, float, float]) -> np.ndarray:
    """Samples `field` (resolution x resolution) at each of `points`
    (N,2), which live in the same coordinate space as `extent` — same
    grid-pixel-to-raster-pixel mapping rasterize_points() uses, just
    read instead of written. Bilinear rather than nearest-pixel: agent
    positions are continuous, and outside_shape_penalty()'s whole point
    is to grade distance smoothly, not snap it to whatever raster cell a
    point happens to round into."""
    resolution = field.shape[0]
    xmin, xmax, ymin, ymax = extent
    scale_x = (resolution - 1) / (xmax - xmin)
    scale_y = (resolution - 1) / (ymax - ymin)
    cx = np.clip((points[:, 0] - xmin) * scale_x, 0.0, resolution - 1)
    cy = np.clip((points[:, 1] - ymin) * scale_y, 0.0, resolution - 1)

    x0 = np.floor(cx).astype(np.int64)
    y0 = np.floor(cy).astype(np.int64)
    x1 = np.minimum(x0 + 1, resolution - 1)
    y1 = np.minimum(y0 + 1, resolution - 1)
    fx = cx - x0
    fy = cy - y0

    top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def outside_shape_penalty(
    points: np.ndarray,
    target_distance_field: np.ndarray,
    extent: tuple[float, float, float, float],
) -> float:
    """Mean squared distance from each of `points` to the nearest pixel
    actually inside the target's footprint, normalized as a *fraction of
    the grid's own span* (extent) rather than of the distance field's
    raster resolution. That distinction matters: the distance field is
    computed on a `resolution`x`resolution` lattice purely as an
    implementation detail (a cheap way to get a Euclidean distance
    transform), but the resulting penalty should mean the same thing
    physically ("this agent is N% of the grid's width away from the
    shape") whatever --raster-resolution happens to be set to — dividing
    by `resolution` instead would make the penalty *shrink* every time
    the lattice got finer, for the same real-world miss.

    0 for a point sitting inside the shape; grows quadratically the
    further outside it strayed, unbounded (unlike raster_distance()'s
    flat, capped-at-1-per-pixel contribution that gets diluted further
    by however many background pixels sit in that mean). See
    build_target_distance_field()'s docstring for why this term exists
    alongside raster_distance() rather than replacing it."""
    if points.shape[0] == 0:
        return 0.0
    resolution = target_distance_field.shape[0]
    xmin, xmax, ymin, ymax = extent
    raster_px_per_extent_unit = (resolution - 1) / (xmax - xmin)
    d_raster = _bilinear_sample(target_distance_field, points, extent)
    d_extent_fraction = d_raster / raster_px_per_extent_unit / (xmax - xmin)
    return float(np.mean(d_extent_fraction * d_extent_fraction))


def training_raster_distance(
    points: np.ndarray,
    target_points: np.ndarray,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
    outside_weight: float = 1.0,
    num_angles: int = TRAIN_NUM_ANGLES,
    track_best_raster: bool = False,
) -> float | tuple[float, np.ndarray | None]:
    """Rotation-search fitness scoring, shaped exactly like
    alignment.training_alignment_distance (same coarse angle grid,
    analytic centroid-matching translation — see that function's own
    docstring for why: nothing pins agent motion to the target's pose,
    so fitness has to search over orientation rather than compare a
    fixed one), but scored via raster_distance() plus
    outside_shape_penalty() against precomputed target_raster /
    target_distance_field, instead of a KD-tree Chamfer distance.

    The returned score is `coverage + outside_weight * penalty`:
    coverage (raster_distance) is what pulls agents *into* the shape and
    spread across its whole footprint (a candidate clustered in one
    corner of an otherwise-covered target still scores badly here, same
    "coverage, not concentration" property rasterize_points()'s max-
    combine gives it); penalty (outside_shape_penalty) is what actually
    punishes straying outside it, growing with distance rather than
    MSE's flat per-pixel contribution. The best-scoring rotation is
    chosen against this *combined* score, not coverage alone, so the
    rotation search itself is already shaped by the outside penalty.

    `track_best_raster`, off by default (the hot training path doesn't
    need it — one less array kept alive per call), returns the winning
    rotation's own raster alongside the distance when set. That's for
    train_server.py's end-of-generation debug snapshot: it's the raster
    "ready for" direct comparison against target_raster, i.e. the same
    one training actually scored the winner against, as opposed to the
    winner's raw (un-rotated) replay positions."""
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        return (float("inf"), None) if track_best_raster else float("inf")

    centered = points - points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    best = float("inf")
    best_raster: np.ndarray | None = None
    for i in range(num_angles):
        theta = 2.0 * np.pi * i / num_angles
        rotated = centered @ _rotation_matrix(theta).T + target_centroid
        candidate_raster = rasterize_points(rotated, resolution, extent, sigma)
        coverage = raster_distance(target_raster, candidate_raster)
        penalty = outside_shape_penalty(rotated, target_distance_field, extent)
        dist = coverage + outside_weight * penalty
        if dist < best:
            best = dist
            if track_best_raster:
                best_raster = candidate_raster

    return (best, best_raster) if track_best_raster else best

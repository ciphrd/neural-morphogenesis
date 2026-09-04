"""Gaussian-splat rasterization + a smooth raster distance — ported from
envnca/raster.py. This is this project's own evolution-training fitness
function (evolve.py's own rollout()/_score_fitness call
training_raster_distance() below directly), in place of
distance.py/alignment.py's Chamfer distance — that strategy was tried
too, and for a while was the live fitness function, before this module
was brought back (see evolve.py's own module docstring). train_server.py's
own _save_generation_images() also calls training_raster_distance()
directly, on each generation's winning positions, to render the
`..._agents.png` debug raster the dashboard shows next to
`..._target.png` — a second, display-only use of the same function
fitness scoring already calls, not a separate code path.
Hard Hamming/IoU scoring gives evolutionary search almost nothing to climb: a
near miss and a far miss both fail until a point enters the exact target cell.
Candidate particles are therefore Gaussian-splatted into weighted density and
smoothly saturated into bounded occupancy. A fine-to-coarse pyramid preserves
the broad attraction basin while retaining high-resolution silhouette detail.
Separate coverage, spill, boundary, and crowding terms prevent background
dilution and stop density collapse from hiding behind occupancy saturation.

Unlike envnca (whose own domain IS its simulation grid, in grid-pixel
units), this project's particle positions and target points already
live in MpmCore's fixed [0,1]^2 domain (see targets.py) — `extent` below
is always evolve.py's own RASTER_EXTENT = (0, 1, 0, 1), not a
resolution-dependent value, but every function here stays generic over
`extent` regardless, unchanged from envnca's own version.

Performance note: this runs in evolve.py's hot training path (every
candidate x every rotation-search angle x every generation x every
near-the-end capture snapshot — see evolve.py's own CAPTURE_OFFSETS), so
rasterize_points()/rasterize_points_sum() below are fully vectorized (no
Python-level loop over points) via broadcasting + np.maximum.at's
scatter-max — a naive per-point Python loop was measured to be the
dominant cost otherwise, in envnca's own version of this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

# Matches alignment.py's own TRAIN_NUM_ANGLES — same reasoning (a coarse,
# cheap rotation grid search is enough for a per-candidate training
# signal).
TRAIN_NUM_ANGLES = 16
FITNESS_MODEL_VERSION = 1

# A fine-to-coarse image pyramid. The finest level supplies detailed
# silhouette pressure while pooled levels keep a useful signal when a
# candidate is still several pixels away from the target.
FITNESS_PYRAMID_FACTORS = (1, 2, 4, 8)
FITNESS_PYRAMID_WEIGHTS = (0.50, 0.25, 0.15, 0.10)


@dataclass(frozen=True)
class RasterFitnessBreakdown:
    """Individually inspectable terms from one aligned raster score."""

    total: float
    coverage: float
    spill: float
    boundary: float
    crowding: float
    angle: float


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
    space `points` are already in (this project's MpmCore [0,1]^2
    domain, same for both target points and particle positions). Each
    point contributes an *unnormalized* kernel — peak exactly 1, not a
    probability density that integrates to 1 — and overlapping points
    combine via max, not sum, so a pixel's brightness always means "how
    close is the nearest point (or its footprint)," never "how many
    points landed here." That's what makes this a fair stand-in for a
    binary occupancy mask: piling particles onto one spot can't inflate
    a cell's value above 1, so the metric can't be gamed by clustering
    instead of actually covering the target shape (the same "coverage,
    not concentration" property distance.py's Chamfer distance gets for
    free from being symmetric — see that module's own docstring).

    `half_size` (in the same units as `extent`/`points`, default 0) is
    the radius of a flat-topped square each point's kernel is 1.0 across
    — not just at the point itself — before the Gaussian falloff (still
    controlled by `sigma`) takes over outside that square. A target
    point represents a whole texel-sized *area* of the original pixel-
    art drawing (see targets.py's texel_size()), not a single location,
    so build_target_raster() passes a nonzero half_size to render solid
    filled blocks matching that actual footprint — a plain peaked
    Gaussian (half_size=0, this function's default, matching a true
    point) would only ever light up a small dot near each texel's
    center and leave most of its real area dim. Particles, by contrast,
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

    # extent/resolution are square in every caller (this project's own
    # RASTER_EXTENT / --raster-resolution) — used interchangeably below
    # to convert half_size into raster-pixel units.
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


def rasterize_points_sum(
    points: np.ndarray,
    resolution: int,
    extent: tuple[float, float, float, float],
    sigma: float,
    particle_weight: float = 1.0,
) -> np.ndarray:
    """Particle-side counterpart to rasterize_points() above, differing
    in exactly one way: overlapping points combine via *sum*, not max.
    See training_raster_distance()'s own docstring for why this exists:
    max-combine's whole point was making the coverage metric insensitive
    to *how many* points landed near a pixel, so clustering couldn't
    inflate a cell's score above what a single point there would already
    give — but that same insensitivity let a genuinely bad failure mode
    hide in plain sight. A trained policy can collapse every particle
    onto one point, and under max-combine that collapse rasterizes as
    "one point's worth" of coverage instead of the disaster it actually
    is — plausible enough to survive as a stable, self-reinforcing local
    optimum instead of being trained away. Sum-combine restores the
    missing penalty: N particles piled on one pixel contribute ~N there,
    and squared-error against the target's own peak-1 reference grows
    with the pile-up, not just with whether *a* point is nearby. A
    properly spread candidate (particles each near a distinct target
    texel, only mild kernel-tail overlap between neighbors) still comes
    out close to the target's own ~1-per-covered-pixel scale, so this
    doesn't meaningfully change what a *good* placement scores — only
    what a collapsed one does.

    No half_size parameter (unlike rasterize_points) and no "force
    nearest pixel to 1.0" override — both belong to the target's *fixed*
    footprint semantics, not to a live candidate raster where letting
    density show through is now the entire point."""
    raster = np.zeros((resolution, resolution), dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return raster

    xmin, xmax, ymin, ymax = extent
    scale_x = (resolution - 1) / (xmax - xmin)
    scale_y = (resolution - 1) / (ymax - ymin)
    cx = (points[:, 0] - xmin) * scale_x
    cy = (points[:, 1] - ymin) * scale_y

    ix = np.round(cx).astype(np.int64)
    iy = np.round(cy).astype(np.int64)

    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    w = offsets.shape[0]

    row_grid = iy[:, None] + offsets[None, :]
    col_grid = ix[:, None] + offsets[None, :]
    dy = row_grid - cy[:, None]
    dx = col_grid - cx[:, None]

    kernel = np.exp(-(dy[:, :, None] ** 2 + dx[:, None, :] ** 2) / (2.0 * sigma * sigma))
    row_idx = np.broadcast_to(row_grid[:, :, None], (n, w, w))
    col_idx = np.broadcast_to(col_grid[:, None, :], (n, w, w))

    valid = (row_idx >= 0) & (row_idx < resolution) & (col_idx >= 0) & (col_idx < resolution)
    # np.add.at, not plain fancy-index assignment — same reasoning as
    # rasterize_points()'s np.maximum.at: overlapping points' windows
    # routinely hit the same pixel, and only .at() correctly accumulates
    # over repeated indices within one call.
    if not np.isfinite(particle_weight) or particle_weight <= 0.0:
        raise ValueError("particle_weight must be finite and positive")
    np.add.at(raster, (row_idx[valid], col_idx[valid]), particle_weight * kernel[valid])

    return raster


def raster_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two same-shaped [0,1] rasters —
    resolution-independent (doesn't scale with pixel count), unlike a
    raw sum of squared differences."""
    diff = a - b
    return float(np.mean(diff * diff))


def occupancy_reference(
    target_raster: np.ndarray,
    sigma: float,
    expected_weighted_particles: float,
    target_occupancy: float = 0.9,
) -> float:
    """Calibrate summed particle density into resolution-independent occupancy.

    The target raster mass changes quadratically with resolution, while a
    peak-one Gaussian's raster mass depends on sigma. Their ratio defines the
    expected interior density for a full target, so increasing resolution can
    sharpen the measurement without silently changing its density semantics.
    """
    if not np.isfinite(expected_weighted_particles) or expected_weighted_particles <= 0.0:
        raise ValueError("expected_weighted_particles must be finite and positive")
    if not 0.0 < target_occupancy < 1.0:
        raise ValueError("target_occupancy must be strictly between zero and one")
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_mass = float(
        np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2.0 * sigma * sigma)).sum()
    )
    target_mass = max(float(target_raster.sum()), np.finfo(np.float64).eps)
    expected_density = expected_weighted_particles * kernel_mass / target_mass
    return expected_density / -np.log1p(-target_occupancy)


def bounded_occupancy(weighted_density: np.ndarray, reference: float) -> np.ndarray:
    """Map additive particle density to a smooth occupancy field in [0, 1]."""
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("occupancy reference must be finite and positive")
    return -np.expm1(-np.maximum(weighted_density, 0.0) / reference)


def _average_pool(field: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return field
    height = field.shape[0] - field.shape[0] % factor
    width = field.shape[1] - field.shape[1] % factor
    if height == 0 or width == 0:
        raise ValueError(f"pooling factor {factor} exceeds raster shape {field.shape}")
    cropped = field[:height, :width]
    return cropped.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _multiscale_shape_terms(
    target: np.ndarray,
    candidate: np.ndarray,
    factors: tuple[int, ...] = FITNESS_PYRAMID_FACTORS,
    weights: tuple[float, ...] = FITNESS_PYRAMID_WEIGHTS,
) -> tuple[float, float]:
    """Return target-normalized missing-coverage and outside-occupancy terms."""
    if len(factors) != len(weights) or not factors:
        raise ValueError("fitness pyramid factors and weights must have equal nonzero lengths")
    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("fitness pyramid weights must sum to a positive value")

    coverage = 0.0
    spill = 0.0
    eps = np.finfo(np.float64).eps
    for factor, weight in zip(factors, weights):
        t = _average_pool(target, factor)
        c = _average_pool(candidate, factor)
        target_mass = max(float(t.sum()), eps)
        missing = np.maximum(t - c, 0.0)
        coverage += weight * float(np.sum(t * missing * missing) / target_mass)
        spill += weight * float(np.sum((1.0 - t) * c * c) / target_mass)
    return coverage / weight_sum, spill / weight_sum


def _boundary_loss(target: np.ndarray, candidate: np.ndarray, sampling_factor: int = 4) -> float:
    """Compare silhouette edges after suppressing individual-particle grain.

    The full-resolution occupancy deliberately preserves fine geometry, but its
    individual Gaussian splats also create interior micro-edges. Pooling only
    for this term makes it measure the tissue silhouette rather than particle
    sampling noise; coverage still retains the full-resolution level.
    """
    target = _average_pool(target, sampling_factor)
    candidate = _average_pool(candidate, sampling_factor)
    target_dy, target_dx = np.gradient(target)
    candidate_dy, candidate_dx = np.gradient(candidate)
    target_edge = np.hypot(target_dx, target_dy)
    candidate_edge = np.hypot(candidate_dx, candidate_dy)
    denominator = max(float(np.sum(target_edge * target_edge)), np.finfo(np.float64).eps)
    diff = target_edge - candidate_edge
    return float(np.sum(diff * diff) / denominator)


def _crowding_loss(
    weighted_density: np.ndarray,
    expected_density: float,
    target_raster: np.ndarray,
    tolerance: float = 2.0,
) -> float:
    """Robustly penalize density spikes without letting one pile-up dominate."""
    relative = weighted_density / max(expected_density, np.finfo(np.float64).eps)
    excess = np.maximum(relative - tolerance, 0.0)
    target_mass = max(float(target_raster.sum()), np.finfo(np.float64).eps)
    return float(np.sum(np.log1p(excess * excess)) / target_mass)


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
    rotation-search angles x snapshots) would be pure waste. Callers
    hold onto the result and pass it into training_raster_distance()
    below.

    `half_size` should normally be `target.texel_size() / 2` (see
    targets.py) — see rasterize_points()'s own docstring for why a
    target point needs a nonzero footprint and a particle doesn't."""
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
    *particle* rather than diluted across the whole lattice's pixel
    count."""
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
    read instead of written. Bilinear rather than nearest-pixel: particle
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
    physically ("this particle is N% of the domain's width away from the
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
    particle_weight: float = 1.0,
    expected_weighted_particles: float | None = None,
    target_occupancy: float = 0.9,
    coverage_weight: float = 1.0,
    spill_weight: float = 1.0,
    boundary_weight: float = 0.25,
    crowding_weight: float = 0.05,
    alignment_refinement_steps: int = 2,
    return_breakdown: bool = False,
) -> float | tuple[float, np.ndarray | None] | tuple[float, np.ndarray | None, RasterFitnessBreakdown | None]:
    """Rotation-search fitness scoring, shaped exactly like
    alignment.training_alignment_distance (same coarse angle grid,
    analytic centroid-matching translation — see that function's own
    docstring for why: nothing pins particle growth to the target's
    pose, so fitness has to search over orientation rather than compare
    a fixed one), but scored via raster_distance() plus
    outside_shape_penalty() against precomputed target_raster /
    target_distance_field, instead of a KD-tree Chamfer distance.

    Weighted summed density is first mapped to bounded occupancy with
    ``1 - exp(-density / reference)``. The reference is calibrated from the
    target raster mass, Gaussian kernel mass, and desired represented particle
    count, making target and candidate comparable across raster resolutions.
    The score combines multiscale missing coverage, multiscale outside
    occupancy plus physical outside distance, fine boundary disagreement, and
    a robust raw-density crowding penalty.

    `track_best_raster`, off by default (the hot training path doesn't
    need it — one less array kept alive per call), returns the winning
    rotation's own raster alongside the distance when set. That's for a
    caller (e.g. train_server.py's end-of-generation debug snapshot)
    that wants the raster "ready for" direct comparison against
    target_raster, i.e. the same one training actually scored the
    winner against, as opposed to the winner's raw (un-rotated) replay
    positions."""
    if points.shape[0] == 0 or target_points.shape[0] == 0:
        if return_breakdown:
            return float("inf"), None, None
        return (float("inf"), None) if track_best_raster else float("inf")

    for name, value in (
        ("coverage_weight", coverage_weight),
        ("spill_weight", spill_weight),
        ("boundary_weight", boundary_weight),
        ("crowding_weight", crowding_weight),
        ("outside_weight", outside_weight),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    desired_particles = (
        float(expected_weighted_particles)
        if expected_weighted_particles is not None
        else float(points.shape[0]) * particle_weight
    )
    reference = occupancy_reference(target_raster, sigma, desired_particles, target_occupancy)
    expected_density = reference * -np.log1p(-target_occupancy)

    centered = points - points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    best = float("inf")
    best_raster: np.ndarray | None = None
    best_breakdown: RasterFitnessBreakdown | None = None

    def evaluate(theta: float) -> None:
        nonlocal best, best_raster, best_breakdown
        rotated = centered @ _rotation_matrix(theta).T + target_centroid
        candidate_density = rasterize_points_sum(
            rotated, resolution, extent, sigma, particle_weight=particle_weight
        )
        candidate_raster = bounded_occupancy(candidate_density, reference)
        coverage, raster_spill = _multiscale_shape_terms(target_raster, candidate_raster)
        distance_spill = outside_shape_penalty(rotated, target_distance_field, extent)
        spill = raster_spill + outside_weight * distance_spill
        boundary = _boundary_loss(target_raster, candidate_raster)
        crowding = _crowding_loss(candidate_density, expected_density, target_raster)
        dist = (
            coverage_weight * coverage
            + spill_weight * spill
            + boundary_weight * boundary
            + crowding_weight * crowding
        )
        if dist < best:
            best = dist
            if track_best_raster or return_breakdown:
                best_raster = candidate_raster
            best_breakdown = RasterFitnessBreakdown(
                total=dist,
                coverage=coverage,
                spill=spill,
                boundary=boundary,
                crowding=crowding,
                angle=theta,
            )

    for i in range(num_angles):
        evaluate(2.0 * np.pi * i / num_angles)

    # Successive local thirds turn the cheap 16-angle search's 22.5-degree
    # bins into roughly 2.5-degree pose precision with only four additional
    # raster evaluations. That prevents orientation quantization from hiding
    # the finer silhouette differences measured by the higher-resolution field.
    if num_angles >= 4 and best_breakdown is not None:
        step = 2.0 * np.pi / num_angles
        for _ in range(max(0, alignment_refinement_steps)):
            step /= 3.0
            center_angle = best_breakdown.angle
            evaluate(center_angle - step)
            evaluate(center_angle + step)

    if return_breakdown:
        return best, best_raster, best_breakdown
    return (best, best_raster) if track_best_raster else best

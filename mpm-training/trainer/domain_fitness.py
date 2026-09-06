"""Material-area fitness, independent of subdivision and numerical capacity.

Pixels are cell averages on [0,1]^2. Triangle/pixel intersections are integrated
exactly before rotation alignment (bilinear image resampling). The unblurred
area fractions drive stopping; multiscale losses drive evolutionary selection.
"""
from config import CONFIG
_DEFAULTS = CONFIG["run"]

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt
from raster import RasterFitnessBreakdown, _average_pool, _boundary_loss
from triangle_vertices import unwrap_vertices

FITNESS_MODEL_VERSION = 3

def target_mask(target, resolution):
    return target.mask(resolution)

def centered_triangles(vertices, center):
    triangles = unwrap_vertices(vertices)
    # Lift the periodic cloud through the largest empty centroid gap per axis.
    # As with the simulator, individual triangles must be smaller than a period/2.
    centers = triangles.mean(axis=1)
    for axis in range(2):
        c = centers[:, axis] % 1
        ordered = np.sort(c)
        gap = np.diff(np.r_[ordered, ordered[0]+1])
        start = ordered[(int(np.argmax(gap))+1) % len(ordered)]
        lifted = start + (c-start) % 1
        triangles[:, :, axis] += (lifted-centers[:, axis])[:, None]
    cross = np.linalg.det(triangles[:, 1:] - triangles[:, :1])
    areas = np.abs(cross)/2
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 1e-16:
        raise ValueError('empty or degenerate material domains')
    centroid = np.sum(triangles.mean(axis=1)*areas[:, None], axis=0)/total
    return triangles-centroid+center, total

def _raster_batches(triangles, resolution):
    scaled = np.asarray(triangles, float).reshape(-1, 3, 2)*resolution
    if not len(scaled):
        return
    low = np.maximum(0, np.floor(scaled.min(axis=1)))
    high = np.minimum(resolution, np.ceil(scaled.max(axis=1)))
    counts = np.maximum(0, high-low).prod(axis=1)
    start = 0
    cells = 0
    for i, count in enumerate(counts):
        if i > start and (cells+count > 32768 or i-start >= 256):
            yield scaled[start:i]
            start, cells = i, 0
        cells += count
    if start < len(scaled):
        yield scaled[start:]

def rasterize_triangles(triangles, resolution):
    """Add exact triangle/pixel intersection areas using batched convex clipping."""
    out = np.zeros((resolution, resolution), dtype=float)
    for chunk in _raster_batches(triangles, resolution):
        low = np.maximum(0, np.floor(chunk.min(axis=1)).astype(int))
        high = np.minimum(resolution, np.ceil(chunk.max(axis=1)).astype(int))
        sizes = np.maximum(0, high-low)
        counts = sizes.prod(axis=1)
        indices = np.repeat(np.arange(len(chunk)), counts)
        if not len(indices):
            continue
        local = np.arange(len(indices))-np.repeat(np.cumsum(counts)-counts, counts)
        xs = low[indices, 0]+local % sizes[indices, 0]
        ys = low[indices, 1]+local // sizes[indices, 0]
        # Convex triangle clipped to four half planes has at most seven vertices.
        poly = np.zeros((len(indices), 8, 2))
        poly[:, :3] = chunk[indices]
        n = np.full(len(indices), 3)
        rows = np.arange(len(indices))[:, None]
        slots = np.arange(8)[None, :]
        for axis, boundary, sign in ((0, xs, 1), (0, xs+1, -1), (1, ys, 1), (1, ys+1, -1)):
            a = poly
            b = poly[rows, (slots+1) % np.maximum(n[:, None], 1)]
            da = sign*(a[:, :, axis]-boundary[:, None])
            db = sign*(b[:, :, axis]-boundary[:, None])
            valid = slots < n[:, None]
            inside_a = da >= 0; inside_b = db >= 0
            crossing = valid & (inside_a != inside_b)
            keep = valid & inside_b
            amount = crossing.astype(int)+keep.astype(int)
            offset = np.cumsum(amount, axis=1)-amount
            clipped = np.zeros_like(poly)
            r, s = np.nonzero(crossing)
            ratio = da[r, s]/(da[r, s]-db[r, s])
            clipped[r, offset[r, s]] = a[r, s]+ratio[:, None]*(b[r, s]-a[r, s])
            r, s = np.nonzero(keep)
            clipped[r, offset[r, s]+crossing[r, s]] = b[r, s]
            n = amount.sum(axis=1)
            poly = clipped
        b = poly[rows, (slots+1) % np.maximum(n[:, None], 1)]
        cross = poly[:, :, 0]*b[:, :, 1]-poly[:, :, 1]*b[:, :, 0]
        area = np.abs(np.sum(np.where(slots < n[:, None], cross, 0), axis=1))/2
        np.add.at(out, (ys, xs), area)
    return out

def rotate_density(density, angle, center):
    # Row-vector geometry uses raster.py's clockwise rotation convention.
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, s], [-s, c]])
    origin = np.array(center)[::-1]*len(density)-0.5
    return affine_transform(density, matrix, offset=origin-matrix@origin,
                            output_shape=density.shape, order=1, mode='grid-constant', cval=0, prefilter=False)

@dataclass(frozen=True)
class MatchMetrics:
    missing: float
    spill: float
    overlap: float

    @property
    def error(self):
        return self.missing+self.spill+self.overlap

@dataclass
class DomainEvaluation:
    total: float
    raster: np.ndarray | None
    breakdown: RasterFitnessBreakdown | None
    match: MatchMetrics

def evaluate_domains(vertices, target, mask, *, coverage_weight=_DEFAULTS["fitnessCoverageWeight"], spill_weight=_DEFAULTS["fitnessSpillWeight"],
                     boundary_weight=_DEFAULTS["fitnessBoundaryWeight"], crowding_weight=_DEFAULTS["fitnessCrowdingWeight"], outside_weight=_DEFAULTS["outsideWeight"],
                     num_angles=16, refinement_steps=2):
    fail = DomainEvaluation(float('inf'), None, None, MatchMetrics(float('inf'), float('inf'), float('inf')))
    vertices = np.asarray(vertices, float).reshape(-1, 3, 2)
    if not len(vertices) or not np.isfinite(vertices).all() or not mask.any():
        return fail
    center = target.center
    try:
        triangles, material_area = centered_triangles(vertices, center)
    except ValueError:
        return fail
    density = rasterize_triangles(triangles, len(mask))
    mass = float(mask.sum())
    material_pixels = material_area*len(mask)**2
    # All nonzero soft-edge support is part of the target for distance
    # purposes; coverage/spill still use its fractional alpha value.
    distances = distance_transform_edt(mask <= 0)/len(mask)
    best = fail
    best_match = fail.match
    match_angle = 0.
    def evaluate(angle):
        nonlocal best, best_match, match_angle
        d = density if angle == 0 else rotate_density(density, angle, center)
        occupancy = np.clip(d, 0, 1)
        # Include cropped material so clipping cannot hide excess geometry.
        escaped = max(0., material_pixels-float(d.sum()))/mass
        match = MatchMetrics(float(np.maximum(mask-occupancy, 0).sum()/mass),
                             float(np.maximum(occupancy-mask, 0).sum()/mass)+escaped,
                             float(np.maximum(d-1, 0).sum()/mass))
        if match.error < best_match.error - 1e-10:
            best_match = match; match_angle = angle
        coverage = spill = 0.
        for factor, weight in zip((1, 2, 4, 8), (.5, .25, .15, .1)):
            t, p = _average_pool(mask, factor), _average_pool(occupancy, factor)
            coverage += weight*float(np.square(np.maximum(t-p, 0)).sum()/t.sum())
            spill += weight*float(np.square(np.maximum(p-t, 0)).sum()/t.sum())
        spill += escaped+outside_weight*float((occupancy*distances**2).sum()/mass)
        boundary = _boundary_loss(mask, occupancy)
        crowding = match.overlap
        score = coverage_weight*coverage+spill_weight*spill+boundary_weight*boundary+crowding_weight*crowding
        if score < best.total:
            best = DomainEvaluation(score, occupancy, RasterFitnessBreakdown(score, coverage, spill, boundary, crowding, angle), match)
    for i in range(num_angles):
        evaluate(2*np.pi*i/num_angles)
    # Refine the physical match pose; this search is identical in browser playback.
    step = 2*np.pi/num_angles
    for _ in range(refinement_steps):
        step /= 3
        previous_match_angle = match_angle
        for angle in (previous_match_angle-step, previous_match_angle+step):
            evaluate(angle)
    best.match = best_match
    return best

class StableMatchStop:
    """External growth controller; failed settling resumes growth on next step."""
    def __init__(self, enabled=_DEFAULTS["stableStop"], interval=_DEFAULTS["shapeCheckInterval"], confirmations=_DEFAULTS["shapeConfirmations"], settle_steps=_DEFAULTS["shapeSettleSteps"],
                 missing=_DEFAULTS["shapeMissingTolerance"], spill=_DEFAULTS["shapeSpillTolerance"], overlap=_DEFAULTS["shapeOverlapTolerance"]):
        self.enabled, self.interval = enabled, interval
        self.confirmations, self.settle_steps = confirmations, settle_steps
        self.limits = (missing, spill, overlap)
        self.streak = 0
        self.settling_since = None
        self.complete = False
        self.settling_scores = []

    @property
    def growth_enabled(self):
        return self.settling_since is None and not self.complete

    def due(self, step):
        return self.enabled and (step % self.interval == 0 or
            (self.settling_since is not None and step-self.settling_since >= self.settle_steps))

    def observe(self, step, match, score, sampling_blocked=False):
        good = not sampling_blocked and all(np.isfinite(v) and v <= limit for v, limit in zip(
            (match.missing, match.spill, match.overlap), self.limits))
        if not good:
            self.streak = 0; self.settling_since = None; self.settling_scores = []
        elif self.settling_since is not None:
            self.settling_scores.append(score)
            self.complete = step-self.settling_since >= self.settle_steps
        else:
            self.streak += 1
            if self.streak >= self.confirmations:
                self.settling_since = step
                self.settling_scores = [score]
        return self.complete

def stopping_from_args(args):
    return StableMatchStop(args.stable_stop, args.shape_check_interval, args.shape_confirmations,
                          args.shape_settle_steps, args.shape_missing_tolerance,
                          args.shape_spill_tolerance, args.shape_overlap_tolerance)

def score_domains(vertices, target, mask, args):
    return evaluate_domains(vertices, target, mask, coverage_weight=args.fitness_coverage_weight,
        spill_weight=args.fitness_spill_weight, boundary_weight=args.fitness_boundary_weight,
        crowding_weight=args.fitness_crowding_weight, outside_weight=args.outside_weight)

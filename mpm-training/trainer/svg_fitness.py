"""Continuous target-pose fitness with a single exact candidate rasterization."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize_scalar

from domain_fitness import (DomainEvaluation, MatchMetrics, centered_triangles,
                            rasterize_triangles, _DEFAULTS)
from raster import RasterFitnessBreakdown, _average_pool
from svg_target import render_svg

FACTORS = (1, 2, 4, 8)
WEIGHTS = (.5, .25, .15, .1)


def _edge(pooled):
    dy, dx = np.gradient(pooled)
    return np.hypot(dx, dy)


def _pose(target, n, angle, cache=False):
    key = (n, float(angle))
    if key in target._svg_pose_cache:
        return target._svg_pose_cache[key]
    mask, rgb = render_svg(target.svg_source, n, angle, target.svg_transform)
    pyramid = tuple(_average_pool(mask, factor) for factor in FACTORS)
    edge = _edge(pyramid[2])
    result = (mask, rgb, pyramid, edge,
              max(float(np.square(edge).sum()), np.finfo(float).eps),
              np.square(distance_transform_edt(mask <= 0)/n))
    # Only coarse poses repeat across candidates. Do not retain arbitrary
    # optimizer angles or let continuous refinement evict the reusable bank.
    if cache:
        if len(target._svg_pose_cache) >= 32:
            target._svg_pose_cache.pop(next(iter(target._svg_pose_cache)))
        target._svg_pose_cache[key] = result
    return result


def evaluate_svg(vertices, target, mask, *,
                 coverage_weight=_DEFAULTS['fitnessCoverageWeight'],
                 spill_weight=_DEFAULTS['fitnessSpillWeight'],
                 boundary_weight=_DEFAULTS['fitnessBoundaryWeight'],
                 crowding_weight=_DEFAULTS['fitnessCrowdingWeight'],
                 outside_weight=_DEFAULTS['outsideWeight'],
                 colors=None, color_weight=1., num_angles=16,
                 angle_tolerance=1e-5, refinement_basins=2):
    """Minimize weighted fitness over continuous clockwise target rotations.

    Two promising coarse local minima are refined independently. This is a
    bounded local search, not a guarantee of the global minimum. Stopping
    metrics describe the same pose as selection. Returned comparison rasters
    are in the candidate's centered frame; breakdown.angle is the inverse
    transform (candidate to canonical target), as in the raster scorer.
    """
    if num_angles < 4 or refinement_basins < 1 or not np.isfinite(angle_tolerance) or angle_tolerance <= 0:
        raise ValueError('invalid SVG alignment search settings')
    fail = DomainEvaluation(float('inf'), None, None,
                            MatchMetrics(float('inf'), float('inf'), float('inf')))
    vertices = np.asarray(vertices, float).reshape(-1, 3, 2)
    if not len(vertices) or not np.isfinite(vertices).all() or not mask.any():
        return fail
    try:
        triangles, area = centered_triangles(vertices, target.center)
    except ValueError:
        return fail
    n = len(mask)
    mass = float(mask.sum())
    candidate_rgb = None
    if colors is not None and color_weight > 0:
        colors = np.asarray(colors, float)
        if colors.shape != (len(vertices), 3) or not np.isfinite(colors).all():
            return fail
        density, rgb = rasterize_triangles(triangles, n, np.clip(colors, 0, 1))
        candidate_rgb = rgb/np.maximum(density[..., None], 1.)
    else:
        density = rasterize_triangles(triangles, n)
    occupancy = np.clip(density, 0, 1)
    pyramid = tuple(_average_pool(occupancy, factor) for factor in FACTORS)
    edge = _edge(pyramid[2])
    escaped = max(0., area*n*n-float(density.sum()))/mass
    overlap = float(np.maximum(density-1, 0).sum()/mass)
    best = fail

    def evaluate(angle, cache=False):
        nonlocal best
        t, rgb, levels, t_edge, edge_mass, distances = _pose(target, n, angle, cache)
        coverage = spill = 0.
        for reference, candidate, weight in zip(levels, pyramid, WEIGHTS):
            difference = reference-candidate
            denominator = mass / (n/reference.shape[0])**2
            coverage += weight*float(np.square(np.maximum(difference, 0)).sum()/denominator)
            spill += weight*float(np.square(np.minimum(difference, 0)).sum()/denominator)
        spill += escaped + outside_weight*float((occupancy*distances).sum()/mass)
        boundary = float(np.square(t_edge-edge).sum()/edge_mass)
        color = 0. if candidate_rgb is None else float(np.square(candidate_rgb-rgb).sum()/(3*mass))
        score = coverage_weight*coverage + spill_weight*spill + boundary_weight*boundary + crowding_weight*overlap + color_weight*color
        if score < best.total:
            match = MatchMetrics(float(np.maximum(t-occupancy, 0).sum()/mass),
                                 float(np.maximum(occupancy-t, 0).sum()/mass)+escaped, overlap)
            best = DomainEvaluation(score, occupancy,
                RasterFitnessBreakdown(score, coverage, spill, boundary, overlap, -float(angle), color),
                match, candidate_rgb, t, rgb)
        return score

    step = 2*np.pi/num_angles
    scores = np.array([evaluate(i*step, cache=True) for i in range(num_angles)])
    minima = [i for i in range(num_angles)
              if scores[i] <= scores[(i-1) % num_angles] and scores[i] <= scores[(i+1) % num_angles]]
    for i in sorted(minima, key=lambda i: scores[i])[:refinement_basins]:
        minimize_scalar(evaluate, bounds=((i-1)*step, (i+1)*step), method='bounded',
                        options={'xatol': angle_tolerance, 'maxiter': 28})
    return best

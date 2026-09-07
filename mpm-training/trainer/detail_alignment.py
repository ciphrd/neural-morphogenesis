"""Opt-in accurate pose scoring of continuous triangle geometry.

The normal raster alignment supplies a coarse pose. This slower reference
rotates vertices before exact pixel integration and optimizes the weighted
fitness locally. The default scorer and browser stopping retain raster
alignment; run metadata identifies this more expensive training mode.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from domain_fitness import centered_triangles, evaluate_domains


def rotate_vertices(vertices, angle, center):
    c, s = np.cos(angle), np.sin(angle)
    # Same clockwise row-vector convention as domain_fitness.rotate_density.
    return (vertices-center) @ np.array([[c, -s], [s, c]]) + center


def evaluate_precise(vertices, target, mask, *, angle_tolerance=1e-5, **kwargs):
    """Return an exactly integrated score near the standard score's best pose.

    This is a local reference, not a proof of globally optimal alignment. The
    coarse search remains useful for arbitrary shape orientations. Both ends,
    the initial pose and the optimizer result are compared to avoid regression
    within the tested bracket. RGB rotates with its parent triangle.
    """
    if not np.isfinite(angle_tolerance) or angle_tolerance <= 0:
        raise ValueError('angle_tolerance must be finite and positive')
    coarse = evaluate_domains(vertices, target, mask, **kwargs)
    if not np.isfinite(coarse.total):
        return coarse
    triangles, _ = centered_triangles(np.asarray(vertices).reshape(-1, 3, 2), target.center)
    center_angle = coarse.breakdown.angle
    candidates = []

    def score(angle):
        result = evaluate_domains(rotate_vertices(triangles, angle, target.center), target, mask,
                                  num_angles=1, refinement_steps=0, **kwargs)
        # The evaluation was performed at angle zero after rotating geometry.
        from dataclasses import replace
        result.breakdown = replace(result.breakdown, angle=float(angle))
        candidates.append(result)
        return result.total

    radius = 2*np.pi/16
    score(center_angle)
    score(center_angle-radius)
    score(center_angle+radius)
    minimize_scalar(score, bounds=(center_angle-radius, center_angle+radius), method='bounded',
                    options={'xatol':angle_tolerance, 'maxiter':32})
    return min(candidates, key=lambda result: result.total)

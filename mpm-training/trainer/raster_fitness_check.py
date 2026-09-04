"""Deterministic behavioral checks for the live bounded raster fitness.

Run from ``trainer`` with ``.venv/bin/python raster_fitness_check.py``.
These are score-ordering tests rather than brittle golden numbers: they define
the selection behavior evolution relies on while permitting calibrated weight
changes later.
"""
from __future__ import annotations

import numpy as np

from raster import build_target_distance_field, build_target_raster, training_raster_distance
from targets import TargetShape, load_target

EXTENT = (0.0, 1.0, 0.0, 1.0)
EXPECTED_PARTICLES = 400


def _sample_target(target: TargetShape, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(target.points.shape[0], size=EXPECTED_PARTICLES)
    jitter = rng.uniform(-0.45, 0.45, size=(EXPECTED_PARTICLES, 2)) * target.texel_size()
    return target.points[indices].astype(np.float64) + jitter


def _score(
    points: np.ndarray,
    target: TargetShape,
    resolution: int,
    *,
    particle_weight: float = 1.0,
) -> float:
    target_raster = build_target_raster(
        target.points,
        resolution,
        EXTENT,
        1.5,
        half_size=target.texel_size() / 2.0,
    )
    target_distance = build_target_distance_field(target_raster)
    return float(training_raster_distance(
        points,
        target.points,
        target_raster,
        target_distance,
        resolution,
        EXTENT,
        1.5,
        num_angles=1,
        particle_weight=particle_weight,
        expected_weighted_particles=EXPECTED_PARTICLES,
    ))


def main() -> None:
    for target_index, name in enumerate(("circle", "donut", "legs", "line2")):
        target = load_target(name)
        rng = np.random.default_rng(1000 + target_index)
        good = _sample_target(target, rng)
        center = target.points.mean(axis=0)

        left = target.points[target.points[:, 0] < np.median(target.points[:, 0])]
        missing_half = left[rng.integers(left.shape[0], size=EXPECTED_PARTICLES)].astype(np.float64)
        missing_half += rng.uniform(-0.45, 0.45, size=missing_half.shape) * target.texel_size()
        enlarged = (good - center) * 1.35 + center
        collapsed = np.repeat(center[None, :], EXPECTED_PARTICLES, axis=0)
        collapsed += rng.normal(0.0, 0.002, size=collapsed.shape)

        good_score = _score(good, target, 256)
        assert good_score < _score(missing_half, target, 256), name
        assert good_score < _score(enlarged, target, 256), name
        assert good_score < _score(collapsed, target, 256), name

        doubled_score = _score(np.repeat(good, 2, axis=0), target, 256, particle_weight=0.5)
        assert np.isclose(good_score, doubled_score, rtol=1e-12, atol=1e-12), name

        if name == "legs":
            theta = 0.137
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.array([[c, -s], [s, c]])
            rotated = (good - center) @ rotation.T + center
            target_raster = build_target_raster(
                target.points, 256, EXTENT, 1.5, half_size=target.texel_size() / 2.0
            )
            aligned_score = float(training_raster_distance(
                rotated,
                target.points,
                target_raster,
                build_target_distance_field(target_raster),
                256,
                EXTENT,
                1.5,
                expected_weighted_particles=EXPECTED_PARTICLES,
            ))
            assert np.isclose(good_score, aligned_score, rtol=0.03, atol=0.005), (
                good_score,
                aligned_score,
            )

    print("[PASS] bounded raster fitness ordering and particle-density invariance")


if __name__ == "__main__":
    main()

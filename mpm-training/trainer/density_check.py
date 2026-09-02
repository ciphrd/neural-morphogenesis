"""Fast CPU checks for the shared particle-density resolver contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from density import (
    DENSITY_MODEL_VERSION,
    DensityReference,
    resolve_density,
    validate_multiplier,
)
from raster import rasterize_points_sum
from agents_gpu import _SPATIAL_HEADING_DOMAIN, _spatial_uniform01_batch


def main() -> None:
    cases = json.loads((Path(__file__).parent.parent / "core" / "density_cases.json").read_text())
    ref = cases["reference"]
    reference = DensityReference(
        particle_cap=ref["particleCap"],
        initial_particles=ref["initialParticles"],
        chemical_field_n=ref["chemicalFieldN"],
        particle_mass=ref["particleMass"],
        particle_volume=ref["particleVolume"],
        deposit_sigma=ref["depositSigma"],
        chemical_gradient_input_scale=ref["chemicalGradientInputScale"],
        repulsion_strength=ref["repulsionStrength"],
        repulsion_max_delta=ref["repulsionMaxDelta"],
    )
    fields = (
        "spacing", "particle_mass", "particle_volume", "deposit_sigma", "chemical_projection_weight", "splat_radius",
        "chemical_gradient_input_scale", "repulsion_strength", "repulsion_max_delta",
    )
    json_names = {
        "particle_mass": "particleMass",
        "particle_volume": "particleVolume",
        "deposit_sigma": "depositSigma",
        "chemical_projection_weight": "chemicalProjectionWeight",
        "splat_radius": "splatRadius",
        "chemical_gradient_input_scale": "chemicalGradientInputScale",
        "repulsion_strength": "repulsionStrength",
        "repulsion_max_delta": "repulsionMaxDelta",
    }
    for expected in cases["cases"]:
        actual = resolve_density(reference, expected["multiplier"])
        assert actual.model_version == DENSITY_MODEL_VERSION
        assert actual.initial_particles == expected["initialParticles"]
        assert actual.particle_cap == expected["particleCap"]
        for field in fields:
            key = json_names.get(field, field)
            assert np.isclose(getattr(actual, field), expected[key], rtol=1e-12, atol=1e-12), (field, actual, expected)

    q1 = resolve_density(reference, 1.0)
    assert q1.spacing == 0.0027
    assert q1.deposit_sigma == 0.324
    assert q1.splat_radius == 0.004
    for invalid in (0.0, -1.0, float("nan"), float("inf"), 0.25, 8.0):
        try:
            validate_multiplier(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid/unsupported multiplier {invalid!r}")
    assert validate_multiplier(0.25, allow_unsafe=True) == 0.25
    points = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64)
    reference_raster = rasterize_points_sum(points, 32, (0.0, 1.0, 0.0, 1.0), 1.5)
    doubled_raster = rasterize_points_sum(
        np.repeat(points, 2, axis=0), 32, (0.0, 1.0, 0.0, 1.0), 1.5,
        particle_weight=0.5,
    )
    assert np.allclose(reference_raster, doubled_raster)
    spatial_points = np.array([
        [0.5001, 0.5001], [0.5002, 0.5003], [0.72, 0.31], [0.5001, 0.5001],
    ], dtype=np.float32)
    spatial = _spatial_uniform01_batch(12345, spatial_points, _SPATIAL_HEADING_DOMAIN)
    # Same spatial cell/value regardless of numerical particle identity/order.
    assert spatial[0] == spatial[1] == spatial[3]
    permuted = _spatial_uniform01_batch(12345, spatial_points[[2, 0]], _SPATIAL_HEADING_DOMAIN)
    assert permuted[0] == spatial[2] and permuted[1] == spatial[0]
    assert _spatial_uniform01_batch(54321, spatial_points[:1], _SPATIAL_HEADING_DOMAIN)[0] != spatial[0]
    print("[PASS] density resolver, q=1 constants, spatial RNG, and validation")


if __name__ == "__main__":
    main()

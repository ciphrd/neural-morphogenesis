"""Regression checks for the circularly clipped hexagonal seed layout."""
from __future__ import annotations

import numpy as np

from density import INITIAL_PACKING_SPACING_SCALE
from training_sim import seed_blob


def check_circular_hexagonal_seed() -> None:
    spacing = 0.01
    packed_spacing = spacing * INITIAL_PACKING_SPACING_SCALE
    center = np.array([0.5, 0.5], dtype=np.float32)

    # The first three exact Euclidean-radius shells contain 1 + 6 + 6 sites:
    # radii 0, spacing, and sqrt(3)*spacing. This distinguishes circular
    # clipping from the old axial-ring fill, whose next contour was hexagonal.
    positions, *_ = seed_blob(13, tuple(center), spacing, seed=17)
    offsets = positions - center
    radii = np.sort(np.linalg.norm(offsets, axis=1))
    np.testing.assert_allclose(radii[:1], 0.0, atol=2e-7)
    np.testing.assert_allclose(radii[1:7], packed_spacing, atol=2e-7)
    np.testing.assert_allclose(radii[7:], np.sqrt(3.0) * packed_spacing, atol=2e-7)
    np.testing.assert_allclose(positions.mean(axis=0), center, atol=2e-7)

    # A larger partial disk remains a genuine hexagonal packing: its closest
    # pair is one lattice edge apart and recentering does not alter spacing.
    positions, *_ = seed_blob(100, tuple(center), spacing, seed=23)
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    assert np.isclose(distances.min(), packed_spacing, atol=2e-7), distances.min()
    np.testing.assert_allclose(positions.mean(axis=0), center, atol=2e-7)

    print("[PASS] initial agents form a centered, circularly clipped hexagonal lattice")


if __name__ == "__main__":
    check_circular_hexagonal_seed()

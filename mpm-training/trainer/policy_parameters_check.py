"""Regression checks for policy-head layout, initialization, and mutation."""
from __future__ import annotations

import numpy as np

from evolve import get_weights, mutate, set_weights
from policy_parameters import mutation_scale_vector, policy_heads, random_flat_policy_weights
from simulation_settings import CHEM_CHANNELS, HIDDEN_DIM
from update_rule import UpdateRule


def main() -> None:
    model = UpdateRule(CHEM_CHANNELS)
    flat = get_weights(model)
    exported = model.export_weights()
    serialized = np.concatenate(
        [
            np.asarray(exported["fc1w"]).ravel(),
            np.asarray(exported["fc1b"]),
            np.asarray(exported["fc2w"]).ravel(),
            np.asarray(exported["fc2b"]),
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(flat, serialized)

    clone = UpdateRule(CHEM_CHANNELS)
    set_weights(clone, flat)
    np.testing.assert_array_equal(get_weights(clone), flat)
    print(f"[PASS] six logical heads round-trip through the {flat.size}-float GPU/checkpoint layout")

    initialized = random_flat_policy_weights(CHEM_CHANNELS, HIDDEN_DIM, np.random.default_rng(11))
    assert initialized.shape == flat.shape and initialized.dtype == np.float32
    print("[PASS] shared head-aware random initializer produces the canonical float32 layout")

    scales = mutation_scale_vector(CHEM_CHANNELS, HIDDEN_DIM)
    seed = 29
    sigma = 0.05
    expected_noise = np.random.default_rng(seed).normal(size=flat.shape).astype(np.float32)
    expected = flat + expected_noise * np.float32(sigma) * scales
    actual = mutate(flat, sigma, np.random.default_rng(seed))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    expected_scales = {1.0, *(head.mutation_scale for head in policy_heads(CHEM_CHANNELS))}
    assert set(np.unique(scales).tolist()) == set(np.asarray(list(expected_scales), dtype=np.float32).tolist())
    print("[PASS] mutation applies the global sigma through exact trunk/head scale buckets")


if __name__ == "__main__":
    main()

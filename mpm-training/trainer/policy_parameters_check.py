"""Regression checks for policy-head layout, initialization, and mutation."""
from __future__ import annotations

import numpy as np

from agents_gpu import weight_layout
from evolve import get_weights, mutate, set_weights
from policy_parameters import (
    CELL_OWNED_PROJECTION_ARCHITECTURE,
    CHEMICAL_COMMUNICATION_ARCHITECTURES,
    PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    STATEFUL_ARCHITECTURE,
    STATEFUL_128_ARCHITECTURE,
    mutation_scale_vector,
    policy_heads,
    policy_hidden_dim,
    random_flat_policy_weights,
    normalize_chemical_communication_architecture,
    resolve_chemical_communication_architecture,
)
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
    print(f"[PASS] logical heads round-trip through the {flat.size}-float GPU/checkpoint layout")

    initialized = random_flat_policy_weights(CHEM_CHANNELS, HIDDEN_DIM, np.random.default_rng(11))
    assert initialized.shape == flat.shape and initialized.dtype == np.float32
    print("[PASS] shared head-aware random initializer produces the canonical float32 layout")

    growth_head = next(head for head in policy_heads(CHEM_CHANNELS) if head.name == "growthVector")
    assert growth_head.bias_center == (0.0, 0.0)
    # The local direction prior must not privilege the heading's forward axis.
    # With a silent trunk, independent symmetric jitter should populate all
    # four quadrants across fresh policies instead of clustering near 0 degrees.
    directions = []
    for seed in range(64):
        candidate = random_flat_policy_weights(CHEM_CHANNELS, HIDDEN_DIM, np.random.default_rng(seed))
        layout = weight_layout(CHEM_CHANNELS, HIDDEN_DIM)
        biases = candidate[layout["fc2b_offset"]:]
        directions.append(biases[CHEM_CHANNELS:CHEM_CHANNELS + 2])
    directions = np.asarray(directions)
    assert set(zip(directions[:, 0] > 0, directions[:, 1] > 0)) == {
        (False, False), (False, True), (True, False), (True, True)
    }
    print("[PASS] growth-vector initialization is zero-centered with no local-forward prior")

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

    stateful_hidden = policy_hidden_dim(STATEFUL_ARCHITECTURE)
    stateful = UpdateRule(CHEM_CHANNELS, STATEFUL_ARCHITECTURE)
    stateful_flat = get_weights(stateful)
    stateful_random = random_flat_policy_weights(
        CHEM_CHANNELS, stateful_hidden, np.random.default_rng(31), STATEFUL_ARCHITECTURE
    )
    assert stateful_flat.size == stateful_random.size
    assert [head.name for head in policy_heads(CHEM_CHANNELS, STATEFUL_ARCHITECTURE)][-2:] == [
        "stateDelta", "stateGate"
    ]
    stateful_mutated = mutate(
        stateful_flat, sigma, np.random.default_rng(32), STATEFUL_ARCHITECTURE
    )
    assert stateful_mutated.shape == stateful_flat.shape
    stateful_layout = weight_layout(CHEM_CHANNELS, stateful_hidden, STATEFUL_ARCHITECTURE)
    assert stateful_flat.size == stateful_layout["total_floats"]
    print("[PASS] stateful-64 layout matches the shared continuous-vector policy specification")

    recurrent_128 = UpdateRule(CHEM_CHANNELS, STATEFUL_128_ARCHITECTURE)
    recurrent_128_flat = get_weights(recurrent_128)
    assert policy_hidden_dim(STATEFUL_128_ARCHITECTURE) == 128
    recurrent_layout = weight_layout(CHEM_CHANNELS, 128, STATEFUL_128_ARCHITECTURE)
    assert recurrent_128_flat.size == recurrent_layout["total_floats"]
    print("[PASS] recurrent policy keeps 128 hidden units and the shared output layout")

    assert CHEMICAL_COMMUNICATION_ARCHITECTURES == (
        PERSISTENT_ENVIRONMENT_ARCHITECTURE,
        CELL_OWNED_PROJECTION_ARCHITECTURE,
    )
    assert normalize_chemical_communication_architecture(None) == CELL_OWNED_PROJECTION_ARCHITECTURE
    for chemical_architecture in CHEMICAL_COMMUNICATION_ARCHITECTURES:
        assert normalize_chemical_communication_architecture(chemical_architecture) == chemical_architecture
    assert resolve_chemical_communication_architecture(None, 0.91) == PERSISTENT_ENVIRONMENT_ARCHITECTURE
    assert resolve_chemical_communication_architecture(None, 0.0) == CELL_OWNED_PROJECTION_ARCHITECTURE
    print("[PASS] chemical communication architecture selection is validated independently of policy shape")


if __name__ == "__main__":
    main()

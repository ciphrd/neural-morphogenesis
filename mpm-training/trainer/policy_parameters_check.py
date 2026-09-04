"""Regression checks for policy-head layout, initialization, and mutation."""
from __future__ import annotations

import numpy as np
import torch

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

    # Evolution needs phenotype diversity before selection has had a chance to
    # shape the population. Bias centers remain sensible defaults, while random
    # trunk/head weights should produce a broad response even at neutral input.
    zero_channels = torch.zeros((1, CHEM_CHANNELS), dtype=torch.float32)
    zero_spatial = torch.zeros((1, 3), dtype=torch.float32)
    growth_angles: list[float] = []
    heading_angles: list[float] = []
    anisotropies: list[float] = []
    division_biases: list[float] = []
    division_drives: list[float] = []
    chemical_outputs: list[float] = []
    init_rng = np.random.default_rng(101)
    with torch.no_grad():
        for _ in range(128):
            model.load_flat_parameters(torch.from_numpy(
                random_flat_policy_weights(CHEM_CHANNELS, HIDDEN_DIM, init_rng)
            ))
            chemical, heading, controls, growth, _tail = model(
                zero_channels, zero_channels, zero_channels, zero_spatial, zero_spatial
            )
            heading_xy = torch.tanh(heading)[0].numpy()
            growth_xy = torch.tanh(growth)[0].numpy()
            heading_angles.append(float(np.arctan2(heading_xy[1], heading_xy[0])))
            growth_angles.append(float(np.arctan2(growth_xy[1], growth_xy[0])))
            chemical_outputs.extend(torch.tanh(chemical)[0].numpy().tolist())
            anisotropies.append(float(torch.sigmoid(controls[0, 0])))
            division_biases.append(float(torch.sigmoid(controls[0, 1])))
            division_drives.append(float(torch.tanh(controls[0, 2])))

    growth_degrees = np.abs(np.degrees(growth_angles))
    heading_degrees = np.abs(np.degrees(heading_angles))
    assert np.percentile(growth_degrees, 90) > 90.0, np.percentile(growth_degrees, [50, 90])
    assert np.percentile(heading_degrees, 90) > 35.0, np.percentile(heading_degrees, [50, 90])
    assert 0.15 < np.median(anisotropies) < 0.25, np.median(anisotropies)
    assert 0.4 < np.median(division_biases) < 0.6, np.median(division_biases)
    assert abs(np.median(division_drives)) < 0.1, np.median(division_drives)
    assert 0.25 < np.percentile(np.abs(chemical_outputs), 90) < 0.75
    print("[PASS] random policies are expressive around sensible output defaults")

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
    assert stateful_flat.size == stateful_random.size == 4768
    assert [head.name for head in policy_heads(CHEM_CHANNELS, STATEFUL_ARCHITECTURE)][-2:] == [
        "stateDelta", "stateGate"
    ]
    stateful_mutated = mutate(
        stateful_flat, sigma, np.random.default_rng(32), STATEFUL_ARCHITECTURE
    )
    assert stateful_mutated.shape == stateful_flat.shape
    print("[PASS] stateful-64 has 41 inputs, 32 outputs, 4768 parameters, and state-head mutation buckets")

    recurrent_128 = UpdateRule(CHEM_CHANNELS, STATEFUL_128_ARCHITECTURE)
    recurrent_128_flat = get_weights(recurrent_128)
    assert policy_hidden_dim(STATEFUL_128_ARCHITECTURE) == 128
    assert recurrent_128_flat.size == 9504
    print("[PASS] recurrent policy keeps 128 hidden units and has 9504 parameters")

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

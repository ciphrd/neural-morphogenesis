"""Capture the tensor-Fg isotropic-equivalence elastic baseline.

The base scenario is policy-independent: zero network weights and a saturated
growth-chemical channel force every eligible particle into a cell cycle. The
``--directional`` selects a constant local-forward axis and saturates the
independent anisotropy and division-bias controls for comparison.

Run from trainer/:
    .venv/bin/python capture_elastic_baseline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agents_gpu import AgentsGPU, weight_layout
from device import pick_device
from elastic_diagnostics import measure_core
from environment_gpu import EnvironmentGPU
from mpm_core import DT, MpmCore
from simulation_settings import (
    ANGULAR_DAMPING,
    CHEM_CHANNELS,
    CHIRALITY,
    DEPOSIT_DISTANCE,
    DEPOSIT_RATE,
    DEPOSIT_SIGMA,
    DIVISION_COOLDOWN,
    DAMPING_LOSS_FRACTION,
    FIELD_N,
    FRICTION,
    GROWTH_MAX,
    GROWTH_THRESHOLD,
    HIDDEN_DIM,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MAX_ACCEL,
    MAX_ANGULAR_ACCEL,
    MAX_ANGULAR_VELOCITY,
    MAX_ENV_WRITE,
    MAX_STRAFE,
    REPULSION_MAX_DELTA,
    SPLAT_RADIUS,
)
from training_sim import TrainingRollout

ROOT = Path(__file__).parent.parent
ISOTROPIC_OUTPUT = Path(__file__).parent / "snapshots" / "tensor_growth_isotropic_equivalence.json"
DIRECTIONAL_OUTPUT = Path(__file__).parent / "snapshots" / "tensor_growth_directional_strafe.json"
SEED = 20260822
MAX_ACTIVE = 16
SUBSTEPS = 16
GROWTH_STEPS = 32
TOTAL_STEPS = 80
CHECKPOINTS = {0, 8, 16, 24, 32, 40, 48, 64, 80}
# Preserve the original diagnostic trajectory exactly. Production growth now
# uses a controller-tick duration; this historical scalar/tensor comparison
# deliberately retains its old low-level rate.
BASELINE_GROWTH_RATE = 50.0
BASELINE_DECAY = 0.91
BASELINE_SPLIT_DISPLACEMENT = 0.01
BASELINE_REPULSION_STRENGTH = 0.2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_agents(device, core: MpmCore, environment: EnvironmentGPU) -> AgentsGPU:
    return AgentsGPU(
        device,
        core,
        environment,
        CHEM_CHANNELS,
        HIDDEN_DIM,
        MAX_ACCEL,
        MAX_STRAFE,
        MAX_ENV_WRITE,
        MAX_ANGULAR_ACCEL,
        ANGULAR_DAMPING,
        MAX_ANGULAR_VELOCITY,
        CHIRALITY,
        DEPOSIT_DISTANCE,
        MAX_ACTIVE,
        BASELINE_SPLIT_DISPLACEMENT,
        DIVISION_COOLDOWN,
        FRICTION,
        DEPOSIT_SIGMA,
        1.0,
        0.5,
        0.5,
    )


def _assert_close(actual, expected, path: str = "snapshot") -> None:
    """Recursively compare a fresh GPU capture to the saved baseline."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for i, value in enumerate(expected):
            _assert_close(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=2e-5, atol=1e-7), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rerun the scenario and compare measurements with the saved snapshot instead of overwriting it",
    )
    parser.add_argument(
        "--directional",
        action="store_true",
        help="bias the former local-forward strafe output to capture directional rather than isotropic tensor growth",
    )
    args = parser.parse_args()
    output_path = DIRECTIONAL_OUTPUT if args.directional else ISOTROPIC_OUTPUT
    device = pick_device()
    core = MpmCore(device)
    environment = EnvironmentGPU(device, CHEM_CHANNELS, FIELD_N, FIELD_N, BASELINE_DECAY, DEPOSIT_RATE)
    agents = _build_agents(device, core, environment)
    weights = np.zeros(agents._total_floats, dtype=np.float32)
    if args.directional:
        layout = weight_layout(CHEM_CHANNELS, HIDDEN_DIM)
        # Saturated anisotropy/polarity plus a constant local-forward axis.
        env_write_dim = CHEM_CHANNELS * 4
        weights[layout["fc2b_offset"] + env_write_dim + 1] = 20.0
        weights[layout["fc2b_offset"] + env_write_dim + 2] = 20.0
        weights[layout["fc2b_offset"] + env_write_dim + 3] = 1.0
    agents.load_weights(weights)

    core.set_material(
        MATERIAL_E,
        MATERIAL_NU,
        MATERIAL_HARDENING,
        MATERIAL_ELASTICITY,
        growth_rate=BASELINE_GROWTH_RATE,
    )
    core.set_damping(DAMPING_LOSS_FRACTION, SUBSTEPS)
    core.set_splat_radius(SPLAT_RADIUS)
    core.set_repulsion_strength(BASELINE_REPULSION_STRENGTH, REPULSION_MAX_DELTA)
    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(0.5, 0.5),
        spawn_half_width=0.0,
        gravity=0.0,
        seed=SEED,
        neural_updates_per_macro=1,
    )

    growth_plane = np.ones(FIELD_N * FIELD_N, dtype=np.float32)
    growth_plane_offset = (CHEM_CHANNELS - 1) * FIELD_N * FIELD_N * 4
    measurements: list[dict] = []

    def capture(step: int) -> None:
        row = measure_core(
            core,
            material_e=MATERIAL_E,
            material_nu=MATERIAL_NU,
            material_hardening=MATERIAL_HARDENING,
        )
        row["macro_step"] = step
        row["simulated_time"] = step * SUBSTEPS * DT
        row["phase"] = "growth" if step <= GROWTH_STEPS else "settling"
        measurements.append(row)
        print(
            f"step={step:2d} phase={row['phase']:<8} n={row['particle_count']:2d} "
            f"cycles={row['active_cycle_count']:2d} radius={row['geometry']['rms_radius']:.6f} "
            f"Eel={row['total_elastic_energy']:.6f} Ekin={row['total_kinetic_energy']:.6f}"
        )

    capture(0)
    for step in range(1, TOTAL_STEPS + 1):
        if step <= GROWTH_STEPS:
            # Refill the currently-readable parity buffer so the experiment
            # measures saturated growth rather than chemical decay kinetics.
            device.queue.write_buffer(environment.buffers[environment.parity], growth_plane_offset, growth_plane)
        sim.macro_step(SUBSTEPS, growth_enabled=step <= GROWTH_STEPS)
        if step in CHECKPOINTS:
            capture(step)

    source_files = [
        ROOT / "core" / "constants.json",
        ROOT / "core" / "agents.wgsl",
        ROOT / "core" / "p2g.wgsl",
        ROOT / "core" / "g2p.wgsl",
        ROOT / "core" / "gridUpdate.wgsl",
        ROOT / "trainer" / "elastic_diagnostics.py",
    ]
    snapshot = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "growth_model": "tensor_Fg_with_network_direction" if args.directional else "tensor_Fg_with_isotropic_increment",
        "diagnostic_definitions": {
            "elastic_decomposition": "Fe=F*inverse(Fg)",
            "deviatoric_log_strain": "||dev(log(Ue))||_F=abs(log(smax/smin))/sqrt(2)",
            "corotated_strain": "||Fe-Re||_F",
            "growth_deviatoric_log_strain": "||dev(log(Ug))||_F=abs(log(gmax/gmin))/sqrt(2)",
            "elastic_energy_density": "mu*||Fe-Re||_F^2 + lambda/2*(det(Fe)-1)^2",
            "distribution_weighting": "particle mass = PARTICLE_MASS*g",
        },
        "scenario": {
            "name": "saturated_growth_then_settle",
            "purpose": (
                "network-directed tensor-Fg growth plus signed division-polarity comparison"
                if args.directional
                else "tensor-Fg isotropic-equivalence, recoil, and residual-strain baseline"
            ),
            "seed": SEED,
            "initial_particles": 2,
            "max_active_particles": MAX_ACTIVE,
            "growth_field": "last substrate channel refilled uniformly to 1 during growth phase",
            "policy_weights": "anisotropy/polarity logits=20, local-forward direction bias=1" if args.directional else "all zero",
            "growth_direction": (
                "normalized local-forward axis with sigmoid anisotropy=1 and division bias=1"
                if args.directional
                else "zero vector"
            ),
            "macro_steps": TOTAL_STEPS,
            "growth_enabled_through_macro_step": GROWTH_STEPS,
            "substeps_per_macro": SUBSTEPS,
            "dt": DT,
            "gravity": 0.0,
            "spawn_half_width": 0.0,
            "material": {
                "E": MATERIAL_E,
                "nu": MATERIAL_NU,
                "hardening": MATERIAL_HARDENING,
                "elasticity": MATERIAL_ELASTICITY,
            },
            "growth": {
                "legacy_internal_rate": BASELINE_GROWTH_RATE,
                "division_area_ratio": GROWTH_MAX,
                "compression_reference": GROWTH_THRESHOLD,
                "split_displacement": BASELINE_SPLIT_DISPLACEMENT,
                "division_cooldown": DIVISION_COOLDOWN,
            },
            "repulsion": {
                "strength": BASELINE_REPULSION_STRENGTH,
                "max_delta": REPULSION_MAX_DELTA,
                "splat_radius": SPLAT_RADIUS,
            },
            "damping_loss_fraction": DAMPING_LOSS_FRACTION,
        },
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
        "checkpoints": measurements,
    }
    if args.verify:
        expected = json.loads(output_path.read_text())
        # Capture time is intentionally provenance, not deterministic state.
        actual_comparable = {key: value for key, value in snapshot.items() if key != "captured_at_utc"}
        expected_comparable = {key: value for key, value in expected.items() if key != "captured_at_utc"}
        _assert_close(actual_comparable, expected_comparable)
        print(f"[PASS] fresh GPU trajectory reproduces {output_path} within tolerance")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

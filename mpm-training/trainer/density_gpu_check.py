"""Short real-GPU smoke test for switching density on persistent pipelines."""
from __future__ import annotations

import numpy as np

from agents_gpu import AgentsGPU
from density import DensityReference, resolve_density
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import PARTICLE_MASS, VOL, MpmCore
from policy_parameters import policy_hidden_dim
from simulation_settings import (
    ANGULAR_DAMPING, CHEM_CHANNELS, CHEMICAL_COMMUNICATION_ARCHITECTURE,
    CHEMICAL_GRADIENT_INPUT_SCALE, CHIRALITY, DECAY, DEPOSIT_DISTANCE,
    DEPOSIT_RATE, DEPOSIT_SIGMA, DIVISION_COOLDOWN, FIELD_N, FRICTION,
    MATERIAL_E, MATERIAL_ELASTICITY, MATERIAL_HARDENING, MATERIAL_NU,
    MAX_ACCEL, MAX_ANGULAR_ACCEL, MAX_ANGULAR_VELOCITY, MAX_ENV_WRITE,
    MAX_STRAFE, POLICY_ARCHITECTURE, REPULSION_MAX_DELTA,
    REPULSION_STRENGTH, SPLIT_DISPLACEMENT,
)
from training_sim import TrainingRollout


def main() -> None:
    reference = DensityReference(
        particle_cap=20,
        initial_particles=4,
        chemical_field_n=FIELD_N,
        particle_mass=PARTICLE_MASS,
        particle_volume=VOL,
        chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
        repulsion_strength=REPULSION_STRENGTH,
        repulsion_max_delta=REPULSION_MAX_DELTA,
    )
    cases = [resolve_density(reference, q) for q in (0.5, 1.0, 2.0)]
    device = pick_device()
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, CHEM_CHANNELS, FIELD_N, FIELD_N, DECAY, DEPOSIT_RATE,
        CHEMICAL_COMMUNICATION_ARCHITECTURE,
    )
    agents = AgentsGPU(
        device, core, environment, CHEM_CHANNELS, policy_hidden_dim(POLICY_ARCHITECTURE),
        MAX_ACCEL, MAX_STRAFE, MAX_ENV_WRITE, MAX_ANGULAR_ACCEL,
        ANGULAR_DAMPING, MAX_ANGULAR_VELOCITY, CHIRALITY, DEPOSIT_DISTANCE,
        max(case.particle_cap for case in cases), SPLIT_DISPLACEMENT,
        DIVISION_COOLDOWN, FRICTION, DEPOSIT_SIGMA, 1.0, 0.5, 0.5,
        policy_architecture=POLICY_ARCHITECTURE,
        chemical_communication_architecture=CHEMICAL_COMMUNICATION_ARCHITECTURE,
    )
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))

    for case in cases:
        core.set_material(
            MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, MATERIAL_ELASTICITY,
            particle_mass=case.particle_mass, particle_volume=case.particle_volume,
        )
        core.set_splat_radius(case.splat_radius)
        core.set_repulsion_strength(case.repulsion_strength, case.repulsion_max_delta)
        agents.set_density_geometry(case.spacing, case.deposit_sigma)
        agents.set_chemical_gradient_input_scale(case.chemical_gradient_input_scale)
        agents.set_max_active_particles(case.particle_cap)
        sim = TrainingRollout(
            core, agents, environment, spawn_center=(0.5, 0.5),
            spawn_half_width=0.0, gravity=0.0, seed=7,
            initial_particle_count=case.initial_particles,
        )
        initial = sim.positions()
        assert initial.shape == (case.initial_particles, 2)
        if len(initial) > 1:
            distances = np.linalg.norm(initial[:, None, :] - initial[None, :, :], axis=2)
            nearest = np.min(np.where(distances > 0, distances, np.inf), axis=1)
            assert np.isclose(nearest.min(), case.spacing, rtol=2e-5, atol=1e-7)
        sim.macro_step(1, growth_enabled=False)
        assert np.isfinite(sim.positions()).all()
        assert core.particle_mass == case.particle_mass
        assert core.particle_volume == case.particle_volume
        assert agents.max_active_particles == case.particle_cap

    print("[PASS] persistent GPU pipelines switch cleanly across q=0.5,1,2")


if __name__ == "__main__":
    main()

"""Short real-GPU smoke test for switching density on persistent pipelines."""
from __future__ import annotations

import numpy as np

from agents_gpu import PARTICLE_META_BUFFER_OFFSET, AgentsGPU
from density import INITIAL_PACKING_SPACING_SCALE, DensityReference, resolve_density
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


def _set_uniform_chemical_state(agents: AgentsGPU, count: int, value: float) -> None:
    size = count * agents._particle_meta_dtype.itemsize
    raw = agents.device.queue.read_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, size)
    meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=count).copy()
    meta["chemicalState"][:] = value
    agents.device.queue.write_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta.tobytes()
    )


def _projected_plane(
    environment: EnvironmentGPU, agents: AgentsGPU, channel: int,
) -> np.ndarray:
    environment.reset()
    encoder = environment.device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    environment.encode_sense(encoder)
    environment.device.queue.submit([encoder.finish()])
    raw = environment.device.queue.read_buffer(environment.buffers[0])
    grid = np.frombuffer(raw, dtype=np.float32).reshape(
        environment.channels, environment.height, environment.width
    )
    return grid[channel].copy()


def _sample_plane(plane: np.ndarray, positions: np.ndarray) -> np.ndarray:
    height, width = plane.shape
    field = np.mod(positions, 1.0) * np.array([width, height], dtype=np.float32)
    x0 = np.floor(field[:, 0]).astype(np.int64) % width
    y0 = np.floor(field[:, 1]).astype(np.int64) % height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    wx1 = field[:, 0] - np.floor(field[:, 0])
    wy1 = field[:, 1] - np.floor(field[:, 1])
    return (
        plane[y0, x0] * (1 - wx1) * (1 - wy1)
        + plane[y0, x1] * wx1 * (1 - wy1)
        + plane[y1, x0] * (1 - wx1) * wy1
        + plane[y1, x1] * wx1 * wy1
    )


def main() -> None:
    reference = DensityReference(
        particle_cap=20,
        initial_particles=4,
        chemical_field_n=FIELD_N,
        particle_mass=PARTICLE_MASS,
        particle_volume=VOL,
        deposit_sigma=DEPOSIT_SIGMA,
        chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
        repulsion_strength=REPULSION_STRENGTH,
        repulsion_max_delta=REPULSION_MAX_DELTA,
    )
    cases = [resolve_density(reference, q) for q in (0.5, 1.0, 2.0, 4.0)]
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

    projected_mass: list[float] = []
    sampled_signal: list[float] = []
    for case in cases:
        core.set_material(
            MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, MATERIAL_ELASTICITY,
            particle_mass=case.particle_mass, particle_volume=case.particle_volume,
        )
        core.set_splat_radius(case.splat_radius)
        core.set_repulsion_strength(case.repulsion_strength, case.repulsion_max_delta)
        agents.set_density_geometry(case.spacing, case.deposit_sigma)
        agents.set_chemical_gradient_input_scale(case.chemical_gradient_input_scale)
        agents.set_chemical_projection_weight(case.chemical_projection_weight)
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
            assert np.isclose(
                nearest.min(),
                case.spacing * INITIAL_PACKING_SPACING_SCALE,
                rtol=2e-5,
                atol=1e-7,
            )
        if CHEMICAL_COMMUNICATION_ARCHITECTURE == "cell-owned-projection":
            _set_uniform_chemical_state(agents, case.initial_particles, 1.0)
            plane = _projected_plane(environment, agents, CHEM_CHANNELS - 1)
            projected_mass.append(float(plane.sum()))
            sampled_signal.append(float(_sample_plane(plane, initial).mean()))
        sim.macro_step(1, growth_enabled=False)
        assert np.isfinite(sim.positions()).all()
        assert core.particle_mass == case.particle_mass
        assert core.particle_volume == case.particle_volume
        assert agents.max_active_particles == case.particle_cap

    if projected_mass:
        mass = np.asarray(projected_mass)
        signal = np.asarray(sampled_signal)
        assert mass.max() / mass.min() < 1.2, mass
        print(
            "[PASS] density-weighted chemical projection "
            f"mass={mass.round(4).tolist()} sampled={signal.round(4).tolist()}"
        )

    # A controlled lifecycle regression: the brain is identically zero, each
    # starting cell owns the same modest growth-channel level, mechanics has no
    # elastic/compression feedback, and density is the only changed variable.
    # Compare represented population N/q rather than raw particle count.
    checkpoints = (12, 24, 36)
    represented_runs: list[list[list[float]]] = [[] for _ in cases]
    radius_runs: list[list[list[float]]] = [[] for _ in cases]
    for case_index, case in enumerate(cases):
        core.set_material(
            0.0, MATERIAL_NU, 0.0, MATERIAL_ELASTICITY,
            growth_duration_macro_steps=8.0,
            substeps_per_macro=1,
            particle_mass=case.particle_mass,
            particle_volume=case.particle_volume,
            growth_compression_feedback=0.0,
        )
        agents.set_density_geometry(case.spacing, case.deposit_sigma)
        agents.set_chemical_gradient_input_scale(case.chemical_gradient_input_scale)
        agents.set_chemical_projection_weight(case.chemical_projection_weight)
        agents.set_max_active_particles(case.particle_cap)
        for seed in (11, 29, 47, 83):
            sim = TrainingRollout(
                core, agents, environment, spawn_center=(0.5, 0.5),
                spawn_half_width=0.0, gravity=0.0, seed=seed,
                initial_particle_count=case.initial_particles,
            )
            _set_uniform_chemical_state(agents, case.initial_particles, 0.1)
            represented: list[float] = []
            radii: list[float] = []
            for step in range(1, checkpoints[-1] + 1):
                sim.macro_step(1)
                if step in checkpoints:
                    represented.append(core.active_count / case.multiplier)
                    positions = sim.positions()
                    delta = (positions - 0.5 + 0.5) % 1.0 - 0.5
                    radii.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
            represented_runs[case_index].append(represented)
            radius_runs[case_index].append(radii)

    represented_mean = np.asarray(represented_runs).mean(axis=1)
    # Branching is stochastic and the coarsest case begins with only two
    # particles, so this is intentionally a broad regression bound. A renewed
    # monotonic q-dependent acceleration is much larger than this tolerance.
    spread = represented_mean.max(axis=0) / np.maximum(represented_mean.min(axis=0), 1.0)
    assert np.all(spread < 1.5), (represented_mean, spread)
    print(
        "[PASS] represented growth N/q stays density-stable "
        f"at steps {checkpoints}: {represented_mean.round(3).tolist()}"
    )
    radius_mean = np.asarray(radius_runs).mean(axis=1)
    radius_spread = radius_mean.max(axis=0) / np.maximum(radius_mean.min(axis=0), 1e-8)
    # With zero stiffness this radius is almost entirely the accumulated
    # daughter offset, which deliberately scales with q. The learned-policy
    # acceptance metric uses robust p95/RMS envelope spread instead; this
    # broad bound catches runaway mechanics without pretending a serial chain
    # of coarse particles has the same raw offset sum as a fine one.
    # The compact initial disk deliberately increases the early local
    # concentration; q=2 resolves that transient more sharply than q=0.5.
    # Keep it bounded while allowing the expected seed-geometry effect.
    assert np.all(radius_spread < 1.6), (radius_mean, radius_spread)
    print(
        "[PASS] controlled-growth RMS radius stays density-stable "
        f"at steps {checkpoints}: {radius_mean.round(6).tolist()}"
    )

    print("[PASS] persistent GPU pipelines switch cleanly across q=0.5,1,2")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from agents_gpu import AgentsGPU
from alignment import best_alignment
from debug_images import rasterize
from density import DensityReference, resolve_checkpoint_density
from device import pick_device
from environment_gpu import EnvironmentGPU
from evolve import CHECKPOINTS_DIR
from mpm_core import PARTICLE_MASS, VOL, MpmCore
from simulation_settings import (
    CHEM_CHANNELS,
    CHEMICAL_GRADIENT_INPUT_SCALE,
    DAMPING_LOSS_FRACTION,
    DECAY,
    DEPOSIT_RATE,
    NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
    FIELD_N,
    FRICTION,
    GROWTH_DURATION_MACRO_STEPS,
    GROWTH_COMPRESSION_START,
    GROWTH_COMPRESSION_STOP,
    GROWTH_ANISOTROPY_AUTHORITY,
    INITIAL_PARTICLE_COUNT,
    INTERNAL_STATE_SPEED,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MAX_ENV_WRITE,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    SAMPLE_SPACING,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
)
from targets import target_from_checkpoint
from chemical_channels import homogeneous_channel_profiles, resolve_channel_profiles
from training_sim import TrainingRollout
from domain_fitness import StableMatchStop, target_mask, evaluate_domains
from policy_parameters import STATELESS_ARCHITECTURE, policy_hidden_dim, resolve_chemical_communication_architecture

def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "render_out"
    out_dir.mkdir(exist_ok=True)

    meta = json.loads((CHECKPOINTS_DIR / "best_meta.json").read_text())
    weights = np.load(CHECKPOINTS_DIR / "best.npy")

    density = resolve_checkpoint_density(
        meta,
        DensityReference(
            particle_cap=int(meta["particles"]),
            initial_particles=int(meta['initial_particle_count']),
            chemical_field_n=int(meta['field_n']),
            particle_mass=PARTICLE_MASS,
            particle_volume=VOL,
            chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
            repulsion_strength=float(meta['repulsion_strength']),
            repulsion_max_delta=float(meta['repulsion_max_delta']),
        ),
    )

    wgpu_device = pick_device()

    core = MpmCore(wgpu_device)
    core.set_morphology(
        meta['morphology_blur_sigma'],
        meta['morphology_density_reference'],
    )
    material_kwargs = {
        "growth_anisotropy": meta['growth_anisotropy_authority'],
        "growth_compression_start": meta['growth_compression_start'],
        "growth_compression_stop": meta['growth_compression_stop'],
        "growth_compression_feedback": meta['growth_compression_feedback'],
    }
    material_kwargs["growth_duration_macro_steps"] = meta['growth_duration_macro_steps']
    material_kwargs["substeps_per_macro"] = meta["substeps_per_macro"]
    core.set_material(
        MATERIAL_E,
        MATERIAL_NU,
        MATERIAL_HARDENING,
        elasticity=meta['material_elasticity'],
        particle_mass=density.particle_mass,
        particle_volume=density.particle_volume,
        **material_kwargs,
    )
    core.set_damping(DAMPING_LOSS_FRACTION, meta["substeps_per_macro"])
    core.set_splat_radius(density.splat_radius)
    core.set_repulsion_strength(density.repulsion_strength, density.repulsion_max_delta)

    architecture = meta['policy_architecture']
    chemical_architecture = resolve_chemical_communication_architecture(meta["chemical_communication_architecture"])
    hidden_dim = int(meta['hidden_dim'])
    environment = EnvironmentGPU(wgpu_device, CHEM_CHANNELS, FIELD_N, FIELD_N, meta['decay'], meta['deposit_rate'], chemical_architecture, meta['normalize_deposits_by_local_density'], grid_velocity=core.grid_vel, channel_profiles=resolve_channel_profiles(CHEM_CHANNELS, meta['chemical_channel_profiles']))
    agents = AgentsGPU(wgpu_device, core, environment, CHEM_CHANNELS, hidden_dim, MAX_ENV_WRITE, density.particle_cap, density.spacing, meta['friction'], 1.0, meta['spawn_x'], meta['spawn_y'], meta['elastic_strain_scale'], meta['elastic_strain_inputs_enabled'], policy_architecture=architecture, internal_state_speed=meta['internal_state_speed'], chemical_communication_architecture=chemical_architecture,   )
    agents.load_weights(weights)
    agents.set_chemical_gradient_input_scale(density.chemical_gradient_input_scale)

    target = target_from_checkpoint(meta)

    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(meta["spawn_x"], meta["spawn_y"]),
        gravity=meta["gravity"],
        seed=meta['winner_seed'],
        # Checkpoints predating multi-rate communication were trained with
        # exactly one neural/environment round per mechanical macro step.
        neural_updates_per_macro=meta['neural_updates_per_macro'],
        communication_speed=meta['communication_speed'],
        initial_particle_count=density.initial_particles,
        initial_condition=meta['initial_condition'],
        initial_condition_strength=meta['initial_condition_strength'],
        initial_condition_channel=meta['initial_condition_channel'],
        initial_spacing=density.initial_spacing,
    )

    shape = meta['shape_settings']
    stopping = StableMatchStop(shape["stableStop"], shape["shapeCheckInterval"],
        shape["shapeConfirmations"], shape["shapeSettleSteps"],
        shape["shapeMissingTolerance"], shape["shapeSpillTolerance"],
        shape["shapeOverlapTolerance"])
    mask = target_mask(target, meta['raster_resolution'])
    target_points = target.overlay_points(meta['raster_resolution'])
    growth_steps = meta['growth_steps']
    for i in range(meta["macro_steps"]):
        sim.macro_step(
            meta["substeps_per_macro"],
            growth_enabled=stopping.growth_enabled and (growth_steps is None or i < growth_steps),
        )
        if stopping.due(i+1):
            evaluation = evaluate_domains(core.read_rest_state()[:, 8:14], target, mask)
            stopping.observe(i+1, evaluation.match, evaluation.total,
                sampling_blocked=core.active_count >= agents.max_active_particles or agents.capacity_blocked or agents.unresolved_samples > 0)
        if i % 4 == 0 or i == meta["macro_steps"] - 1 or stopping.complete:
            pos = sim.positions()
            _, aligned = best_alignment(pos, target_points)
            img = rasterize(aligned, target_points)
            path = out_dir / f"rollout_{i:03d}.png"
            img.save(path)
            print(f"wrote {path}")
        if stopping.complete:
            print(f"Stable match after settling at step {i+1}")
            break

    print(f"\nDone (target={meta['target']!r}, checkpoint fitness={meta['fitness']:.4f}) — gray=target, white=grown")
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""Measure current checkpoint NN growth proposals in a headless replay.
Run from the repository root: trainer/.venv/bin/python trainer/measure_growth_output.py
Samples all active material points every 10 macro steps, before physics.
"""
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

from agents_gpu import PARTICLE_META_BUFFER_OFFSET

def main():
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
        meta['material_e'],
        meta['material_nu'],
        meta['material_hardening'],
        elasticity=meta['material_elasticity'],
        particle_mass=density.particle_mass,
        particle_volume=density.particle_volume,
        **material_kwargs,
    )
    core.set_damping(meta['damping_loss_fraction'], meta["substeps_per_macro"])
    core.set_splat_radius(density.splat_radius)
    core.set_repulsion_strength(density.repulsion_strength, density.repulsion_max_delta)

    architecture = meta['policy_architecture']
    chemical_architecture = resolve_chemical_communication_architecture(meta["chemical_communication_architecture"])
    hidden_dim = int(meta['hidden_dim'])
    environment = EnvironmentGPU(wgpu_device, int(meta['channels']), int(meta['field_n']), int(meta['field_n']), meta['decay'], meta['deposit_rate'], chemical_architecture, meta['normalize_deposits_by_local_density'], grid_velocity=core.grid_vel, channel_profiles=resolve_channel_profiles(int(meta['channels']), meta['chemical_channel_profiles']))
    agents = AgentsGPU(wgpu_device, core, environment, int(meta['channels']), hidden_dim, meta['max_env_write'], density.particle_cap, density.spacing, meta['friction'], 1.0, meta['spawn_x'], meta['spawn_y'], meta['elastic_strain_scale'], meta['elastic_strain_inputs_enabled'], policy_architecture=architecture, internal_state_speed=meta['internal_state_speed'], chemical_communication_architecture=chemical_architecture,   )
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

    rows = []
    step_index = 0
    original_step = core.step
    def measured_step(substeps):
        if step_index == 1 or step_index % 10 == 0:
            rest = core.read_rest_state().astype(float)
            n = len(rest)
            raw = wgpu_device.queue.read_buffer(agents._agent_state_buffer,
                PARTICLE_META_BUFFER_OFFSET, n * agents._particle_meta_dtype.itemsize)
            state = np.frombuffer(raw, dtype=agents._particle_meta_dtype)
            vector = rest[:, 5:7]
            magnitude = np.linalg.norm(vector, axis=1)
            heading = state['alignment'].astype(float)
            angle = np.where(np.linalg.norm(heading, axis=1)>1e-10, np.arctan2(heading[:,1], heading[:,0]), 0.0)
            local = np.column_stack((vector[:,0]*np.cos(angle)+vector[:,1]*np.sin(angle),
                                     -vector[:,0]*np.sin(angle)+vector[:,1]*np.cos(angle)))
            area = rest[:, 14] * np.linalg.det(rest[:, :4].reshape(-1,2,2))
            # originalArea/original world area is index 8 in RestState.
            weight = area / area.sum()
            f = core.read_deformation().astype(float) if hasattr(core, 'read_deformation') else np.frombuffer(wgpu_device.queue.read_buffer(core.F, 0, n*16), np.float32).reshape(n,4).astype(float)
            je = np.linalg.det(f.reshape(-1,2,2))/np.linalg.det(rest[:,:4].reshape(-1,2,2))
            compression = np.maximum(0, -np.log(np.maximum(je, 1e-6)))
            row = dict(step=step_index, count=n, mean=float(magnitude.mean()),
                p05=float(np.percentile(magnitude,5)), median=float(np.median(magnitude)),
                p95=float(np.percentile(magnitude,95)), max=float(magnitude.max()),
                fraction_norm_ge_1=float(np.mean(magnitude>=1)),
                area_weighted_norm=float(weight@magnitude),
                area_weighted_clamped_rate=float(weight@np.minimum(magnitude,1)),
                fraction_component_abs_ge_095=float(np.mean(np.abs(local)>=.95)),
                fraction_component_abs_ge_099=float(np.mean(np.abs(local)>=.99)),
                component_mean=local.mean(axis=0).tolist(),
                area=float(area.sum()),
                fraction_compressed_ge_stop=float(np.mean(compression>=meta['growth_compression_stop'])),
                magnitude_metadata_max_error=float(np.max(np.abs(magnitude-state['growthMagnitude']))))
            rows.append(row)
            if step_index == 1 or step_index % 100 == 0:
                print(json.dumps(row), flush=True)
        original_step(substeps)
    core.step = measured_step
    for step_index in range(1, int(meta['macro_steps'])+1):
        sim.macro_step(meta['substeps_per_macro'], growth_enabled=step_index<=meta['growth_steps'])
    out = Path(__file__).parent / 'experiments' / 'growth_output'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint_generation=meta['generation'], seed=meta['winner_seed'],
        metadata=meta, note='All points after final neural round and resampling, before physics; component statistics undo alignment rotation; temporal snapshots equally spaced.', rows=rows)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print('Report: '+str(out/'report.json'), flush=True)

if __name__ == '__main__':
    main()

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
    ANGULAR_DAMPING,
    CHEM_CHANNELS,
    CHEMICAL_GRADIENT_INPUT_SCALE,
    CHIRALITY,
    DAMPING_LOSS_FRACTION,
    DECAY,
    DEPOSIT_DISTANCE,
    DEPOSIT_RATE,
    NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
    DEPOSIT_DENSITY_REFERENCE,
    DEPOSIT_SIGMA,
    DIVISION_COOLDOWN,
    DIVISION_DRIVE_BOOST,
    DIVISION_DIRECTIONALITY,
    FIELD_N,
    FRICTION,
    GROWTH_MAX,
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
    MAX_ACCEL,
    MAX_ANGULAR_ACCEL,
    MAX_ANGULAR_VELOCITY,
    MAX_ENV_WRITE,
    MAX_STRAFE,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    SPLIT_DISPLACEMENT,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
)
from targets import load_target
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
            initial_particles=int(meta.get("initial_particle_count", INITIAL_PARTICLE_COUNT)),
            chemical_field_n=int(meta.get("field_n", FIELD_N)),
            particle_mass=PARTICLE_MASS,
            particle_volume=VOL,
            deposit_sigma=DEPOSIT_SIGMA,
            chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
            repulsion_strength=float(meta.get("repulsion_strength", REPULSION_STRENGTH)),
            repulsion_max_delta=float(meta.get("repulsion_max_delta", REPULSION_MAX_DELTA)),
        ),
        legacy_split_displacement=SPLIT_DISPLACEMENT,
        legacy_deposit_sigma=DEPOSIT_SIGMA,
        legacy_splat_radius=SPLAT_RADIUS,
    )

    wgpu_device = pick_device()

    core = MpmCore(wgpu_device)
    core.set_morphology(
        meta.get("morphology_blur_sigma", MORPHOLOGY_BLUR_SIGMA),
        meta.get("morphology_density_reference", MORPHOLOGY_DENSITY_REFERENCE),
    )
    material_kwargs = {
        "growth_max": meta.get("growth_max", GROWTH_MAX),
        "growth_anisotropy": meta.get(
            "growth_anisotropy_authority", GROWTH_ANISOTROPY_AUTHORITY
        ),
        "growth_compression_start": meta.get("growth_compression_start", GROWTH_COMPRESSION_START),
        "growth_compression_stop": meta.get("growth_compression_stop", GROWTH_COMPRESSION_STOP),
        "growth_compression_feedback": meta.get("growth_compression_feedback", 0.0),
    }
    if "growth_rate" in meta and "growth_duration_macro_steps" not in meta:
        # Preserve exact playback of checkpoints created before growth was
        # expressed in controller ticks.
        material_kwargs["growth_rate"] = meta["growth_rate"]
    else:
        material_kwargs["growth_duration_macro_steps"] = meta.get(
            "growth_duration_macro_steps", GROWTH_DURATION_MACRO_STEPS
        )
        material_kwargs["substeps_per_macro"] = meta["substeps_per_macro"]
    core.set_material(
        meta.get("material_e", MATERIAL_E),
        meta.get("material_nu", MATERIAL_NU),
        meta.get("material_hardening", MATERIAL_HARDENING),
        elasticity=meta.get("material_elasticity", MATERIAL_ELASTICITY),
        particle_mass=density.particle_mass,
        particle_volume=density.particle_volume,
        **material_kwargs,
    )
    core.set_damping(meta.get("damping", DAMPING_LOSS_FRACTION), meta["substeps_per_macro"])
    core.set_splat_radius(density.splat_radius)
    core.set_repulsion_strength(density.repulsion_strength, density.repulsion_max_delta)

    architecture = meta.get("policy_architecture", STATELESS_ARCHITECTURE)
    chemical_architecture = resolve_chemical_communication_architecture(
        meta.get("chemical_communication_architecture"), meta.get("decay", DECAY)
    )
    hidden_dim = int(meta.get("hidden_dim", policy_hidden_dim(architecture)))
    environment = EnvironmentGPU(
        wgpu_device, int(meta["channels"]), int(meta["field_n"]), int(meta["field_n"]), meta.get("decay", DECAY),
        meta.get("deposit_rate", DEPOSIT_RATE), chemical_architecture,
        meta.get("normalize_deposits_by_local_density", NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY),
        meta.get("deposit_density_reference", DEPOSIT_DENSITY_REFERENCE),
        grid_velocity=core.grid_vel,
        channel_profiles=resolve_channel_profiles(
            int(meta["channels"]),
            meta.get("chemical_channel_profiles", homogeneous_channel_profiles(int(meta["channels"]))),
        ),
    )
    agents = AgentsGPU(
        wgpu_device,
        core,
        environment,
        int(meta["channels"]),
        hidden_dim,
        meta.get("max_accel", MAX_ACCEL),
        meta.get("max_strafe", MAX_STRAFE),
        meta.get("max_env_write", MAX_ENV_WRITE),
        meta.get("max_angular_accel", MAX_ANGULAR_ACCEL),
        meta.get("angular_damping", ANGULAR_DAMPING),
        meta.get("max_angular_velocity", MAX_ANGULAR_VELOCITY),
        # Falls back to the current constant/value for older checkpoints
        # saved before "chirality"/"deposit_distance"/"split_displacement"/
        # "division_cooldown"/"friction"/"mass_ramp_macro_steps" rode along in
        # best_meta.json —
        # "particles" itself has ALWAYS been recorded, but meant "starting
        # count" on any checkpoint trained before growth existed; using it
        # as the growth cap here regardless is still correct (evolve.py's
        # own module docstring: a policy that never learns to use the
        # division drive just stays at 1 particle forever either way, same
        # as this used to be the ONLY option for an old, pre-growth
        # checkpoint).
        meta.get("chirality", CHIRALITY),
        meta.get("deposit_distance", DEPOSIT_DISTANCE),
        density.particle_cap,
        density.spacing,
        meta.get("division_cooldown", DIVISION_COOLDOWN),
        meta.get("friction", FRICTION),
        density.deposit_sigma,
        1.0,
        meta["spawn_x"],
        meta["spawn_y"],
        meta.get("elastic_strain_scale", 0.15),
        meta.get("elastic_strain_inputs_enabled", False),
        policy_architecture=architecture,
        internal_state_speed=meta.get("internal_state_speed", INTERNAL_STATE_SPEED),
        division_directionality=meta.get("division_directionality", DIVISION_DIRECTIONALITY),
        division_drive_boost=meta.get("division_drive_boost", DIVISION_DRIVE_BOOST),
        chemical_communication_architecture=chemical_architecture,
        growth_compression_start=meta.get("growth_compression_start", GROWTH_COMPRESSION_START),
        growth_compression_stop=meta.get("growth_compression_stop", GROWTH_COMPRESSION_STOP),
        growth_compression_feedback=meta.get("growth_compression_feedback", 0.0),
    )
    agents.load_weights(weights)
    agents.set_chemical_gradient_input_scale(density.chemical_gradient_input_scale)
    agents.set_chemical_projection_weight(density.chemical_projection_weight)

    target = load_target(meta["target"])

    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(meta["spawn_x"], meta["spawn_y"]),
        spawn_half_width=meta["spawn_half_width"],
        gravity=meta["gravity"],
        seed=meta.get("winner_seed", meta["seed"]),
        # Checkpoints predating multi-rate communication were trained with
        # exactly one neural/environment round per mechanical macro step.
        neural_updates_per_macro=meta.get("neural_updates_per_macro", 1),
        communication_speed=meta.get("communication_speed", 1.0),
        initial_particle_count=density.initial_particles,
        initial_condition=meta.get("initial_condition", "none"),
        initial_condition_strength=meta.get("initial_condition_strength", 0.3),
        initial_condition_channel=meta.get("initial_condition_channel", 0),
        material_area_budget=meta.get("material_area_budget", 0.0),
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
            area = rest[:, 8] * np.linalg.det(rest[:, :4].reshape(-1,2,2))
            # divisionBias/original world area is index 8 in RestState.
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
                magnitude_metadata_max_error=float(np.max(np.abs(magnitude-state['mitosisPropensity']))))
            rows.append(row)
            if step_index == 1 or step_index % 100 == 0:
                print(json.dumps(row), flush=True)
        original_step(substeps)
    core.step = measured_step
    for step_index in range(1, int(meta['macro_steps'])+1):
        sim.macro_step(meta['substeps_per_macro'], growth_enabled=step_index<=meta.get('growth_steps',meta['macro_steps']))
    out = Path(__file__).parent / 'experiments' / 'growth_output'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint_generation=meta['generation'], seed=meta.get('winner_seed',meta['seed']),
        metadata=meta, note='All points after final neural round and resampling, before physics; component statistics undo alignment rotation; temporal snapshots equally spaced.', rows=rows)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print('Report: '+str(out/'report.json'), flush=True)

if __name__ == '__main__':
    main()

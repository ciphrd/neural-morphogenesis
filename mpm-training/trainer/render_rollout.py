"""One-off visual sanity check for evolve.py's checkpoints — not part of
the eventual trainer/server (see ../viewer/README.md's own staging).
Loads checkpoints/best.npy + best_meta.json, replays that rollout, and
rasterizes the *aligned* final particle positions (via alignment.py's
own Chamfer-based best_alignment — the SAME rotation-search algorithm
evolve.py's own _score_fitness() currently scores/selects checkpoints
with, see that module's own module docstring for the current Chamfer/
raster back-and-forth; if fitness is ever switched back to raster.py's
own rotation search instead, this file's own alignment would once again
be a reasonable-but-not-necessarily-exact stand-in, not the literal pose
that scored the checkpoint) overlaid on the target point cloud — showing
the raw, un-transformed positions instead would make a perfectly-scoring
but rotated/translated result look wrong by eye, since fitness is
pose-invariant but a flat image isn't. Mirrors render_check.py's own
role for feasibility_check.py.

Usage:
    python render_rollout.py [out_dir]
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
    DEPOSIT_SIGMA,
    DIVISION_COOLDOWN,
    DIVISION_DIRECTIONALITY,
    FIELD_N,
    FRICTION,
    GROWTH_MAX,
    GROWTH_DURATION_MACRO_STEPS,
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
from training_sim import TrainingRollout
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
        MATERIAL_E,
        MATERIAL_NU,
        MATERIAL_HARDENING,
        elasticity=meta.get("material_elasticity", MATERIAL_ELASTICITY),
        particle_mass=density.particle_mass,
        particle_volume=density.particle_volume,
        **material_kwargs,
    )
    core.set_damping(DAMPING_LOSS_FRACTION, meta["substeps_per_macro"])
    core.set_splat_radius(density.splat_radius)
    core.set_repulsion_strength(density.repulsion_strength, density.repulsion_max_delta)

    architecture = meta.get("policy_architecture", STATELESS_ARCHITECTURE)
    chemical_architecture = resolve_chemical_communication_architecture(
        meta.get("chemical_communication_architecture"), meta.get("decay", DECAY)
    )
    hidden_dim = int(meta.get("hidden_dim", policy_hidden_dim(architecture)))
    environment = EnvironmentGPU(
        wgpu_device, CHEM_CHANNELS, FIELD_N, FIELD_N, meta.get("decay", DECAY),
        meta.get("deposit_rate", DEPOSIT_RATE), chemical_architecture,
    )
    agents = AgentsGPU(
        wgpu_device,
        core,
        environment,
        CHEM_CHANNELS,
        hidden_dim,
        MAX_ACCEL,
        MAX_STRAFE,
        MAX_ENV_WRITE,
        MAX_ANGULAR_ACCEL,
        ANGULAR_DAMPING,
        MAX_ANGULAR_VELOCITY,
        # Falls back to the current constant/value for older checkpoints
        # saved before "chirality"/"deposit_distance"/"split_displacement"/
        # "division_cooldown"/"friction"/"mass_ramp_macro_steps" rode along in
        # best_meta.json —
        # "particles" itself has ALWAYS been recorded, but meant "starting
        # count" on any checkpoint trained before growth existed; using it
        # as the growth cap here regardless is still correct (evolve.py's
        # own module docstring: a policy that never learns to use the
        # growth channel just stays at 1 particle forever either way, same
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
        chemical_communication_architecture=chemical_architecture,
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
    )

    growth_steps = meta.get("growth_steps")
    for i in range(meta["macro_steps"]):
        sim.macro_step(
            meta["substeps_per_macro"],
            growth_enabled=growth_steps is None or i < growth_steps,
        )
        if i % 4 == 0 or i == meta["macro_steps"] - 1:
            pos = sim.positions()
            _, aligned = best_alignment(pos, target.points)
            img = rasterize(aligned, target.points)
            path = out_dir / f"rollout_{i:03d}.png"
            img.save(path)
            print(f"wrote {path}")

    print(f"\nDone (target={meta['target']!r}, checkpoint fitness={meta['fitness']:.4f}) — gray=target, white=grown")
    return 0


if __name__ == "__main__":
    sys.exit(main())

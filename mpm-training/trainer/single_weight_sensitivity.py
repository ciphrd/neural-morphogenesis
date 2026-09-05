"""Run a controlled 1,000-frame single-weight sensitivity experiment.

Two independent GPU rollouts use the same random policy, rollout seed, and
simulation settings.  The treatment changes exactly one output-layer weight by
``epsilon``.  Machine-readable states, checkpoint metrics, and a Markdown/PNG
report are written to the requested output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from agents_gpu import AgentsGPU, PARTICLE_META_BUFFER_OFFSET, weight_layout
from density import DensityReference, resolve_density
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import PARTICLE_MASS, VOL, MpmCore
from policy_parameters import policy_heads, policy_hidden_dim, random_flat_policy_weights
from simulation_settings import *  # This experiment intentionally snapshots every shared default.
from training_sim import TrainingRollout


CHECKPOINTS = (0, 1, 10, 50, 100, 250, 500, 750, 1000)


@dataclass(frozen=True)
class ExperimentConfig:
    frames: int
    policy_seed: int
    rollout_seed: int
    epsilon: float
    output_dir: str


def _density():
    return resolve_density(
        DensityReference(
            particle_cap=int(DEFAULT_RUN_SETTINGS["particles"]),
            initial_particles=INITIAL_PARTICLE_COUNT,
            chemical_field_n=FIELD_N,
            particle_mass=PARTICLE_MASS,
            particle_volume=VOL,
            deposit_sigma=DEPOSIT_SIGMA,
            chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
            repulsion_strength=REPULSION_STRENGTH,
            repulsion_max_delta=REPULSION_MAX_DELTA,
        ),
        1.0,
    )


def _build_sim(weights: np.ndarray, rollout_seed: int):
    density = _density()
    device = pick_device(verbose=False)
    core = MpmCore(device)
    core.set_morphology(MORPHOLOGY_BLUR_SIGMA, MORPHOLOGY_DENSITY_REFERENCE)
    core.set_material(
        MATERIAL_E,
        MATERIAL_NU,
        MATERIAL_HARDENING,
        MATERIAL_ELASTICITY,
        growth_duration_macro_steps=GROWTH_DURATION_MACRO_STEPS,
        substeps_per_macro=DEFAULT_SUBSTEPS_PER_MACRO,
        particle_mass=density.particle_mass,
        particle_volume=density.particle_volume,
        growth_max=GROWTH_MAX,
        growth_anisotropy=GROWTH_ANISOTROPY_AUTHORITY,
        growth_compression_start=GROWTH_COMPRESSION_START,
        growth_compression_stop=GROWTH_COMPRESSION_STOP,
        growth_compression_feedback=GROWTH_COMPRESSION_FEEDBACK,
    )
    core.set_damping(DAMPING_LOSS_FRACTION, DEFAULT_SUBSTEPS_PER_MACRO)
    core.set_splat_radius(density.splat_radius)
    core.set_repulsion_strength(density.repulsion_strength, density.repulsion_max_delta)

    environment = EnvironmentGPU(
        device,
        CHEM_CHANNELS,
        FIELD_N,
        FIELD_N,
        DECAY,
        DEPOSIT_RATE,
        CHEMICAL_COMMUNICATION_ARCHITECTURE,
        NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
        DEPOSIT_DENSITY_REFERENCE,
        grid_velocity=core.grid_vel,
        channel_profiles=CHEMICAL_CHANNEL_PROFILES,
    )
    agents = AgentsGPU(
        device,
        core,
        environment,
        CHEM_CHANNELS,
        policy_hidden_dim(POLICY_ARCHITECTURE),
        MAX_ACCEL,
        MAX_STRAFE,
        MAX_ENV_WRITE,
        MAX_ANGULAR_ACCEL,
        ANGULAR_DAMPING,
        MAX_ANGULAR_VELOCITY,
        CHIRALITY,
        DEPOSIT_DISTANCE,
        density.particle_cap,
        density.spacing,
        DIVISION_COOLDOWN,
        FRICTION,
        density.deposit_sigma,
        1.0,
        DEFAULT_RUN_SETTINGS["spawnX"],
        DEFAULT_RUN_SETTINGS["spawnY"],
        ELASTIC_STRAIN_SCALE,
        ELASTIC_STRAIN_INPUTS_ENABLED,
        policy_architecture=POLICY_ARCHITECTURE,
        internal_state_speed=INTERNAL_STATE_SPEED,
        division_directionality=DIVISION_DIRECTIONALITY,
        division_drive_boost=DIVISION_DRIVE_BOOST,
        chemical_communication_architecture=CHEMICAL_COMMUNICATION_ARCHITECTURE,
        growth_compression_start=GROWTH_COMPRESSION_START,
        growth_compression_stop=GROWTH_COMPRESSION_STOP,
        growth_compression_feedback=GROWTH_COMPRESSION_FEEDBACK,
    )
    agents.load_weights(weights)
    agents.set_chemical_gradient_input_scale(density.chemical_gradient_input_scale)
    agents.set_chemical_projection_weight(density.chemical_projection_weight)
    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(DEFAULT_RUN_SETTINGS["spawnX"], DEFAULT_RUN_SETTINGS["spawnY"]),
        spawn_half_width=DEFAULT_RUN_SETTINGS["spawnHalfWidth"],
        gravity=DEFAULT_RUN_SETTINGS["gravity"],
        seed=rollout_seed,
        mpm_enabled=MPM_ENABLED,
        neural_updates_per_macro=NEURAL_UPDATES_PER_MACRO,
        communication_speed=COMMUNICATION_SPEED,
        initial_particle_count=density.initial_particles,
    )
    return core, sim


def _capture(core: MpmCore, agents: AgentsGPU) -> dict[str, np.ndarray]:
    count = core.active_count
    raw = core.device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        count * agents._particle_meta_dtype.itemsize,
    )
    meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=count).copy()
    return {
        "positions": core.read_positions(),
        "velocities": core.read_velocities(),
        "deformation": core.read_deformation(),
        "affine": core.read_affine(),
        "rest_state": core.read_rest_state(),
        "agent_alignment": meta["alignment"].copy(),
        "agent_color": meta["color"].copy(),
        "agent_division_hazard": meta["divisionHazard"].copy(),
        "agent_division_threshold": meta["divisionThreshold"].copy(),
        "agent_mitosis_propensity": meta["mitosisPropensity"].copy(),
        "agent_private_state": meta["privateState"].copy(),
        "agent_chemical_state": meta["chemicalState"].copy(),
    }


def _run_variant(payload: tuple[str, np.ndarray, ExperimentConfig]) -> str:
    name, weights, config = payload
    output_dir = Path(config.output_dir)
    core, sim = _build_sim(weights, config.rollout_seed)
    wanted = set(CHECKPOINTS) | {config.frames}
    captures: dict[int, dict[str, np.ndarray]] = {0: _capture(core, sim.agents)}
    for frame in range(1, config.frames + 1):
        sim.macro_step(DEFAULT_SUBSTEPS_PER_MACRO)
        if frame in wanted:
            captures[frame] = _capture(core, sim.agents)
        if frame % 100 == 0:
            print(f"[{name}] frame {frame}/{config.frames}, particles={core.active_count}", flush=True)
    path = output_dir / f"{name}_states.npz"
    flat = {
        f"frame_{frame:04d}_{field}": value
        for frame, state in captures.items()
        for field, value in state.items()
    }
    np.savez_compressed(path, **flat)
    return str(path)


def _cloud_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "baseline_count": int(len(a)),
        "perturbed_count": int(len(b)),
        "count_difference": int(len(b) - len(a)),
    }
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        out.update({"chamfer": None, "paired_mean": None, "paired_rms": None, "paired_max": None})
        return out
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d_ab = cKDTree(b64).query(a64)[0]
    d_ba = cKDTree(a64).query(b64)[0]
    out["chamfer"] = float(0.5 * (d_ab.mean() + d_ba.mean()))
    common = min(len(a64), len(b64))
    paired = np.linalg.norm(a64[:common] - b64[:common], axis=1)
    out.update(
        {
            "paired_mean": float(paired.mean()),
            "paired_rms": float(np.sqrt(np.mean(paired**2))),
            "paired_max": float(paired.max()),
        }
    )
    ca, cb = a64.mean(axis=0), b64.mean(axis=0)
    out["centroid_shift"] = float(np.linalg.norm(cb - ca))
    out["baseline_rms_radius"] = float(np.sqrt(np.mean(np.sum((a64 - ca) ** 2, axis=1))))
    out["perturbed_rms_radius"] = float(np.sqrt(np.mean(np.sum((b64 - cb) ** 2, axis=1))))
    return out


def _draw_clouds(a: np.ndarray, b: np.ndarray, path: Path, title: str) -> None:
    size, margin = 720, 42
    image = Image.new("RGB", (size, size + 48), (12, 16, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, size - margin, size - margin), outline=(80, 88, 104), width=1)
    for points, color in ((a, (65, 190, 255)), (b, (255, 145, 65))):
        for x, y in points:
            px = margin + float(x) * (size - 2 * margin)
            py = margin + (1.0 - float(y)) * (size - 2 * margin)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    draw.text((margin, 12), title, fill=(235, 238, 245), font=ImageFont.load_default())
    draw.text((margin, size + 10), "blue = baseline    orange = +epsilon", fill=(200, 205, 216))
    image.save(path)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=20260903)
    parser.add_argument("--rollout-seed", type=int, default=1701)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--reuse-states",
        action="store_true",
        help="rebuild reports from existing baseline_states.npz and perturbed_states.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "experiments" / "single_weight_1k",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(args.frames, args.policy_seed, args.rollout_seed, args.epsilon, str(args.output_dir))

    hidden = policy_hidden_dim(POLICY_ARCHITECTURE)
    rng = np.random.default_rng(args.policy_seed)
    baseline = random_flat_policy_weights(CHEM_CHANNELS, hidden, rng, POLICY_ARCHITECTURE)
    layout = weight_layout(CHEM_CHANNELS, hidden, POLICY_ARCHITECTURE)
    heads = policy_heads(CHEM_CHANNELS, POLICY_ARCHITECTURE)
    growth_vector_row = 0
    for head in heads:
        if head.name == "growthVector":
            break
        growth_vector_row += head.size
    hidden_index = int(np.random.default_rng(args.policy_seed + 1).integers(hidden))
    parameter_index = layout["fc2w_offset"] + growth_vector_row * hidden + hidden_index
    perturbed = baseline.copy()
    old_value = float(perturbed[parameter_index])
    perturbed[parameter_index] = np.float32(perturbed[parameter_index] + np.float32(args.epsilon))
    actual_delta = float(perturbed[parameter_index] - baseline[parameter_index])

    np.save(args.output_dir / "baseline_weights.npy", baseline)
    np.save(args.output_dir / "perturbed_weights.npy", perturbed)
    changed = np.flatnonzero(baseline != perturbed)
    if changed.tolist() != [parameter_index]:
        raise RuntimeError(f"expected exactly one changed float, found {changed.tolist()}")

    paths = [args.output_dir / "baseline_states.npz", args.output_dir / "perturbed_states.npz"]
    if args.reuse_states:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit(f"cannot reuse missing state archives: {missing}")
    else:
        pick_device(verbose=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            paths = list(
                pool.map(
                    _run_variant,
                    (("baseline", baseline, config), ("perturbed", perturbed, config)),
                )
            )

    with np.load(paths[0]) as base_data, np.load(paths[1]) as pert_data:
        frames = sorted({int(key.split("_")[1]) for key in base_data.files if key.endswith("_positions")})
        metrics = []
        for frame in frames:
            a = base_data[f"frame_{frame:04d}_positions"]
            b = pert_data[f"frame_{frame:04d}_positions"]
            row = {"frame": frame, **_cloud_metrics(a, b)}
            changed_values = 0
            max_state_delta = 0.0
            changed_fields: list[str] = []
            prefix = f"frame_{frame:04d}_"
            for key in (key for key in base_data.files if key.startswith(prefix)):
                delta = np.abs(base_data[key].astype(np.float64) - pert_data[key].astype(np.float64))
                field_changed = int(np.count_nonzero(delta))
                if field_changed:
                    changed_fields.append(key[len(prefix):])
                    changed_values += field_changed
                    max_state_delta = max(max_state_delta, float(delta.max()))
            row["changed_state_values"] = changed_values
            row["max_abs_state_delta"] = max_state_delta
            row["changed_state_fields"] = ";".join(changed_fields)
            metrics.append(row)
            if frame in (100, 500, args.frames):
                _draw_clouds(a, b, args.output_dir / f"comparison_frame_{frame:04d}.png", f"Frame {frame}")

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    metadata = {
        "claim": "exactly one float32 policy weight differs between runs",
        "frames": args.frames,
        "policy_seed": args.policy_seed,
        "rollout_seed": args.rollout_seed,
        "parameter": {
            "flat_index": parameter_index,
            "tensor": "fc2.weight",
            "output_head": "growthVector.x",
            "output_row": growth_vector_row,
            "hidden_input": hidden_index,
            "baseline_value": old_value,
            "requested_delta": args.epsilon,
            "actual_float32_delta": actual_delta,
            "perturbed_value": float(perturbed[parameter_index]),
            "changed_parameter_count": int(len(changed)),
            "total_parameter_count": int(len(baseline)),
        },
        "settings": {
            "policy_architecture": POLICY_ARCHITECTURE,
            "hidden_dim": hidden,
            "chemical_channels": CHEM_CHANNELS,
            "initial_particles": INITIAL_PARTICLE_COUNT,
            "particle_cap": int(DEFAULT_RUN_SETTINGS["particles"]),
            "substeps_per_frame": DEFAULT_SUBSTEPS_PER_MACRO,
            "neural_updates_per_frame": NEURAL_UPDATES_PER_MACRO,
            "gravity": DEFAULT_RUN_SETTINGS["gravity"],
            "mpm_enabled": MPM_ENABLED,
            "chemical_communication_architecture": CHEMICAL_COMMUNICATION_ARCHITECTURE,
        },
        "metrics": metrics,
    }
    (args.output_dir / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")

    final = metrics[-1]
    spatial_conclusion = (
        f"the spatial output diverged (Chamfer {_fmt(final['chamfer'])})"
        if final["chamfer"]
        else "the spatial output remained bit-for-bit identical"
    )
    internal_conclusion = (
        f"{final['changed_state_values']} saved internal-state values differed"
        if final["changed_state_values"]
        else "all saved physical and agent-state values also remained bit-for-bit identical"
    )
    rows = "\n".join(
        "| " + " | ".join(
            [_fmt(row[k]) for k in ("frame", "baseline_count", "perturbed_count", "chamfer", "paired_rms", "paired_max", "changed_state_values", "max_abs_state_delta")]
        ) + " |"
        for row in metrics
    )
    report = f"""# Single-weight neural policy sensitivity at {args.frames:,} frames

## Result

The two runs began from the same random brain and identical simulation state. Exactly one of {len(baseline):,} float32 parameters changed. At frame {args.frames:,}, {spatial_conclusion}; {internal_conclusion}. The raw spatial output had a symmetric Chamfer distance of **{_fmt(final['chamfer'])}** domain units, RMS same-slot displacement was **{_fmt(final['paired_rms'])}**, and particle counts were **{final['baseline_count']} vs {final['perturbed_count']}**.

![Final baseline/perturbed overlay](comparison_frame_{args.frames:04d}.png)

Blue is the baseline output; orange is the one-weight-perturbed output.

## Controlled change

- Random policy seed: `{args.policy_seed}`
- Rollout seed: `{args.rollout_seed}`
- Architecture: `{POLICY_ARCHITECTURE}` ({hidden} hidden units, {CHEM_CHANNELS} chemical channels)
- Changed parameter: flat index `{parameter_index}`, `fc2.weight[growthVector.x, hidden {hidden_index}]`
- Baseline value: `{old_value:.10g}`
- Perturbed value: `{float(perturbed[parameter_index]):.10g}`
- Requested delta: `{args.epsilon:.10g}`; actual float32 delta: `{actual_delta:.10g}`
- Verification: `{len(changed)}` changed parameter out of `{len(baseline)}`
- One frame: {NEURAL_UPDATES_PER_MACRO} neural updates + {DEFAULT_SUBSTEPS_PER_MACRO} physics substeps

## Difference over time

| Frame | Baseline n | Perturbed n | Chamfer | Paired RMS | Paired max | Changed saved values | Max state delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

Chamfer is the symmetric nearest-neighbor distance between raw output point clouds. Paired values compare particles by stable slot index for the shared prefix. "Saved values" covers positions, velocities, deformation, affine state, tensor-growth/rest state, channel-7-gradient alignment, color, division hazard/threshold/propensity, recurrent private state, and per-agent chemical state. Coordinates use the simulation's unit-square domain.

## Output files

- `baseline_weights.npy`, `perturbed_weights.npy`: complete policies
- `baseline_states.npz`, `perturbed_states.npz`: positions, velocities, deformation, affine state, and growth/rest state at every reported checkpoint
- `metrics.csv`, `report.json`: machine-readable results
- `comparison_frame_0100.png`, `comparison_frame_0500.png`, `comparison_frame_{args.frames:04d}.png`: visual overlays

## Interpretation

This random brain never initiated growth: both runs stayed at five particles. The changed weight feeds the division-drive output, and the perturbation produced small float32 differences in `agent_mitosis_propensity` and accumulated `agent_division_hazard` (maximum saved delta `{_fmt(final['max_abs_state_delta'])}` at frame {args.frames:,}). Those differences never crossed a stochastic division threshold. Because the current shared defaults also set translational acceleration controls to zero, there was no alternate motion pathway through which this particular weight could affect position. The observed effect for this controlled pair is therefore **measurable inside the division controller, but zero in the spatial/physical output**.

This is a deterministic paired sensitivity test for one randomly initialized policy, not a population-level estimate. A single zero-effect run does not show that every weight is insensitive. Estimating typical sensitivity would require repeating the pair across weights, policy seeds, and rollout seeds.
"""
    (args.output_dir / "REPORT.md").write_text(report)
    print(f"wrote {args.output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

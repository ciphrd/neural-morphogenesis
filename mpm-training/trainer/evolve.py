from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from agents_gpu import AgentsGPU
from chemical_channels import profiles_to_wire
from density import DENSITY_MODEL_VERSION, INITIAL_SPACING_IN_SAMPLE_SPACINGS, DensityReference, ResolvedDensity, parse_multipliers, resolve_density
from environment_gpu import EnvironmentGPU
from mpm_core import MAX_PARTICLES, PARTICLE_MASS, VOL, MpmCore
from parallel_workers import build_pool, worker_rollout
from rollout_snapshot import RolloutSnapshot
from timing import Timings, aggregate_rollouts
from gpu_timing import GpuTimings
from time import perf_counter
from raster import build_target_distance_field
from domain_fitness import FITNESS_MODEL_VERSION, target_mask, score_domains
from simulation_settings import (
    CHEM_CHANNELS,
    CHEMICAL_CHANNEL_PROFILES,
    CHEMICAL_COMMUNICATION_ARCHITECTURE,
    CHEMICAL_GRADIENT_INPUT_SCALE,
    CHEMICAL_VALUE_INPUT_MULTIPLIER,
    COMMUNICATION_SPEED,
    DAMPING_LOSS_FRACTION,
    DECAY,
    DEPOSIT_RATE,
    NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
    DEFAULT_RUN_SETTINGS,
    DEFAULT_SUBSTEPS_PER_MACRO,
    ELASTIC_STRAIN_SCALE,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    FIELD_N,
    GROWTH_DURATION_MACRO_STEPS,
    GROWTH_MODEL_VERSION,
    GROWTH_COMPRESSION_FEEDBACK,
    GROWTH_COMPRESSION_START,
    GROWTH_COMPRESSION_STOP,
    GROWTH_ANISOTROPY_AUTHORITY,
    INITIAL_PARTICLE_COUNT,
    INTERNAL_STATE_SPEED,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MPM_ENABLED,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    NEURAL_UPDATES_PER_MACRO,
    POLICY_ARCHITECTURE,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
)
from policy_parameters import (
    CELL_MEMORY_OPTIONS,
    CHEMICAL_COMMUNICATION_ARCHITECTURES,
    POLICY_ARCHITECTURES,
    STATELESS_ARCHITECTURE,
    architecture_for_cell_memory,
    cell_memory_for_architecture,
    mutation_scale_vector,
    mutation_scales,
    policy_hidden_dim,
)
from targets import TargetShape, available_targets, load_target
from training_sim import TrainingRollout
from update_rule import UpdateRule

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

def density_reference(args: argparse.Namespace) -> DensityReference:
    """The run's q=1 settings; public particle counts remain reference counts."""
    return DensityReference(
        particle_cap=int(args.particles),
        initial_particles=int(args.initial_particles),
        chemical_field_n=FIELD_N,
        particle_mass=PARTICLE_MASS,
        particle_volume=VOL,
        chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
        repulsion_strength=REPULSION_STRENGTH,
        repulsion_max_delta=REPULSION_MAX_DELTA,
    )

def resolve_run_density(args: argparse.Namespace, multiplier: float) -> ResolvedDensity:
    return resolve_density(
        density_reference(args), multiplier,
        allow_unsafe=bool(getattr(args, "allow_unsafe_density", False)),
    )

# Target points and particle positions both already live in MpmCore's
# fixed [0,1]^2 domain (see targets.py) — unlike envnca's own raster
# extent, which depends on that project's own --grid-size, this is
# always the same fixed unit square, so it's a constant, not a function
# of any CLI arg.
RASTER_EXTENT = (0.0, 1.0, 0.0, 1.0)

# Fractional offsets *before* the end of a rollout's own macro_steps at
# which a fitness-scoring snapshot is taken — mirrors envnca/evolve.py's
# own CAPTURE_OFFSETS (see that module's rollout() docstring for the
# full reasoning): scoring only the very final step lets a candidate
# "pose for one known instant" — converging into shape right on cue and
# immediately drifting apart after, which would score perfectly under a
# single-snapshot fitness while looking wrong at any other moment. Taking
# several evenly-spaced snapshots across the last 10% of the rollout and
# blending their mean with the worst (highest, since lower is better) keeps
# distinctions across the whole window while strongly penalizing immediate
# destabilization. Unlike
# envnca, this project does NOT also jitter the rollout's own total
# macro_steps count — not asked for, and this project's macro_steps is
# already a small, fixed count (tens, not hundreds), so the marginal
# "don't let it pose for one known instant" value of also randomizing
# total length is much smaller here than the five-snapshot spread alone
# already provides.
CAPTURE_OFFSETS = (0.10, 0.075, 0.05, 0.025, 0.0)

def get_weights(model: UpdateRule) -> np.ndarray:
    """Also the exact flat layout agents_gpu.AgentsGPU.load_weights()
    expects — see that method's own docstring for why (nn.Linear's own
    fc1w/fc1b followed by the logical heads concatenated into fc2w/fc2b,
    matching core/agents.wgsl's FC1W_OFFSET/etc. indexing)."""
    return model.flat_parameters().detach().cpu().numpy()

def set_weights(model: UpdateRule, flat: np.ndarray) -> None:
    """CPU-only — `model` (a scratch UpdateRule) is never used for a live
    forward pass anymore (see training_sim.py's own module docstring),
    only for export_weights()'s JSON shape at checkpoint time, so there's
    no reason to pay a device transfer here."""
    model.load_flat_parameters(torch.from_numpy(flat).float())

def mutate(
    weights: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    architecture: str = STATELESS_ARCHITECTURE,
) -> np.ndarray:
    """Gaussian mutation with a global sigma and semantic bucket scales.

    The trunk keeps the CLI sigma. Output heads use smaller multipliers because
    each row directly controls a bounded behavior and receives 128 hidden
    contributions; equal elementwise noise there caused disproportionately
    large behavioral jumps, especially for persistent direction/control state.
    """
    scales = mutation_scale_vector(CHEM_CHANNELS, policy_hidden_dim(architecture), architecture)
    if weights.shape != scales.shape:
        raise ValueError(f"expected {scales.size} policy parameters, got {weights.size}")
    noise = rng.normal(size=weights.shape).astype(np.float32)
    return (weights.astype(np.float32, copy=False) + noise * np.float32(sigma) * scales).astype(np.float32)

def estimated_sample_capacity(args, target):
    # Two samples per spacing-squared is an allocation estimate, not a
    # physical material limit. Anisotropy and conforming refinement need slack.
    spacing = resolve_run_density(args, 1.0).spacing
    return max(2 * args.initial_particles, int(np.ceil(2 * target.filled_area() / spacing**2)))

def offspring_sigma(args, index):
    factors = getattr(args, "mutation_factors", (1.,))
    return args.mutation_sigma * factors[index % len(factors)]

def initial_population(args, rng):
    """Start a new objective from random policies or an explicitly supplied parent."""
    path = getattr(args, "initial_weights", None)
    if path is None:
        return [get_weights(UpdateRule(CHEM_CHANNELS, args.policy_architecture)) for _ in range(args.population)]
    weights = np.load(path, allow_pickle=False)
    expected = mutation_scale_vector(CHEM_CHANNELS, policy_hidden_dim(args.policy_architecture), args.policy_architecture)
    if weights.shape != expected.shape or not np.isfinite(weights).all():
        raise ValueError(f"initial weights must be {expected.size} finite parameters for {args.policy_architecture}")
    weights = weights.astype(np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("initial weights overflow float32")
    return [weights.copy()] + [mutate(weights, offspring_sigma(args, i), rng, args.policy_architecture)
                               for i in range(args.population-1)]

def report_shape_capacity(args, target):
    if target.filled_area() <= 0:
        raise SystemExit("training target must contain at least one filled texel")
    estimate = estimated_sample_capacity(args, target)
    print(f"Sample capacity: {args.particles}; estimated target sampling capacity: ~{estimate} at 1x density")
    if args.particles < estimate:
        print(f"The current sampling cap may pause growth before the target is filled. "
              f"Consider --particles {estimate}; adding --particle-densities 0.5 reduces its sampling cost.")

def shape_settings(args, target):
    """Wire settings also embed target geometry for autonomous offline playback."""
    return {
        "estimatedSampleCapacity": estimated_sample_capacity(args, target),
        "stableStop": False,
        "shapeCheckInterval": args.shape_check_interval,
        "shapeConfirmations": args.shape_confirmations,
        "shapeSettleSteps": args.shape_settle_steps,
        "shapeMissingTolerance": args.shape_missing_tolerance,
        "shapeSpillTolerance": args.shape_spill_tolerance,
        "shapeOverlapTolerance": args.shape_overlap_tolerance,
        # Embed the resolved fitness mask. Replays therefore remain identical
        # even if the source PNG is subsequently edited or removed.
        "shapeTarget": target.wire(args.raster_resolution),
    }

def _aggregate_scores(scores, args):
    return ((1-args.fitness_temporal_worst_weight)*float(np.mean(scores))
            + args.fitness_temporal_worst_weight*max(scores))

def rollout(
    weights: np.ndarray,
    target: TargetShape,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    core: MpmCore,
    agents: AgentsGPU,
    environment: EnvironmentGPU,
    return_positions: bool = False,
    density_multiplier: float = 1.0,
    return_snapshot: bool = False,
) -> float | tuple[float, np.ndarray] | RolloutSnapshot:
    """Score late-window and terminal states; stop at capacity or low growth.

    Optional snapshots retain the final scoring rasters for server previews.
    """
    timings = Timings()
    if "gpu_timings" not in core.__dict__:
        core.gpu_timings = GpuTimings(core.device)
    gpu_timings = core.gpu_timings
    gpu_timings.interval = getattr(args, "gpu_timing_interval", 20)
    gpu_timings.reset()
    agents.load_weights(weights)
    density = resolve_run_density(args, density_multiplier)
    if density.particle_cap > agents.particle_capacity:
        raise ValueError(
            f"resolved particle cap {density.particle_cap} exceeds worker capacity {agents.particle_capacity}"
        )

    core.set_material(
        MATERIAL_E,
        MATERIAL_NU,
        MATERIAL_HARDENING,
        MATERIAL_ELASTICITY,
        growth_duration_macro_steps=GROWTH_DURATION_MACRO_STEPS,
        substeps_per_macro=args.substeps_per_macro,
        particle_mass=density.particle_mass,
        particle_volume=density.particle_volume,
    )
    core.set_damping(DAMPING_LOSS_FRACTION, args.substeps_per_macro)
    core.set_splat_radius(density.splat_radius)
    core.set_repulsion_strength(density.repulsion_strength, density.repulsion_max_delta)
    agents.set_density_geometry(density.spacing)
    agents.set_chemical_gradient_input_scale(density.chemical_gradient_input_scale)

    agents.set_max_active_particles(density.particle_cap)

    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(args.spawn_x, args.spawn_y),
        gravity=args.gravity,
        seed=seed,
        mpm_enabled=MPM_ENABLED,
        initial_particle_count=density.initial_particles,
        initial_condition=getattr(args, "initial_condition", "none"),
        initial_condition_strength=getattr(args, "initial_condition_strength", 0.3),
        initial_condition_channel=getattr(args, "initial_condition_channel", 0),
        initial_spacing=density.initial_spacing,
    )

    timings.add("setup", perf_counter()-timings.started)
    sim.timings = timings
    sim.gpu_timings = gpu_timings

    def score_snapshot():
        with timings.measure("geometryReadback"):
            vertices = core.read_rest_state()[:, 8:14]
        colors = None
        if train_color:
            with timings.measure("colorReadback"):
                colors = agents.read_colors(core.active_count)
        with timings.measure("fitness"):
            return score_domains(vertices, target, target_raster, args, colors)

    checkpoint_steps = {max(1, round(args.macro_steps * (1.0 - offset))) for offset in CAPTURE_OFFSETS}
    scores = []
    last_evaluation = None
    step = 0
    train_color = target.has_color and args.fitness_color_weight > 0
    initial_sample_count = core.active_count
    stopped_for_low_growth = False
    stopped_at_capacity = core.active_count >= agents.max_active_particles
    if stopped_at_capacity:
        # An exactly-full initial seed is already terminal. This geometry read
        # is the one required final fitness sample, not a per-step poll.
        last_evaluation = score_snapshot()
        scores.append(last_evaluation.total)
    else:
        for step in range(1, args.macro_steps + 1):
            sim.macro_step(args.substeps_per_macro, growth_enabled=(args.growth_steps is None or step <= args.growth_steps))
            # macro_step() already read and cached these values to size the next
            # GPU dispatch. Capacity termination therefore adds no bridge read.
            stopped_at_capacity = (
                core.active_count >= agents.max_active_particles or agents.capacity_blocked
            )
            stopped_for_low_growth = step == 200 and core.active_count*10 < initial_sample_count*11
            if stopped_at_capacity or stopped_for_low_growth or step in checkpoint_steps:
                last_evaluation = score_snapshot()
                scores.append(last_evaluation.total)
                if stopped_at_capacity or stopped_for_low_growth:
                    break
    fitness = _aggregate_scores(scores, args)
    # Preserve replay diagnostics and existing scalar/position callers.
    core.rollout_diagnostics = {
        "steps": step, "stableMatch": False,
        "settling": False,
        "capacityBlocked": bool(agents.capacity_blocked),
        "atCapacity": core.active_count >= agents.max_active_particles,
        "stopReason": "capacity" if stopped_at_capacity else ("low-growth" if stopped_for_low_growth else "horizon"),
        "initialSamples": initial_sample_count,
        "finalSamples": core.active_count,
        "unresolvedSamples": int(agents.unresolved_samples),
        "missing": last_evaluation.match.missing,
        "spill": last_evaluation.match.spill,
        "overlap": last_evaluation.match.overlap,
    }
    positions = None
    if return_snapshot or return_positions:
        with timings.measure("positionsReadback"):
            positions = sim.positions()
    core.rollout_timings = timings.report()
    core.rollout_timings["gpu"] = gpu_timings.report()
    if return_snapshot:
        return RolloutSnapshot(fitness, last_evaluation, positions, dict(core.rollout_diagnostics), core.rollout_timings)
    return (fitness, positions) if return_positions else fitness

def run_generation(
    population: list[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
    pool: ProcessPoolExecutor,
    return_snapshot: bool = False,
) -> tuple:
    """Evaluates every candidate on the same rotating seed batch.

    A fresh batch is drawn from the run RNG once per generation, then every
    candidate is evaluated on every seed in that batch. Candidate fitness is
    the arithmetic mean across its rollouts. Sharing seeds within a generation
    removes seed luck from pairwise selection; rotating the batch between
    generations prevents elites from specializing to one fixed starting
    trajectory. ``--seeds-per-candidate=1`` preserves the old rollout cost
    while still making that generation's single seed common to all candidates.

    Rollouts are fanned out across
    `pool`'s worker processes (see parallel_workers.py's own module
    docstring for why: a single process evaluating candidates
    sequentially, or even several MpmCore instances batched within one
    process, both measured as CPU-bound on one core — see this module's
    own module docstring) — then sorts best-first and refills back up to
    `args.population` via elitism + Gaussian mutation of a randomly-
    chosen elite — plain (mu, lambda) ES, no memetic refinement. Returns
    the next population, sorted fitnesses, representative seed/density, seed
    batch, and per-density scores. With return_snapshot, also return the
    winning candidate's worst-scoring seed/density terminal snapshot, already
    rasterized by its worker. Ties preserve the original task order.

    `target` is NOT passed here — it's baked into each worker's own
    globals once, at pool creation (parallel_workers.build_pool()'s own
    initializer), since it never changes generation to generation and
    re-sending it with every task would be pure waste."""
    generation_started = perf_counter()
    seeds_per_candidate = max(1, int(getattr(args, "seeds_per_candidate", 1)))
    evaluation_seeds = [
        int(seed) for seed in rng.integers(0, 2**31 - 1, size=seeds_per_candidate)
    ]
    densities = tuple(float(q) for q in getattr(args, "particle_densities", (1.0,)))
    task_weights = [
        weights for weights in population for _q in densities for _seed in evaluation_seeds
    ]
    task_densities = [
        q for _weights in population for q in densities for _seed in evaluation_seeds
    ]
    task_seeds = [
        seed for _weights in population for _q in densities for seed in evaluation_seeds
    ]
    pool_started = perf_counter()
    if return_snapshot:
        results = list(pool.map(worker_rollout, task_weights, task_seeds, task_densities,
                                [True] * len(task_weights)))
        values = [result.fitness for result in results]
    else:
        values = list(pool.map(worker_rollout, task_weights, task_seeds, task_densities))
    pool_seconds = perf_counter()-pool_started
    rollout_fitnesses = np.asarray(values, dtype=np.float64).reshape(
        len(population), len(densities), seeds_per_candidate)
    per_density_fitnesses = np.mean(rollout_fitnesses, axis=2)
    fitnesses = (
        np.max(per_density_fitnesses, axis=1)
        if getattr(args, "density_aggregation", "worst") == "worst"
        else np.mean(per_density_fitnesses, axis=1)
    )
    # Replaying the hardest seed is more informative than choosing an
    # arbitrary or lucky member of the batch. np.argmax is stable and picks
    # the first seed when several scores tie.
    representative_cases = []
    for candidate_scores in rollout_fitnesses:
        density_index, seed_index = np.unravel_index(np.argmax(candidate_scores), candidate_scores.shape)
        representative_cases.append((evaluation_seeds[int(seed_index)], densities[int(density_index)]))

    order = np.argsort(fitnesses, kind="stable")
    winner_density_fitnesses = {
        f"{q:g}": float(per_density_fitnesses[int(order[0]), density_index])
        for density_index, q in enumerate(densities)
    }
    winner_snapshot = None
    if return_snapshot:
        candidate_index = int(order[0])
        case_index = int(np.argmax(rollout_fitnesses[candidate_index]))
        winner_snapshot = results[candidate_index * len(densities) * seeds_per_candidate + case_index]
    population = [population[i] for i in order]
    fitnesses = [float(fitnesses[i]) for i in order]
    representative_cases = [representative_cases[i] for i in order]

    elites = population[: args.elites]
    next_population = list(elites)
    while len(next_population) < args.population:
        parent = elites[rng.integers(len(elites))]
        next_population.append(mutate(parent, offspring_sigma(args, len(next_population)-len(elites)), rng, args.policy_architecture))

    winner_seed, winner_density = representative_cases[0]
    result = (
        next_population, fitnesses, winner_seed, winner_density,
        evaluation_seeds, winner_density_fitnesses,
    )
    if return_snapshot:
        winner_snapshot.generation_timings = {
            "poolSeconds": pool_seconds,
            "selectionSeconds": perf_counter()-generation_started-pool_seconds,
            "rollouts": aggregate_rollouts([item.timings for item in results]),
            "winner": winner_snapshot.timings,
        }
    return (*result, winner_snapshot) if return_snapshot else result

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults live in core/config.json; both architecture axes are public
    # run selections. The hidden alias remains exclusively for the paired
    # comparison utility's existing subprocess interface.
    parser.set_defaults(policy_architecture=None, cell_memory=None)
    parser.add_argument(
        "--cell-memory",
        choices=CELL_MEMORY_OPTIONS,
        help="private neural memory: none uses a reactive policy; recurrent adds gated per-cell state",
    )
    parser.add_argument(
        "--_comparison-policy-architecture",
        dest="policy_architecture",
        choices=POLICY_ARCHITECTURES,
        help=argparse.SUPPRESS,
    )
    from initial_conditions import PRESETS
    parser.add_argument("--initial-condition", choices=PRESETS, default=DEFAULT_RUN_SETTINGS["initialCondition"],
                        help="One-time asymmetry applied at every rollout reset")
    parser.add_argument("--initial-condition-strength", type=float, default=DEFAULT_RUN_SETTINGS["initialConditionStrength"])
    parser.add_argument("--initial-condition-channel", type=int, default=DEFAULT_RUN_SETTINGS["initialConditionChannel"])
    parser.add_argument(
        "--chemical-communication-architecture",
        choices=CHEMICAL_COMMUNICATION_ARCHITECTURES,
        default=CHEMICAL_COMMUNICATION_ARCHITECTURE,
        help=(
            "persistent-environment keeps a diffusing/decaying spatial field; "
            "cell-owned-projection stores chemistry per cell and rebuilds the field each round"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=CHECKPOINTS_DIR,
        help="checkpoint destination (useful for paired architecture comparisons)",
    )
    parser.add_argument(
        "--target", default=DEFAULT_RUN_SETTINGS["target"], choices=available_targets(),
        help="target shape name (an RGBA .png, or legacy .json, in ./targets/)"
    )
    parser.add_argument("--population", type=int, default=DEFAULT_RUN_SETTINGS["population"], help="number of candidate weight-sets per generation")
    parser.add_argument(
        "--seeds-per-candidate",
        type=int,
        default=DEFAULT_RUN_SETTINGS["seedsPerCandidate"],
        help=(
            "shared rollout seeds used to evaluate every candidate each generation; "
            "the batch rotates between generations and candidate fitness is its mean "
            "across the batch (default: 1, preserving the previous rollout cost)"
        ),
    )
    parser.add_argument("--elites", type=int, default=DEFAULT_RUN_SETTINGS["elites"], help="top performers carried into the next generation unmutated")
    parser.add_argument("--generations", type=int, default=DEFAULT_RUN_SETTINGS["totalGenerations"])
    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_RUN_SETTINGS["particles"],
        help="maximum particle count a rollout can grow to via splitting",
    )
    parser.add_argument(
        "--particle-densities",
        type=float,
        nargs="+",
        default=DEFAULT_RUN_SETTINGS["trainingDensityMultipliers"],
        metavar="Q",
        help="particle sampling-density multipliers evaluated for every candidate (default: 1.0)",
    )
    parser.add_argument(
        "--density-aggregation",
        choices=("worst", "mean"),
        default=DEFAULT_RUN_SETTINGS["densityAggregation"],
        help="combine one candidate's mean-per-density fitnesses (default: worst)",
    )
    parser.add_argument(
        "--allow-unsafe-density",
        action="store_true",
        help="allow multipliers outside core/config.json's calibrated range",
    )
    parser.add_argument(
        "--initial-particles",
        type=int,
        default=INITIAL_PARTICLE_COUNT,
        help="number of material seed cells; each emits two half-weight triangles (2*count must fit --particles)",
    )
    parser.add_argument(
        "--macro-steps",
        type=int,
        default=DEFAULT_RUN_SETTINGS["macroSteps"],
        help="total NN sense/act interventions per rollout",
    )
    parser.add_argument(
        "--growth-steps",
        type=int,
        default=None,
        help="optional last macro step in which agents may start new cell cycles; omitted means no time cutoff",
    )
    parser.add_argument("--stable-stop", action=argparse.BooleanOptionalAction, default=False,
                        help="deprecated compatibility option; training no longer checks stable matches")
    parser.add_argument("--shape-check-interval", type=int, default=DEFAULT_RUN_SETTINGS["shapeCheckInterval"])
    parser.add_argument("--shape-confirmations", type=int, default=DEFAULT_RUN_SETTINGS["shapeConfirmations"])
    parser.add_argument("--shape-settle-steps", type=int, default=DEFAULT_RUN_SETTINGS["shapeSettleSteps"])
    parser.add_argument("--shape-missing-tolerance", type=float, default=DEFAULT_RUN_SETTINGS["shapeMissingTolerance"])
    parser.add_argument("--shape-spill-tolerance", type=float, default=DEFAULT_RUN_SETTINGS["shapeSpillTolerance"])
    parser.add_argument("--shape-overlap-tolerance", type=float, default=DEFAULT_RUN_SETTINGS["shapeOverlapTolerance"])
    parser.add_argument(
        "--substeps-per-macro",
        type=int,
        default=DEFAULT_SUBSTEPS_PER_MACRO,
        help="MLS-MPM physics substeps run between each NN intervention (mpm_core.MpmCore.step's own substep unit)",
    )
    parser.add_argument("--gravity", type=float, default=DEFAULT_RUN_SETTINGS["gravity"])
    # Center of MpmCore's own [0,1]^2 domain — particles start in the
    # middle of the space, not offset toward any one wall.
    parser.add_argument("--spawn-x", type=float, default=DEFAULT_RUN_SETTINGS["spawnX"])
    parser.add_argument("--spawn-y", type=float, default=DEFAULT_RUN_SETTINGS["spawnY"])
    parser.add_argument("--mutation-sigma", type=float, default=DEFAULT_RUN_SETTINGS["mutationSigma"])
    parser.add_argument("--mutation-factors", type=float, nargs="+", default=[1.],
        help="cycle offspring through these sigma multipliers, e.g. 1 .1 .01 .001; elites are preserved")
    parser.add_argument("--initial-weights", type=Path,
        help="start a new run from this flat .npy policy; select its matching architecture; scores are reevaluated")
    parser.add_argument("--fitness-alignment", choices=("raster", "geometry"), default="raster",
        help="raster uses fast image rotation; geometry refines pose with exact triangle integration (slower)")
    parser.add_argument(
        "--raster-resolution",
        type=int,
        default=DEFAULT_RUN_SETTINGS["rasterResolution"],
        help="side length of the cell-average material coverage raster",
    )
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=DEFAULT_RUN_SETTINGS["outsideWeight"],
        help=(
            "weight of the material-coverage distance penalty outside the target footprint "
            "inside the spill term (0 keeps occupancy spill but disables distance growth)"
        ),
    )
    parser.add_argument("--gpu-timing-interval", type=int, default=20,
        help="sample GPU stage intervals every N macro steps; 0 disables GPU timestamps")
    parser.add_argument("--deterministic-reference", action="store_true",
        help="slow native-GPU reference: order physics/chemical float reductions by sample index; not cross-device bit parity")
    parser.add_argument("--fitness-color-weight", type=float, default=1.0,
        help="weight of triangle RGB error against PNG targets; zero disables color")
    parser.add_argument(
        "--fitness-coverage-weight", type=float,
        default=DEFAULT_RUN_SETTINGS["fitnessCoverageWeight"],
        help="weight of multiscale missing-target coverage",
    )
    parser.add_argument(
        "--fitness-spill-weight", type=float,
        default=DEFAULT_RUN_SETTINGS["fitnessSpillWeight"],
        help="weight of occupancy and distance outside the target",
    )
    parser.add_argument(
        "--fitness-boundary-weight", type=float,
        default=DEFAULT_RUN_SETTINGS["fitnessBoundaryWeight"],
        help="weight of fine silhouette-edge disagreement",
    )
    parser.add_argument(
        "--fitness-crowding-weight", type=float,
        default=DEFAULT_RUN_SETTINGS["fitnessCrowdingWeight"],
        help="weight of overlapping material area",
    )
    parser.add_argument(
        "--fitness-temporal-worst-weight", type=float,
        default=DEFAULT_RUN_SETTINGS["fitnessTemporalWorstWeight"],
        help="blend between mean late-snapshot score (0) and worst late snapshot (1)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN_SETTINGS["runSeed"])
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_RUN_SETTINGS["checkpointEvery"])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "worker processes evaluating candidates concurrently, each with its own wgpu device (see "
            "parallel_workers.py) — defaults to min(os.cpu_count(), --population), since more workers than "
            "candidates in a generation can't ever all be busy at once"
        ),
    )
    return parser

def finalize_policy_configuration(args: argparse.Namespace) -> None:
    """Resolve memory and explicit comparison architecture selection."""
    if args.policy_architecture is not None:
        selected_memory = cell_memory_for_architecture(args.policy_architecture)
        if args.cell_memory is not None and args.cell_memory != selected_memory:
            raise SystemExit("--cell-memory conflicts with the comparison architecture value")
        args.cell_memory = selected_memory
    else:
        args.cell_memory = args.cell_memory or cell_memory_for_architecture(POLICY_ARCHITECTURE)
        args.policy_architecture = architecture_for_cell_memory(args.cell_memory)
    args.hidden_layers = [policy_hidden_dim(args.policy_architecture)]

def finalize_density_configuration(args: argparse.Namespace) -> None:
    try:
        args.particle_densities = list(parse_multipliers(
            args.particle_densities, allow_unsafe=args.allow_unsafe_density
        ))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.particle_capacity = max(resolve_run_density(args, q).particle_cap for q in args.particle_densities)
    if args.particle_capacity > MAX_PARTICLES:
        raise SystemExit(
            f"maximum resolved particle cap {args.particle_capacity} exceeds GPU capacity {MAX_PARTICLES}"
        )

def validate_fitness_configuration(args: argparse.Namespace) -> None:
    if not np.isfinite(args.mutation_sigma) or args.mutation_sigma < 0:
        raise SystemExit("--mutation-sigma must be finite and non-negative")
    factors = getattr(args, "mutation_factors", [1.])
    if not factors or any(not np.isfinite(f) or f <= 0 for f in factors):
        raise SystemExit("--mutation-factors must be finite and positive")
    if args.gpu_timing_interval < 0:
        raise SystemExit("--gpu-timing-interval must be non-negative")
    from initial_conditions import validate_initial_condition
    from policy_parameters import policy_has_recurrence
    try:
        validate_initial_condition(getattr(args, "initial_condition", "none"),
                                   getattr(args, "initial_condition_strength", .3),
                                   getattr(args, "initial_condition_channel", 0),
                                   CHEM_CHANNELS, policy_has_recurrence(args.policy_architecture))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for name in ("macro_steps", "shape_check_interval", "shape_confirmations", "shape_settle_steps"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("shape_missing_tolerance", "shape_spill_tolerance", "shape_overlap_tolerance"):
        if not 0 <= getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0,1)")
    if args.raster_resolution < 8:
        raise SystemExit("--raster-resolution must be at least 8")
    for name in (
        "outside_weight",
        "fitness_color_weight",
        "fitness_coverage_weight",
        "fitness_spill_weight",
        "fitness_boundary_weight",
        "fitness_crowding_weight",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not 0.0 <= args.fitness_temporal_worst_weight <= 1.0:
        raise SystemExit("--fitness-temporal-worst-weight must be between 0 and 1")

def checkpoint_metadata(args, target, generation, best_fitness, best_winner_seed, best_winner_density, best_density_fitnesses, best_evaluation_seeds):
    """Complete checkpoint settings shared by CLI training and the server."""
    return {
        'generation': generation,
        'fitness': best_fitness,
        'fitness_model_version': FITNESS_MODEL_VERSION,
        'target': args.target,
        'particles': args.particles,
        'initial_particle_count': args.initial_particles,
        'initial_spacing': INITIAL_SPACING_IN_SAMPLE_SPACINGS * DEFAULT_RUN_SETTINGS["sampleSpacing"],
        'sample_spacing': DEFAULT_RUN_SETTINGS["sampleSpacing"],
        'initial_condition': args.initial_condition,
        'initial_condition_strength': args.initial_condition_strength,
        'initial_condition_channel': args.initial_condition_channel,
        'density_model_version': DENSITY_MODEL_VERSION,
        'particle_density_multipliers': args.particle_densities,
        'density_aggregation': args.density_aggregation,
        'particle_capacity': args.particle_capacity,
        'particle_mass': PARTICLE_MASS,
        'particle_volume': VOL,
        'chemical_value_input_multiplier': CHEMICAL_VALUE_INPUT_MULTIPLIER,
        'chemical_gradient_input_scale': CHEMICAL_GRADIENT_INPUT_SCALE,
        'winner_seed': best_winner_seed,
        'winner_density_multiplier': best_winner_density,
        'density_fitnesses': best_density_fitnesses,
        'macro_steps': args.macro_steps,
        'growth_steps': args.growth_steps,
        'substeps_per_macro': args.substeps_per_macro,
        'growth_model_version': GROWTH_MODEL_VERSION,
        'domain_geometry': 'triangle-vertices',
        'shape_settings': shape_settings(args, target),
        'growth_duration_macro_steps': GROWTH_DURATION_MACRO_STEPS,
        'growth_compression_start': GROWTH_COMPRESSION_START,
        'growth_compression_stop': GROWTH_COMPRESSION_STOP,
        'growth_compression_feedback': GROWTH_COMPRESSION_FEEDBACK,
        'growth_anisotropy_authority': GROWTH_ANISOTROPY_AUTHORITY,
        'morphology_blur_sigma': MORPHOLOGY_BLUR_SIGMA,
        'morphology_density_reference': MORPHOLOGY_DENSITY_REFERENCE,
        'neural_updates_per_macro': NEURAL_UPDATES_PER_MACRO,
        'communication_speed': COMMUNICATION_SPEED,
        'internal_state_speed': INTERNAL_STATE_SPEED,
        'elastic_strain_scale': ELASTIC_STRAIN_SCALE,
        'elastic_strain_inputs_enabled': ELASTIC_STRAIN_INPUTS_ENABLED,
        'gravity': args.gravity,
        'spawn_x': args.spawn_x,
        'spawn_y': args.spawn_y,
        'channels': CHEM_CHANNELS,
        'field_n': FIELD_N,
        'chemical_channel_profiles': profiles_to_wire(CHEMICAL_CHANNEL_PROFILES),
        'population': args.population,
        'seeds_per_candidate': args.seeds_per_candidate,
        'evaluation_seeds': best_evaluation_seeds,
        'elites': args.elites,
        'mutation_sigma': args.mutation_sigma,
        'mutation_factors': list(getattr(args, 'mutation_factors', [1.])),
        'initial_weights': str(args.initial_weights) if getattr(args, 'initial_weights', None) else None,
        'fitness_alignment': getattr(args, 'fitness_alignment', 'raster'),
        'deterministic_reference': getattr(args, 'deterministic_reference', False),
        'policy_architecture': args.policy_architecture,
        'cell_memory': args.cell_memory,
        'hidden_layers': args.hidden_layers,
        'chemical_communication_architecture': args.chemical_communication_architecture,
        'decay': DECAY,
        'friction': DEFAULT_RUN_SETTINGS["friction"],
        'max_env_write': DEFAULT_RUN_SETTINGS["maxEnvWrite"],
        'deposit_rate': DEPOSIT_RATE,
        'normalize_deposits_by_local_density': NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
        'hidden_dim': policy_hidden_dim(args.policy_architecture),
        'mutation_scales': mutation_scales(args.policy_architecture),
        'raster_resolution': args.raster_resolution,
        'outside_weight': args.outside_weight,
        'fitness_color_weight': args.fitness_color_weight,
        'fitness_coverage_weight': args.fitness_coverage_weight,
        'fitness_spill_weight': args.fitness_spill_weight,
        'fitness_boundary_weight': args.fitness_boundary_weight,
        'fitness_crowding_weight': args.fitness_crowding_weight,
        'fitness_temporal_worst_weight': args.fitness_temporal_worst_weight,
        'seed': args.seed,
        'damping_loss_fraction': DAMPING_LOSS_FRACTION,
        'material_e': MATERIAL_E,
        'material_nu': MATERIAL_NU,
        'material_hardening': MATERIAL_HARDENING,
        'material_elasticity': MATERIAL_ELASTICITY,
        'splat_radius': SPLAT_RADIUS,
        'repulsion_strength': REPULSION_STRENGTH,
        'repulsion_max_delta': REPULSION_MAX_DELTA,
    }

def main() -> None:
    args = build_arg_parser().parse_args()
    finalize_policy_configuration(args)

    if not 1 <= args.elites <= args.population:
        raise SystemExit("--elites must be between 1 and --population")
    if args.seeds_per_candidate < 1:
        raise SystemExit("--seeds-per-candidate must be at least 1")
    if not 1 <= args.initial_particles <= args.particles // 2:
        raise SystemExit("--initial-particles seed cells must be between 1 and floor(--particles/2)")
    validate_fitness_configuration(args)
    finalize_density_configuration(args)
    if args.growth_steps is not None and not 0 <= args.growth_steps <= args.macro_steps:
        raise SystemExit("--growth-steps must be between 0 and --macro-steps")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    target = load_target(args.target)
    report_shape_capacity(args, target)
    # Physical target footprint, precomputed once for every worker.
    target_raster = target_mask(target, args.raster_resolution)
    target_distance_field = build_target_distance_field(target_raster)

    # A persistent pool of worker processes (see parallel_workers.py's
    # own module docstring for why this replaced a single reused
    # MpmCore/AgentsGPU/EnvironmentGPU — each worker builds and keeps its
    # own triple, baked in once at pool creation, same "wgpu pipeline
    # compilation is real, avoidable overhead" reasoning a single process
    # already applied, just paid --workers times at startup instead of
    # once). `update_rule` is a CPU-only scratch nn.Module used ONLY for
    # random weight initialization (below) and checkpoint JSON export (in
    # the checkpoint block below) — it never runs a live forward pass,
    # see training_sim.py's own module docstring for why.
    num_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, args.population)
    population = initial_population(args, rng)
    pool = build_pool(num_workers, args.particle_capacity, target, target_raster, target_distance_field, args)
    update_rule = UpdateRule(CHEM_CHANNELS, args.policy_architecture)

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_fitness = float("inf")
    best_weights = population[0]
    best_evaluation_seeds: list[int] = []
    best_winner_seed = args.seed
    best_winner_density = args.particle_densities[0]
    best_density_fitnesses: dict[str, float] = {}

    for generation in range(args.generations):
        population, fitnesses, winner_seed, winner_density, evaluation_seeds, density_fitnesses = run_generation(
            population, args, rng, pool
        )

        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_weights = population[0].copy()
            best_evaluation_seeds = list(evaluation_seeds)
            best_winner_seed = winner_seed
            best_winner_density = winner_density
            best_density_fitnesses = dict(density_fitnesses)

        finite = [f for f in fitnesses if np.isfinite(f)]
        print(
            f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(finite) if finite else float('inf'):.4f}  "
            f"worst {fitnesses[-1]:.4f}  (all-time best {best_fitness:.4f})"
        )

        if (generation + 1) % args.checkpoint_every == 0 or generation == args.generations - 1:
            set_weights(update_rule, best_weights)
            np.save(checkpoint_dir / "best.npy", best_weights)
            (checkpoint_dir / "best_weights.json").write_text(json.dumps(update_rule.export_weights()))
            (checkpoint_dir / "best_meta.json").write_text(
                json.dumps(
                    checkpoint_metadata(args, target, generation, best_fitness, best_winner_seed, best_winner_density, best_density_fitnesses, best_evaluation_seeds),
                    indent=2,
                )
            )

    pool.shutdown()
    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {checkpoint_dir / 'best.npy'}")

if __name__ == "__main__":
    main()

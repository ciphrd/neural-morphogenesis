"""Population-based (mu, lambda) evolutionary training for UpdateRule,
with elitism — the mpm-training analogue of envnca/evolve.py and
trainer/backend/evolve.py, adapted to real MLS-MPM particle physics via
training_sim.TrainingRollout instead of either project's own simpler
rollout substrate.

No gradient descent anywhere (MpmCore's physics isn't wired for
autograd) — a plain (mu, lambda) ES, same as this repo's other
evolve.py's minus envnca's own optional memetic refinement (not ported
here). population size here means the *evolutionary* population
(candidate weight-sets), same meaning it has in every other evolve.py in
this repo — not the particle count: every rollout starts with
--initial-particles agents and grows via splitting (see training_sim.py's
own module docstring and core/agents.wgsl's own growth design);
--particles is the CAP that growth can reach, not a fixed per-rollout
count.

Fitness is raster.py's bounded, multiscale occupancy comparison. Weighted
Gaussian particle density is saturated into [0,1] occupancy, then scored for
missing coverage, outside spill, fine silhouette disagreement, and excessive
crowding after centroid matching and a coarse-to-fine rotation search. Several
snapshots near the rollout's end are blended as mean plus worst-case pressure;
see _score_fitness() and CAPTURE_OFFSETS. The earlier aligned symmetric Chamfer
metric remains available in alignment.py for one-off point-cloud diagnostics,
but it no longer selects candidates.

Rollouts run across a persistent pool of worker PROCESSES (see
parallel_workers.py's own module docstring), each with its own wgpu
device and a single MpmCore/AgentsGPU/EnvironmentGPU triple, one
candidate rollout per task, `--workers` of them running truly
concurrently. This replaced an earlier single-process sequential loop
after profiling (cProfile against a real generation) showed that loop
was CPU-bound on a single core — wgpu-py's own per-compute-pass FFI call
overhead plus its GPU-sync poll loop, together over 75% of wall time —
while a 400-particle rollout barely touches the GPU's actual throughput
and every OTHER CPU core sat idle. Multiple OS processes, each doing the
exact same cheap per-candidate work concurrently on separate cores,
sharing the same GPU (which has plenty of spare capacity for this), maps
directly onto that profile in a way an in-process design can't: a single
Python process is fundamentally bounded by one core's worth of FFI/poll
overhead no matter how the GPU work within it is organized.

A batched, single-process alternative was tried and measured FIRST
(every candidate getting its own MpmCore/AgentsGPU/EnvironmentGPU, all
advanced together within one process so per-macro-step GPU syncs are
paid once for the whole population instead of once per candidate), on
the theory that sync *count* was the dominant cost. It wasn't: at
population=8, 16, and 32 (particles=150, macro_steps=16) it measured no
reliable speedup — 0.8x-1.2x, noisy, one config outright slower — and
crashed the device once (a real, confirmed instance of the wgpu-native
command-buffer-count bug mpm_core.MpmCore's own docstring already
describes, from an under-counted pass budget spanning several MpmCore
instances). The profiling that followed explained why: the dominant
costs (FFI call overhead, GPU-sync poll wait) are both proportional to
total work done, not to how many syncs that work is grouped into —
batching within one process changes the grouping, not the total, so it
couldn't have helped. Multiprocessing instead adds a second (and third,
...) core actually doing that work concurrently — the only lever that
was ever going to move the needle for a single-core-CPU-bound loop.

Fitness scoring is pure NumPy/SciPy and separate from the wgpu simulation. Its
local Gaussian scatter is vectorized, and worker processes evaluate candidates
concurrently; raster.py documents the hot path and its complexity.

Usage:
    python evolve.py --target puddle --generations 50 --population 16
"""
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
from density import DENSITY_MODEL_VERSION, DensityReference, ResolvedDensity, parse_multipliers, resolve_density
from environment_gpu import EnvironmentGPU
from mpm_core import MAX_PARTICLES, PARTICLE_MASS, VOL, MpmCore
from parallel_workers import build_pool, worker_rollout
from raster import FITNESS_MODEL_VERSION, build_target_distance_field, build_target_raster, training_raster_distance
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
    DEPOSIT_DENSITY_REFERENCE,
    DEPOSIT_SIGMA,
    DEFAULT_RUN_SETTINGS,
    DEFAULT_SUBSTEPS_PER_MACRO,
    ELASTIC_STRAIN_SCALE,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    FIELD_N,
    GROWTH_DURATION_MACRO_STEPS,
    GROWTH_MODEL_VERSION,
    MATERIAL_AREA_BUDGET,
    GROWTH_COMPRESSION_FEEDBACK,
    GROWTH_COMPRESSION_START,
    GROWTH_COMPRESSION_STOP,
    GROWTH_ANISOTROPY_AUTHORITY,
    INITIAL_PARTICLE_COUNT,
    INTERNAL_STATE_SPEED,
    DIVISION_DRIVE_BOOST,
    DIVISION_DIRECTIONALITY,
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
        deposit_sigma=DEPOSIT_SIGMA,
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


def _score_fitness(
    snapshots: list[np.ndarray],
    target: TargetShape,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    args: argparse.Namespace,
    density_multiplier: float,
) -> float:
    """Bounded multiscale raster fitness over the rollout's late snapshots.

    The mean preserves distinctions across the whole late window while a
    configurable worst-snapshot contribution still penalizes transient poses.
    A diverged or empty snapshot fails softly with positive infinity.
    """
    scores: list[float] = []
    for positions in snapshots:
        if positions.shape[0] == 0 or not np.isfinite(positions).all():
            return float("inf")
        scores.append(float(training_raster_distance(
            positions,
            target.points,
            target_raster,
            target_distance_field,
            args.raster_resolution,
            RASTER_EXTENT,
            args.raster_sigma,
            outside_weight=args.outside_weight,
            particle_weight=1.0 / density_multiplier,
            expected_weighted_particles=float(args.particles),
            target_occupancy=args.fitness_target_occupancy,
            coverage_weight=args.fitness_coverage_weight,
            spill_weight=args.fitness_spill_weight,
            boundary_weight=args.fitness_boundary_weight,
            crowding_weight=args.fitness_crowding_weight,
        )))
    worst_weight = args.fitness_temporal_worst_weight
    return (1.0 - worst_weight) * float(np.mean(scores)) + worst_weight * max(scores)


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
) -> float | tuple[float, np.ndarray]:
    """Runs one seed-to-`args.macro_steps` rollout with `weights` loaded
    into the (reused, GPU-resident) `agents` and `core`, and returns a
    fitness score — lower is better. Rather than reading positions once
    at the very end, a snapshot is captured at each of CAPTURE_OFFSETS
    (the last 10% of `args.macro_steps`, five evenly-spaced points), each
    snapshot is scored by raster.training_raster_distance's bounded multiscale
    occupancy metric. The arithmetic mean across captures is blended with their
    worst score, retaining granular temporal information while punishing an
    unstable one-instant pose. `target_raster` and `target_distance_field` are
    the run-level precomputations used directly by that metric.
    See CAPTURE_OFFSETS's own comment for the full "don't let a
    candidate learn to pose for one known instant" reasoning. A diverged
    (non-finite, or fully emptied-out) snapshot scores +inf rather than
    crashing the generation, same "fail soft" backstop every other
    evolve.py in this repo has.

    `return_positions`, off by default, additionally returns the
    rollout's final (last-captured) positions — for callers that need
    them for something other than scoring (e.g. train_server.py's
    end-of-generation debug images), not the hot per-candidate training
    path, which only ever wants the scalar fitness."""
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
    agents.set_density_geometry(density.spacing, density.deposit_sigma)
    agents.set_chemical_gradient_input_scale(density.chemical_gradient_input_scale)
    agents.set_chemical_projection_weight(density.chemical_projection_weight)
    agents.set_max_active_particles(density.particle_cap)

    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(args.spawn_x, args.spawn_y),
        spawn_half_width=args.spawn_half_width,
        gravity=args.gravity,
        seed=seed,
        mpm_enabled=MPM_ENABLED,
        initial_particle_count=density.initial_particles,
    )

    # Deduplicated and sorted ascending — offsets can collide onto the
    # same integer macro step for a short enough rollout, and scoring the
    # same step twice would just waste work, not change the result (max
    # of a value with itself is itself). offset=0.0 always maps to
    # args.macro_steps itself, guaranteeing this set is never empty and
    # its last (chronological) member is always the final step.
    checkpoint_steps = {max(1, round(args.macro_steps * (1.0 - offset))) for offset in CAPTURE_OFFSETS}
    snapshots: list[np.ndarray] = []

    for step in range(1, args.macro_steps + 1):
        sim.macro_step(
            args.substeps_per_macro,
            growth_enabled=args.growth_steps is None or step <= args.growth_steps,
        )
        if step in checkpoint_steps:
            snapshots.append(sim.positions())

    final_positions = snapshots[-1]
    fitness = _score_fitness(
        snapshots, target, target_raster, target_distance_field, args, density_multiplier
    )

    return (fitness, final_positions) if return_positions else fitness


def run_generation(
    population: list[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
    pool: ProcessPoolExecutor,
) -> tuple[list[np.ndarray], list[float], int, float, list[int], dict[str, float]]:
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
    (next_population, fitnesses, winner_seed, evaluation_seeds) — `fitnesses` are for the
    population just evaluated, sorted ascending (lower raster distance is
    better — see raster.py); `next_population[0]` is this generation's
    winning weights, carried over unmutated; `winner_seed` is the winning
    candidate's worst-scoring seed from the shared batch — needed by callers
    (train_server.py)
    that want to reproduce this generation's *actual* winning rollout,
    not just its weights, for a debug render (via rollout(), a single,
    non-pooled replay — see that function's own docstring).

    `target` is NOT passed here — it's baked into each worker's own
    globals once, at pool creation (parallel_workers.build_pool()'s own
    initializer), since it never changes generation to generation and
    re-sending it with every task would be pure waste."""
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
    rollout_fitnesses = np.asarray(
        list(pool.map(worker_rollout, task_weights, task_seeds, task_densities)), dtype=np.float64
    ).reshape(len(population), len(densities), seeds_per_candidate)
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
    population = [population[i] for i in order]
    fitnesses = [float(fitnesses[i]) for i in order]
    representative_cases = [representative_cases[i] for i in order]

    elites = population[: args.elites]
    next_population = list(elites)
    while len(next_population) < args.population:
        parent = elites[rng.integers(len(elites))]
        next_population.append(mutate(parent, args.mutation_sigma, rng, args.policy_architecture))

    winner_seed, winner_density = representative_cases[0]
    return (
        next_population, fitnesses, winner_seed, winner_density,
        evaluation_seeds, winner_density_fitnesses,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults live in core/constants.json; both architecture axes are public
    # run selections. The hidden alias remains exclusively for the paired
    # comparison utility's existing subprocess interface.
    parser.set_defaults(policy_architecture=None, cell_memory=None)
    parser.add_argument(
        "--cell-memory",
        choices=CELL_MEMORY_OPTIONS,
        help="private neural memory: none uses a reactive policy; recurrent adds gated per-cell state",
    )
    parser.add_argument(
        "--policy-architecture",
        dest="policy_architecture",
        choices=POLICY_ARCHITECTURES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_comparison-policy-architecture",
        dest="policy_architecture",
        choices=POLICY_ARCHITECTURES,
        help=argparse.SUPPRESS,
    )
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
        "--target", default=DEFAULT_RUN_SETTINGS["target"], choices=available_targets(), help="target shape name (a .json file in ./targets/)"
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
        help="allow multipliers outside core/density.json's calibrated range",
    )
    parser.add_argument(
        "--initial-particles",
        type=int,
        default=INITIAL_PARTICLE_COUNT,
        help="number of agents seeded at the beginning of each rollout (must not exceed --particles)",
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
    parser.add_argument("--spawn-half-width", type=float, default=DEFAULT_RUN_SETTINGS["spawnHalfWidth"])
    parser.add_argument("--mutation-sigma", type=float, default=DEFAULT_RUN_SETTINGS["mutationSigma"])
    parser.add_argument(
        "--raster-resolution",
        type=int,
        default=DEFAULT_RUN_SETTINGS["rasterResolution"],
        help="side length of the square lattice target/particle point clouds are splatted onto for fitness — see raster.py",
    )
    parser.add_argument(
        "--raster-sigma",
        type=float,
        default=DEFAULT_RUN_SETTINGS["rasterSigma"],
        help="Gaussian splat width, in raster pixels (not domain units) — see raster.rasterize_points",
    )
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=DEFAULT_RUN_SETTINGS["outsideWeight"],
        help=(
            "weight of the distance-transform penalty for particles landing outside the target's footprint "
            "inside the spill term (0 keeps occupancy spill but disables distance growth)"
        ),
    )
    parser.add_argument(
        "--fitness-target-occupancy", type=float,
        default=DEFAULT_RUN_SETTINGS["fitnessTargetOccupancy"],
        help="desired bounded occupancy in a uniformly filled target interior",
    )
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
        help="weight of excessive raw particle-density spikes",
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
    """Resolve the public memory switch while retaining legacy CLI support."""
    if args.policy_architecture is not None:
        legacy_memory = cell_memory_for_architecture(args.policy_architecture)
        if args.cell_memory is not None and args.cell_memory != legacy_memory:
            raise SystemExit("--cell-memory conflicts with the legacy --policy-architecture value")
        args.cell_memory = legacy_memory
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
    if args.raster_resolution < 8:
        raise SystemExit("--raster-resolution must be at least 8")
    if not np.isfinite(args.raster_sigma) or args.raster_sigma <= 0.0:
        raise SystemExit("--raster-sigma must be finite and positive")
    if not 0.0 < args.fitness_target_occupancy < 1.0:
        raise SystemExit("--fitness-target-occupancy must be strictly between 0 and 1")
    for name in (
        "outside_weight",
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


def main() -> None:
    args = build_arg_parser().parse_args()
    finalize_policy_configuration(args)

    if not 1 <= args.elites <= args.population:
        raise SystemExit("--elites must be between 1 and --population")
    if args.seeds_per_candidate < 1:
        raise SystemExit("--seeds-per-candidate must be at least 1")
    if not 1 <= args.initial_particles <= args.particles:
        raise SystemExit("--initial-particles must be between 1 and --particles")
    validate_fitness_configuration(args)
    finalize_density_configuration(args)
    if args.growth_steps is not None and not 0 <= args.growth_steps <= args.macro_steps:
        raise SystemExit("--growth-steps must be between 0 and --macro-steps")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    target = load_target(args.target)
    # Fixed for the whole run — precomputed once rather than on every
    # rollout's training_raster_distance() call (population x generations
    # x rotation-search angles x snapshots otherwise recomputing the
    # exact same thing). See raster.build_target_raster()'s own
    # docstring.
    target_raster = build_target_raster(
        target.points, args.raster_resolution, RASTER_EXTENT, args.raster_sigma, half_size=target.texel_size() / 2.0
    )
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
    pool = build_pool(num_workers, args.particle_capacity, target, target_raster, target_distance_field, args)
    update_rule = UpdateRule(CHEM_CHANNELS, args.policy_architecture)

    population = [get_weights(UpdateRule(CHEM_CHANNELS, args.policy_architecture)) for _ in range(args.population)]

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
                    {
                        "generation": generation,
                        "fitness": best_fitness,
                        "fitness_model_version": FITNESS_MODEL_VERSION,
                        "target": args.target,
                        "particles": args.particles,
                        "initial_particle_count": args.initial_particles,
                        "density_model_version": DENSITY_MODEL_VERSION,
                        "particle_density_multipliers": args.particle_densities,
                        "density_aggregation": args.density_aggregation,
                        "particle_capacity": args.particle_capacity,
                        "particle_mass": PARTICLE_MASS,
                        "particle_volume": VOL,
                        "deposit_sigma": DEPOSIT_SIGMA,
                        "chemical_projection_weight": 1.0,
                        "chemical_value_input_multiplier": CHEMICAL_VALUE_INPUT_MULTIPLIER,
                        "chemical_gradient_input_scale": CHEMICAL_GRADIENT_INPUT_SCALE,
                        "winner_seed": best_winner_seed,
                        "winner_density_multiplier": best_winner_density,
                        "density_fitnesses": best_density_fitnesses,
                        "macro_steps": args.macro_steps,
                        "growth_steps": args.growth_steps,
                        "substeps_per_macro": args.substeps_per_macro,
                        "growth_model_version": GROWTH_MODEL_VERSION,
                        "material_area_budget": MATERIAL_AREA_BUDGET,
                        "growth_duration_macro_steps": GROWTH_DURATION_MACRO_STEPS,
                        "growth_compression_start": GROWTH_COMPRESSION_START,
                        "growth_compression_stop": GROWTH_COMPRESSION_STOP,
                        "growth_compression_feedback": GROWTH_COMPRESSION_FEEDBACK,
                        "growth_anisotropy_authority": GROWTH_ANISOTROPY_AUTHORITY,
                        "morphology_blur_sigma": MORPHOLOGY_BLUR_SIGMA,
                        "morphology_density_reference": MORPHOLOGY_DENSITY_REFERENCE,
                        "neural_updates_per_macro": NEURAL_UPDATES_PER_MACRO,
                        "communication_speed": COMMUNICATION_SPEED,
                        "internal_state_speed": INTERNAL_STATE_SPEED,
                        "division_directionality": DIVISION_DIRECTIONALITY,
                        "division_drive_boost": DIVISION_DRIVE_BOOST,
                        "elastic_strain_scale": ELASTIC_STRAIN_SCALE,
                        "elastic_strain_inputs_enabled": ELASTIC_STRAIN_INPUTS_ENABLED,
                        "gravity": args.gravity,
                        "spawn_x": args.spawn_x,
                        "spawn_y": args.spawn_y,
                        "spawn_half_width": args.spawn_half_width,
                        "channels": CHEM_CHANNELS,
                        "field_n": FIELD_N,
                        "chemical_channel_profiles": profiles_to_wire(CHEMICAL_CHANNEL_PROFILES),
                        "population": args.population,
                        "seeds_per_candidate": args.seeds_per_candidate,
                        # These belong to best_weights, which may come from an
                        # earlier generation than the checkpoint write.
                        "evaluation_seeds": best_evaluation_seeds,
                        "elites": args.elites,
                        "mutation_sigma": args.mutation_sigma,
                        "policy_architecture": args.policy_architecture,
                        "cell_memory": args.cell_memory,
                        "hidden_layers": args.hidden_layers,
                        "chemical_communication_architecture": args.chemical_communication_architecture,
                        "decay": DECAY,
                        "deposit_rate": DEPOSIT_RATE,
                        "normalize_deposits_by_local_density": NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
                        "deposit_density_reference": DEPOSIT_DENSITY_REFERENCE,
                        "hidden_dim": policy_hidden_dim(args.policy_architecture),
                        "mutation_scales": mutation_scales(args.policy_architecture),
                        "raster_resolution": args.raster_resolution,
                        "raster_sigma": args.raster_sigma,
                        "outside_weight": args.outside_weight,
                        "fitness_target_occupancy": args.fitness_target_occupancy,
                        "fitness_coverage_weight": args.fitness_coverage_weight,
                        "fitness_spill_weight": args.fitness_spill_weight,
                        "fitness_boundary_weight": args.fitness_boundary_weight,
                        "fitness_crowding_weight": args.fitness_crowding_weight,
                        "fitness_temporal_worst_weight": args.fitness_temporal_worst_weight,
                        "seed": args.seed,
                        # Not CLI args (nothing above this line is) — the
                        # simulation_settings.py values this run actually
                        # simulated under. Recorded here too so this
                        # checkpoint's own metadata is a complete,
                        # standalone description of the run, without
                        # requiring a cross-reference to whatever
                        # simulation_settings.py happened to say at some
                        # other point in time.
                        "damping_loss_fraction": DAMPING_LOSS_FRACTION,
                        "material_e": MATERIAL_E,
                        "material_nu": MATERIAL_NU,
                        "material_hardening": MATERIAL_HARDENING,
                        "material_elasticity": MATERIAL_ELASTICITY,
                        "splat_radius": SPLAT_RADIUS,
                        "repulsion_strength": REPULSION_STRENGTH,
                        "repulsion_max_delta": REPULSION_MAX_DELTA,
                    },
                    indent=2,
                )
            )

    pool.shutdown()
    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {checkpoint_dir / 'best.npy'}")


if __name__ == "__main__":
    main()

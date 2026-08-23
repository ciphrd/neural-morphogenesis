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
this repo — not the particle count: every rollout currently starts
with a coordinated two-particle seed and grows via splitting (see training_sim.py's
own module docstring and core/agents.wgsl's own growth design);
--particles is the CAP that growth can reach, not a fixed per-rollout
count.

Fitness is alignment.py's own training_alignment_distance — distance.py's
own KD-tree symmetric nearest-neighbor Chamfer distance, wrapped with a
coarse rotation-search + centroid-matching-translation best-fit (lowest
distance after that transform, since nothing pins particle growth to the
target's exact pose) — see _score_fitness()'s own docstring, scored at
several snapshots near the end of the rollout and scored by the *worst*
of them, not just the final step — see CAPTURE_OFFSETS. This project has
gone back and forth on this more than once: raster.py's own Gaussian-
splat rasterization + distance-transform comparison (raster.
training_raster_distance) was the live fitness function for a while
instead — it's still fully wired, just no longer what candidates are
scored/selected by: train_server.py's own end-of-generation debug images
(the "Target"/"Agents (aligned)"/"Grown (raw)" snapshots) render via
that same raster distance regardless of which strategy fitness itself
uses, and target_raster/target_distance_field (raster.
build_target_raster()/build_target_distance_field(), precomputed once
per run) still get built and threaded through rollout()/worker_rollout()
for exactly that reason — see rollout()'s own docstring for why removing
them from that plumbing isn't worth it. If raster-based fitness is worth
revisiting again, _score_fitness() is the one place to swap back.

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

Fitness scoring (raster.training_raster_distance's rotation search, back
when raster distance was the live fitness function — see this module's
own "Fitness is..." paragraph above for the current Chamfer/raster
back-and-forth) was the other ~27% of profiled time, entirely separate
from any of the above — pure NumPy, no wgpu/GPU involvement at all. See
raster.py's own module docstring for the GPU port of the hot inner loop
(rasterize_points_sum's scatter + the MSE/distance-transform reduction)
that addressed it; alignment.py's own Chamfer distance (a KD-tree
nearest-neighbor query, not this GPU-ported raster path) hasn't been
profiled/optimized under this same worker-pool setup.

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
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from agents_gpu import AgentsGPU
from alignment import training_alignment_distance
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore
from parallel_workers import build_pool, worker_rollout
from raster import build_target_distance_field, build_target_raster
from simulation_settings import (
    CHEM_CHANNELS,
    COMMUNICATION_SPEED,
    DAMPING_LOSS_FRACTION,
    DEFAULT_SUBSTEPS_PER_MACRO,
    ELASTIC_STRAIN_SCALE,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    FIELD_N,
    GROWTH_DURATION_MACRO_STEPS,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MPM_ENABLED,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    NEURAL_UPDATES_PER_MACRO,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
)
from targets import TargetShape, available_targets, load_target
from training_sim import TrainingRollout
from update_rule import UpdateRule

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

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
# scoring by the *worst* (highest, since lower is better) of them means a
# candidate that reaches a good shape and then immediately destabilizes
# scores exactly as badly as one that never reached it at all. Unlike
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
    weight-then-bias parameter order + Sequential's own fc1-then-fc2
    order + parameters_to_vector's own row-major flatten already matches
    core/agents.wgsl's FC1W_OFFSET/etc. indexing, no restructuring
    needed)."""
    return parameters_to_vector(model.parameters()).detach().cpu().numpy()


def set_weights(model: UpdateRule, flat: np.ndarray) -> None:
    """CPU-only — `model` (a scratch UpdateRule) is never used for a live
    forward pass anymore (see training_sim.py's own module docstring),
    only for export_weights()'s JSON shape at checkpoint time, so there's
    no reason to pay a device transfer here."""
    vector_to_parameters(torch.from_numpy(flat).float(), model.parameters())


def mutate(weights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return weights + rng.normal(scale=sigma, size=weights.shape)


def _score_fitness(snapshots: list[np.ndarray], target: TargetShape) -> float:
    """Called by rollout() (via the worker pool's own worker_rollout()) —
    the *worst* (highest, since lower is better) Chamfer distance across
    `snapshots` (one rollout's own CAPTURE_OFFSETS captures), via
    alignment.training_alignment_distance's rotation-search + centroid-
    matching-translation Chamfer distance (distance.py's own KD-tree
    symmetric nearest-neighbor distance) — see this module's own module
    docstring for the raster/Chamfer back-and-forth; this is Chamfer
    again, not raster.py's own Gaussian-splat raster distance (that stays
    wired for train_server.py's own debug-image rendering — see
    rollout()'s own docstring for why it still accepts target_raster/
    target_distance_field despite not using them for scoring anymore). A
    diverged (non-finite, or fully emptied-out) snapshot scores +inf
    rather than crashing the generation, same "fail soft" backstop every
    other evolve.py in this repo has. See CAPTURE_OFFSETS's own comment
    for why the worst-of-several-snapshots scoring exists at all."""
    fitness = 0.0
    for positions in snapshots:
        if positions.shape[0] == 0 or not np.isfinite(positions).all():
            return float("inf")
        distance = training_alignment_distance(positions, target.points)
        fitness = max(fitness, distance)
    return fitness


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
) -> float | tuple[float, np.ndarray]:
    """Runs one seed-to-`args.macro_steps` rollout with `weights` loaded
    into the (reused, GPU-resident) `agents` and `core`, and returns a
    fitness score — lower is better. Rather than reading positions once
    at the very end, a snapshot is captured at each of CAPTURE_OFFSETS
    (the last 10% of `args.macro_steps`, five evenly-spaced points), each
    snapshot is scored via alignment.training_alignment_distance (a
    rotation-search, centroid-matching-translation Chamfer distance —
    see _score_fitness()'s own docstring), and the *worst* (highest) of
    those scores is what's actually returned.

    `target_raster`/`target_distance_field` are accepted but NOT used for
    scoring anymore (see _score_fitness()'s own docstring for the raster/
    Chamfer back-and-forth) — kept in this signature anyway since every
    caller (parallel_workers.py's own worker_rollout(), train_server.py's
    own debug-image replay) already threads them through positionally,
    and train_server.py's own callers still need them regardless, for
    their own SEPARATE raster.training_raster_distance(...,
    track_best_raster=True) call on this function's returned positions
    (a debug-image render, not fitness) — removing them here would just
    relocate the same values one call frame up for no benefit, and would
    make toggling back to raster-based fitness later (see this module's
    own module docstring) a bigger diff than it needs to be.
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

    core.set_material(
        MATERIAL_E,
        MATERIAL_NU,
        MATERIAL_HARDENING,
        MATERIAL_ELASTICITY,
        growth_duration_macro_steps=GROWTH_DURATION_MACRO_STEPS,
        substeps_per_macro=args.substeps_per_macro,
    )
    core.set_damping(DAMPING_LOSS_FRACTION, args.substeps_per_macro)
    core.set_splat_radius(SPLAT_RADIUS)
    core.set_repulsion_strength(REPULSION_STRENGTH, REPULSION_MAX_DELTA)

    sim = TrainingRollout(
        core,
        agents,
        environment,
        spawn_center=(args.spawn_x, args.spawn_y),
        spawn_half_width=args.spawn_half_width,
        gravity=args.gravity,
        seed=seed,
        mpm_enabled=MPM_ENABLED,
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
    fitness = _score_fitness(snapshots, target)

    return (fitness, final_positions) if return_positions else fitness


def run_generation(
    population: list[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
    pool: ProcessPoolExecutor,
) -> tuple[list[np.ndarray], list[float], int]:
    """Evaluates every candidate — one rollout each, fanned out across
    `pool`'s worker processes (see parallel_workers.py's own module
    docstring for why: a single process evaluating candidates
    sequentially, or even several MpmCore instances batched within one
    process, both measured as CPU-bound on one core — see this module's
    own module docstring) — then sorts best-first and refills back up to
    `args.population` via elitism + Gaussian mutation of a randomly-
    chosen elite — plain (mu, lambda) ES, no memetic refinement. Returns
    (next_population, fitnesses, winner_seed) — `fitnesses` are for the
    population just evaluated, sorted ascending (lower raster distance is
    better — see raster.py); `next_population[0]` is this generation's
    winning weights, carried over unmutated; `winner_seed` is the seed
    that produced `fitnesses[0]` — needed by callers (train_server.py)
    that want to reproduce this generation's *actual* winning rollout,
    not just its weights, for a debug render (via rollout(), a single,
    non-pooled replay — see that function's own docstring).

    `target` is NOT passed here — it's baked into each worker's own
    globals once, at pool creation (parallel_workers.build_pool()'s own
    initializer), since it never changes generation to generation and
    re-sending it with every task would be pure waste."""
    seeds = rng.integers(0, 2**31 - 1, size=len(population))
    fitnesses = list(pool.map(worker_rollout, population, (int(s) for s in seeds)))

    order = np.argsort(fitnesses)
    population = [population[i] for i in order]
    fitnesses = [fitnesses[i] for i in order]
    seeds = [int(seeds[i]) for i in order]

    elites = population[: args.elites]
    next_population = list(elites)
    while len(next_population) < args.population:
        parent = elites[rng.integers(len(elites))]
        next_population.append(mutate(parent, args.mutation_sigma, rng))

    return next_population, fitnesses, seeds[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", default="circle", choices=available_targets(), help="target shape name (a .json file in ./targets/)"
    )
    parser.add_argument("--population", type=int, default=16, help="number of candidate weight-sets per generation")
    parser.add_argument("--elites", type=int, default=3, help="top performers carried into the next generation unmutated")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument(
        "--particles",
        type=int,
        default=400,
        help=(
            "maximum particle count a rollout can grow to via splitting — every rollout currently starts with "
            "a coordinated two-particle seed (see training_sim.py)"
        ),
    )
    parser.add_argument(
        "--macro-steps",
        type=int,
        default=160,
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
    parser.add_argument("--gravity", type=float, default=200.0)
    # Center of MpmCore's own [0,1]^2 domain — particles start in the
    # middle of the space, not offset toward any one wall.
    parser.add_argument("--spawn-x", type=float, default=0.5)
    parser.add_argument("--spawn-y", type=float, default=0.5)
    parser.add_argument("--spawn-half-width", type=float, default=0.08)
    parser.add_argument("--mutation-sigma", type=float, default=0.05)
    parser.add_argument(
        "--raster-resolution",
        type=int,
        default=128,
        help="side length of the square lattice target/particle point clouds are splatted onto for fitness — see raster.py",
    )
    parser.add_argument(
        "--raster-sigma",
        type=float,
        default=1.5,
        help="Gaussian splat width, in raster pixels (not domain units) — see raster.rasterize_points",
    )
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=1.0,
        help=(
            "weight of the distance-transform penalty for particles landing outside the target's footprint "
            "(0 disables it, falling back to raster coverage MSE alone) — see raster.outside_shape_penalty"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
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


def main() -> None:
    args = build_arg_parser().parse_args()

    if not 1 <= args.elites <= args.population:
        raise SystemExit("--elites must be between 1 and --population")
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
    pool = build_pool(num_workers, args.particles, target, target_raster, target_distance_field, args)
    update_rule = UpdateRule(CHEM_CHANNELS)

    population = [get_weights(UpdateRule(CHEM_CHANNELS)) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    best_fitness = float("inf")
    best_weights = population[0]

    for generation in range(args.generations):
        population, fitnesses, _ = run_generation(population, args, rng, pool)

        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_weights = population[0].copy()

        finite = [f for f in fitnesses if np.isfinite(f)]
        print(
            f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(finite) if finite else float('inf'):.4f}  "
            f"worst {fitnesses[-1]:.4f}  (all-time best {best_fitness:.4f})"
        )

        if (generation + 1) % args.checkpoint_every == 0 or generation == args.generations - 1:
            set_weights(update_rule, best_weights)
            np.save(CHECKPOINTS_DIR / "best.npy", best_weights)
            (CHECKPOINTS_DIR / "best_weights.json").write_text(json.dumps(update_rule.export_weights()))
            (CHECKPOINTS_DIR / "best_meta.json").write_text(
                json.dumps(
                    {
                        "generation": generation,
                        "fitness": best_fitness,
                        "target": args.target,
                        "particles": args.particles,
                        "macro_steps": args.macro_steps,
                        "growth_steps": args.growth_steps,
                        "substeps_per_macro": args.substeps_per_macro,
                        "growth_duration_macro_steps": GROWTH_DURATION_MACRO_STEPS,
                        "morphology_blur_sigma": MORPHOLOGY_BLUR_SIGMA,
                        "morphology_density_reference": MORPHOLOGY_DENSITY_REFERENCE,
                        "neural_updates_per_macro": NEURAL_UPDATES_PER_MACRO,
                        "communication_speed": COMMUNICATION_SPEED,
                        "elastic_strain_scale": ELASTIC_STRAIN_SCALE,
                        "elastic_strain_inputs_enabled": ELASTIC_STRAIN_INPUTS_ENABLED,
                        "gravity": args.gravity,
                        "spawn_x": args.spawn_x,
                        "spawn_y": args.spawn_y,
                        "spawn_half_width": args.spawn_half_width,
                        "channels": CHEM_CHANNELS,
                        "field_n": FIELD_N,
                        "population": args.population,
                        "elites": args.elites,
                        "mutation_sigma": args.mutation_sigma,
                        "raster_resolution": args.raster_resolution,
                        "raster_sigma": args.raster_sigma,
                        "outside_weight": args.outside_weight,
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
    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


if __name__ == "__main__":
    main()

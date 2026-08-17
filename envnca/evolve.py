"""Population-based (mu, lambda) evolutionary training for UpdateRule,
with elitism — same approach as trainer/backend/evolve.py (no backprop:
each generation, every candidate's weights are evaluated by running a
full rollout from a fresh seed, scored by a rotation/translation-
invariant Gaussian-splat raster distance against a target — see
raster.py; the next generation is the elites plus Gaussian-mutated
copies of the elites). Random search, not gradient descent, exactly
like the other project's.

Adapted to this project's own rollout: a *fixed* population of GPU-
batched agents reading/depositing a GPU-resident chemical field, instead
of a growing node graph relaxed by CPU physics. Concretely, each
candidate's "shape" is just its agents' final positions after `--steps`
— there's no splitting/growth here to produce more points than the
agent count you start with (see simulation.py's own "No growth" note).

Rollouts run sequentially within a single process, not fanned out across
a ProcessPoolExecutor the way trainer/backend's does. That's a CPU-
parallelism story that doesn't map cleanly onto one shared GPU device —
several processes fighting over a single MPS/CUDA context isn't the
reliable win independent CPU cores are — and at this project's measured
per-step cost (sub-millisecond to a few ms even at 1000 agents) a
sequential loop over a candidate population is already fast; a
24-candidate x 100-generation run is a couple thousand rollouts, not the
kind of count where that matters yet. Batching the whole *population of
candidates* into one shared GPU tensor dimension (the natural GPU-native
analogue of process-level parallelism) is a real option if this ever
becomes the bottleneck — not built here.

Headless — no window, no plot. An earlier pass of this had a live
matplotlib fitness-chart-plus-replay window; that's been pulled out in
favor of a separate web-based frontend (built elsewhere) driving this
same training loop, so this module's only job now is the actual
(console-logged, checkpointed) search itself, same role
trainer/backend/evolve.py's own CLI plays for that project.

Usage:
    python evolve.py --target circle --generations 100 --population 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from device import pick_device
from environment import Environment
from raster import build_target_distance_field, build_target_raster, training_raster_distance
from simulation import Simulation
from target import TargetShape
from update_rule import UpdateRule

TARGETS_DIR = Path(__file__).parent / "targets"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def load_target(name: str, grid_size: int) -> TargetShape:
    path = TARGETS_DIR / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"unknown target '{name}' (looked for {path})")
    return TargetShape.from_export(json.loads(path.read_text()), grid_size)


def get_weights(model: UpdateRule) -> np.ndarray:
    return parameters_to_vector(model.parameters()).detach().cpu().numpy()


def set_weights(model: UpdateRule, flat: np.ndarray, device: torch.device) -> None:
    vector_to_parameters(torch.from_numpy(flat).float().to(device), model.parameters())


def mutate(weights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return weights + rng.normal(scale=sigma, size=weights.shape)


def raster_extent(grid_size: int) -> tuple[float, float, float, float]:
    """Shared (xmin, xmax, ymin, ymax) both target points and agent
    positions get rasterized against — they already live in the same
    grid-pixel coordinate space (0..grid_size), so this is just that
    space, not the target's own (usually much coarser) native authoring
    resolution. See raster.py."""
    return (0.0, float(grid_size), 0.0, float(grid_size))


# Fractional offsets *before* the end of a rollout's own (already
# +/-10%-jittered) step count at which a fitness-scoring snapshot is
# taken — see rollout()'s own docstring for why several, and why the
# worst of them (not e.g. their mean) is what's actually used.
CAPTURE_OFFSETS = (0.10, 0.075, 0.05, 0.025, 0.0)


def rollout(
    weights: np.ndarray,
    target: TargetShape,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    update_rule: UpdateRule,
    return_positions: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Runs one seed-to-`args.steps` simulation with `weights` loaded into
    the (reused, scratch) `update_rule`, and returns a fitness score —
    lower is better. `update_rule` is passed in and reused across calls
    purely to avoid rebuilding/re-`.to(device)`-ing the network on every
    rollout; its weights are fully overwritten by `weights` before use.

    `seed` drives a fresh, explicit torch.Generator (see agent_state.py's
    seed() docstring) rather than the implicit global RNG state — the
    only randomness anywhere in this simulation is the initial agent
    jitter plus this function's own step-count/checkpoint jitter below
    (simulation.py's step() is otherwise fully deterministic given agent
    state), so this one seed is all that's needed for "same weights,
    same fitness every time," the exact property
    trainer/backend/evolve.py's own docstring had to fix a bug to get.

    Two layers of "don't let a candidate learn to pose for one known
    instant" stack here:

    1. This rollout's own total length (`capture_steps`) is `args.steps`
       jittered +/-10%, deterministically from `rng`. Without this,
       "look good at exactly step N" would itself be learnable —
       e.g. converging into shape right on cue and immediately drifting
       apart after, which would score perfectly under a fixed step count
       while looking wrong at any other moment.
    2. Within *that* (already-randomized) length, the fitness snapshot
       isn't taken once at the very end — it's taken at each of
       `CAPTURE_OFFSETS` (the last 10% of the rollout, five evenly-spaced
       points), and the *worst* (highest, since lower is better) of the
       five scores is what's actually used as this candidate's fitness.
       A candidate that reaches a good shape and then immediately
       destabilizes scores exactly as badly as one that never reached it
       at all — this is what makes the old explicit `motion_weight`/
       settle-penalty mechanism (an earlier, cruder attempt at the same
       goal: penalizing continued motion once "close enough") redundant.
       Holding still isn't rewarded *directly* anymore; it falls out for
       free from needing to look good at every one of five nearby
       moments, not just one.

    `return_positions`, off by default, additionally returns the
    rollout's final (offset=0.0, raw, un-rotated) agent positions — for
    callers that need them for something other than scoring (e.g.
    train_server.py's end-of-generation debug images), not the hot
    per-candidate training path, which only ever wants the scalar
    fitness."""
    set_weights(update_rule, weights, device)
    rng = torch.Generator().manual_seed(seed)

    # +/-10%, drawn before Simulation below consumes `rng` for agent
    # jitter — order doesn't affect reproducibility (same seed always
    # draws the same sequence), just needs to be consistent call to call.
    step_jitter = 0.9 + 0.2 * torch.rand((), generator=rng).item()
    capture_steps = max(1, round(args.steps * step_jitter))

    env = Environment(height=args.grid_size, width=args.grid_size, channels=args.channels, device=device)
    sim = Simulation(env, update_rule, device, population=args.agents, spawn_spread=args.spawn_spread, rng=rng)

    # Deduplicated and sorted ascending — offsets can collide onto the
    # same integer step for a short enough rollout, and scoring the same
    # step twice would just waste work, not change the result (max of a
    # value with itself is itself). offset=0.0 always maps to
    # `capture_steps` itself, guaranteeing this set is never empty and
    # its last (chronological) member is always the final step.
    checkpoint_steps = {max(1, round(capture_steps * (1.0 - offset))) for offset in CAPTURE_OFFSETS}
    snapshots: list[np.ndarray] = []

    for step in range(1, capture_steps + 1):
        sim.step()
        if step in checkpoint_steps:
            snapshots.append(sim.agents.positions.detach().cpu().numpy())

    final_positions = snapshots[-1]

    # Rotation/translation-invariant: nothing pins agent motion to the
    # target's pose (there's no physics anchoring anything in place), and
    # the target's own orientation is an arbitrary artifact of however it
    # was drawn — fitness should reward getting the *shape* right, not
    # accidentally landing in the same pose as the target. Combines raster
    # coverage (MSE) with an explicit, distance-transform-based penalty for
    # landing outside the target's footprint — see raster.py.
    fitness = 0.0
    for positions in snapshots:
        if positions.shape[0] == 0 or not np.isfinite(positions).all():
            # A diverged/emptied-out candidate should score terribly, not
            # crash the run — this is the backstop.
            fitness = float("inf")
            break
        distance = training_raster_distance(
            positions,
            target.points,
            target_raster,
            target_distance_field,
            args.raster_resolution,
            raster_extent(args.grid_size),
            args.raster_sigma,
            outside_weight=args.outside_weight,
        )
        fitness = max(fitness, distance)

    return (fitness, final_positions) if return_positions else fitness


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="circle", help="target name in envnca/targets/ (without .json)")
    parser.add_argument("--population", type=int, default=24, help="number of candidate weight-sets per generation")
    parser.add_argument(
        "--elites", type=int, default=4, help="top performers carried into the next generation unmutated"
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--agents", type=int, default=200, help="agent count per rollout (fixed population)")
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="nominal simulation steps per rollout — the actual fitness-capture step is jittered +/-10% (see rollout())",
    )
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--spawn-spread", type=float, default=4.0)
    parser.add_argument("--mutation-sigma", type=float, default=0.05)
    parser.add_argument(
        "--raster-resolution",
        type=int,
        default=128,
        help="side length of the square lattice target/agent point clouds are splatted onto for fitness — see raster.py",
    )
    parser.add_argument(
        "--raster-sigma",
        type=float,
        default=1.5,
        help="Gaussian splat width, in raster pixels (not grid pixels) — see raster.rasterize_points",
    )
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=1.0,
        help=(
            "weight of the distance-transform penalty for agents landing outside the target's footprint "
            "(0 disables it, falling back to raster coverage MSE alone) — see raster.outside_shape_penalty"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser


def run_generation(
    population: list[np.ndarray],
    target: TargetShape,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
    update_rule: UpdateRule,
) -> tuple[list[np.ndarray], list[float], int]:
    """Evaluates every candidate in `population` (one rollout each,
    sequentially — see module docstring), sorts best-first, and refills
    back up to `args.population` via elitism + Gaussian mutation.

    Each candidate's rollout gets its own seed, deterministically derived
    from the caller's `rng`, so a full run is reproducible given the same
    top-level --seed.

    Returns `(next_population, fitnesses, winner_seed)` — `fitnesses` are
    for the population just evaluated, sorted best (lowest raster
    distance — see raster.py) first; `next_population[0]` is this
    generation's winning weights, carried over unmutated as the top
    elite; `winner_seed` is the seed that produced `fitnesses[0]` —
    needed by callers (e.g. train_server.py) that want to reproduce this
    generation's *actual* winning rollout, not just its weights,
    elsewhere (e.g. a browser replay seeded to match)."""
    seeds = rng.integers(0, 2**31 - 1, size=len(population))
    fitnesses = [
        rollout(w, target, target_raster, target_distance_field, args, int(s), device, update_rule)
        for w, s in zip(population, seeds)
    ]

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


def main() -> None:
    args = build_arg_parser().parse_args()

    if not 1 <= args.elites <= args.population:
        raise SystemExit("--elites must be between 1 and --population")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    target = load_target(args.target, args.grid_size)
    # Fixed for the whole run — precomputed once rather than on every
    # rollout's training_raster_distance() call (population x generations
    # x rotation-search angles otherwise recomputing the exact same
    # thing). See raster.build_target_raster()'s own docstring.
    target_raster = build_target_raster(
        target.points,
        args.raster_resolution,
        raster_extent(args.grid_size),
        args.raster_sigma,
        half_size=target.texel_size(args.grid_size) / 2.0,
    )
    target_distance_field = build_target_distance_field(target_raster)
    update_rule = UpdateRule(num_channels=args.channels).to(device)
    population = [get_weights(UpdateRule(num_channels=args.channels)) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    best_fitness = float("inf")
    best_weights = population[0]

    for generation in range(args.generations):
        population, fitnesses, winner_seed = run_generation(
            population, target, target_raster, target_distance_field, args, rng, device, update_rule
        )

        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_weights = population[0].copy()

        finite = [f for f in fitnesses if np.isfinite(f)]
        print(
            f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(finite) if finite else float('inf'):.4f}  "
            f"worst {fitnesses[-1]:.4f}  (all-time best {best_fitness:.4f})"
        )

        if (generation + 1) % args.checkpoint_every == 0 or generation == args.generations - 1:
            np.save(CHECKPOINTS_DIR / "best.npy", best_weights)
            (CHECKPOINTS_DIR / "best_meta.json").write_text(
                json.dumps(
                    {
                        "generation": generation,
                        "fitness": best_fitness,
                        "target": args.target,
                        "agents": args.agents,
                        "steps": args.steps,
                        "grid_size": args.grid_size,
                        "channels": args.channels,
                        "spawn_spread": args.spawn_spread,
                        "population": args.population,
                        "elites": args.elites,
                        "mutation_sigma": args.mutation_sigma,
                        "raster_resolution": args.raster_resolution,
                        "raster_sigma": args.raster_sigma,
                        "outside_weight": args.outside_weight,
                        "seed": args.seed,
                        "winner_seed": winner_seed,
                    },
                    indent=2,
                )
            )

    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


if __name__ == "__main__":
    main()

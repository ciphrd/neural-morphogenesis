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

Optional memetic (Lamarckian) refinement, --memetic-steps > 0 (0, the
default, is pure Darwinian ES — nothing below changes): each candidate,
before being scored, gets a short truncated-BPTT gradient-descent
refinement pass (gradient_refine(), reusing train_gd.py's own per-window
mechanics — differentiable rollout via raster_torch's loss, Adam,
grad-clipping) applied directly to its weights. run_generation() then
scores and selects on the *refined* result, not the pre-refinement one —
"Lamarckian" in the classic memetic-algorithm sense: acquired
improvement (from gradient descent) is what gets inherited, not just
whatever survived random mutation + selection alone. This exists because
plain gradient descent (train_gd.py) has no population and no selection
pressure — a single trajectory that can only exploit whatever basin it's
already in, never explore several at once the way ES's population does;
plain ES, in turn, only ever finds a better basin through random
mutation, never through following a local gradient once it's in one.
Memetic refinement is meant to get both: ES's population-level breadth
plus gradient descent's cheap, fast local convergence within whichever
basin each individual already landed in.

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
from typing import Optional

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from device import pick_device
from environment import Environment
from raster import build_target_distance_field, build_target_raster, training_raster_distance
from raster_torch import target_rasters_to_torch, training_raster_distance_torch
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

    # Explicit opt-out of gradient tracking — simulation.py's step() no
    # longer forces this itself (see that method's own docstring), since
    # it also now backs a differentiable training path elsewhere. This ES
    # path never calls .backward() (fitness is scored as plain NumPy
    # below), so tracking a graph through hundreds of untracked steps
    # would be pure wasted memory/compute with zero benefit — this
    # context is what keeps rollout()'s performance/memory profile
    # identical to before.
    with torch.no_grad():
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


def gradient_refine(
    weights: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    update_rule: UpdateRule,
    target_points_t: torch.Tensor,
    target_raster_t: torch.Tensor,
    target_distance_field_t: torch.Tensor,
) -> np.ndarray:
    """The "acquired-trait" half of memetic/Lamarckian refinement — see
    this module's own docstring. Runs `args.memetic_steps` of truncated-
    BPTT gradient descent starting from `weights`, mirroring train_gd.py's
    own per-window mechanics (windowed backward + detach, so the graph
    never exceeds one window regardless of how many windows run — see
    that module's own docstring for the full memory reasoning) rather
    than importing it, since this needs its own fresh Adam optimizer per
    call: each candidate here is a genuinely different individual, not a
    continuation of whichever candidate last used this same scratch
    `update_rule`, so no momentum should carry over between them.

    Deliberately a *simpler*, cheaper rollout than this module's own
    rollout() — no step-count jitter, no multi-point worst-of-5 capture
    (see that function's own docstring for why that robustness matters
    for scoring) — because this function's own loss is never what a
    candidate gets selected or ranked on; run_generation() always
    re-scores whatever this returns through the real, robust rollout()
    afterward. This only has to be a reasonable *nudge* toward a better
    local optimum, not a gameable fitness signal in its own right.

    Returns the refined flat weight vector — or, if the rollout diverges
    partway through (non-finite positions, an empty population), whatever
    `update_rule` held at the last finite window, same "fail soft, don't
    crash the generation" backstop rollout() has."""
    set_weights(update_rule, weights, device)
    optimizer = torch.optim.Adam(update_rule.parameters(), lr=args.memetic_lr)
    rng = torch.Generator().manual_seed(seed)

    env = Environment(height=args.grid_size, width=args.grid_size, channels=args.channels, device=device)
    sim = Simulation(env, update_rule, device, population=args.agents, spawn_spread=args.spawn_spread, rng=rng)

    step = 0
    while step < args.memetic_steps:
        window_end = min(step + args.memetic_bptt_steps, args.memetic_steps)
        for _ in range(window_end - step):
            sim.step()
        step = window_end

        positions = sim.agents.positions
        if positions.shape[0] == 0 or not torch.isfinite(positions).all():
            break

        loss = training_raster_distance_torch(
            positions,
            target_points_t,
            target_raster_t,
            target_distance_field_t,
            args.raster_resolution,
            raster_extent(args.grid_size),
            args.raster_sigma,
            outside_weight=args.outside_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(update_rule.parameters(), args.memetic_grad_clip)
        if torch.isfinite(grad_norm):
            optimizer.step()

        # Sever the graph before the next window — same reasoning as
        # train_gd.py's own truncated-BPTT loop.
        sim.agents.positions = sim.agents.positions.detach()
        sim.agents.velocity = sim.agents.velocity.detach()
        sim.env.grid = sim.env.grid.detach()

    return get_weights(update_rule)


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
    parser.add_argument(
        "--memetic-steps",
        type=int,
        default=0,
        help=(
            "if > 0, each candidate gets this many steps of gradient-descent refinement (truncated BPTT, "
            "raster_torch loss, its own fresh Adam optimizer) before being scored — see gradient_refine()'s "
            "own docstring and this module's own docstring for the memetic/Lamarckian reasoning. 0 (the "
            "default) is pure Darwinian ES, unchanged from before this existed"
        ),
    )
    parser.add_argument(
        "--memetic-bptt-steps",
        type=int,
        default=20,
        help="truncated-BPTT window size for memetic refinement — see train_gd.py's own module docstring "
        "for what this trades off (only relevant when --memetic-steps > 0)",
    )
    parser.add_argument("--memetic-lr", type=float, default=2e-4, help="Adam learning rate for memetic refinement")
    parser.add_argument(
        "--memetic-grad-clip",
        type=float,
        default=0.25,
        help="max gradient norm during memetic refinement — see train_gd.py's own --grad-clip for the same "
        "BPTT-instability reasoning",
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
    target_points_t: Optional[torch.Tensor] = None,
    target_raster_t: Optional[torch.Tensor] = None,
    target_distance_field_t: Optional[torch.Tensor] = None,
) -> tuple[list[np.ndarray], list[float], int]:
    """Evaluates every candidate in `population` (one rollout each,
    sequentially — see module docstring), sorts best-first, and refills
    back up to `args.population` via elitism + Gaussian mutation.

    Each candidate's rollout gets its own seed, deterministically derived
    from the caller's `rng`, so a full run is reproducible given the same
    top-level --seed.

    Memetic refinement (`args.memetic_steps > 0` — see gradient_refine()'s
    own docstring): every candidate is gradient-refined *before* scoring,
    using the same seed its scoring rollout will use, so what gets
    selected/mutated below is already the refined weights, not the
    pre-refinement ones — Lamarckian, not just Darwinian, inheritance.
    Requires target_points_t/target_raster_t/target_distance_field_t (the
    torch-side target constants — see raster_torch.target_rasters_to_torch);
    callers that never enable memetic mode can leave them None (the
    default) and nothing here changes.

    Returns `(next_population, fitnesses, winner_seed)` — `fitnesses` are
    for the population just evaluated, sorted best (lowest raster
    distance — see raster.py) first; `next_population[0]` is this
    generation's winning weights, carried over unmutated as the top
    elite; `winner_seed` is the seed that produced `fitnesses[0]` —
    needed by callers (e.g. train_server.py) that want to reproduce this
    generation's *actual* winning rollout, not just its weights,
    elsewhere (e.g. a browser replay seeded to match)."""
    seeds = rng.integers(0, 2**31 - 1, size=len(population))

    if args.memetic_steps > 0:
        assert target_points_t is not None and target_raster_t is not None and target_distance_field_t is not None, (
            "run_generation() needs the torch-side target constants when --memetic-steps > 0"
        )
        population = [
            gradient_refine(w, args, int(s), device, update_rule, target_points_t, target_raster_t, target_distance_field_t)
            for w, s in zip(population, seeds)
        ]

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
    # Only needed for memetic refinement (gradient_refine()) — skipped
    # (left None) for a plain --memetic-steps=0 run so a pure-ES
    # invocation never pays for a torch tensor conversion it won't use.
    target_points_t = target_raster_t = target_distance_field_t = None
    if args.memetic_steps > 0:
        target_points_t = torch.tensor(target.points, dtype=torch.float32, device=device)
        target_raster_t, target_distance_field_t = target_rasters_to_torch(target_raster, target_distance_field, device)
    update_rule = UpdateRule(num_channels=args.channels).to(device)
    population = [get_weights(UpdateRule(num_channels=args.channels)) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    best_fitness = float("inf")
    best_weights = population[0]

    for generation in range(args.generations):
        population, fitnesses, winner_seed = run_generation(
            population,
            target,
            target_raster,
            target_distance_field,
            args,
            rng,
            device,
            update_rule,
            target_points_t=target_points_t,
            target_raster_t=target_raster_t,
            target_distance_field_t=target_distance_field_t,
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
                        "memetic_steps": args.memetic_steps,
                        "memetic_bptt_steps": args.memetic_bptt_steps,
                        "memetic_lr": args.memetic_lr,
                        "memetic_grad_clip": args.memetic_grad_clip,
                        "seed": args.seed,
                        "winner_seed": winner_seed,
                    },
                    indent=2,
                )
            )

    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


if __name__ == "__main__":
    main()

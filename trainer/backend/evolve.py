"""Population-based (mu, lambda) evolutionary training for UpdateRule,
with elitism. No backprop: each generation, every candidate's weights
are evaluated by running a full rollout (update rule + physics relax)
from a fresh seed, scored against a target by a rotation/translation-
invariant Chamfer distance (see alignment.py); the next generation is the
elites plus Gaussian-mutated copies of the elites.
Random search, not gradient descent — see trainer/README.md's "Training"
section for where this sits in the larger plan (this is the first,
simplest version of it).

Usage:
    python evolve.py --target circle --generations 100 --population 24
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from alignment import training_alignment_distance
from graph import Graph
from physics import relax
from target import TargetShape
from update_rule import UpdateRule
from update_rule import step as apply_update_rule

TARGETS_DIR = Path(__file__).parent / "targets"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def load_target(name: str) -> TargetShape:
    path = TARGETS_DIR / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"unknown target '{name}' (looked for {path})")
    return TargetShape.from_export(json.loads(path.read_text()))


def get_weights(model: UpdateRule) -> np.ndarray:
    return parameters_to_vector(model.parameters()).detach().numpy()


def set_weights(model: UpdateRule, flat: np.ndarray) -> None:
    vector_to_parameters(torch.from_numpy(flat).float(), model.parameters())


def mutate(weights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return weights + rng.normal(scale=sigma, size=weights.shape)


def rollout(
    weights: np.ndarray,
    target: TargetShape,
    steps: int,
    damage_prob: float,
    damage_fraction: float,
    rng: np.random.Generator,
    model: UpdateRule,
) -> float:
    """Runs one seed-to-`steps` simulation with `weights` loaded into the
    (reused, scratch) `model`, and returns the best-orientation Chamfer
    distance to `target` (see alignment.training_alignment_distance) —
    lower is better. `model` is passed in and reused across calls
    purely to avoid rebuilding the network's structure on every rollout;
    its weights are fully overwritten by `weights` before use."""
    set_weights(model, weights)
    graph = Graph.seed(rng=rng)

    for _ in range(steps):
        # rng, not the module's implicit global random state: without
        # this, the same weights + the same nominal seed produce a
        # *different* fitness every single call (confirmed directly:
        # identical weights/seed, 10 rollouts at steps=200, fitness
        # ranged 3.14-3.75, std ~0.2) — the growth/split process was
        # silently drawing from numpy's global RNG regardless of what
        # seed the caller asked for. That's pure simulation noise
        # layered on top of whatever signal a candidate's weights
        # actually carry, and it's large enough to swamp real
        # generation-to-generation improvement, especially on a target
        # (like a thin line) where the resulting shape is highly
        # sensitive to exactly when/where each split happens.
        changed = apply_update_rule(graph, model, rng=rng)

        if damage_prob > 0 and rng.random() < damage_prob:
            candidates = [i for i in range(len(graph.positions)) if i not in graph.pinned]
            if candidates:
                num_to_damage = max(1, int(len(candidates) * damage_fraction))
                victims = rng.choice(candidates, size=min(num_to_damage, len(candidates)), replace=False)
                graph.remove_nodes(set(victims.tolist()))
                changed = True

        # Nothing to re-settle if the node set didn't change this step —
        # positions were already at equilibrium from the previous relax.
        if changed:
            relaxed = relax(graph.positions_array(), graph.pinned, graph.id_array())
            graph.set_positions(relaxed)

    positions = graph.positions_array()
    if len(graph.positions) == 0:
        return float("inf")

    # Defense in depth, not the primary fix: update_rule.py's
    # CHEMICAL_CLIP / id renormalization is what actually prevents state
    # from diverging to inf/NaN in the first place. This is a backstop
    # for whatever future numerical edge case that primary fix doesn't
    # anticipate — chamfer_distance's cKDTree raises on non-finite input,
    # and letting that exception escape a worker process kills the whole
    # ProcessPoolExecutor generation (and, previously, the training
    # server's background task silently and permanently). A candidate
    # that diverged is exactly the kind of candidate that should score
    # terribly, not crash the run.
    if not np.isfinite(positions).all():
        return float("inf")

    # Rotation/translation-invariant: growth isn't anchored to a fixed
    # pose (nothing pins the structure in place, so it can drift/rotate
    # freely during physics relax), and the target's own orientation is
    # an arbitrary artifact of however it was drawn — fitness should
    # reward getting the *shape* right, not accidentally landing in the
    # same pose as the target. See alignment.py.
    return training_alignment_distance(positions, target.points)


# Population members are fully independent rollouts (different weights,
# no shared mutable state) — evaluating them one at a time in a single
# thread leaves every other CPU core idle for no reason. These two
# module-level pieces (picklable, needed by ProcessPoolExecutor) are what
# let run_generation fan the population out across worker processes.
_worker_model: Optional[UpdateRule] = None


def _init_worker() -> None:
    global _worker_model
    # One core, deliberately: parallelism happens at the process level
    # (one candidate per worker), so letting torch *also* spread each
    # tiny MLP forward pass across threads inside every worker would
    # oversubscribe the same cores against itself for no benefit — the
    # network here is far too small (37->128->18) for intra-op threading
    # to pay for its own overhead.
    torch.set_num_threads(1)
    _worker_model = UpdateRule()


def _rollout_worker(job: tuple) -> float:
    weights, target, steps, damage_prob, damage_fraction, seed = job
    assert _worker_model is not None, "worker called without _init_worker"
    rng = np.random.default_rng(seed)
    try:
        return rollout(weights, target, steps, damage_prob, damage_fraction, rng, _worker_model)
    except Exception as exc:
        # Broadest possible safety net: rollout() already guards the
        # specific non-finite-positions case, but an uncaught exception
        # anywhere in a worker propagates through executor.map() and
        # kills the *entire* generation for every other candidate too
        # (and previously, train_server.py's whole background task,
        # silently — see its own try/except around this generation's
        # caller). One candidate misbehaving should never be able to do
        # that; scoring it as the worst possible outcome and moving on
        # is always the right response for population-based search.
        print(f"[evolve] rollout failed, scoring as worst-possible: {exc!r}")
        return float("inf")


def build_arg_parser() -> argparse.ArgumentParser:
    """Shared flag definitions — used by this script's own CLI and by
    train_server.py, so the two never drift out of sync on what a given
    flag means or defaults to."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="circle", help="target name in backend/targets/ (without .json)")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument(
        "--elites", type=int, default=4, help="top performers carried into the next generation unmutated"
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=15, help="simulation steps per rollout")
    parser.add_argument("--mutation-sigma", type=float, default=0.05)
    parser.add_argument(
        "--damage-prob", type=float, default=0.0, help="probability, per simulation step, of a damage event"
    )
    parser.add_argument(
        "--damage-fraction",
        type=float,
        default=0.1,
        help="fraction of (non-pinned) nodes removed on a damage event",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="parallel worker processes for evaluating the population (default: all cores)",
    )
    return parser


def run_generation(
    population: list[np.ndarray],
    target: TargetShape,
    args: argparse.Namespace,
    rng: np.random.Generator,
    model: UpdateRule,
    executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Evaluates every candidate in `population` (one rollout each) — in
    parallel across `executor`'s worker processes when given, since
    candidates are fully independent; falls back to sequential
    evaluation in the calling process (reusing `model` as scratch space)
    when `executor` is None, e.g. for tests or a single-candidate debug
    run where spinning up a process pool isn't worth it. Sorts
    best-first, and refills back up to `args.population` via elitism +
    Gaussian mutation.

    Each candidate gets its own RNG, seeded deterministically from the
    caller's `rng` — necessary since a single mutable Generator can't be
    shared across process boundaries the way it could when everything
    ran in one thread. Reproducible given the same top-level --seed, but
    not bit-identical to a purely-sequential run's exact random draws.

    Returns `(next_population, fitnesses)` — `fitnesses` are for the
    population just evaluated, sorted best (lowest Chamfer distance)
    first; `next_population[0]` is this generation's winning weights,
    carried over unmutated as the top elite, so callers that want "the
    best candidate from this generation" (e.g. to replay/visualize it)
    can use `next_population[0]` directly rather than re-deriving it.
    """
    seeds = rng.integers(0, 2**31 - 1, size=len(population))

    if executor is not None:
        jobs = [
            (w, target, args.steps, args.damage_prob, args.damage_fraction, int(s))
            for w, s in zip(population, seeds)
        ]
        fitnesses = list(executor.map(_rollout_worker, jobs))
    else:
        fitnesses = [
            rollout(
                w,
                target,
                args.steps,
                args.damage_prob,
                args.damage_fraction,
                np.random.default_rng(int(s)),
                model,
            )
            for w, s in zip(population, seeds)
        ]

    order = np.argsort(fitnesses)
    population = [population[i] for i in order]
    fitnesses = [fitnesses[i] for i in order]

    elites = population[: args.elites]
    next_population = list(elites)
    while len(next_population) < args.population:
        parent = elites[rng.integers(len(elites))]
        next_population.append(mutate(parent, args.mutation_sigma, rng))

    return next_population, fitnesses


def main() -> None:
    args = build_arg_parser().parse_args()

    if not 1 <= args.elites <= args.population:
        raise SystemExit("--elites must be between 1 and --population")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    target = load_target(args.target)
    model = UpdateRule()
    population = [get_weights(UpdateRule()) for _ in range(args.population)]

    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    best_fitness = float("inf")
    best_weights = population[0]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as executor:
        for generation in range(args.generations):
            population, fitnesses = run_generation(population, target, args, rng, model, executor)

            if fitnesses[0] < best_fitness:
                best_fitness = fitnesses[0]
                best_weights = population[0].copy()

            print(
                f"gen {generation:4d}  best {fitnesses[0]:.4f}  mean {np.mean(fitnesses):.4f}  "
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
                            "steps": args.steps,
                            "population": args.population,
                            "elites": args.elites,
                            "mutation_sigma": args.mutation_sigma,
                            "damage_prob": args.damage_prob,
                            "damage_fraction": args.damage_fraction,
                            "seed": args.seed,
                        },
                        indent=2,
                    )
                )

    print(f"done. best fitness: {best_fitness:.4f}. weights saved to {CHECKPOINTS_DIR / 'best.npy'}")


if __name__ == "__main__":
    main()

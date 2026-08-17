"""Wires physics_torch.py + graph_torch.py + distance_torch.py + the
existing (already-torch) UpdateRule into one differentiable rollout, and
a small demo training loop that runs real gradient descent against a
target shape — the actual proof that the three prototype pieces hang
together end to end, not just individually correct in isolation. See
those three files' own docstrings for what each replaces and why.

Not a production training entry point — evolve.py's evolutionary search
remains what actually trains the checkpoints used elsewhere in this
project. This is a standalone demonstration that backprop-based training
is *possible* on this simulation now, answering the question the rest of
this differentiable-* effort was building toward. Promoting any of this
to a real trainer (batching a population, checkpointing, wiring into
train_server.py) is future work, deliberately out of scope here.

Usage:
    python diff_rollout_demo.py
"""

from __future__ import annotations

import numpy as np
import torch

from distance_torch import training_alignment_distance_torch
from graph_torch import seed_graph_torch, step_torch
from physics_torch import relax_torch
from update_rule import UpdateRule

# Deliberately small next to production defaults (MAX_NODES=400,
# SETTLE_ITERATIONS=100, CLEANUP_ITERATIONS=400, evolve.py's own default
# --steps=15 is actually comparable) — this demo unrolls the *entire*
# rollout into one backward pass, so its cost is steps * (relax
# iterations) * (dense O(max_nodes^2) physics), all multiplied again by
# however many outer training iterations run. Small enough here to
# finish in well under a minute; scaling any of these up is a compute
# tradeoff, not a correctness one — nothing about the approach changes.
MAX_NODES = 20
ROLLOUT_STEPS = 18
SETTLE_ITERATIONS = 15
CLEANUP_ITERATIONS = 30
TRAINING_ITERATIONS = 40
LEARNING_RATE = 0.02


def rollout_torch(
    model: UpdateRule,
    target_points: torch.Tensor,
    steps: int = ROLLOUT_STEPS,
    max_nodes: int = MAX_NODES,
    rng: "np.random.Generator | None" = None,
    settle_iterations: int = SETTLE_ITERATIONS,
    cleanup_iterations: int = CLEANUP_ITERATIONS,
) -> torch.Tensor:
    """One full seed-to-`steps` differentiable rollout, scored the same
    way evolve.py's rollout() scores its (non-differentiable) one — same
    metric (training_alignment_distance), just computed in torch so
    gradient survives back to `model`'s weights. Physics runs every step
    unconditionally (no "did anything change, skip the relax" branch
    like the production step()/rollout() have) — a data-dependent skip
    is exactly the kind of thing a static, backpropagable loop can't
    have; see physics_torch.py's own docstring for the fuller version of
    this same point."""
    graph = seed_graph_torch(max_nodes=max_nodes, rng=rng)

    for _ in range(steps):
        graph = step_torch(graph, model, rng=rng)
        graph.positions = relax_torch(
            graph.positions,
            graph.pinned,
            graph.id_vectors,
            alive=graph.alive,
            settle_iterations=settle_iterations,
            cleanup_iterations=cleanup_iterations,
        )

    alive_mask = graph.alive > 0.5
    points = graph.positions[alive_mask]
    return training_alignment_distance_torch(points, target_points)


def _make_demo_target() -> torch.Tensor:
    """A small, deliberately simple target (a short line of points) —
    reachable within ROLLOUT_STEPS/MAX_NODES's modest budget, and simple
    enough that "did the loss actually go down" is easy to sanity-check
    by eye against the growth topology, same spirit as this whole
    project's own line/circle/donut targets but hand-built here so this
    demo has no file-path dependency on trainer/backend/targets/."""
    xs = torch.linspace(-2.5, 2.5, steps=9, dtype=torch.float64)
    ys = torch.zeros_like(xs)
    return torch.stack([xs, ys], dim=-1)


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = UpdateRule().to(torch.float64)
    target = _make_demo_target()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"target: {target.shape[0]} points, demo config: max_nodes={MAX_NODES}, "
          f"rollout_steps={ROLLOUT_STEPS}, training_iterations={TRAINING_ITERATIONS}\n")

    losses = []
    for iteration in range(TRAINING_ITERATIONS):
        # Fresh seed per iteration so the optimizer can't just overfit
        # one lucky root chemical/id draw — same reproducibility
        # machinery evolve.py's rollout() now uses (see cell_state.py's
        # random_id/random_chemical), just re-seeded every iteration
        # instead of once per candidate.
        iter_rng = np.random.default_rng(1000 + iteration)

        optimizer.zero_grad()
        loss = rollout_torch(model, target, rng=iter_rng)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"iter {iteration:3d}  loss {loss.item():.4f}")

    first_five = np.mean(losses[:5])
    last_five = np.mean(losses[-5:])
    print(f"\nmean loss, first 5 iterations: {first_five:.4f}")
    print(f"mean loss, last 5 iterations:  {last_five:.4f}")
    print(f"{'IMPROVED' if last_five < first_five else 'DID NOT IMPROVE'} "
          f"({100 * (first_five - last_five) / first_five:.1f}% change)")


if __name__ == "__main__":
    main()

"""Fixed-capacity, differentiable graph state — torch port of graph.py +
update_rule.py's step(), the last of the three prototype pieces (see
physics_torch.py and distance_torch.py) needed to backprop through a
full rollout. Not wired into any live path; graph.py/update_rule.py
remain authoritative for evolve.py/main.py/train_server.py.

Two changes from the production system, both required for autograd —
same spirit as physics_torch.py's dense pairwise ops and fixed iteration
count, this is the graph-representation half of that same problem:

1. Fixed (MAX_NODES, ...) tensors instead of graph.py's Python list that
   grows one node at a time. A dynamically-growing list of tensors with
   a different shape every step can't be traced through a static
   backward graph the way an in-place update to a fixed-shape tensor
   can — and it can't be batched across a population at all. Every slot
   exists from the start; `alive` (continuous, in [0, 1], not boolean)
   is what actually represents "has this node been born yet," and
   everything downstream (physics_torch.relax_torch's own `alive` param,
   sensing below, eventually the loss) is built to respect it as a soft
   gate rather than a hard filter.

2. update_rule.py's step()'s hard `should_split = rng.random() <
   effective_split_prob` Bernoulli draw becomes a straight-through
   estimator (STE): the forward pass still makes a crisp yes/no decision
   (behavior doesn't get mushy), but the backward pass pretends the
   decision *was* the network's own continuous split probability, so a
   gradient on "should this node have wanted to split more/less" can
   actually reach the weights that produced it. Same trick Growing
   NCA's own alpha-threshold "born" transition uses, adapted for a
   Bernoulli decision instead of a fixed threshold.

One deliberate simplification beyond those two, worth being explicit
about: only ONE new child slot is claimed per step — whichever alive
node has the single highest split probability, if it clears the gate —
not "every alive node independently rolls its own Bernoulli draw and
potentially spawns simultaneously" like the production step(). Assigning
several simultaneous new children to specific free slots is itself a
discrete scheduling problem, and solving *that* differentiably is
meaningfully more work than this prototype's actual goal (prove gradient
reaches the weights across a full rollout). A production version of this
would need to revisit it — e.g. claiming the top-K slots by split
probability each step, K bounded by however many free slots remain, with
the same STE gate per claimed slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from cell_state import ID_DIM, INITIAL_ENERGY, NUM_CHEMICAL_CHANNELS, SPAWN_DIR_DIM, random_chemical, random_id
from physics import CONTACT_DISTANCE
from update_rule import (
    CHEMICAL_CLIP,
    ENERGY_INJECTION,
    ENERGY_INJECTION_NOISE,
    MAX_ENERGY,
    MAX_NODES,
    MIN_SPLIT_ENERGY,
    SENSING_SIGMA,
    UpdateRule,
)


@dataclass
class TorchGraph:
    """All tensors are (MAX_NODES, ...), always — growth changes `alive`
    and `next_free_slot`, never a tensor's shape. `next_free_slot` and
    `pinned` are plain Python state (scheduling/bookkeeping, not learned
    values), same treatment as physics_torch.relax_torch's `active`
    pairwise mask — discrete and gradient-free by design, not an
    oversight."""

    positions: torch.Tensor  # (MAX_NODES, 2)
    chemicals: torch.Tensor  # (MAX_NODES, NUM_CHEMICAL_CHANNELS)
    id_vectors: torch.Tensor  # (MAX_NODES, ID_DIM)
    energy: torch.Tensor  # (MAX_NODES,)
    alive: torch.Tensor  # (MAX_NODES,) in [0, 1]
    pinned: torch.Tensor  # (MAX_NODES,) bool
    next_free_slot: int


def seed_graph_torch(
    max_nodes: int = MAX_NODES,
    initial_energy: float = INITIAL_ENERGY,
    rng: Optional[np.random.Generator] = None,
    dtype: torch.dtype = torch.float64,
) -> TorchGraph:
    """Slot 0 is the seed node (alive=1, random chemicals/id — via the
    same rng-aware cell_state helpers evolve.py's rollout() now uses, so
    a differentiable rollout can be made just as reproducible given a
    seed). Every other slot starts fully dead (alive=0) at the origin —
    an arbitrary placeholder position with no effect on anything, since
    physics_torch.relax_torch's alive-masking excludes dead slots from
    every pairwise interaction and this file's own sensing step below
    excludes them from the field. Nothing is pinned, matching graph.py's
    own actual default (see this file's module docstring) — production
    never pins the seed node either, which is exactly why alignment.py's
    rotation/translation search exists."""
    positions = torch.zeros(max_nodes, 2, dtype=dtype)
    chemicals = torch.zeros(max_nodes, NUM_CHEMICAL_CHANNELS, dtype=dtype)
    id_vectors = torch.zeros(max_nodes, ID_DIM, dtype=dtype)
    energy = torch.zeros(max_nodes, dtype=dtype)
    alive = torch.zeros(max_nodes, dtype=dtype)

    chemicals[0] = torch.from_numpy(random_chemical(rng)).to(dtype)
    id_vectors[0] = torch.from_numpy(random_id(rng)).to(dtype)
    energy[0] = initial_energy
    alive[0] = 1.0

    pinned = torch.zeros(max_nodes, dtype=torch.bool)

    return TorchGraph(
        positions=positions,
        chemicals=chemicals,
        id_vectors=id_vectors,
        energy=energy,
        alive=alive,
        pinned=pinned,
        next_free_slot=1,
    )


def _weighted_field_and_gradient_torch(
    query_points: torch.Tensor, source_points: torch.Tensor, weights: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-bandwidth case of substrate.py's weighted_field_and_gradient
    — the only case update_rule.py's step() actually uses. Pure
    elementwise/einsum math, no discrete decisions, so this is a direct
    numpy-to-torch swap, not a relaxation of anything (unlike the split
    gate above)."""
    diff = query_points.unsqueeze(1) - source_points.unsqueeze(0)  # (Q, S, 2)
    sq_dist = (diff * diff).sum(-1)  # (Q, S)
    kernel = torch.exp(-sq_dist / (2.0 * sigma * sigma))  # (Q, S)

    values = kernel @ weights  # (Q, S) @ (S, K) -> (Q, K)
    weighted_kernel = kernel.unsqueeze(-1) * weights.unsqueeze(0) / (sigma * sigma)  # (Q, S, K)
    gradients = -torch.einsum("qsk,qsd->qkd", weighted_kernel, diff)  # (Q, K, 2)
    return values, gradients


def _straight_through_gate(soft: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Forward value is the hard 0/1 threshold; backward gradient is
    exactly as if this were the identity function on `soft` — see this
    module's docstring, point 2."""
    hard = (soft > threshold).to(soft.dtype)
    return hard + (soft - soft.detach())


def step_torch(
    graph: TorchGraph,
    update_rule: UpdateRule,
    rng: Optional[np.random.Generator] = None,
    sensing_sigma: float = SENSING_SIGMA,
    energy_injection: float = ENERGY_INJECTION,
    energy_injection_noise: float = ENERGY_INJECTION_NOISE,
    min_split_energy: float = MIN_SPLIT_ENERGY,
    max_energy: float = MAX_ENERGY,
) -> TorchGraph:
    """One differentiable simulation step: sense -> decide -> act,
    mirroring update_rule.py's step() (see its own docstring for the
    energy mechanism's rationale, unchanged here) modulo the two
    representational differences this module's docstring documents.
    Does not relax physics — same division of labor as the production
    step(), the caller does that with physics_torch.relax_torch.

    Functional, not in-place: returns a new TorchGraph rather than
    mutating `graph`, since PyTorch autograd doesn't like in-place
    mutation of tensors that are part of an active graph — the caller's
    per-step loop should rebind (`graph = step_torch(graph, ...)`), the
    same pattern relax_torch's own settle/cleanup loops use internally.
    """
    n = graph.positions.shape[0]
    dtype = graph.positions.dtype
    alive_mask = graph.alive > 0.5

    draw_uniform = rng.uniform if rng is not None else np.random.uniform
    draw_random = rng.random if rng is not None else np.random.random

    noise = torch.from_numpy(np.asarray(draw_uniform(-energy_injection_noise, energy_injection_noise, size=n))).to(
        dtype
    )
    injected_energy = (graph.energy + energy_injection + noise).clamp(0.0, max_energy)
    normalized_energy = (injected_energy / max_energy) * 2.0 - 1.0

    # Dead slots contribute nothing to the field they'd otherwise
    # diffuse — same alive-gating principle as physics_torch.relax_torch's
    # pairwise interactions, just applied to sensing instead of force.
    weighted_chemicals = graph.chemicals * graph.alive.unsqueeze(-1)
    _, gradients = _weighted_field_and_gradient_torch(
        graph.positions, graph.positions, weighted_chemicals, sensing_sigma
    )
    grad_x = gradients[:, :, 0]
    grad_y = gradients[:, :, 1]

    split_logit, chemical_delta, id_delta, spawn_direction = update_rule(
        graph.chemicals, grad_x, grad_y, normalized_energy
    )
    split_prob = torch.sigmoid(split_logit)

    energy_weight = ((injected_energy - min_split_energy) / (max_energy - min_split_energy)).clamp(0.0, 1.0)
    effective_split_prob = split_prob * energy_weight

    new_chemicals = (graph.chemicals + chemical_delta).clamp(-CHEMICAL_CLIP, CHEMICAL_CLIP)
    new_id_raw = graph.id_vectors + id_delta
    id_norm = new_id_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    new_id = new_id_raw / id_norm

    # Only alive slots actually update — a dead slot's chemicals/id/energy
    # are arbitrary placeholders, so leaving them untouched (rather than
    # applying a delta that was computed from meaningless input) keeps
    # them exactly that: inert until the step they're actually born.
    alive_f = graph.alive
    chemicals_next = alive_f.unsqueeze(-1) * new_chemicals + (1 - alive_f).unsqueeze(-1) * graph.chemicals
    id_vectors_next = alive_f.unsqueeze(-1) * new_id + (1 - alive_f).unsqueeze(-1) * graph.id_vectors
    energy_next = alive_f * injected_energy + (1 - alive_f) * graph.energy

    positions_next = graph.positions
    alive_next = graph.alive
    next_free_slot = graph.next_free_slot

    # Single-spawner-per-step simplification (see module docstring) —
    # pick whichever alive node most wants to split, hard argmax (a
    # selection decision, not a learned value, same treatment as
    # physics_torch's pairwise proximity cutoffs), then gate whether that
    # split actually happens through the STE.
    candidate_prob = torch.where(alive_mask, effective_split_prob, torch.full_like(effective_split_prob, -1.0))
    if bool(alive_mask.any()) and next_free_slot < n:
        spawner = int(torch.argmax(candidate_prob).item())
        gate = _straight_through_gate(effective_split_prob[spawner])

        # Hard control flow (does a slot actually get claimed this step)
        # is a Python-level bookkeeping decision — see module docstring —
        # so it reads the STE's *hard* forward value, not the soft one.
        if float(gate.detach()) > 0.5:
            new_slot = next_free_slot
            next_free_slot += 1

            direction = spawn_direction[spawner]
            dir_norm = direction.norm().clamp_min(1e-9)
            unit = direction / dir_norm
            child_position = graph.positions[spawner] + unit * CONTACT_DISTANCE

            split_energy = energy_next[spawner] / 2.0

            # Scatter the new slot's state in without in-place mutation
            # of a tensor autograd is tracking — build a one-hot slot
            # mask and blend, same pattern as the alive-gated updates
            # above, rather than `positions_next[new_slot] = ...`.
            slot_onehot = torch.zeros(n, dtype=dtype)
            slot_onehot[new_slot] = 1.0

            positions_next = positions_next * (1 - slot_onehot).unsqueeze(-1) + torch.outer(
                slot_onehot, child_position
            )
            chemicals_next = chemicals_next * (1 - slot_onehot).unsqueeze(-1) + torch.outer(
                slot_onehot, new_chemicals[spawner]
            )
            id_vectors_next = id_vectors_next * (1 - slot_onehot).unsqueeze(-1) + torch.outer(
                slot_onehot, new_id[spawner]
            )
            energy_next = energy_next * (1 - slot_onehot) + slot_onehot * split_energy

            # `gate` (soft-valued via STE) is what actually carries
            # gradient back to the split decision — everything else
            # above is written at full value gated only by the hard
            # onehot, but alive is where the *confidence* of the
            # decision lives, exactly mirroring physics_torch's use of
            # alive as the one continuous quantity mediating a slot's
            # influence on the rest of the system.
            alive_next = alive_next * (1 - slot_onehot) + slot_onehot * gate.clamp(0.0, 1.0)

            # Parent's own energy is only actually spent if the split
            # gate is live — an STE-gated blend between "kept full
            # energy" (gate=0) and "spent half on the child" (gate=1),
            # not a hard overwrite, so gradient still reaches
            # energy_next[spawner] through the same gate.
            spawner_onehot = torch.zeros(n, dtype=dtype)
            spawner_onehot[spawner] = 1.0
            spawner_energy_blend = (1 - gate) * energy_next[spawner] + gate * split_energy
            energy_next = energy_next * (1 - spawner_onehot) + spawner_onehot * spawner_energy_blend

    return TorchGraph(
        positions=positions_next,
        chemicals=chemicals_next,
        id_vectors=id_vectors_next,
        energy=energy_next,
        alive=alive_next,
        pinned=graph.pinned,
        next_free_slot=next_free_slot,
    )

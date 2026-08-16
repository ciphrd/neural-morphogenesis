"""The per-node update rule: Dense(128) -> tanh -> Dense(18), evaluated
identically at every node from its own sensed local chemical gradient
(substrate.weighted_field_and_gradient) and its own energy level. See
trainer/README.md for the full spec this implements.

Forward-only scaffold: weights are randomly initialized and never
optimized here — training (loss, optimizer, backprop through the
simulation) is a deliberately separate, larger piece of work. Behavior
will look undirected/chaotic until that exists to shape it; that's
expected, not a bug in this module.

Update semantics (see README's "Open questions", now resolved):
- Every node's sense/decide is computed from one consistent snapshot of
  the graph (Jacobi-style, like physics.py) — not sequentially, so
  growth doesn't depend on node iteration order.
- A node that splits this step passes its *post-delta* state (chemicals,
  id, and its share of energy) to the child, not the state from before
  this step's update.
- Split is gated by energy, not a free-standing Bernoulli draw on the
  network's raw output — see "Energy" below.

Spawn direction: a 2D vector the network emits every step (not just on
steps where it actually splits) saying which way it *would* place a
child right now — always computed, since a node can't know in advance
whether this step's energy-gated Bernoulli draw (see "Energy" below)
will land on split, and the direction still needs to be there the
instant it does. Graph.spawn_directions holds the latest one per node,
normalized to unit length, purely for consumption (spawning) and display
(see render/GraphRenderer.tsx's "direction" color mode) — it isn't part
of the id/chemicals state vector the network reads back next step, so
there's no accumulation or feedback loop to bound here the way
CHEMICAL_CLIP/id-renormalization bound those. A degenerate (near-zero)
raw output has no direction to normalize; Graph.add_child falls back to
a random angle in that case, same as it always did before this existed.

Energy: a per-node growth budget (Graph.energy), not part of the state
vector the network reads and writes freely — the network can *sense* its
own energy (one more input) and *emit a desire to split* (still just a
probability, from the same output slot as before), but growth itself is
externally rate-limited, not something the network's output alone can
override.
- Every node receives a flat injection each step (ENERGY_INJECTION, ±
  ENERGY_INJECTION_NOISE), clamped to [0, MAX_ENERGY] — this is what
  bounds how fast the *whole organism* can grow regardless of how many
  nodes are simultaneously trying to split, since it's a per-node flat
  rate rather than something that scales with population.
- Below MIN_SPLIT_ENERGY, a node cannot split at all (effective
  probability forced to exactly 0) — a hard gate, not a soft
  discouragement.
- Above that threshold, the network's own split probability is scaled by
  how far above threshold the node's energy sits (linear ramp from 0 at
  MIN_SPLIT_ENERGY to 1 at MAX_ENERGY) — "the higher the energy, the
  more chances it has," without ever letting the network exceed its own
  stated probability.
- On an actual split, the (post-injection) energy is divided evenly
  between parent and child — mirroring cell division consuming the
  resources it took to reach the split threshold in the first place.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cell_state import ID_DIM, NUM_CHEMICAL_CHANNELS, SPAWN_DIR_DIM
from graph import Graph
from substrate import weighted_field_and_gradient

HIDDEN_DIM = 128

# Kernel bandwidth for the chemical-gradient sensing step. Standalone
# (not imported from physics.TENSION_RANGE) so it can be tuned
# independently later, but starts near that value: sensing radius
# roughly matching the radius nodes can actually physically interact
# within is a reasonable default until there's a reason to diverge.
SENSING_SIGMA = 1.15

# Scaffold-only safety valve, not part of the spec: MAX_NODES is a hard
# backstop against the O(n^2) physics solver blowing up, independent of
# (and in addition to) the energy gate below — split decisions past this
# cap are still made and logged into state but simply not acted on.
MAX_NODES = 400

# --- Energy: a per-node growth budget, gating (not replacing) the
# network's own split probability. See module docstring above. ---
MIN_SPLIT_ENERGY = 75.0
MAX_ENERGY = 100.0
ENERGY_INJECTION = 1.0
ENERGY_INJECTION_NOISE = 0.75

# Numerical safety: an untrained network's per-step deltas are
# completely unbounded, and both chemicals and id feed additively into
# next step's own input — a plain feedback loop with no cap, which
# reliably diverges given enough steps (confirmed directly: id norm grew
# 0.97 -> 5.45 over 15 steps from a *freshly random* — not even
# mutated — network). Once id_vectors overflow to inf,
# physics._tension_compatibility's norm-based division produces
# inf/inf = NaN, which propagates into positions and crashes
# chamfer_distance's cKDTree on the resulting non-finite input — this is
# what actually took down a live training run.
#
# chemicals are clipped to a generous bounded range (they're read as
# plain values, so a hard clip is the natural fix). id_vectors are
# renormalized to unit length instead: every consumer only ever reads id
# via cosine similarity (direction, never magnitude — see
# physics.py's _tension_compatibility and cell_state.py's docstring), so
# normalizing doesn't discard anything meaningful and makes overflow
# structurally impossible rather than just delayed.
CHEMICAL_CLIP = 10.0


class UpdateRule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = 3 * NUM_CHEMICAL_CHANNELS + 1  # + energy
        output_dim = 1 + NUM_CHEMICAL_CHANNELS + ID_DIM + SPAWN_DIR_DIM
        # tanh, not ReLU: a hidden unit dominated by a large bias is
        # unbounded on ReLU's positive side (feeds fc2 an arbitrarily
        # large, near-input-independent value) but capped at [-1, 1] on
        # tanh regardless of how large its pre-activation gets — fc2 (see
        # export_weights' docstring, still unactivated) can only ever see
        # a bounded contribution from a saturated unit instead of an
        # unbounded one. Doesn't cap fc2's own output, just shrinks the
        # space of ways a single dominant hidden unit can blow it up. The
        # usual reason to prefer ReLU (avoiding vanishing gradients deep
        # in a backprop-trained net) doesn't apply — this net is trained
        # by evolutionary mutation/selection, not backprop, and is one
        # hidden layer deep either way.
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(
        self,
        chemicals: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([chemicals, grad_x, grad_y, energy.unsqueeze(-1)], dim=-1)
        out = self.net(x)
        split_logit = out[:, 0]
        chemical_delta = out[:, 1 : 1 + NUM_CHEMICAL_CHANNELS]
        id_delta = out[:, 1 + NUM_CHEMICAL_CHANNELS : 1 + NUM_CHEMICAL_CHANNELS + ID_DIM]
        spawn_direction = out[:, 1 + NUM_CHEMICAL_CHANNELS + ID_DIM :]
        return split_logit, chemical_delta, id_delta, spawn_direction

    @torch.no_grad()
    def step_numpy(
        self,
        chemicals: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        energy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """numpy in, numpy out, no grad tracking — this is a forward-only
        scaffold, not a training loop. `energy` is expected pre-normalized
        (see step()'s normalized_energy) — this method doesn't know or
        care what it represents, same as chemicals/grad_x/grad_y.
        `spawn_direction` is returned raw (un-normalized) — normalizing a
        possibly-near-zero vector is the caller's call to make (see
        step()'s dir_norms_safe), not this method's."""
        chemicals_t = torch.from_numpy(chemicals).float()
        grad_x_t = torch.from_numpy(grad_x).float()
        grad_y_t = torch.from_numpy(grad_y).float()
        energy_t = torch.from_numpy(energy).float()
        split_logit, chemical_delta, id_delta, spawn_direction = self.forward(
            chemicals_t, grad_x_t, grad_y_t, energy_t
        )
        return (
            torch.sigmoid(split_logit).numpy(),
            chemical_delta.numpy(),
            id_delta.numpy(),
            spawn_direction.numpy(),
        )

    def export_weights(self) -> dict:
        """JSON-ready weights for a from-scratch forward-pass
        reimplementation elsewhere (the frontend's sim/updateRule.ts) —
        nn.Linear stores weight as (out_features, in_features), i.e.
        `y = x @ W.T + b`; keep that orientation on the receiving end
        rather than transposing here, so both sides can cite this same
        shape convention."""
        fc1, fc2 = self.net[0], self.net[2]
        return {
            "fc1w": fc1.weight.detach().tolist(),
            "fc1b": fc1.bias.detach().tolist(),
            "fc2w": fc2.weight.detach().tolist(),
            "fc2b": fc2.bias.detach().tolist(),
        }


def step(graph: Graph, update_rule: UpdateRule, rng: Optional[np.random.Generator] = None) -> bool:
    """Run one autonomous simulation step: every node senses, decides,
    and acts from the same pre-step snapshot; mutates `graph` in place
    (updated chemicals/id/energy, plus any new children) but does not
    relax physics — the caller does that afterward, same as a manual
    split_node, and only when this returns True (nothing moved if
    nothing split, so re-relaxing an already-settled graph is wasted
    work — see physics.py callers).

    `rng` controls the two stochastic decisions below (energy injection
    noise, split Bernoulli draw) — pass a seeded `np.random.Generator`
    from any caller that needs a rollout to be reproducible given a seed
    (evolve.py's rollout(), so a candidate's fitness reflects its
    weights and not which way the coin flips landed this run — see
    evolve.py's own docstring on why fitness noise defeats selection).
    Omit it (the default) for callers that don't care, e.g. main.py's
    live interactive websocket, which wants fresh randomness on every
    step, not reproducibility — falls back to numpy's global random
    state, exactly this function's original behavior."""
    n = len(graph.positions)
    if n == 0:
        return False

    draw_uniform = rng.uniform if rng is not None else np.random.uniform
    draw_random = rng.random if rng is not None else np.random.random

    positions = graph.positions_array()
    chemicals = graph.chemicals_array()
    id_vectors = graph.id_array()

    # Energy regenerates before this step's decision is made, so a node
    # senses (and can act on) its own post-injection level, not last
    # step's stale one.
    noise = draw_uniform(-ENERGY_INJECTION_NOISE, ENERGY_INJECTION_NOISE, size=n)
    injected_energy = np.clip(graph.energy_array() + ENERGY_INJECTION + noise, 0.0, MAX_ENERGY)
    normalized_energy = (injected_energy / MAX_ENERGY) * 2.0 - 1.0

    _, gradients = weighted_field_and_gradient(positions, positions, chemicals, SENSING_SIGMA)
    grad_x = gradients[:, :, 0]
    grad_y = gradients[:, :, 1]

    split_prob, chemical_delta, id_delta, spawn_direction = update_rule.step_numpy(
        chemicals, grad_x, grad_y, normalized_energy
    )

    # The network's own probability is a ceiling, not the final word:
    # energy_weight is 0 below MIN_SPLIT_ENERGY (hard gate) and ramps
    # linearly to 1 at MAX_ENERGY, so low-energy nodes can *never* split
    # regardless of how confident the network is.
    energy_weight = np.clip(
        (injected_energy - MIN_SPLIT_ENERGY) / (MAX_ENERGY - MIN_SPLIT_ENERGY), 0.0, 1.0
    )
    effective_split_prob = split_prob * energy_weight
    should_split = draw_random(n) < effective_split_prob

    new_chemicals = np.clip(chemicals + chemical_delta, -CHEMICAL_CLIP, CHEMICAL_CLIP)

    new_id_raw = id_vectors + id_delta
    id_norms = np.linalg.norm(new_id_raw, axis=1, keepdims=True)
    id_norms_safe = np.where(id_norms < 1e-9, 1.0, id_norms)
    new_id = new_id_raw / id_norms_safe

    # Unlike id, spawn_direction isn't added to a running state — it's a
    # fresh reading every step ("which way would I spawn right now"), so
    # there's no previous value to add to, just this step's raw output
    # normalized for storage/display. A near-zero raw output has no
    # direction; graph.add_child (below) falls back to a random angle for
    # the degenerate case rather than dividing by ~0 here.
    dir_norms = np.linalg.norm(spawn_direction, axis=1, keepdims=True)
    dir_norms_safe = np.where(dir_norms < 1e-9, 1.0, dir_norms)
    spawn_dir_unit = spawn_direction / dir_norms_safe

    for i in range(n):
        graph.chemicals[i] = new_chemicals[i]
        graph.id_vectors[i] = new_id[i]
        graph.energy[i] = float(injected_energy[i])
        graph.spawn_directions[i] = spawn_dir_unit[i]
        graph.split_probs[i] = float(split_prob[i])

    # Splitting happens after every node's delta/energy is written back,
    # using the now-current (post-delta, post-injection) state — and
    # iterates only over the original snapshot's node count, so children
    # spawned this step don't themselves get a chance to split again
    # until next step.
    did_split = False
    for i in range(n):
        if should_split[i] and len(graph.positions) < MAX_NODES:
            split_energy = graph.energy[i] / 2.0
            graph.energy[i] = split_energy
            graph.add_child(
                i,
                new_id[i].copy(),
                new_chemicals[i].copy(),
                split_energy,
                direction=spawn_direction[i],
                rng=rng,
            )
            did_split = True

    return did_split

"""The per-node update rule: Dense(128) -> tanh -> Dense(20), evaluated
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

Velocity & heading: every node has a persistent 2D velocity
(Graph.velocity) — the *only* motion state stored; heading isn't stored
at all, it's derived on demand as `atan2(vy, vx)` (a node at rest,
velocity exactly (0, 0), has an arbitrary-but-well-defined heading of 0
until it actually starts moving — same convention atan2 itself uses).
Replaces the earlier heading+speed+angular_velocity representation (3
persistent scalars, heading/speed integrated from two separate
accelerations via an extra hop through angular_velocity) with a single
2D vector and a direct one-hop integration — physically, "facing" was
never really independent information, it's just velocity's direction,
so storing it separately was redundant state that could (and did) drift
inconsistently from the thing actually driving motion.

- The network's last MOTION_DIM output slots are a 2D *acceleration*,
  expressed in the node's own *local* frame (forward = current heading
  direction, lateral = 90° left of it) — not world x/y. Each component
  is independently tanh-squashed then scaled by MAX_ACCEL before being
  rotated into world coordinates (the inverse of the sensing rotation
  below — same heading, opposite direction) and added to velocity. A
  near-zero raw output means "barely change velocity," not "snap to some
  fixed rate."
- velocity's *magnitude* is clamped to MAX_SPEED every step — rescaling
  the whole vector when it's over the limit, not clamping vx/vy
  independently (which would let the diagonal case exceed MAX_SPEED by
  up to sqrt(2)x). Unbounded accumulation would otherwise diverge
  exactly like chemicals/id would without their own clip/renormalization
  — see CHEMICAL_CLIP's comment below for the same failure mode.
- Motion is derived from the *updated* velocity, not read back as raw
  network output — `graph.positions[i] += velocity` — applied straight
  to position the same way an earlier "strafe" mechanism used to
  (bypassing physics.relax() entirely; the node moving itself, as
  opposed to relax() moving it in reaction to neighbors). Pinned nodes
  still have their velocity integrated (so a later "Move" tool release
  doesn't jump-start motion from stale state) but are excluded from the
  position write itself, for the same reason strafe was: relax()'s
  free_mask only ever stops a *relax* correction from moving a pinned
  node, it never undoes a displacement already written straight to
  graph.positions before relax() runs.
- Sensing reads the chemical gradient in each node's own *local* frame
  (forward = heading direction, lateral = 90° left of it) instead of
  world x/y — see step()'s rotation of substrate.weighted_field_and_gradient's
  output before it reaches the network. This is the actual point of
  deriving a heading at all: a node doesn't need to separately learn
  "world gradient direction X means turn this way" for every possible
  absolute orientation, only "gradient forward-left means turn left,"
  which transfers regardless of which way it's currently facing —
  rotation-equivariant sensing, the same reason a lot of steering/flocking
  models (boids, ant pheromone-following) work in a body-relative frame
  rather than a world-fixed one. The network's own output acceleration is
  expressed in this *same* frame (see above), so the two rotations
  (sensing in, acceleration out) are exact inverses of each other, using
  the identical heading computed once at the top of the step.

Energy: a per-node growth budget (Graph.energy), not part of the state
vector the network reads and writes freely, and not something it can
sense at all — the network only *emits a desire to split* (a
probability, from the same output slot as before), completely blind to
its own energy level; growth itself is externally rate-limited, not
something the network's output alone can override or even see coming.
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

from cell_state import ID_DIM, MOTION_DIM, NUM_CHEMICAL_CHANNELS, SPAWN_DIR_DIM
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
MAX_NODES = 600

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

# MAX_SPEED "tiny": a fraction of CONTACT_DISTANCE (the resting gap
# between two touching nodes), so one step's self-motion at full speed
# is a subtle nudge relative to the scale nodes actually interact/
# collide at, not a jump that could leapfrog a neighbor in one step.
# Physics still resolves whatever overlap self-motion causes on the next
# relax, same as any other position change. Bounds velocity's
# *magnitude*, not each component independently — see "Velocity &
# heading" in the module docstring. Not imported from
# physics.CONTACT_DISTANCE to avoid a dependency in that direction — see
# SENSING_SIGMA's own comment for why this module prefers a standalone
# constant here over reaching into physics.py for one.
MAX_SPEED = 0.05

# Reaching MAX_SPEED from a dead stop takes a few steps of sustained
# full acceleration (MAX_SPEED / MAX_ACCEL = 4), not one — that's the
# actual point of velocity being persistent state instead of an
# instantaneous per-step nudge like an earlier "strafe" mechanism was:
# motion has inertia, so a node has to "commit" to a direction for a few
# steps rather than being able to reverse itself completely step to
# step. Applied to each of the network's two local-frame acceleration
# components independently (tanh-squash then scale — see step()), not as
# a magnitude clamp on the raw acceleration vector.
MAX_ACCEL = MAX_SPEED / 4.0


class UpdateRule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = 3 * NUM_CHEMICAL_CHANNELS  # chemicals + grad_forward + grad_lateral, no energy
        output_dim = 1 + NUM_CHEMICAL_CHANNELS + ID_DIM + SPAWN_DIR_DIM + MOTION_DIM
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
        grad_forward: torch.Tensor,
        grad_lateral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`grad_forward`/`grad_lateral` are the chemical-gradient
        sensing input in the node's own local frame (forward = heading
        direction, lateral = 90° left of it), already rotated by the
        caller (step()) — this method itself is frame-agnostic, it just
        consumes whatever two gradient components it's handed. No energy
        input — see the module docstring's "Energy" section: the network
        is deliberately blind to its own energy level."""
        x = torch.cat([chemicals, grad_forward, grad_lateral], dim=-1)
        out = self.net(x)
        split_logit = out[:, 0]
        chemical_delta = out[:, 1 : 1 + NUM_CHEMICAL_CHANNELS]
        id_end = 1 + NUM_CHEMICAL_CHANNELS + ID_DIM
        id_delta = out[:, 1 + NUM_CHEMICAL_CHANNELS : id_end]
        spawn_dir_end = id_end + SPAWN_DIR_DIM
        spawn_direction = out[:, id_end:spawn_dir_end]
        # local_accel: a 2D acceleration in the node's own local frame
        # (column 0 = forward, column 1 = lateral), applied to the
        # persistent velocity after being rotated into world coordinates
        # — see "Velocity & heading" in the module docstring. Same
        # (N, 2)-block treatment as spawn_direction just above, not two
        # separate scalar outputs.
        local_accel = out[:, spawn_dir_end : spawn_dir_end + MOTION_DIM]
        return split_logit, chemical_delta, id_delta, spawn_direction, local_accel

    @torch.no_grad()
    def step_numpy(
        self,
        chemicals: np.ndarray,
        grad_forward: np.ndarray,
        grad_lateral: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """numpy in, numpy out, no grad tracking — this is a forward-only
        scaffold, not a training loop. `spawn_direction`/`local_accel`
        are both returned raw (un-normalized/un-squashed) — turning them
        into an actual direction or a bounded, world-frame acceleration
        is the caller's call to make (see step()'s dir_norms_safe and
        MAX_ACCEL usage), not this method's."""
        chemicals_t = torch.from_numpy(chemicals).float()
        grad_forward_t = torch.from_numpy(grad_forward).float()
        grad_lateral_t = torch.from_numpy(grad_lateral).float()
        split_logit, chemical_delta, id_delta, spawn_direction, local_accel = self.forward(
            chemicals_t, grad_forward_t, grad_lateral_t
        )
        return (
            torch.sigmoid(split_logit).numpy(),
            chemical_delta.numpy(),
            id_delta.numpy(),
            spawn_direction.numpy(),
            local_accel.numpy(),
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
    (updated chemicals/id/energy/heading/speed/position, plus any new
    children) but does not relax physics — the caller does that
    afterward, same as a manual split_node, and only when this returns
    True. Returns whether anything actually moved/changed this step (a
    split, or any unpinned node's own motion — see "Heading & speed" in
    the module docstring, which makes this true on almost every step now,
    not just split steps) — False only when there's nothing to relax
    (empty graph, or somehow every node is pinned), so the caller can
    skip pointless relax work.

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
    velocity = graph.velocity_array()

    # heading is never stored — derived fresh each step from the
    # *current* velocity (see "Velocity & heading" in the module
    # docstring). A node at rest (velocity exactly (0, 0)) gets heading
    # 0, the same convention np.arctan2 itself uses — arbitrary but
    # well-defined, and only matters until the node actually starts
    # moving under its own acceleration.
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])

    # Energy regenerates before this step's split-gate is computed, so a
    # node's effective_split_prob below reflects its post-injection
    # level, not last step's stale one — the network itself never sees
    # this value (see "Energy" in the module docstring), only the
    # external gate does.
    noise = draw_uniform(-ENERGY_INJECTION_NOISE, ENERGY_INJECTION_NOISE, size=n)
    injected_energy = np.clip(graph.energy_array() + ENERGY_INJECTION + noise, 0.0, MAX_ENERGY)

    _, gradients = weighted_field_and_gradient(positions, positions, chemicals, SENSING_SIGMA)
    grad_x = gradients[:, :, 0]
    grad_y = gradients[:, :, 1]

    # Rotate the world-frame gradient into each node's own local frame
    # (forward = current heading, lateral = 90° left of it) before it
    # reaches the network — see "Velocity & heading" in the module
    # docstring for why. This is an ordinary 2D rotation by -heading,
    # applied per node (broadcasting heading's (n,) shape against
    # grad_x/grad_y's (n, NUM_CHEMICAL_CHANNELS)) and per chemical
    # channel identically.
    cos_h = np.cos(heading)[:, None]
    sin_h = np.sin(heading)[:, None]
    grad_forward = grad_x * cos_h + grad_y * sin_h
    grad_lateral = -grad_x * sin_h + grad_y * cos_h

    split_prob, chemical_delta, id_delta, spawn_direction, local_accel = update_rule.step_numpy(
        chemicals, grad_forward, grad_lateral
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

    # tanh-squash each of the two local-frame acceleration components
    # independently before scaling — a near-zero raw output should mean
    # "barely change velocity," not snap to some fixed rate — see
    # "Velocity & heading" in the module docstring. Rotate the resulting
    # local (forward, lateral) acceleration into world (x, y) using the
    # *same* heading sensing used, the exact inverse of that rotation
    # (world = R(+heading) . local, vs. sensing's world = R(-heading)...
    # local): accel_x = forward*cos - lateral*sin,
    # accel_y = forward*sin + lateral*cos.
    accel_local = np.tanh(local_accel) * MAX_ACCEL
    accel_forward = accel_local[:, 0]
    accel_lateral = accel_local[:, 1]
    accel_world = np.stack(
        [
            accel_forward * cos_h[:, 0] - accel_lateral * sin_h[:, 0],
            accel_forward * sin_h[:, 0] + accel_lateral * cos_h[:, 0],
        ],
        axis=-1,
    )

    # Clamp velocity's *magnitude*, not vx/vy independently (which would
    # let the diagonal case exceed MAX_SPEED by up to sqrt(2)x) — rescale
    # the whole vector when it's over the limit, direction unchanged.
    new_velocity_raw = velocity + accel_world
    speed = np.linalg.norm(new_velocity_raw, axis=1, keepdims=True)
    speed_safe = np.where(speed < 1e-9, 1.0, speed)
    scale = np.minimum(1.0, MAX_SPEED / speed_safe)
    new_velocity = new_velocity_raw * scale

    for i in range(n):
        graph.chemicals[i] = new_chemicals[i]
        graph.id_vectors[i] = new_id[i]
        graph.energy[i] = float(injected_energy[i])
        graph.spawn_directions[i] = spawn_dir_unit[i]
        graph.split_probs[i] = float(split_prob[i])
        # velocity updates regardless of pinned status (same as
        # chemicals/id/energy above) — only the resulting *position*
        # write is skipped for a pinned node, exactly the pattern an
        # earlier "strafe" mechanism used. relax()'s free_mask only ever
        # stops a pinned node from being moved by a relax correction, it
        # never undoes a displacement already written straight to
        # graph.positions before relax() even runs — skipping the write
        # here is what actually keeps a pinned node fixed.
        graph.velocity[i] = new_velocity[i]
        graph.accel[i] = accel_world[i]
        if i not in graph.pinned:
            graph.positions[i] = positions[i] + new_velocity[i]

    # Splitting happens after every node's delta/energy is written back,
    # using the now-current (post-delta, post-injection, post-motion)
    # state — and iterates only over the original snapshot's node count,
    # so children spawned this step don't themselves get a chance to
    # split again until next step.
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

    changed = did_split or any(i not in graph.pinned for i in range(n))
    return changed

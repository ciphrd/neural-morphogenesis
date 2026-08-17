from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cell_state import (
    ID_DIM,
    INITIAL_ENERGY,
    NUM_CHEMICAL_CHANNELS,
    SPAWN_DIR_DIM,
    random_chemical,
    random_id,
)
from physics import CONTACT_DISTANCE


@dataclass
class Graph:
    """Owns node positions plus each node's internal state (`id_vectors`,
    `chemicals`, `energy`, `spawn_directions` — see cell_state.py /
    update_rule.py) — no topology to track: growth is "split a node,
    physics figures out the rest," so the graph is a point set,
    per-point internal state, and which points are pinned in place.
    """

    positions: list[np.ndarray] = field(default_factory=list)
    id_vectors: list[np.ndarray] = field(default_factory=list)
    chemicals: list[np.ndarray] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    # Latest spawn-direction reading per node (unit 2D vector, or the
    # zero vector for a node that hasn't run through the update rule
    # yet) — see update_rule.py's module docstring for why this isn't
    # part of id/chemicals' accumulated state.
    spawn_directions: list[np.ndarray] = field(default_factory=list)
    # Latest split-probability reading per node (the network's own raw
    # sigmoid(split_logit), *before* update_rule.py's energy gate scales
    # it down) — 0.0 for a node that hasn't run through the update rule
    # yet, same "fresh reading every step, not accumulated state"
    # treatment as spawn_directions and for the same reason: it's not
    # something the network reads back next step.
    split_probs: list[float] = field(default_factory=list)
    # Persistent, accumulated state — same footing as id_vectors/chemicals,
    # not a fresh-every-step reading like spawn_directions/split_probs.
    # Each entry is a (2,) array [vx, vy] in world coordinates — the
    # *only* motion state stored; there is no separate heading field.
    # "Which way is this node facing" is always derived on demand as
    # atan2(vy, vx) (see update_rule.py's step()), never read from here
    # directly. See update_rule.py's "Velocity & heading" docstring
    # section for the full mechanism.
    velocity: list[np.ndarray] = field(default_factory=list)
    # Latest world-frame acceleration reading per node — the network's
    # raw per-step output (after tanh-squash + MAX_ACCEL scaling +
    # local-to-world rotation, see update_rule.py's step()), *before*
    # it's added to velocity. Same "fresh reading every step, not
    # accumulated state" treatment as spawn_directions/split_probs — this
    # value itself isn't read back next step, only used to derive the
    # new velocity, which *is* the accumulated state. Exists purely for
    # display (see render/GraphRenderer.tsx's always-on heading/accel
    # ticks) — nothing computational reads it back from here.
    accel: list[np.ndarray] = field(default_factory=list)
    pinned: set[int] = field(default_factory=set)

    @classmethod
    def seed(cls, rng: Optional[np.random.Generator] = None) -> "Graph":
        # `rng` makes the seed node's own random id/chemicals
        # reproducible given a seed — without it, this was the dominant
        # remaining source of a rollout not being reproducible even after
        # update_rule.step()'s own randomness was fixed (confirmed
        # directly: same weights, same seed, still non-deterministic
        # fitness, traced to Graph.seed() drawing fresh random state
        # every call). See random_id/random_chemical's own docstrings.
        graph = cls()
        graph._add_node(np.array([0.0, 0.0]), rng=rng)
        return graph

    def _add_node(
        self,
        position: np.ndarray,
        id_vector: Optional[np.ndarray] = None,
        chemicals: Optional[np.ndarray] = None,
        energy: Optional[float] = None,
        spawn_direction: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        node_id = len(self.positions)
        self.positions.append(np.asarray(position, dtype=np.float64))
        self.id_vectors.append(
            np.asarray(id_vector, dtype=np.float64) if id_vector is not None else random_id(rng)
        )
        self.chemicals.append(
            np.asarray(chemicals, dtype=np.float64) if chemicals is not None else random_chemical(rng)
        )
        self.energy.append(energy if energy is not None else INITIAL_ENERGY)
        self.spawn_directions.append(
            np.asarray(spawn_direction, dtype=np.float64)
            if spawn_direction is not None
            else np.zeros(SPAWN_DIR_DIM)
        )
        self.split_probs.append(0.0)
        # Every new node starts at rest (velocity (0, 0), no randomness)
        # — heading being purely derived means there's no separate
        # "facing" to randomize the way random_id/random_chemical
        # randomize identity/chemicals; a stationary node's heading is
        # just atan2(0, 0) = 0 until it actually starts accelerating.
        self.velocity.append(np.asarray(velocity, dtype=np.float64) if velocity is not None else np.zeros(2))
        self.accel.append(np.asarray(accel, dtype=np.float64) if accel is not None else np.zeros(2))
        return node_id

    def add_child(
        self,
        parent_id: int,
        id_vector: np.ndarray,
        chemicals: np.ndarray,
        energy: float,
        direction: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Spawn a new node touching `parent_id`, with explicit state —
        used by the learned update rule, which computes the child's
        post-delta state (and its share of the parent's energy) itself
        rather than getting a fresh random one. `direction` (raw, not
        assumed normalized) steers where the child lands — the update
        rule's own learned spawn-direction output; a random angle is
        used instead when `direction` is omitted (the manual "Add node"
        tool never had a direction to offer) or too close to zero to
        normalize (the network expressed no preference this step).
        `rng`, when given, makes that fallback angle reproducible too
        (see random_id's docstring for why this matters for evolve.py) —
        drawn from numpy's Generator rather than the stdlib `random`
        module so it shares the same seeded stream as everything else a
        caller like update_rule.step() controls.

        The child starts at rest (velocity (0, 0)) regardless of which
        way `unit` pointed it — heading being purely derived from
        velocity means there's no "facing" to inherit separately from
        motion the way an earlier design had; it'll pick up its own
        heading the moment it actually starts accelerating, same as
        every other node."""
        origin = self.positions[parent_id]
        unit = None
        if direction is not None:
            norm = float(np.linalg.norm(direction))
            if norm >= 1e-9:
                unit = np.asarray(direction, dtype=np.float64) / norm
        if unit is None:
            draw_uniform = rng.uniform if rng is not None else random.uniform
            angle = draw_uniform(0.0, 2.0 * math.pi)
            unit = np.array([math.cos(angle), math.sin(angle)])
        offset = unit * CONTACT_DISTANCE
        return self._add_node(
            origin + offset,
            id_vector=id_vector,
            chemicals=chemicals,
            energy=energy,
            rng=rng,
        )

    def split_node(self, node_id: int) -> Optional[int]:
        """Externally-triggered split (the "Add node" tool): spawns a
        child that copies the parent's current state, same as a
        learned-rule-driven split would — including splitting the
        parent's current energy in half between the two, though this
        manual tool bypasses the energy *threshold* (an explicit
        override always succeeds, regardless of budget). No learned
        spawn direction to offer here, so the child lands at a random
        angle, same as before this existed. Physics (collision +
        tension) resolves whatever overlap or rearrangement that causes
        on the next relax."""
        if node_id < 0 or node_id >= len(self.positions):
            return None
        child_energy = self.energy[node_id] / 2.0
        self.energy[node_id] = child_energy
        return self.add_child(
            node_id, self.id_vectors[node_id].copy(), self.chemicals[node_id].copy(), child_energy
        )

    def remove_nodes(self, indices: set[int]) -> None:
        """Drops the given node indices (e.g. for damage training),
        keeping relative order of the survivors and remapping `pinned`
        to their new indices. Pinned nodes are never removable — silently
        excluded from `indices` rather than erroring, so callers don't
        need to filter them out themselves."""
        indices = indices - self.pinned
        if not indices:
            return
        keep = [i for i in range(len(self.positions)) if i not in indices]
        remap = {old: new for new, old in enumerate(keep)}
        self.positions = [self.positions[i] for i in keep]
        self.id_vectors = [self.id_vectors[i] for i in keep]
        self.chemicals = [self.chemicals[i] for i in keep]
        self.energy = [self.energy[i] for i in keep]
        self.spawn_directions = [self.spawn_directions[i] for i in keep]
        self.split_probs = [self.split_probs[i] for i in keep]
        self.velocity = [self.velocity[i] for i in keep]
        self.accel = [self.accel[i] for i in keep]
        self.pinned = {remap[i] for i in self.pinned}

    def positions_array(self) -> np.ndarray:
        if not self.positions:
            return np.zeros((0, 2))
        return np.stack(self.positions)

    def id_array(self) -> np.ndarray:
        if not self.id_vectors:
            return np.zeros((0, ID_DIM))
        return np.stack(self.id_vectors)

    def chemicals_array(self) -> np.ndarray:
        if not self.chemicals:
            return np.zeros((0, NUM_CHEMICAL_CHANNELS))
        return np.stack(self.chemicals)

    def energy_array(self) -> np.ndarray:
        return np.array(self.energy, dtype=np.float64)

    def spawn_direction_array(self) -> np.ndarray:
        if not self.spawn_directions:
            return np.zeros((0, SPAWN_DIR_DIM))
        return np.stack(self.spawn_directions)

    def velocity_array(self) -> np.ndarray:
        if not self.velocity:
            return np.zeros((0, 2))
        return np.stack(self.velocity)

    def set_positions(self, positions: np.ndarray) -> None:
        self.positions = [positions[i] for i in range(positions.shape[0])]

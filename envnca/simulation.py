"""Ties Environment + AgentState + UpdateRule into one autonomous step:
sense (from the environment), decide (the network), move, write back into
the environment. Same Jacobi-style "everyone senses/decides from one
consistent snapshot" semantics and the same local-frame sensing/
acceleration rotation as trainer/backend/update_rule.py's step(), but
agents read from and write to a shared GPU grid instead of an implicit
point-cloud field, and everything is expressed as batched torch ops over
the whole population at once (no Python loop over agents in the hot
path) rather than the old project's per-node numpy arrays, since that's
the only way this is actually fast on a GPU.

No growth in this version: the population is fixed at whatever size it's
seeded with (AgentState.seed, called once in __init__) and never changes
— no splitting, no energy budget, no death. This was pulled out
deliberately (it existed in an earlier pass of this project) rather than
left half-built: growth adds a dynamically-*shaped* population, which
means a GPU->CPU sync every step just to find out how many agents split
before torch.cat-ing the new ones on — see agent_state.py's own note on
this. With a fixed population, step() never needs to know anything about
tensor shapes on the CPU side at all; it's a pure same-shape-in,
same-shape-out GPU op end to end, no sync points. Reintroducing growth
later is a real option, but it comes back with that sync cost (or a
fixed-size-buffer-plus-alive-mask redesign to avoid it) — not free.

Not built yet, on purpose — this remains a first cut, scoped to what was
actually asked (env-based chemicals, gradient sensing, NN-driven writes,
GPU-resident, 512x512):
- No physics/collision pass. Agents can and will overlap; nothing pushes
  them apart. trainer/backend/physics.py's tension+collision solver is a
  substantial separate subsystem — adding it here (in grid-space, on GPU)
  is a reasonable next step but a distinct piece of work.
- No `id` vector / adhesion — see agent_state.py's docstring.
- Positions are clamped to stay inside the grid (there's a hard edge here
  the old project's boundless graph-space never had), rather than any
  more sophisticated boundary handling (wrap-around, reflection).
"""

from __future__ import annotations

from typing import Optional

import torch

from agent_state import AgentState
from environment import Environment
from update_rule import MAX_ACCEL, MAX_SPEED, MAX_STRAFE, UpdateRule

# Population size and seed-cluster jitter — see agent_state.py's seed()
# for why a nonzero spread matters.
DEFAULT_POPULATION = 1000
DEFAULT_SPAWN_SPREAD = 4.0

# Keeps every agent's sampled/deposited position strictly inside the
# grid's interior — see environment.py's deposit(), whose corner math
# assumes x1 = floor(x)+1 and y1 = floor(y)+1 are always valid indices.
# Public (not `_`-prefixed) so train_server.py can forward it to the
# frontend, which needs the exact same clamp to replicate this
# simulation's positions bit-for-bit.
EDGE_MARGIN = 1.001


class Simulation:
    def __init__(
        self,
        env: Environment,
        update_rule: UpdateRule,
        device: torch.device,
        population: int = DEFAULT_POPULATION,
        spawn_spread: float = DEFAULT_SPAWN_SPREAD,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        self.env = env
        self.update_rule = update_rule
        self.device = device
        center = (env.width / 2.0, env.height / 2.0)
        self.agents = AgentState.seed(population, center, spawn_spread, device, rng=rng)

    def _clamp_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.clone()
        positions[:, 0] = positions[:, 0].clamp(0.0, self.env.width - EDGE_MARGIN)
        positions[:, 1] = positions[:, 1].clamp(0.0, self.env.height - EDGE_MARGIN)
        return positions

    @torch.no_grad()
    def step(self) -> None:
        agents = self.agents
        if agents.n == 0:
            return

        # heading never stored — derived fresh from velocity, same
        # convention as trainer/backend (a resting agent, velocity exactly
        # zero, has heading 0 until it actually starts moving).
        heading = torch.atan2(agents.velocity[:, 1], agents.velocity[:, 0])
        cos_h = torch.cos(heading).unsqueeze(1)
        sin_h = torch.sin(heading).unsqueeze(1)

        value, grad_x, grad_y = self.env.sample_value_and_gradient(agents.positions)
        # Rotate world-frame gradient into each agent's local frame
        # (forward = heading, lateral = 90° left) — identical rotation to
        # trainer/backend/update_rule.py's step(), same rationale
        # (rotation-equivariant sensing).
        grad_forward = grad_x * cos_h + grad_y * sin_h
        grad_lateral = -grad_x * sin_h + grad_y * cos_h

        env_write, local_accel, local_strafe = self.update_rule(value, grad_forward, grad_lateral)

        # Local-frame acceleration -> world, exact inverse of the sensing
        # rotation above, then clamp the *magnitude* (not each component)
        # of the resulting velocity — same reasoning as trainer/backend.
        accel_local = torch.tanh(local_accel) * MAX_ACCEL
        accel_forward = accel_local[:, 0]
        accel_lateral = accel_local[:, 1]
        accel_world = torch.stack(
            [
                accel_forward * cos_h.squeeze(1) - accel_lateral * sin_h.squeeze(1),
                accel_forward * sin_h.squeeze(1) + accel_lateral * cos_h.squeeze(1),
            ],
            dim=-1,
        )
        new_velocity_raw = agents.velocity + accel_world
        speed = torch.linalg.norm(new_velocity_raw, dim=1, keepdim=True)
        speed_safe = torch.clamp(speed, min=1e-9)
        scale = torch.clamp(MAX_SPEED / speed_safe, max=1.0)
        new_velocity = new_velocity_raw * scale

        # Strafe: same local -> world rotation as accel (identical
        # heading), but squashed by *magnitude* rather than per-component
        # (see update_rule.py's MAX_STRAFE comment) since there's no
        # downstream integration step to fix up a diagonal overshoot the
        # way velocity's own magnitude clamp does for accel. Added
        # straight onto position — never onto velocity, never persisted —
        # see update_rule.py's "Strafe" docstring section for why.
        strafe_mag = torch.linalg.norm(local_strafe, dim=1, keepdim=True)
        strafe_mag_safe = torch.clamp(strafe_mag, min=1e-9)
        strafe_local = local_strafe * (torch.tanh(strafe_mag_safe) * MAX_STRAFE / strafe_mag_safe)
        strafe_forward = strafe_local[:, 0]
        strafe_lateral = strafe_local[:, 1]
        strafe_world = torch.stack(
            [
                strafe_forward * cos_h.squeeze(1) - strafe_lateral * sin_h.squeeze(1),
                strafe_forward * sin_h.squeeze(1) + strafe_lateral * cos_h.squeeze(1),
            ],
            dim=-1,
        )

        new_positions = self._clamp_to_grid(agents.positions + new_velocity + strafe_world)

        agents.velocity = new_velocity
        agents.positions = new_positions

        # Write to the environment at each agent's *new* position, then
        # let the grid's own dynamics (diffuse + decay) run once — mirrors
        # the old project's ordering (state updates fully written back
        # before anything downstream reacts to them).
        self.env.deposit(new_positions, env_write)
        self.env.step_dynamics()

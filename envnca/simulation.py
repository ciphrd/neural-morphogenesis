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
- No *real* physics/collision pass — trainer/backend/physics.py's
  tension+collision solver is a substantial separate subsystem, still not
  built here. What step() below does have, since repulsion.py: a cheap,
  soft, field-mediated push-apart force (see that module's own docstring
  for why this stopped being optional — a gradient-descent-trained
  policy was observed collapsing every agent onto a single point, a
  stable fixed point nothing else could break). Agents can still overlap
  more than a real collision solver would ever allow; this only makes
  full coincidence a repelled, unstable state instead of an attractive
  one, not physically forbidden.
- No `id` vector / adhesion — see agent_state.py's docstring.
"""

from __future__ import annotations

from typing import Optional

import torch

from agent_state import AgentState
from constants import (
    MAX_ACCEL,
    MAX_ENV_WRITE,
    MAX_SPEED,
    MAX_STRAFE,
    REPULSION_RESOLUTION,
    REPULSION_SIGMA,
    REPULSION_STRENGTH,
)
from environment import Environment
from repulsion import RepulsionField
from update_rule import UpdateRule

# Population size and seed-cluster jitter — see agent_state.py's seed()
# for why a nonzero spread matters.
DEFAULT_POPULATION = 1000
DEFAULT_SPAWN_SPREAD = 4.0


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
        # Own dedicated (coarser) field, independent of env's own (C,H,W)
        # grid — see repulsion.py's module docstring for why. Assumes a
        # square grid (env.width == env.height), same assumption the rest
        # of this codebase's CLI/training already makes.
        self.repulsion = RepulsionField(REPULSION_RESOLUTION, env.width, device)

    def _wrap_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        """Toroidal wrap, not a clamp — the grid has no edge (see
        environment.py's module docstring). torch.remainder is
        floor-style (result always in [0, size)), so this is a true wrap
        even for a position that overshot by more than one grid width in
        a single step (never expected given MAX_SPEED/MAX_STRAFE are both
        tiny relative to grid size, but correct regardless)."""
        positions = positions.clone()
        positions[:, 0] = torch.remainder(positions[:, 0], self.env.width)
        positions[:, 1] = torch.remainder(positions[:, 1], self.env.height)
        return positions

    def step(self) -> None:
        """No longer wrapped in @torch.no_grad() — that decorator used to
        make every call here untracked unconditionally, which is exactly
        wrong for gradient-based (backprop-through-time) training: it
        silently discarded the computation graph every step regardless
        of what the caller actually wanted. Whether this step is tracked
        is now the *caller's* choice, made the normal PyTorch way (an
        ambient `torch.no_grad()`/`torch.inference_mode()` context), not
        baked into this method. The existing evolutionary training path
        (evolve.py's rollout()) explicitly opts back into the old
        untracked-and-fast behavior by wrapping its own stepping loop in
        `torch.no_grad()` — nothing about its performance or memory use
        changes. A caller building a differentiable rollout instead calls
        this with gradients enabled (the default) and never detaches
        agents.positions/env.grid until it actually wants to stop
        backpropagating through them."""
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

        # cos_h/sin_h aren't passed into the network (see update_rule.py's
        # own "Heading" docstring section) — they're only used here, for
        # the sensing rotation above and the accel/strafe rotation below.
        env_write, local_accel, local_strafe = self.update_rule(value, grad_forward, grad_lateral)

        # Repulsion force, from this same pre-move snapshot of positions
        # (same Jacobi-style "everyone acts on one consistent snapshot"
        # semantics as sensing above) — see repulsion.py's own docstring.
        # World-frame already (this force has no notion of an agent's own
        # heading/local frame, unlike accel/strafe), so it's folded
        # straight into accel_world below with no rotation step of its
        # own.
        repulsion_world = self.repulsion.compute(agents.positions, REPULSION_SIGMA, REPULSION_STRENGTH)

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
        new_velocity_raw = agents.velocity + accel_world + repulsion_world
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

        new_positions = self._wrap_to_grid(agents.positions + new_velocity + strafe_world)

        agents.velocity = new_velocity
        agents.positions = new_positions

        # Write to the environment at each agent's *new* position, then
        # let the grid's own dynamics (diffuse + decay) run once — mirrors
        # the old project's ordering (state updates fully written back
        # before anything downstream reacts to them). Squashed here, not
        # inside UpdateRule.forward() — same division of labor
        # local_accel/local_strafe already have (the network returns raw
        # outputs, this method squashes/applies them) — see
        # constants.MAX_ENV_WRITE's own docstring for why this needed to
        # be bounded at all.
        env_write_squashed = torch.tanh(env_write) * MAX_ENV_WRITE
        self.env.deposit(new_positions, env_write_squashed)
        self.env.step_dynamics()

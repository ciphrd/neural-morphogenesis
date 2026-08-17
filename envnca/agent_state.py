"""Per-agent state living on the GPU: position and velocity — nothing
else. No chemicals, which now live in the environment (environment.py) —
that's the deliberate inversion from trainer/backend/cell_state.py, whose
`chemicals` field this project drops entirely. No `id` vector: the old
project's id-based adhesion is a physics mechanism, and this project has
no physics/collision pass at all yet — see simulation.py's module
docstring. No energy/growth state either — this version has a fixed
population, set once at seed() and never changed (no splitting, no
death) — see simulation.py's module docstring for why growth was pulled
out.

Every field is a single (N, ...) tensor, not a Python list of per-agent
arrays like the old cell_state.py's list-of-ndarrays — batching every
agent into one tensor is what lets update_rule.py evaluate the whole
population's forward pass as a handful of matmuls instead of a Python
loop, which is the only way this is actually fast on a GPU (a Python
loop over agents would serialize thousands of tiny GPU calls and be
*slower* than doing the same thing on CPU). With a fixed population,
these tensors also never change shape after seed() — every step is a
same-shape op on the same buffers, so there's nothing here that ever
needs a GPU->CPU sync.

velocity/heading follow trainer/backend's proven design unchanged: a
single persistent 2D velocity, heading always derived on demand as
atan2(vy, vx), never stored — see update_rule.py's own docstring here
for the full mechanism.
"""

from __future__ import annotations

from typing import Optional

import torch

MOTION_DIM = 2
# Same footing as MOTION_DIM: not persisted agent state, just network
# output width bookkeeping shared with update_rule.py/simulation.py. See
# simulation.py's step() for what strafe actually does — a per-step
# displacement applied straight to position, independent of (and never
# folded into) velocity.
STRAFE_DIM = 2


class AgentState:
    def __init__(self, positions: torch.Tensor, velocity: torch.Tensor) -> None:
        self.positions = positions  # (N, 2) float32, pixel coords
        self.velocity = velocity  # (N, 2) float32

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @staticmethod
    def seed(
        n: int,
        center: tuple[float, float],
        spread: float,
        device: torch.device,
        rng: Optional[torch.Generator] = None,
    ) -> "AgentState":
        """`n` agents jittered uniformly within `spread` pixels of `center`
        — not all placed exactly on top of each other. With identical
        starting positions every agent would sense an identical value/
        gradient (heading also starts at 0 for all), and a deterministic
        network maps identical input to identical output — the whole
        population would move in lockstep forever, one point acting as
        N, rather than N agents actually spreading out and diverging.
        This jitter is the only source of initial variation; everything
        after that (deposits landing at slightly different points, hence
        slightly different future readings) is what actually pulls
        trajectories apart.

        `rng` (a CPU torch.Generator, moved to `device` after drawing —
        not every backend's random op supports a device-local generator,
        and this is a one-off (n, 2) draw, not worth the uncertainty) is
        for callers that need this seed reproducible given the same
        weights + same seed, e.g. evolve.py's rollout() — the simulation
        itself has no other source of randomness (see simulation.py's
        step(), fully deterministic given agent state), so this jitter is
        the *only* thing standing between "same weights, same fitness
        every time" and per-run noise large enough to swamp real
        selection signal — trainer/backend/evolve.py hit exactly this
        problem once, from a different source (unseeded split draws), and
        its own docstring documents the fitness noise that caused. Omit
        (the default) for callers that don't need reproducibility, e.g.
        an interactive/live caller that wants fresh randomness on every
        restart."""
        cx, cy = center
        offsets = (torch.rand((n, 2), generator=rng) - 0.5) * 2.0 * spread
        base = torch.tensor([[cx, cy]], dtype=torch.float32)
        positions = (base + offsets).to(device)
        velocity = torch.zeros((n, 2), dtype=torch.float32, device=device)
        return AgentState(positions, velocity)

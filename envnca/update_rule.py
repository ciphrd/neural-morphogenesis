"""The per-agent update rule: Dense(128) -> tanh -> Dense(output), evaluated
identically for every agent from its own sensed local chemical value +
gradient (environment.Environment.sample_value_and_gradient). Same
architecture and forward-only-scaffold status as
trainer/backend/update_rule.py (weights are randomly initialized and
never optimized here — training is separate, later work), but the
input/output contract is inverted to match where chemicals now live:

- Input: value (C) + grad_forward (C) + grad_lateral (C) — the agent's own
  chemical *state* is gone from the input entirely (there isn't one
  anymore; see agent_state.py), replaced one-for-one by what the
  environment reads at the agent's position. Still 3*C total, same shape
  the old input had, just a different C-vector in the first third.
- Output: env_write (C) + local_accel (2) + local_strafe (2). `env_write`
  replaces the old `chemical_delta` + `id_delta` — instead of updating
  its own persistent chemicals/id, the agent decides what to *deposit
  into the environment* at its current position
  (environment.Environment.deposit). No split_logit and no
  spawn_direction: this version has a fixed population (no growth at
  all — see simulation.py's module docstring), so there's nothing for
  either output to drive.

Velocity & heading and the local-frame acceleration/sensing rotation are
carried over unchanged in spirit from trainer/backend/update_rule.py —
that mechanism has nothing to do with where chemicals live, so there was
no reason to redesign it. simulation.py does the per-step orchestration
(the rotation) the same way trainer/backend's step() function does; this
module only owns the network itself and the constants specific to it.

Strafe: `local_strafe`, a second, *independent* 2D local-frame vector —
same local (forward, lateral) convention as local_accel, but applied
straight to position every step (see simulation.py's step()), never
folded into velocity and never persisted. Where local_accel has inertia
(has to accumulate over several steps via velocity before it moves
anything much), strafe is an instant, un-accumulating nudge: whatever
this step's network output says, that's exactly how far the agent moves
this step from strafe, and it's forgotten immediately after — next
step's strafe is computed fresh from scratch, with no memory of this
one. Mirrors an equivalent mechanism trainer/backend used to have
(direct self-motion, bypassing physics entirely) before that project
replaced it with pure velocity/acceleration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agent_state import MOTION_DIM, STRAFE_DIM

HIDDEN_DIM = 128

# Grid-space equivalent of trainer/backend's MAX_SPEED/MAX_ACCEL — that
# project's world is a handful of unit-radius circles, this one is a
# 512px grid, so the constants are re-derived in this project's own
# units rather than reused: MAX_SPEED is still "a subtle nudge," here
# expressed as a small fraction of one grid cell's diagonal per step, and
# MAX_ACCEL keeps the same MAX_SPEED/4 relationship (reach full speed
# from rest in ~4 steps of sustained acceleration).
MAX_SPEED = 0.6
MAX_ACCEL = MAX_SPEED / 4.0

# Strafe doesn't accumulate step to step the way accel -> velocity does
# (see module docstring's "Strafe" section), so there's no multi-step
# buildup toward a ceiling the way velocity has — this is directly "how
# far can one step's strafe move an agent," same order of magnitude as
# MAX_SPEED (the velocity cap) rather than MAX_ACCEL.
MAX_STRAFE = 0.5


class UpdateRule(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        input_dim = 3 * num_channels  # value + grad_forward + grad_lateral
        output_dim = num_channels + MOTION_DIM + STRAFE_DIM
        # tanh hidden activation for the same reason as trainer/backend's
        # UpdateRule: bounds a single dominant unit's contribution to the
        # output layer, and this net is evolved (not backprop-trained), so
        # ReLU's usual vanishing-gradient advantage doesn't apply here.
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(
        self,
        value: torch.Tensor,
        grad_forward: torch.Tensor,
        grad_lateral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All inputs (N, C); local-frame rotation of grad_forward/
        grad_lateral is the caller's job (simulation.py), same as the old
        project — this method is frame-agnostic. Returns (env_write,
        local_accel, local_strafe), all raw/un-squashed — squashing and
        rotating both local_accel and local_strafe into world coordinates
        is the caller's job (simulation.py), same division of
        responsibility as the old project."""
        x = torch.cat([value, grad_forward, grad_lateral], dim=-1)
        out = self.net(x)
        c = self.num_channels
        env_write = out[:, :c]
        accel_end = c + MOTION_DIM
        local_accel = out[:, c:accel_end]
        local_strafe = out[:, accel_end : accel_end + STRAFE_DIM]
        return env_write, local_accel, local_strafe

    def export_weights(self) -> dict:
        """JSON-ready weights for a from-scratch forward-pass
        reimplementation elsewhere (the frontend's WebGPU agent pass) —
        nn.Linear stores weight as (out_features, in_features), i.e.
        `y = x @ W.T + b`; keep that orientation on the receiving end
        rather than transposing here, so both sides can cite this same
        shape convention. Same shape/convention as
        trainer/backend/update_rule.py's own export_weights()."""
        fc1, fc2 = self.net[0], self.net[2]
        return {
            "fc1w": fc1.weight.detach().cpu().tolist(),  # (HIDDEN_DIM, 3*num_channels)
            "fc1b": fc1.bias.detach().cpu().tolist(),  # (HIDDEN_DIM,)
            "fc2w": fc2.weight.detach().cpu().tolist(),  # (num_channels+4, HIDDEN_DIM)
            "fc2b": fc2.bias.detach().cpu().tolist(),  # (num_channels+4,)
        }

    @torch.no_grad()
    def randomize(self) -> None:
        """Re-draw every weight/bias from scratch, in place — same
        distribution nn.Linear itself uses at construction time
        (`reset_parameters()`, Kaiming-uniform), just triggered again on
        demand. In place on the existing tensors (not a fresh module) so
        the caller doesn't need to worry about re-attaching this to
        whatever device it was already on. Not called anywhere in this
        module itself — a capability for whatever's driving the agent
        (a training loop, or a future server endpoint) to reset a
        policy's weights without disturbing anything else (environment,
        agent positions) it's plugged into."""
        for module in self.net.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

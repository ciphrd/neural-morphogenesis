"""The evolved per-particle policy: Dense(128) -> sin -> Dense(16) —
architecture/shape reference and a CPU-only utility class (random weight
init via a fresh instance's own initialized parameters, JSON export via
export_weights()), NOT the live forward pass anymore. That now runs
entirely as a wgpu compute shader, core/agents.wgsl (see agents_gpu.py's
own AgentsGPU, and training_sim.py's own module docstring for why:
running this as torch/MPS ops required a real, blocking host round-trip
every macro step to bridge wgpu-native's own physics device and torch's
own MPS/CUDA device, which share no buffers). forward() below is kept as
a readable, executable reference for the exact math core/agents.wgsl's
own agentStep() implements — evolve.py/train_server.py never call it.
Hidden activation swapped from tanh to sin (a periodic activation) per
this project's own explicit request — see self.net's own comment below
for why, and why this doesn't need SIREN's own specific init/frequency
scheme to remain sound under this project's own mutation-based (not
gradient-based) optimizer.

Same architecture, and the same LOCAL (heading-relative) frame
convention envnca's own agents use: heading is core/agents.wgsl's own
persistent per-particle state (NOT derived from velocity — see that
file's own module docstring for why), which that shader uses to rotate
the sensed gradient into forward/lateral before this net ever sees it,
and this net's own strafe output is rotated back out to world frame by
that same shader afterward — see core/agents.wgsl's own module docstring
for the exact rotation (training_sim.py, unlike an earlier revision, no
longer does any of this itself — it only orchestrates GPU buffers/
pipelines now).

- Input: value (C) + grad_forward (C) + grad_lateral (C) — the
  *rotated*, local-frame gradient the caller (core/agents.wgsl, or this
  method's own torch equivalent if called directly) computes, 3*C total —
  plus the agent's own (x,y) domain position (2), RELATIVE TO THE
  ROLLOUT'S OWN SPAWN CENTER (not rotated into local frame — there's no
  meaningful "local-frame position" the way a gradient has one), for
  3*C+2 total. This module itself is frame-agnostic (it just concatenates
  whatever it's given) — the rotation (and spawn-center subtraction) is
  entirely the caller's job.
- Output: env_write (C * DEPOSIT_SPOTS) — one independent deposit value
  per channel for EACH of DEPOSIT_SPOTS=4 spots around the particle's own
  position (front/left/back/right relative to its own heading — see
  core/agents.wgsl's own agentStep() comment for the exact angles and
  DEPOSIT_DISTANCE for how far out), not just a single deposit at the
  particle's own position the way this used to work — plus angular_accel
  (1), accel (2), and strafe (2), all raw/local-frame. angular_accel
  nudges the particle's own persistent angular velocity (which in turn
  nudges its persistent heading — see core/agents.wgsl's own module
  docstring for why heading is no longer derived from velocity); strafe
  is rotated to world frame and nudges VELOCITY — an acceleration, damped
  by a FRICTION retention fraction, integrated forward into position by
  MpmCore's own physics (see core/agents.wgsl's own module docstring for
  the full history: this has flipped between velocity and a direct
  position nudge twice now). accel is a separate output channel, still
  produced (output width is unchanged) but currently unused. C=8
  (simulation_settings.CHEM_CHANNELS) and
  DEPOSIT_SPOTS=4 (simulation_settings.DEPOSIT_SPOTS) are what pin the
  output width at exactly 37: 8*4 + 1 + 2 + 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from simulation_settings import ACCEL_DIM, ANGULAR_DIM, CHEM_CHANNELS, DEPOSIT_SPOTS, HIDDEN_DIM, STRAFE_DIM

# The agent's own (x,y) spawn-center-relative position, appended after
# the per-channel value/grad_forward/grad_lateral triples — see this
# module's own module docstring and core/agents.wgsl's own IN_DIM
# comment. Hardcoded rather than added to simulation_settings.py, same
# convention ANGULAR_DIM/ACCEL_DIM/STRAFE_DIM already follow (fixed
# architecture constants, not CLI-configurable).
POSITION_DIM = 2


class Sin(nn.Module):
    """torch has no built-in sin activation module (unlike Tanh/ReLU) —
    this is the whole layer, nothing to configure. See UpdateRule.__init__'s
    own comment for why this replaced Tanh here specifically."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class UpdateRule(nn.Module):
    def __init__(self, num_channels: int = CHEM_CHANNELS) -> None:
        super().__init__()
        self.num_channels = num_channels
        input_dim = 3 * num_channels + POSITION_DIM
        output_dim = num_channels * DEPOSIT_SPOTS + ANGULAR_DIM + ACCEL_DIM + STRAFE_DIM
        # sin hidden activation, not ReLU or tanh: this net is evolved
        # (mutation + selection), never backprop-trained, so ReLU's usual
        # vanishing-gradient advantage doesn't apply, and neither does
        # tanh's own "bounds a single dominant unit's contribution"
        # rationale specifically — sin was swapped in instead per this
        # project's own explicit request, on the theory that a PERIODIC
        # activation is a more natural fit for a domain where heading/
        # rotation (themselves sin/cos-parameterized throughout
        # core/agents.wgsl's own sensing rotation, deposit-spot angles,
        # strafe rotation) are everywhere. Deliberately NOT a full SIREN
        # architecture (multiple sin-activated layers + SIREN's own
        # frequency-scaled init, ω₀) — SIREN's actual value proposition is
        # enabling STABLE, GRADIENT-BASED training of deep sinusoidal
        # stacks representing high-frequency continuous signals; this
        # network has neither property (shallow — one hidden layer — and
        # never gradient-trained at all), so that machinery wouldn't buy
        # anything here. This is a controlled, single-layer activation
        # swap only.
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            Sin(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(
        self, value: torch.Tensor, grad_forward: torch.Tensor, grad_lateral: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """value/grad_forward/grad_lateral are (N, C); position is (N, 2)
        — the agent's own (x,y), already spawn-center-relative and NOT
        rotated into local frame (see this module's own module docstring
        for why position has no local-frame rotation the way a gradient
        does). The local-frame rotation of grad_forward/grad_lateral, and
        the spawn-center subtraction for position, are both the caller's
        job (training_sim.py/core/agents.wgsl) — this method is frame-
        agnostic, it just concatenates whatever it's given. Returns
        (env_write, angular_accel, accel, strafe), all raw/un-squashed
        and still in LOCAL frame — squashing (tanh + the MAX_ANGULAR_ACCEL/
        MAX_ACCEL/MAX_STRAFE/MAX_ENV_WRITE scale), reshaping env_write's
        own flat (N, C*DEPOSIT_SPOTS) output into the (N, DEPOSIT_SPOTS, C)
        callers actually want one-spot-at-a-time, and rotating accel/
        strafe to world frame are all training_sim.py's/core/agents.wgsl's
        own job (this reference forward() only knows raw tensor shapes,
        not deposit-spot geometry), same division of responsibility
        envnca's own UpdateRule/Simulation split. angular_accel needs no
        rotation either way — it nudges the particle's own angular
        velocity directly, there's no "world frame" for a scalar turn
        rate to be rotated into."""
        x = torch.cat([value, grad_forward, grad_lateral, position], dim=-1)
        out = self.net(x)
        c = self.num_channels
        d = c * DEPOSIT_SPOTS
        env_write = out[:, :d]  # (N, DEPOSIT_SPOTS*C), spot-major — reshape to (N, DEPOSIT_SPOTS, C) if needed
        angular_accel = out[:, d : d + ANGULAR_DIM]
        accel = out[:, d + ANGULAR_DIM : d + ANGULAR_DIM + ACCEL_DIM]
        strafe = out[:, d + ANGULAR_DIM + ACCEL_DIM : d + ANGULAR_DIM + ACCEL_DIM + STRAFE_DIM]
        return env_write, angular_accel, accel, strafe

    def export_weights(self) -> dict:
        """JSON-ready weights for a from-scratch forward-pass
        reimplementation elsewhere (an eventual frontend replay, mirroring
        ../viewer/README.md's own staging) — nn.Linear stores weight as
        (out_features, in_features), i.e. `y = x @ W.T + b`; keep that
        orientation on the receiving end rather than transposing here,
        same convention envnca/update_rule.py's own export_weights()
        documents."""
        fc1, fc2 = self.net[0], self.net[2]
        return {
            "fc1w": fc1.weight.detach().cpu().tolist(),  # (HIDDEN_DIM, 3*num_channels+POSITION_DIM)
            "fc1b": fc1.bias.detach().cpu().tolist(),  # (HIDDEN_DIM,)
            "fc2w": fc2.weight.detach().cpu().tolist(),  # (num_channels*DEPOSIT_SPOTS+5, HIDDEN_DIM)
            "fc2b": fc2.bias.detach().cpu().tolist(),  # (num_channels*DEPOSIT_SPOTS+5,)
        }

"""The evolved per-particle policy: Dense(128) -> sin -> Dense(4*C+8) —
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
and this net's two former strafe outputs are rotated back out to world
frame by that same shader afterward — see core/agents.wgsl's own module docstring
for the exact rotation (training_sim.py, unlike an earlier revision, no
longer does any of this itself — it only orchestrates GPU buffers/
pipelines now).

- Input: value (C) + grad_forward (C) + grad_lateral (C), followed by
  morphology occupancy/forward-gradient/lateral-gradient (3) — the
  *rotated*, local-frame gradient the caller (core/agents.wgsl, or this
  method's own torch equivalent if called directly) computes, for 3*C+3 total.
  There is no absolute or spawn-relative position input. This module itself
  is frame-agnostic; the gradient rotation is entirely the caller's job.
- Output: env_write (4*C) — one independent deposit value per channel at
  each heading-relative cardinal spot (front/left/back/right) — plus angular_accel (1),
  anisotropy/polarity logits (2), direction (2), and RGB color logits (3),
  all raw/local-frame. angular_accel
  nudges the particle's own persistent angular velocity (which in turn
  nudges its persistent heading — see core/agents.wgsl's own module
  docstring for why heading is no longer derived from velocity); the two
  former strafe channels now encode a normalized local growth direction.
  The two former acceleration channels independently control tensor
  anisotropy and signed division bias through sigmoid. C=8
  (simulation_settings.CHEM_CHANNELS) and
  the output width is exactly 40: 4*8 + 1 + 2 + 2 + 3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from simulation_settings import ACCEL_DIM, ANGULAR_DIM, CHEM_CHANNELS, DEPOSIT_SPOTS, HIDDEN_DIM, STRAFE_DIM

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
        input_dim = 3 * num_channels + 6
        output_dim = num_channels * DEPOSIT_SPOTS + ANGULAR_DIM + ACCEL_DIM + STRAFE_DIM + 3
        # sin hidden activation, not ReLU or tanh: this net is evolved
        # (mutation + selection), never backprop-trained, so ReLU's usual
        # vanishing-gradient advantage doesn't apply, and neither does
        # tanh's own "bounds a single dominant unit's contribution"
        # rationale specifically — sin was swapped in instead per this
        # project's own explicit request, on the theory that a PERIODIC
        # activation is a more natural fit for a domain where heading/
        # rotation (themselves sin/cos-parameterized throughout
        # core/agents.wgsl's own sensing and strafe rotations) are
        # everywhere. Deliberately NOT a full SIREN
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
        self,
        value: torch.Tensor,
        grad_forward: torch.Tensor,
        grad_lateral: torch.Tensor,
        morphology: torch.Tensor,
        elastic_strain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chemical tensors are (N, C); morphology is (N, 3) containing
        occupancy, forward gradient, and lateral gradient; elastic_strain is
        (N, 3): volumetric, axial, and shear Hencky strain. The local-frame
        rotation of the gradients is the caller's job
        (training_sim.py/core/agents.wgsl); this method is frame-agnostic and
        simply concatenates the three channel blocks. Returns
        (env_write, angular_accel, growth_controls, direction, color), all raw/un-squashed
        and still in LOCAL frame — squashing (tanh + the MAX_ANGULAR_ACCEL/
        MAX_ACCEL/MAX_ENV_WRITE scale; controls use sigmoid and direction
        uses normalized tanh outputs), and rotating direction to world frame are all training_sim.py's/core/agents.wgsl's
        own job (this reference forward() only knows raw tensor shapes,
        not spatial deposit geometry), same division of responsibility
        envnca's own UpdateRule/Simulation split. angular_accel needs no
        rotation either way — it nudges the particle's own angular
        velocity directly, there's no "world frame" for a scalar turn
        rate to be rotated into."""
        x = torch.cat([value, grad_forward, grad_lateral, morphology, elastic_strain], dim=-1)
        out = self.net(x)
        c = self.num_channels
        d = c * DEPOSIT_SPOTS
        # (N, DEPOSIT_SPOTS*C), spot-major: front, left, back, right.
        env_write = out[:, :d]
        angular_accel = out[:, d : d + ANGULAR_DIM]
        growth_controls = out[:, d + ANGULAR_DIM : d + ANGULAR_DIM + ACCEL_DIM]
        direction = out[:, d + ANGULAR_DIM + ACCEL_DIM : d + ANGULAR_DIM + ACCEL_DIM + STRAFE_DIM]
        color_start = d + ANGULAR_DIM + ACCEL_DIM + STRAFE_DIM
        color = out[:, color_start : color_start + 3]
        return env_write, angular_accel, growth_controls, direction, color

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
            "fc1w": fc1.weight.detach().cpu().tolist(),  # (HIDDEN_DIM, 3*num_channels+6)
            "fc1b": fc1.bias.detach().cpu().tolist(),  # (HIDDEN_DIM,)
            "fc2w": fc2.weight.detach().cpu().tolist(),  # (num_channels*DEPOSIT_SPOTS+8, HIDDEN_DIM)
            "fc2b": fc2.bias.detach().cpu().tolist(),  # (num_channels*DEPOSIT_SPOTS+8,)
        }

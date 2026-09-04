"""The evolved stateless-128/stateful-64 policies with logical output heads —
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
The hidden activation is bounded, monotonic tanh. This replaces the earlier
experimental sine activation to make evolved responses smoother under input
changes and mutation.

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
  method's own torch equivalent if called directly) computes, followed by
  three elastic-strain inputs, for 3*C+6 total.
  There is no absolute or spawn-relative position input. This module itself
  is frame-agnostic; the gradient rotation and robust input normalization are
  entirely the caller's job.
- Output: env_write (C) — retained ABI name for one bounded signed chemical
  expression target per channel — plus desired heading (2), anisotropy/polarity logits (2),
  desired growth direction (2), and a signed division drive (1). Stateless-128 ends with RGB logits (3);
  stateful-64 instead ends with private-state residuals (8) and gates (8),
  all raw/local-frame. C=9 gives 33 inputs and 19 stateless outputs, or 41
  inputs and 32 stateful outputs. The desired-heading vector is converted by the shader
  into angular acceleration from its shortest local angular error; the two
  former strafe channels encode a desired local growth direction.
  The two former acceleration channels independently control tensor
  anisotropy and division bias through sigmoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from simulation_settings import CHEM_CHANNELS
from policy_parameters import (
    STATELESS_ARCHITECTURE,
    normalize_architecture,
    policy_heads,
    policy_has_recurrence,
    policy_hidden_dim,
    policy_input_dim,
    trunk_initialization,
)

class UpdateRule(nn.Module):
    def __init__(self, num_channels: int = CHEM_CHANNELS, architecture: str = STATELESS_ARCHITECTURE) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.architecture = normalize_architecture(architecture)
        self.hidden_dim = policy_hidden_dim(self.architecture)
        input_dim = policy_input_dim(num_channels, self.architecture)
        # Bounded, monotonic, zero-centered hidden response. This controller
        # is evolved, so smooth local behavior under mutation is preferable
        # to the periodic phase wrapping of the earlier sine experiment.
        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.activation = nn.Tanh()
        self.heads = nn.ModuleDict(
            {head.name: nn.Linear(self.hidden_dim, head.size) for head in policy_heads(num_channels, self.architecture)}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Head-aware initialization shared conceptually with the browser.

        Bias priors keep the population centered on forward directions,
        anisotropy near 0.2, and neutral scalar controls. Random trunk bias and
        head-aware Xavier gains provide phenotype diversity around those
        defaults without forcing safety-sensitive outputs to saturation.
        """
        trunk_gain, trunk_bias_jitter = trunk_initialization()
        nn.init.xavier_uniform_(self.input_layer.weight, gain=trunk_gain)
        nn.init.uniform_(self.input_layer.bias, -trunk_bias_jitter, trunk_bias_jitter)
        with torch.no_grad():
            for spec in policy_heads(self.num_channels, self.architecture):
                layer = self.heads[spec.name]
                nn.init.xavier_uniform_(layer.weight, gain=spec.weight_gain)
                center = torch.tensor(spec.bias_center, dtype=layer.bias.dtype, device=layer.bias.device)
                layer.bias.copy_(center)
                if spec.bias_jitter > 0.0:
                    layer.bias.add_(torch.empty_like(layer.bias).uniform_(-spec.bias_jitter, spec.bias_jitter))

    def forward(
        self,
        value: torch.Tensor,
        grad_forward: torch.Tensor,
        grad_lateral: torch.Tensor,
        morphology: torch.Tensor,
        elastic_strain: torch.Tensor,
        private_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chemical tensors are (N, C); morphology is (N, 3) containing
        occupancy, forward gradient, and lateral gradient; elastic_strain is
        (N, 3): volumetric, axial, and shear Hencky strain. The local-frame
        rotation of the gradients is the caller's job
        (training_sim.py/core/agents.wgsl); this method is frame-agnostic and
        simply concatenates the three channel blocks. Returns
        (env_write, heading_target, growth_controls, direction, tail), all raw/un-squashed;
        tail is RGB for stateless-128 or concatenated state residual/gate for stateful-64
        and still in LOCAL frame — squashing (tanh for vectors and chemical targets;
        sigmoid for scalar controls), conversion of the heading target to
        angular acceleration, and rotating growth direction to world frame are all training_sim.py's/core/agents.wgsl's
        own job (this reference forward() only knows raw tensor shapes,
        not transient spatial splat geometry), same division of responsibility
        envnca's own UpdateRule/Simulation split."""
        inputs = [value, grad_forward, grad_lateral, morphology, elastic_strain]
        if policy_has_recurrence(self.architecture):
            if private_state is None:
                raise ValueError("stateful policy requires private_state")
            inputs.append(private_state)
        x = torch.cat(inputs, dim=-1)
        hidden = self.activation(self.input_layer(x))
        env_write = self.heads["chemical"](hidden)
        heading_target = self.heads["heading"](hidden)
        growth_controls = torch.cat(
            [
                self.heads["anisotropy"](hidden),
                self.heads["division"](hidden),
                self.heads["divisionDrive"](hidden),
            ],
            dim=-1,
        )
        direction = self.heads["growthDirection"](hidden)
        tail = (
            self.heads["color"](hidden)
            if not policy_has_recurrence(self.architecture)
            else torch.cat([self.heads["stateDelta"](hidden), self.heads["stateGate"](hidden)], dim=-1)
        )
        return env_write, heading_target, growth_controls, direction, tail

    def concatenated_output_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logical heads concatenated in the stable GPU output order."""
        ordered = [self.heads[spec.name] for spec in policy_heads(self.num_channels, self.architecture)]
        return torch.cat([layer.weight for layer in ordered]), torch.cat([layer.bias for layer in ordered])

    def flat_parameters(self) -> torch.Tensor:
        """Canonical fc1w/fc1b/fc2w/fc2b representation used by WebGPU."""
        output_weight, output_bias = self.concatenated_output_parameters()
        return torch.cat(
            [
                self.input_layer.weight.reshape(-1),
                self.input_layer.bias,
                output_weight.reshape(-1),
                output_bias,
            ]
        )

    def load_flat_parameters(self, flat: torch.Tensor) -> None:
        """Inverse of flat_parameters(), preserving the public checkpoint ABI."""
        flat = flat.to(dtype=self.input_layer.weight.dtype, device=self.input_layer.weight.device).reshape(-1)
        input_weight_count = self.input_layer.weight.numel()
        input_bias_count = self.input_layer.bias.numel()
        output_weight_count = sum(layer.weight.numel() for layer in self.heads.values())
        output_bias_count = sum(layer.bias.numel() for layer in self.heads.values())
        expected = input_weight_count + input_bias_count + output_weight_count + output_bias_count
        if flat.numel() != expected:
            raise ValueError(f"expected {expected} policy parameters, got {flat.numel()}")

        cursor = 0
        with torch.no_grad():
            self.input_layer.weight.copy_(flat[cursor : cursor + input_weight_count].view_as(self.input_layer.weight))
            cursor += input_weight_count
            self.input_layer.bias.copy_(flat[cursor : cursor + input_bias_count])
            cursor += input_bias_count
            for spec in policy_heads(self.num_channels, self.architecture):
                layer = self.heads[spec.name]
                count = layer.weight.numel()
                layer.weight.copy_(flat[cursor : cursor + count].view_as(layer.weight))
                cursor += count
            for spec in policy_heads(self.num_channels, self.architecture):
                layer = self.heads[spec.name]
                count = layer.bias.numel()
                layer.bias.copy_(flat[cursor : cursor + count])
                cursor += count

    def export_weights(self) -> dict:
        """JSON-ready weights for a from-scratch forward-pass
        reimplementation elsewhere (an eventual frontend replay, mirroring
        ../viewer/README.md's own staging) — nn.Linear stores weight as
        (out_features, in_features), i.e. `y = x @ W.T + b`; keep that
        orientation on the receiving end rather than transposing here,
        same convention envnca/update_rule.py's own export_weights()
        documents."""
        fc2w, fc2b = self.concatenated_output_parameters()
        return {
            "fc1w": self.input_layer.weight.detach().cpu().tolist(),  # (HIDDEN_DIM, 3*num_channels+6)
            "fc1b": self.input_layer.bias.detach().cpu().tolist(),  # (HIDDEN_DIM,)
            "fc2w": fc2w.detach().cpu().tolist(),  # (architecture output width, hidden_dim)
            "fc2b": fc2b.detach().cpu().tolist(),  # (architecture output width,)
        }

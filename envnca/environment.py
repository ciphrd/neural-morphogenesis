"""GPU-resident chemical field: the thing agents sense and write to. This
is the architectural inversion from trainer/backend/substrate.py — there,
each node carries its own chemical vector and the field at any point is an
implicit Gaussian sum over all nodes, recomputed from scratch every query.
Here the field *is* the state: a dense (C, H, W) tensor living on the GPU,
mutated in place by whatever agents write to it, with its own dynamics
(diffusion + decay, see step_dynamics) independent of how many agents
exist or where they are. Agents no longer carry chemicals at all — see
agent_state.py.

Two things this module has to do that the old point-cloud field never
needed to, because "agent position" and "grid cell" are no longer the
same thing:
- sample_value_and_gradient(): read the field at arbitrary continuous
  positions, not just grid cell centers — bilinear interpolation via
  grid_sample, the standard differentiable way to do this on GPU.
- deposit(): the inverse — scatter a value *into* the field at a
  continuous position, splatted across the 4 surrounding cells weighted
  by the same bilinear coefficients a sample() at that position would
  use to read it back (so writing and then immediately reading close to
  where you wrote lands close to the written value, rather than
  snapping to a cell and losing sub-pixel position). deposit()'s
  scatter-add is the mathematical transpose of grid_sample's gather —
  same 4 corners, same weights, opposite direction.

Gradient is computed once per step for the *entire* grid via a fixed 3x3
Sobel-style depthwise convolution (one conv2d call, cost independent of
agent count) rather than per-agent finite differences — the same trick
Growing NCA's own perception step uses Sobel filters for, and the only
way "gradient sensing" stays cheap when thousands of agents might be
sampling it every step.

Diffusion + decay (step_dynamics) is this module's replacement for what
the old Gaussian-sum field gave for free: smooth, spatially-extended
influence that doesn't just sit exactly where a source last wrote it. A
dense grid with no spreading mechanism would only ever be non-zero
exactly where an agent has stood — nothing for a *nearby* agent's
gradient sensing to pick up. A mass-preserving blur (kernel sums to 1)
followed by a multiplicative decay is the standard reaction-diffusion /
pheromone-trail treatment (physarum sims, ant-trail models) for exactly
this problem.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DECAY = 0.98

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_BLUR = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


class Environment:
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        device: torch.device,
        decay: float = DECAY,
    ) -> None:
        self.height = height
        self.width = width
        self.channels = channels
        self.device = device
        self.decay = decay
        self.grid = torch.zeros((channels, height, width), dtype=torch.float32, device=device)

        # Depthwise: one (1,1,3,3) kernel repeated per channel, applied via
        # groups=channels — every channel diffuses/gradients independently,
        # channels never mix (matches the old field's per-channel Gaussian
        # sums, which also never mixed channels).
        self._kernel_x = _SOBEL_X.view(1, 1, 3, 3).repeat(channels, 1, 1, 1).to(device)
        self._kernel_y = _SOBEL_X.t().contiguous().view(1, 1, 3, 3).repeat(channels, 1, 1, 1).to(device)
        self._blur_kernel = _BLUR.view(1, 1, 3, 3).repeat(channels, 1, 1, 1).to(device)

    def _normalize(self, positions: torch.Tensor) -> torch.Tensor:
        """(M,2) pixel coords -> grid_sample's (1,1,M,2) normalized-[-1,1]
        coords. align_corners=True throughout this module so pixel index i
        in [0, size-1] maps to exactly 2*i/(size-1) - 1 — the mapping
        deposit()'s corner math below assumes."""
        x = positions[:, 0]
        y = positions[:, 1]
        nx = 2.0 * x / (self.width - 1) - 1.0
        ny = 2.0 * y / (self.height - 1) - 1.0
        return torch.stack([nx, ny], dim=-1).view(1, 1, -1, 2)

    def _sample_grid(self, grid_1chw: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        coords = self._normalize(positions)
        sampled = F.grid_sample(grid_1chw, coords, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.view(grid_1chw.shape[1], -1).T  # (M, C)

    def sample_value_and_gradient(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """positions: (M,2) pixel coords. Returns (value, grad_x, grad_y),
        each (M, C) — grad_x/grad_y in *world* frame; rotating into an
        agent's local frame is the caller's job (mirrors
        update_rule.py's step() rotating substrate.py's world-frame
        gradient), same reasoning as the old project: this module doesn't
        know anything about headings."""
        g = self.grid.unsqueeze(0)
        gx = F.conv2d(g, self._kernel_x, padding=1, groups=self.channels)
        gy = F.conv2d(g, self._kernel_y, padding=1, groups=self.channels)
        value = self._sample_grid(g, positions)
        grad_x = self._sample_grid(gx, positions)
        grad_y = self._sample_grid(gy, positions)
        return value, grad_x, grad_y

    def deposit(self, positions: torch.Tensor, values: torch.Tensor) -> None:
        """Scatter-add `values` (M, C) into the grid at `positions` (M, 2),
        bilinearly splatted across the 4 surrounding cells — see module
        docstring. Multiple agents landing in the same cell simply sum,
        the same "contributions add" convention the old
        weighted_field_and_gradient used."""
        x, y = positions[:, 0], positions[:, 1]
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1
        wx1 = x - x0
        wx0 = 1.0 - wx1
        wy1 = y - y0
        wy0 = 1.0 - wy1
        x0i = x0.long().clamp(0, self.width - 1)
        x1i = x1.long().clamp(0, self.width - 1)
        y0i = y0.long().clamp(0, self.height - 1)
        y1i = y1.long().clamp(0, self.height - 1)

        grid_flat = self.grid.view(self.channels, -1)
        values_t = values.T  # (C, M)
        for yi, xi, w in (
            (y0i, x0i, wx0 * wy0),
            (y0i, x1i, wx1 * wy0),
            (y1i, x0i, wx0 * wy1),
            (y1i, x1i, wx1 * wy1),
        ):
            flat_idx = yi * self.width + xi
            grid_flat.index_add_(1, flat_idx, values_t * w)

    def step_dynamics(self) -> None:
        """Mass-preserving blur (kernel sums to 1) then a flat decay — see
        module docstring's "Diffusion + decay" section for why this
        exists at all."""
        g = self.grid.unsqueeze(0)
        blurred = F.conv2d(g, self._blur_kernel, padding=1, groups=self.channels)
        self.grid = (blurred * self.decay).squeeze(0)

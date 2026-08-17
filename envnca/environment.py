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

The grid is toroidal: both axes wrap. There is no edge — position (-1, y)
and (width-1, y) are the same point, and sensing/deposit/diffusion all
treat it that way (no agent ever "hits a wall"; a Sobel/blur neighborhood
that would run off one side reads in from the other). This is why
sampling and deposit are hand-rolled bilinear gather/scatter below
instead of grid_sample: torch's grid_sample has no circular padding
mode, only zeros/border/reflection, none of which wrap.

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

from constants import DECAY

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

    def _corners(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(M,2) pixel coords -> the 4 surrounding integer cells + their
        bilinear weights, corner indices wrapped (not clamped) into
        [0, size) — the one piece of index math sample_value_and_gradient
        and deposit share, since (per module docstring) they're the
        mathematical transpose of each other and must agree on exactly
        which 4 cells a given position touches. `%` here is torch's
        floor-style remainder (always non-negative for a positive
        divisor), so this is correct even for the not-actually-expected
        case of a negative coordinate, not just the [0, size) values
        simulation.py's own wrap already guarantees."""
        x, y = positions[:, 0], positions[:, 1]
        x0f = torch.floor(x)
        y0f = torch.floor(y)
        wx1 = x - x0f
        wx0 = 1.0 - wx1
        wy1 = y - y0f
        wy0 = 1.0 - wy1
        x0i = x0f.long() % self.width
        x1i = (x0f.long() + 1) % self.width
        y0i = y0f.long() % self.height
        y1i = (y0f.long() + 1) % self.height
        return x0i, x1i, y0i, y1i, wx0, wx1, wy0, wy1

    def _sample_grid(self, grid_chw: torch.Tensor, corners) -> torch.Tensor:
        x0i, x1i, y0i, y1i, wx0, wx1, wy0, wy1 = corners
        flat = grid_chw.reshape(self.channels, -1)  # (C, H*W)
        v00 = flat[:, y0i * self.width + x0i]
        v10 = flat[:, y0i * self.width + x1i]
        v01 = flat[:, y1i * self.width + x0i]
        v11 = flat[:, y1i * self.width + x1i]
        out = v00 * (wx0 * wy0) + v10 * (wx1 * wy0) + v01 * (wx0 * wy1) + v11 * (wx1 * wy1)
        return out.T  # (M, C)

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
        g_wrapped = F.pad(g, (1, 1, 1, 1), mode="circular")
        gx = F.conv2d(g_wrapped, self._kernel_x, groups=self.channels).squeeze(0)
        gy = F.conv2d(g_wrapped, self._kernel_y, groups=self.channels).squeeze(0)
        corners = self._corners(positions)
        value = self._sample_grid(self.grid, corners)
        grad_x = self._sample_grid(gx, corners)
        grad_y = self._sample_grid(gy, corners)
        return value, grad_x, grad_y

    def deposit(self, positions: torch.Tensor, values: torch.Tensor) -> None:
        """Scatter-add `values` (M, C) into the grid at `positions` (M, 2),
        bilinearly splatted across the 4 surrounding cells — see module
        docstring. Multiple agents landing in the same cell simply sum,
        the same "contributions add" convention the old
        weighted_field_and_gradient used."""
        x0i, x1i, y0i, y1i, wx0, wx1, wy0, wy1 = self._corners(positions)
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
        g_wrapped = F.pad(g, (1, 1, 1, 1), mode="circular")
        blurred = F.conv2d(g_wrapped, self._blur_kernel, groups=self.channels)
        self.grid = (blurred * self.decay).squeeze(0)

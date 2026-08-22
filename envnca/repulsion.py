"""Cheap O(N) stand-in for pairwise agent-agent repulsion — splats every
agent's position as a Gaussian blob onto a dedicated, low-resolution
density field (independent of the main (C,H,W) chemical grid), computes
that field's gradient once via a fixed Sobel-style convolution (the same
technique environment.py's own sample_value_and_gradient uses for the
chemical field), and returns each agent's own local negative gradient —
"move away from wherever density is locally high" — sampled back at its
position. No agent ever looks at another agent directly: total cost is
O(N * window) for splatting plus O(resolution^2) for the gradient, not
O(N^2) for a real pairwise force.

Ported from exploration/pages/repulsion-field.html, a standalone JS
sandbox built to interactively tune this idea before committing to any
specific constants — see that page for the live version this mirrors.

Fills a gap simulation.py's own module docstring already named up front:
"No physics/collision pass. Agents can and will overlap; nothing pushes
them apart." This is that missing push — a soft, field-mediated one, not
a real collision solver — and it turned out not to be optional: a
gradient-descent-trained policy was observed collapsing every agent onto
a single point, a genuinely *stable fixed point* of the shared-weights,
purely-local-sensing dynamics. Every agent runs the exact same
deterministic function; two agents at (nearly) the same position sense
(nearly) identically and therefore move (nearly) identically, forever —
no loss-function change can break that symmetry on its own (see
raster.py's rasterize_points_sum docstring for the fitness-side half of
this fix, which makes collapse score badly but can't prevent it from
happening). Only something that lets an agent's own motion depend on
*where other agents are* can actually break it — this module is that.

Independent field resolution, not reused from the main grid: the whole
point is staying cheap, and a repulsion signal doesn't need anywhere
near the fidelity the chemical field does — a coarse, blurry "which
direction is crowded" is enough. `--repulsion-resolution` (or
constants.REPULSION_RESOLUTION as a starting default) controls exactly
that resolution/cost tradeoff.

Differentiable end to end (scatter_add is out-of-place, conv2d is an
ordinary autograd op, bilinear sampling below matches environment.py's
own _corners/_sample_grid pattern) — sigma/strength aren't learned
parameters, but agent positions flow through this every step of a
gradient-descent rollout and need to stay traceable for BPTT to keep
working past it, the same requirement that drove environment.py's own
deposit() fix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0


class RepulsionField:
    """Owns one dedicated density-field buffer's worth of scratch state
    (really just the fixed Sobel kernels — the density field itself is
    rebuilt fresh from scratch on every compute() call, no persistence
    across steps, matching a plain instantaneous repulsion force rather
    than a decaying/trailing one). `resolution`/`grid_size` are
    structural (fix the field's own lattice size and its mapping to the
    simulation's world-pixel space) and set once at construction, same
    footing as Environment's own height/width/channels; `sigma`/
    `strength` are passed into compute() itself, not fixed here, since
    those are exactly the two knobs meant to be live-tunable (see
    simulation.py's own step() for how they get there)."""

    def __init__(self, resolution: int, grid_size: float, device: torch.device) -> None:
        self.resolution = resolution
        self.grid_size = grid_size
        self.device = device
        # World-pixel-space (envnca's main grid coordinates) -> this
        # field's own (independent, usually coarser) cell space.
        self._scale = resolution / grid_size
        self._kernel_x = _SOBEL_X.view(1, 1, 3, 3).to(device)
        self._kernel_y = _SOBEL_X.t().contiguous().view(1, 1, 3, 3).to(device)

    def _splat(self, positions: torch.Tensor, sigma: float) -> torch.Tensor:
        """Sum-combined Gaussian splat of `positions` onto a fresh
        (resolution, resolution) density field, toroidal-wrapped to
        match environment.py's own no-edge convention. Sum, not max —
        see raster.rasterize_points_sum's own docstring for why sum is
        the right combine mode whenever density itself (not just "is
        anyone nearby at all") is the actual quantity being measured;
        that reasoning applies here even more directly, since the entire
        point of this field is to measure crowding."""
        device = positions.device
        dtype = positions.dtype
        resolution = self.resolution
        n = positions.shape[0]
        if n == 0:
            return torch.zeros((resolution, resolution), dtype=dtype, device=device)

        cx = positions[:, 0] * self._scale
        cy = positions[:, 1] * self._scale
        # Discrete window center — like environment.py's own bilinear
        # corner indices, this is a discrete choice of *which* cells a
        # point's kernel can touch, not a value being learned; only the
        # kernel weights computed from cx/cy below need to stay
        # traceable for backprop.
        ix = torch.round(cx).detach().long()
        iy = torch.round(cy).detach().long()

        radius = max(1, math.ceil(3.0 * sigma))
        offsets = torch.arange(-radius, radius + 1, device=device)
        w = offsets.shape[0]

        row_grid = iy[:, None] + offsets[None, :]
        col_grid = ix[:, None] + offsets[None, :]
        dy = row_grid.to(dtype) - cy[:, None]
        dx = col_grid.to(dtype) - cx[:, None]
        kernel = torch.exp(-(dy[:, :, None] ** 2 + dx[:, None, :] ** 2) / (2.0 * sigma * sigma))

        # Toroidal wrap (`%`), not clamp — the field covers the same
        # no-edge world the main grid does.
        row_idx = row_grid[:, :, None].expand(n, w, w) % resolution
        col_idx = col_grid[:, None, :].expand(n, w, w) % resolution
        flat_idx = (row_idx * resolution + col_idx).reshape(-1)
        flat_kernel = kernel.reshape(-1)

        field_flat = torch.zeros(resolution * resolution, dtype=dtype, device=device)
        field_flat = field_flat.scatter_add(0, flat_idx, flat_kernel)
        return field_flat.view(resolution, resolution)

    def compute(self, positions: torch.Tensor, sigma: float, strength: float) -> torch.Tensor:
        """Returns a (N, 2) world-frame force: `strength` times the
        negative gradient of the density field, bilinearly sampled at
        each agent's own position — "push away from wherever it's
        locally crowded," O(N) per call after the one-time O(resolution^2)
        gradient pass. `strength == 0` (the default everywhere this is
        wired in) short-circuits to an exact zero force without ever
        touching the field, so existing behavior is completely unchanged
        until a caller explicitly turns this on."""
        if positions.shape[0] == 0 or strength == 0.0:
            return torch.zeros_like(positions)

        density = self._splat(positions, sigma)
        d = density.unsqueeze(0).unsqueeze(0)
        d_wrapped = F.pad(d, (1, 1, 1, 1), mode="circular")
        gx = F.conv2d(d_wrapped, self._kernel_x).squeeze(0).squeeze(0)
        gy = F.conv2d(d_wrapped, self._kernel_y).squeeze(0).squeeze(0)

        # Same bilinear-corner sampling pattern as environment.py's own
        # _corners/_sample_grid, just against this field's own
        # (independent) resolution/scale instead of the main grid's.
        cx = positions[:, 0] * self._scale
        cy = positions[:, 1] * self._scale
        x0f = torch.floor(cx)
        y0f = torch.floor(cy)
        wx1 = cx - x0f
        wx0 = 1.0 - wx1
        wy1 = cy - y0f
        wy0 = 1.0 - wy1
        x0i = x0f.long() % self.resolution
        x1i = (x0f.long() + 1) % self.resolution
        y0i = y0f.long() % self.resolution
        y1i = (y0f.long() + 1) % self.resolution

        def _sample(field: torch.Tensor) -> torch.Tensor:
            return (
                field[y0i, x0i] * (wx0 * wy0)
                + field[y0i, x1i] * (wx1 * wy0)
                + field[y1i, x0i] * (wx0 * wy1)
                + field[y1i, x1i] * (wx1 * wy1)
            )

        grad_x = _sample(gx)
        grad_y = _sample(gy)
        return torch.stack([-grad_x, -grad_y], dim=-1) * strength

"""GPU-resident chemical field, Python (wgpu-py) port of
../viewer/src/gpu/environment.ts, wrapping the exact same shared shader
module (../core/environment.wgsl) mpm_core.py already loads its own
core/*.wgsl passes from via shader_template.load_core_shader(). Replaces
environment.py's own torch/MPS implementation — see training_sim.py's
own module docstring for why: torch/MPS and wgpu-native/Metal are two
separate GPU frameworks that share no buffers, so bridging them required
a real, blocking host round-trip every macro step. Running this on the
SAME wgpu device/queue as MpmCore's own physics removes that crossing
entirely.

One instance is built ONCE per training run (like MpmCore itself — see
evolve.py's own module docstring on why rebuilding wgpu pipelines per
candidate is real, avoidable overhead) and reset() between rollouts,
mirroring environment.ts's own instance lifetime exactly (constructed
once in simulation.ts's rebuild(), reset via restartRollout()) rather
than environment.py's own "a fresh Environment each rollout" pattern
(which was fine for a cheap torch tensor, not for wgpu shader modules/
pipelines).
"""
from __future__ import annotations

import numpy as np
import wgpu

from chemical_channels import (
    ChemicalChannelProfile,
    channel_shader_constants,
    homogeneous_channel_profiles,
    packed_offsets,
    resolve_channel_profiles,
    resolved_dimensions,
)
from mpm_core import GRID_N, ceil_div, flat_dispatch_2d
from policy_parameters import (
    CELL_OWNED_PROJECTION_ARCHITECTURE,
    PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    normalize_chemical_communication_architecture,
)
from shader_template import load_core_shader

CLEAR_WORKGROUP = 256
GRID_WORKGROUP = 16
class EnvironmentGPU:
    """Public buffers (`buffers`, `gradient`, `deposit_scratch`) mirror
    environment.ts's own public surface — agents_gpu.py's own AgentsGPU
    builds its bind groups directly against these, same reasoning
    environment.ts's own docstring gives for why they're public there."""

    def __init__(
        self, device: wgpu.GPUDevice, channels: int, width: int, height: int,
        decay: float, deposit_rate: float,
        chemical_communication_architecture: str = CELL_OWNED_PROJECTION_ARCHITECTURE,
        normalize_deposits_by_local_density: bool = False,
        deposit_density_reference: float = 1.0,
        *,
        grid_velocity: wgpu.GPUBuffer | None = None,
        advection_dt: float = 0.0,
        channel_profiles: tuple[ChemicalChannelProfile, ...] | None = None,
    ) -> None:
        self.device = device
        self.channels = channels
        self.width = width
        self.height = height
        # Explicit profiles opt into the developmental layout.  Omission stays
        # homogeneous for legacy checkpoints and focused single-field checks.
        self.channel_profiles = resolve_channel_profiles(
            channels,
            channel_profiles if channel_profiles is not None else homogeneous_channel_profiles(channels),
        )
        self.channel_widths, self.channel_heights = resolved_dimensions(
            width, height, self.channel_profiles
        )
        self.channel_offsets, self.total_values = packed_offsets(
            self.channel_widths, self.channel_heights
        )
        self.shader_constants = channel_shader_constants(width, height, self.channel_profiles)
        self.max_width = max(self.channel_widths)
        self.max_height = max(self.channel_heights)
        self.chemical_communication_architecture = normalize_chemical_communication_architecture(
            chemical_communication_architecture
        )
        self.base_decay = float(decay)
        self.base_deposit_rate = float(deposit_rate)
        self.normalize_deposits_by_local_density = bool(normalize_deposits_by_local_density)
        self.deposit_density_reference = max(0.0, float(deposit_density_reference))
        self.advection_dt = max(0.0, float(advection_dt))
        # Small standalone shader checks do not always construct an MpmCore.
        # A zero fallback retains their old diffusion-only behavior.
        self._owns_grid_velocity = grid_velocity is None
        self.grid_velocity = grid_velocity if grid_velocity is not None else device.create_buffer(
            size=(GRID_N + 1) * (GRID_N + 1) * 2 * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        total = self.total_values
        # Numerator + matched density + one shared adaptive fixed-point scale.
        scratch_total = total * 2 + 1
        f32 = 4

        self.buffers = [
            device.create_buffer(
                size=total * f32,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            ),
            device.create_buffer(
                size=total * f32,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            ),
        ]
        self.gradient = device.create_buffer(size=total * 2 * f32, usage=wgpu.BufferUsage.STORAGE)
        self.deposit_scratch = device.create_buffer(size=scratch_total * f32, usage=wgpu.BufferUsage.STORAGE)
        self._physics_uniform = device.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.set_communication_timestep(1, 1.0)

        module = device.create_shader_module(code=load_core_shader(
            "environment.wgsl",
            {"CHANNELS": channels, "GRID_N": GRID_N, **self.shader_constants},
        ))

        self._clear_scratch_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "clearScratch"}
        )
        self._clear_scratch_bind_group = device.create_bind_group(
            layout=self._clear_scratch_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 2, "resource": {"buffer": self.deposit_scratch, "offset": 0, "size": self.deposit_scratch.size}}],
        )
        self._materialize_splat_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "materializeSplat"}
        )
        self._materialize_splat_bind_group = device.create_bind_group(
            layout=self._materialize_splat_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.buffers[0], "offset": 0, "size": self.buffers[0].size}},
                {"binding": 2, "resource": {"buffer": self.deposit_scratch, "offset": 0, "size": self.deposit_scratch.size}},
                {"binding": 4, "resource": {"buffer": self._physics_uniform, "offset": 0, "size": self._physics_uniform.size}},
            ],
        )

        self._compute_gradient_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "computeGradient"}
        )
        self._merge_deposit_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "mergeDeposit"}
        )
        self._diffuse_decay_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "diffuseDecay"}
        )

        self._compute_gradient_bind_groups = [
            device.create_bind_group(
                layout=self._compute_gradient_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.buffers[p], "offset": 0, "size": self.buffers[p].size}},
                    {"binding": 1, "resource": {"buffer": self.gradient, "offset": 0, "size": self.gradient.size}},
                ],
            )
            for p in (0, 1)
        ]
        self._merge_deposit_bind_groups = [
            device.create_bind_group(
                layout=self._merge_deposit_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.buffers[p], "offset": 0, "size": self.buffers[p].size}},
                    {"binding": 2, "resource": {"buffer": self.deposit_scratch, "offset": 0, "size": self.deposit_scratch.size}},
                    {"binding": 4, "resource": {"buffer": self._physics_uniform, "offset": 0, "size": self._physics_uniform.size}},
                ],
            )
            for p in (0, 1)
        ]
        self._diffuse_decay_bind_groups = [
            device.create_bind_group(
                layout=self._diffuse_decay_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.buffers[p], "offset": 0, "size": self.buffers[p].size}},
                    {"binding": 3, "resource": {"buffer": self.buffers[1 - p], "offset": 0, "size": self.buffers[1 - p].size}},
                    {"binding": 4, "resource": {"buffer": self._physics_uniform, "offset": 0, "size": self._physics_uniform.size}},
                    {"binding": 5, "resource": {"buffer": self.grid_velocity, "offset": 0, "size": self.grid_velocity.size}},
                ],
            )
            for p in (0, 1)
        ]

        self._clear_dispatch = flat_dispatch_2d(scratch_total, CLEAR_WORKGROUP)
        self._grid_dispatch = (
            ceil_div(self.max_width, GRID_WORKGROUP),
            ceil_div(self.max_height, GRID_WORKGROUP),
            channels,
        )

        self._parity = 0

    @property
    def parity(self) -> int:
        return self._parity

    def set_communication_timestep(self, rounds: int, speed: float) -> float:
        """Configure one field evolution and return each neural round's dt."""
        macro_dt = max(0.0, float(speed))
        neural_dt = macro_dt / max(1, int(rounds))
        decay = max(0.0, min(1.0, self.base_decay)) ** macro_dt
        self.device.queue.write_buffer(
            self._physics_uniform,
            0,
            np.array(
                [
                    decay,
                    self.base_deposit_rate * macro_dt,
                    min(macro_dt, 1.0),
                    1.0 if self.normalize_deposits_by_local_density else 0.0,
                    self.deposit_density_reference,
                    self.advection_dt, 0.0, 0.0,
                ],
                dtype=np.float32,
            ),
        )
        return neural_dt

    def set_deposit_normalization(self, enabled: bool, density_reference: float) -> None:
        """Configure matching-kernel capacity normalization."""
        self.normalize_deposits_by_local_density = bool(enabled)
        self.deposit_density_reference = max(0.0, float(density_reference))

    def set_advection_timestep(self, dt: float) -> None:
        """Set the elapsed mechanical time represented by the velocity grid."""
        self.advection_dt = max(0.0, float(dt))
        self.device.queue.write_buffer(
            self._physics_uniform, 5 * 4,
            np.array([self.advection_dt], dtype=np.float32),
        )

    def reset(self) -> None:
        """Zeroes both grid buffers and resets parity to 0 — call at the
        start of every rollout, same as environment.ts's own reset()
        (and what a fresh trainer/environment.py Environment instance
        used to give for free each Python-side rollout, before this
        class started being reused across rollouts instead)."""
        zeros = np.zeros(self.total_values, dtype=np.float32)
        self.device.queue.write_buffer(self.buffers[0], 0, zeros)
        self.device.queue.write_buffer(self.buffers[1], 0, zeros)
        self._parity = 0

    def encode_clear(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Remove every splat from the previous communication round."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._clear_scratch_pipeline)
        p.set_bind_group(0, self._clear_scratch_bind_group)
        p.dispatch_workgroups(*self._clear_dispatch)
        p.end()

    def encode_sense(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Materialize current cell splats and compute their shared gradient."""
        if self.chemical_communication_architecture == CELL_OWNED_PROJECTION_ARCHITECTURE:
            p = encoder.begin_compute_pass()
            p.set_pipeline(self._materialize_splat_pipeline)
            p.set_bind_group(0, self._materialize_splat_bind_group)
            p.dispatch_workgroups(*self._clear_dispatch)
            p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(self._compute_gradient_pipeline)
        p.set_bind_group(0, self._compute_gradient_bind_groups[self._parity])
        p.dispatch_workgroups(*self._grid_dispatch)
        p.end()

    def encode_prepare_persistent(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Advect/diffuse/decay the field before the policy senses it."""
        if self.chemical_communication_architecture != PERSISTENT_ENVIRONMENT_ARCHITECTURE:
            return
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._diffuse_decay_pipeline)
        p.set_bind_group(0, self._diffuse_decay_bind_groups[self._parity])
        p.dispatch_workgroups(*self._grid_dispatch)
        p.end()
        self._parity = 1 - self._parity

    def encode_merge_persistent(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Add this tick's final signed policy deltas to the prepared field."""
        if self.chemical_communication_architecture != PERSISTENT_ENVIRONMENT_ARCHITECTURE:
            return
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._merge_deposit_pipeline)
        p.set_bind_group(0, self._merge_deposit_bind_groups[self._parity])
        p.dispatch_workgroups(*self._clear_dispatch)
        p.end()

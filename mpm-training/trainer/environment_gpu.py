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

from mpm_core import ceil_div
from shader_template import load_core_shader

CLEAR_WORKGROUP = 256
GRID_WORKGROUP = 16


class EnvironmentGPU:
    """Public buffers (`buffers`, `gradient`, `deposit_scratch`) mirror
    environment.ts's own public surface — agents_gpu.py's own AgentsGPU
    builds its bind groups directly against these, same reasoning
    environment.ts's own docstring gives for why they're public there."""

    def __init__(self, device: wgpu.GPUDevice, channels: int, width: int, height: int, decay: float, deposit_rate: float) -> None:
        self.device = device
        self.channels = channels
        self.width = width
        self.height = height

        total = width * height * channels
        f32 = 4

        self.buffers = [
            device.create_buffer(size=total * f32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            device.create_buffer(size=total * f32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        ]
        self.gradient = device.create_buffer(size=total * 2 * f32, usage=wgpu.BufferUsage.STORAGE)
        self.deposit_scratch = device.create_buffer(size=total * f32, usage=wgpu.BufferUsage.STORAGE)
        # 8 bytes — core/environment.wgsl's own EnvPhysics struct is
        # {decay, depositRate}, both f32, in that order.
        self._physics_uniform = device.create_buffer(size=8, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_physics(decay, deposit_rate)

        module = device.create_shader_module(code=load_core_shader("environment.wgsl", {"CHANNELS": channels, "WIDTH": width, "HEIGHT": height}))

        self._clear_scratch_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "clearScratch"}
        )
        self._clear_scratch_bind_group = device.create_bind_group(
            layout=self._clear_scratch_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 2, "resource": {"buffer": self.deposit_scratch, "offset": 0, "size": self.deposit_scratch.size}}],
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
                ],
            )
            for p in (0, 1)
        ]

        self._clear_dispatch = ceil_div(total, CLEAR_WORKGROUP)
        self._grid_dispatch = (ceil_div(width, GRID_WORKGROUP), ceil_div(height, GRID_WORKGROUP), channels)

        self._parity = 0

    @property
    def parity(self) -> int:
        return self._parity

    def set_physics(self, decay: float, deposit_rate: float) -> None:
        """Writes both EnvPhysics fields at once (they share one small
        uniform buffer — see core/environment.wgsl's own struct) — safe
        to call any time, a plain buffer write, no pipeline recreation."""
        self.device.queue.write_buffer(self._physics_uniform, 0, np.array([decay, deposit_rate], dtype=np.float32))

    def reset(self) -> None:
        """Zeroes both grid buffers and resets parity to 0 — call at the
        start of every rollout, same as environment.ts's own reset()
        (and what a fresh trainer/environment.py Environment instance
        used to give for free each Python-side rollout, before this
        class started being reused across rollouts instead)."""
        total = self.width * self.height * self.channels
        zeros = np.zeros(total, dtype=np.float32)
        self.device.queue.write_buffer(self.buffers[0], 0, zeros)
        self.device.queue.write_buffer(self.buffers[1], 0, zeros)
        self._parity = 0

    def encode_sense(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Sense: clearScratch + computeGradient over the current grid.
        Call once per macro step, before the NN forward pass reads it.
        Encodes into `encoder`, does not submit."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._clear_scratch_pipeline)
        p.set_bind_group(0, self._clear_scratch_bind_group)
        p.dispatch_workgroups(self._clear_dispatch)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(self._compute_gradient_pipeline)
        p.set_bind_group(0, self._compute_gradient_bind_groups[self._parity])
        p.dispatch_workgroups(*self._grid_dispatch)
        p.end()

    def encode_merge_and_decay(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Diffuse+decay the CURRENT grid (as left by the previous macro
        step, before this step's own deposit touches anything) into the
        other buffer, then merge the NN forward pass's fresh deposit
        directly on top of that already-decayed result — deliberately
        decay-THEN-deposit, not deposit-then-decay (this used to run in
        the opposite order, decaying a step's own brand-new deposit
        before it was ever sensed by anyone, so a deposit's own value
        never actually reached its own full depositRate*value magnitude
        at any sensed step — see this method's own git history/PR
        discussion for the exact math). Flips parity at the end, same as
        before. Call once per macro step, after the NN forward pass has
        written into deposit_scratch. Encodes into `encoder`, does not
        submit.

        Both passes' own WGSL bodies (core/environment.wgsl's own
        mergeDeposit()/diffuseDecay()) are UNCHANGED — mergeDeposit's own
        binding 0 doesn't know or care which physical buffer it's bound
        to, it just adds scratch onto whatever's there. The entire
        reordering lives here: diffuseDecay dispatches FIRST now (reading
        buffers[self._parity], the pre-deposit "current" grid, writing
        the decayed+blurred result into buffers[1-self._parity]), then
        mergeDeposit dispatches SECOND, using
        self._merge_deposit_bind_groups[1-self._parity] — NOT
        self._parity — so its own binding 0 targets that same
        just-decayed buffer (what diffuseDecay just wrote), adding this
        step's own deposit on top of it, undecayed."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._diffuse_decay_pipeline)
        p.set_bind_group(0, self._diffuse_decay_bind_groups[self._parity])
        p.dispatch_workgroups(*self._grid_dispatch)
        p.end()

        p = encoder.begin_compute_pass()
        p.set_pipeline(self._merge_deposit_pipeline)
        p.set_bind_group(0, self._merge_deposit_bind_groups[1 - self._parity])
        p.dispatch_workgroups(self._clear_dispatch)
        p.end()

        self._parity = 1 - self._parity

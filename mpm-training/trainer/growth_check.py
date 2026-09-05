"""Focused GPU checks for field-integrated growth and resampling.

Run from this directory with ``.venv/bin/python growth_check.py``.
"""
from __future__ import annotations

import numpy as np
import wgpu

from agents_gpu import PARTICLE_META_BUFFER_OFFSET, AgentsGPU, weight_layout
from device import pick_device
from environment_gpu import EnvironmentGPU
from density import SPATIAL_RANDOM_CELLS
from mpm_core import DT, GRID_N, REPULSION_FIELD_N, MpmCore, ceil_div
from policy_parameters import PERSISTENT_ENVIRONMENT_ARCHITECTURE, STATEFUL_128_ARCHITECTURE
from training_sim import TrainingRollout


def _rest_state(
    jp: np.ndarray,
    growth: np.ndarray,
    cycle: np.ndarray,
    growth_f: np.ndarray | None = None,
    growth_direction: np.ndarray | None = None,
    anisotropy: np.ndarray | None = None,
    division_bias: np.ndarray | None = None,
    appearance_scale: np.ndarray | None = None,
) -> np.ndarray:
    count = len(jp)
    out = np.zeros((count, 12), dtype=np.float32)
    if growth_f is None:
        root = np.sqrt(np.asarray(growth, dtype=np.float32))
        out[:, 0] = root
        out[:, 3] = root
    else:
        out[:, :4] = np.asarray(growth_f, dtype=np.float32).reshape(count, 4)
    out[:, 4] = jp
    out[:, 5] = cycle
    if growth_direction is not None:
        direction = np.asarray(growth_direction, dtype=np.float32)
        out[:, 6] = np.arctan2(direction[:, 1], direction[:, 0])
    if anisotropy is not None:
        out[:, 7] = np.asarray(anisotropy, dtype=np.float32)
    if division_bias is not None:
        out[:, 8] = np.asarray(division_bias, dtype=np.float32)
    out[:, 10] = 1.0 if appearance_scale is None else np.asarray(appearance_scale, dtype=np.float32)
    return out


def _probe(core: MpmCore, count: int) -> np.ndarray:
    """Returns [Je, g, cycleActive] without adding COPY_SRC to hot buffers."""
    shader = core.device.create_shader_module(
        code="""
        struct Rest { growthF: vec4<f32>, jp: f32, cycleActive: f32, growthAngle: f32, growthAnisotropy: f32, divisionBias: f32, growthFrameAngle: f32, appearanceScale: f32, _padding: f32, }
        @group(0) @binding(0) var<storage, read> particleF: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read> rest: array<Rest>;
        @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let f = particleF[gid.x];
          let fg = rest[gid.x].growthF;
          let g = max(fg.x * fg.w - fg.y * fg.z, 1e-6);
          let invFg = vec4<f32>(fg.w, -fg.y, -fg.z, fg.x) / g;
          let fe = vec4<f32>(
            f.x * invFg.x + f.y * invFg.z,
            f.x * invFg.y + f.y * invFg.w,
            f.z * invFg.x + f.w * invFg.z,
            f.z * invFg.y + f.w * invFg.w
          );
          out[gid.x] = vec4<f32>(
            fe.x * fe.w - fe.y * fe.z,
            g,
            rest[gid.x].cycleActive,
            0.0
          );
        }
        """
    )
    pipeline = core.device.create_compute_pipeline(
        layout=wgpu.AutoLayoutMode.auto,
        compute={"module": shader, "entry_point": "main"},
    )
    out = core.device.create_buffer(
        size=count * 16,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = core.device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": core.F}},
            {"binding": 1, "resource": {"buffer": core.rest}},
            {"binding": 2, "resource": {"buffer": out}},
        ],
    )
    encoder = core.device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups(count)
    compute.end()
    core.device.queue.submit([encoder.finish()])
    return np.frombuffer(core.device.queue.read_buffer(out), np.float32).reshape(count, 4)[:, :3]


def _load_two(core: MpmCore) -> None:
    positions = np.array([[0.495, 0.5], [0.505, 0.5]], dtype=np.float32)
    velocities = np.zeros((2, 2), dtype=np.float32)
    deformation = np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (2, 1))
    affine = np.zeros((2, 4), dtype=np.float32)
    core.load_scene(positions, velocities, deformation, affine, np.ones(2, dtype=np.float32))


def check_morphology_occupancy(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    _load_two(core)
    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    device.queue.submit([encoder.finish()])
    occupancy = core.read_morphology()
    assert np.isfinite(occupancy).all()
    assert 0.0 <= float(occupancy.min()) <= float(occupancy.max()) < 1.0
    assert float(occupancy.max()) > 0.0
    assert int(np.count_nonzero(occupancy > occupancy.max() * 0.05)) > 4

    core.set_morphology(0.01, 10.0)
    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    device.queue.submit([encoder.finish()])
    weaker = core.read_morphology()
    assert float(weaker.max()) < float(occupancy.max())
    print(f"[PASS] morphology_occupancy bounded=yes blurred=yes reference_response=yes peak={occupancy.max():.6f}")


def check_single_cell_rollout_seed(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0)
    agents = AgentsGPU(
        device, core, environment, 1, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        4, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5,
    )
    TrainingRollout(
        core, agents, environment,
        spawn_center=(0.5, 0.5), spawn_half_width=0.0,
        gravity=0.0, seed=17, initial_particle_count=1,
    )
    assert core.read_positions().shape == (1, 2)
    assert agents.read_grown_count() == 1
    print("[PASS] rollout starts with exactly one seeded particle")


def check_supersampled_communication_rounds(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 64, 64, 0.5, 1.0)
    agents = AgentsGPU(
        device, core, environment, 1, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 2.0,
        2, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    agents.set_active_count(1)
    agents.set_growth_enabled(False)
    layout = weight_layout(1, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"]] = 1.0
    agents.load_weights(weights)
    readback = device.create_buffer(
        size=environment.buffers[0].size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    copy_module = device.create_shader_module(code="""
        @group(0) @binding(0) var<storage, read> source: array<f32>;
        @group(0) @binding(1) var<storage, read_write> destination: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          if (gid.x < arrayLength(&source)) { destination[gid.x] = source[gid.x]; }
        }
    """)
    copy_pipeline = device.create_compute_pipeline(
        layout=wgpu.AutoLayoutMode.auto, compute={"module": copy_module, "entry_point": "main"}
    )
    copy_groups = [
        device.create_bind_group(
            layout=copy_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": environment.buffers[p]}},
                {"binding": 1, "resource": {"buffer": readback}},
            ],
        )
        for p in range(2)
    ]

    def field_sum(rounds: int) -> float:
        environment.reset()
        agents.reset_state(23)
        communication_dt = environment.set_communication_timestep(rounds, 1.0)
        agents.set_communication_timestep(communication_dt)
        encoder = device.create_command_encoder()
        core.encode_morphology(encoder)
        for communication_round in range(rounds):
            environment.encode_clear(encoder)
            agents.encode_splat_chemical_state(encoder)
            environment.encode_sense(encoder)
            agents.encode_step(encoder, environment.parity, commit_lifecycle=communication_round == rounds - 1)
        device.queue.submit([encoder.finish()])
        raw = device.queue.read_buffer(
            agents._agent_state_buffer,
            PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        )
        return float(np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1)["chemicalState"].sum())

    once = field_sum(1)
    four = field_sum(4)
    assert np.isclose(once, four, atol=2e-6), (once, four)
    print(f"[PASS] supersampled_communication cell_state1={once:.3f} cell_state4={four:.3f}")


def check_growth_without_repulsion(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(1e4, 0.2, 3.0, 0.2, 400.0, 2.0)
    _load_two(core)
    # cycleActive=1 for both particles: growth must not require prior dilation.
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(2), np.ones(2), np.ones(2)))
    core.step(40)
    positions = core.read_positions()
    delta = np.abs(positions[1] - positions[0])
    separation = float(np.linalg.norm(np.minimum(delta, 1.0 - delta)))
    state = _probe(core, 2)
    assert separation > 0.0105, separation
    assert np.all(state[:, 1] > 1.9), state
    print(f"[PASS] growth_without_repulsion separation={separation:.6f}")


def check_resampling_uses_one_sample_volume_target(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(
        0.0, 0.2, 3.0, 0.0,
        growth_rate=1000.0,
        growth_compression_feedback=0.0,
    )
    positions = np.array([[0.4, 0.5], [0.5, 0.5], [0.6, 0.5]], dtype=np.float32)
    core.load_scene(
        positions,
        np.zeros((3, 2), dtype=np.float32),
        np.repeat(np.eye(2, dtype=np.float32).reshape(1, 4), 3, axis=0),
        np.zeros((3, 4), dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )
    spreads = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    device.queue.write_buffer(
        core.rest, 0,
        _rest_state(
            np.ones(3), np.ones(3), np.ones(3),
            anisotropy=1.0 - spreads,
            division_bias=spreads,
        ),
    )
    core.step(32)
    areas = np.linalg.det(core.read_rest_state()[:, :4].reshape(-1, 2, 2))
    assert np.allclose(areas, [2.0, 2.0, 2.0], atol=2e-5), areas
    print("[PASS] resampling target is one sample-volume and independent of legacy spread")


def check_compressed_growth_pauses_and_resumes(device: wgpu.GPUDevice) -> None:
    """Compression arrests active rest growth; release resumes without loss."""
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(
        0.0, 0.2, 3.0, 0.2, growth_rate=50.0,
        growth_compression_start=0.10,
        growth_compression_stop=0.10,
        growth_compression_feedback=1.0,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.ones(1), np.ones(1)))
    core.step(1)
    arrested = float(_probe(core, 1)[0, 1])
    assert np.isclose(arrested, 1.0, atol=2e-6), arrested

    # Release the elastic compression while preserving the active cycle.
    device.queue.write_buffer(
        core.F, 0, np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    )
    core.step(1)
    resumed = float(_probe(core, 1)[0, 1])
    expected = np.exp(50.0 * DT)
    assert np.isclose(resumed, expected, atol=2e-6), (resumed, expected)

    # Strength zero is the exact compatibility/ablation path.
    core.set_material(
        0.0, 0.2, 3.0, 0.2, growth_rate=50.0,
        growth_compression_start=0.10,
        growth_compression_stop=0.10,
        growth_compression_feedback=0.0,
    )
    device.queue.write_buffer(
        core.F, 0,
        np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], dtype=np.float32),
    )
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.ones(1), np.ones(1)))
    core.step(1)
    legacy = float(_probe(core, 1)[0, 1])
    assert np.isclose(legacy, expected, atol=2e-6), (legacy, expected)
    print(
        "[PASS] compression feedback arrests/resumes growth and strength=0 "
        f"preserves legacy rate g={resumed:.6f}"
    )


def check_transient_cell_chemical_splats(device: wgpu.GPUDevice) -> None:
    """Cell deltas persist locally; rebuilt Gaussian fields do not persist."""
    channels = 8
    width = height = 32
    core = MpmCore(device)
    environment = EnvironmentGPU(device, channels, width, height, 0.91, 1.0)
    agents = AgentsGPU(
        device, core, environment, channels, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False,
        0.0,  # legacy deposit-distance ABI slot; centered writes ignore it
        2, 0.01, 1.0, 1.0, 0.01, 1.0, 0.5, 0.5,
    )
    # Exactly the center of texel (x=8,y=24). Keep the diagnostic kernel
    # sub-texel so fixed-point quantization cannot flatten neighboring peaks.
    position = np.array([[(8.5 / width), (24.5 / height)]], dtype=np.float32)
    core.load_scene(
        position,
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    agents.set_active_count(1)
    agents.reset_state(5)
    layout = weight_layout(channels, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Give the first four channels saturated deltas. On the following round
    # their cell-owned levels must splat at the particle, independent of heading.
    for channel in range(4):
        weights[layout["fc2b_offset"] + channel] = 20.0
    env_write_dim = channels
    weights[layout["fc2b_offset"] + env_write_dim + 5] = 20.0
    weights[layout["fc2b_offset"] + env_write_dim + 6] = -20.0
    # Blue stays at logit 0 -> sigmoid 0.5.
    agents.load_weights(weights)

    count = channels * width * height
    readback = device.create_buffer(
        size=count * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    probe = device.create_shader_module(
        code="""
        @group(0) @binding(0) var<storage, read> source: array<i32>;
        @group(0) @binding(1) var<storage, read_write> destination: array<i32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          if (gid.x < arrayLength(&source)) { destination[gid.x] = source[gid.x]; }
        }
        """
    )
    pipeline = device.create_compute_pipeline(
        layout=wgpu.AutoLayoutMode.auto,
        compute={"module": probe, "entry_point": "main"},
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": environment.deposit_scratch}},
            {"binding": 1, "resource": {"buffer": readback}},
        ],
    )

    encoder = device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups((count + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])

    scratch = np.frombuffer(device.queue.read_buffer(readback), np.int32).reshape(channels, height, width)
    target = (8, 24)
    for channel in range(4):
        max_y, max_x = np.unravel_index(np.argmax(scratch[channel]), scratch[channel].shape)
        assert (max_x, max_y) == target, (channel, (max_x, max_y), target)
        assert scratch[channel, max_y, max_x] > 0
    assert not np.any(scratch[4:]), "one output channel leaked into another"

    # Newborn appearance is rendering-only: a physically present daughter must
    # publish its full baseline chemistry immediately after division.
    device.queue.write_buffer(
        core.rest,
        0,
        _rest_state(np.ones(1), np.ones(1), np.ones(1), appearance_scale=np.array([0.25])),
    )
    encoder = device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups((count + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])
    faded_scratch = np.frombuffer(
        device.queue.read_buffer(readback), np.int32
    ).reshape(channels, height, width)
    np.testing.assert_array_equal(faded_scratch, scratch)

    # The substrate footprint follows stress-free material growth. Isotropic
    # area doubling must retain the peak while increasing the projected area,
    # even when the renderer is still fading the particle in.
    root2 = np.float32(np.sqrt(2.0))
    device.queue.write_buffer(
        core.rest,
        0,
        _rest_state(
            np.ones(1), np.array([2.0]), np.ones(1),
            growth_f=np.array([[root2, 0.0, 0.0, root2]], dtype=np.float32),
            appearance_scale=np.array([0.25]),
        ),
    )
    encoder = device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups((count + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])
    grown_scratch = np.frombuffer(
        device.queue.read_buffer(readback), np.int32
    ).reshape(channels, height, width)
    assert grown_scratch[0, target[1], target[0]] >= scratch[0, target[1], target[0]]
    assert grown_scratch[0].sum() > scratch[0].sum() * 1.5
    assert not np.any(grown_scratch[4:])

    meta = np.frombuffer(
        device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        ),
        dtype=agents._particle_meta_dtype, count=1,
    ).copy()
    expected_relaxed_level = 1.0
    np.testing.assert_allclose(
        meta["chemicalState"][0, :4], expected_relaxed_level, atol=1e-7
    )
    meta["chemicalState"][:] = 0.0
    device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta.tobytes())
    encoder = device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups((count + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])
    cleared = np.frombuffer(device.queue.read_buffer(readback), np.int32)
    assert not np.any(cleared), "transient field retained a prior round's splat"
    print("[PASS] cell chemistry persists locally; substrate follows material growth, ignores visual fade, and discards old writes")

    meta_raw = device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        agents._particle_meta_dtype.itemsize,
    )
    meta = np.frombuffer(meta_raw, dtype=agents._particle_meta_dtype, count=1)
    assert np.allclose(meta["color"][0, :3], [1.0, 0.0, 0.5], atol=1e-6), meta["color"][0]
    assert np.isclose(meta["color"][0, 3], 1.0), meta["color"][0]
    print("[PASS] sigmoid RGB outputs are stored in particle state")


def check_persistent_environment_chemistry(device: wgpu.GPUDevice) -> None:
    """Only the final NN output is deposited before one field evolution."""
    channels = 1
    width = height = 16
    decay = 0.81
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, channels, width, height, decay, 1.0,
        PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    )
    agents = AgentsGPU(
        device, core, environment, channels, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        2, 0.01, 1.0, 1.0, 0.2, 0.0, 0.5, 0.5,
        chemical_communication_architecture=PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    core.set_active_count(1)
    agents.set_active_count(1)
    agents.reset_state(23)
    layout = weight_layout(channels, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"]] = 20.0
    agents.load_weights(weights)

    def macro_tick(rounds: int) -> np.ndarray:
        communication_dt = environment.set_communication_timestep(rounds, 1.0)
        agents.set_communication_timestep(communication_dt)
        encoder = device.create_command_encoder()
        environment.encode_prepare_persistent(encoder)
        for communication_round in range(rounds):
            final_round = communication_round == rounds - 1
            if final_round:
                environment.encode_clear(encoder)
            environment.encode_sense(encoder)
            agents.encode_step(
                encoder, environment.parity, commit_lifecycle=final_round
            )
        environment.encode_merge_persistent(encoder)
        device.queue.submit([encoder.finish()])
        return np.frombuffer(
            device.queue.read_buffer(environment.buffers[environment.parity]), np.float32
        ).copy()

    def reset_cellular_chemistry() -> None:
        raw = device.queue.read_buffer(
            agents._agent_state_buffer,
            PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        )
        meta = np.frombuffer(
            raw, dtype=agents._particle_meta_dtype, count=1
        ).copy()
        meta["chemicalState"][0] = 0.0
        device.queue.write_buffer(
            agents._agent_state_buffer,
            PARTICLE_META_BUFFER_OFFSET,
            meta.tobytes(),
        )

    environment.reset()
    reset_cellular_chemistry()
    deposited_once = macro_tick(1)
    assert deposited_once.max() > 0.0 and deposited_once.sum() > 0.0, deposited_once

    # Four deliberation rounds must still produce exactly one deposit from the
    # final output, not four accumulated writes or four diffusion/decay steps.
    environment.reset()
    reset_cellular_chemistry()
    deposited_four = macro_tick(4)
    np.testing.assert_allclose(deposited_four, deposited_once, rtol=2e-5, atol=2e-5)

    agents.load_weights(np.zeros_like(weights))
    reset_cellular_chemistry()
    agents.set_active_count(0)
    aged = macro_tick(4)
    np.testing.assert_allclose(aged.sum(), deposited_four.sum() * decay, rtol=2e-5, atol=2e-5)
    assert aged.max() < deposited_four.max(), (aged.max(), deposited_four.max())
    print("[PASS] persistent environment holds a frozen field, deposits the final NN output once, then diffuses/decays once")


def check_persistent_substrate_advection(device: wgpu.GPUDevice) -> None:
    """A uniform MPM velocity translates persistent chemistry one texel."""
    width = height = 16
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, 1, width, height, 1.0, 1.0,
        PERSISTENT_ENVIRONMENT_ARCHITECTURE,
        grid_velocity=core.grid_vel,
    )
    # Disable diffusion and decay to isolate semi-Lagrangian transport.
    environment.set_communication_timestep(1, 0.0)
    environment.set_advection_timestep(1.0)
    substrate = np.zeros((height, width), dtype=np.float32)
    substrate[8, 4] = 1.0
    device.queue.write_buffer(environment.buffers[0], 0, substrate)
    velocity = np.zeros(((GRID_N + 1) * (GRID_N + 1), 2), dtype=np.float32)
    velocity[:, 0] = 1.0 / width
    device.queue.write_buffer(core.grid_vel, 0, velocity)

    encoder = device.create_command_encoder()
    environment.encode_prepare_persistent(encoder)
    device.queue.submit([encoder.finish()])
    moved = np.frombuffer(
        device.queue.read_buffer(environment.buffers[environment.parity]), np.float32
    ).reshape(height, width)
    assert np.isclose(moved[8, 5], 1.0, atol=2e-5), moved[8]
    assert np.isclose(moved.sum(), 1.0, atol=2e-5), moved.sum()
    print("[PASS] persistent substrate is advected with the MPM velocity field")


def check_elastic_strain_policy_inputs(device: wgpu.GPUDevice) -> None:
    """Route each new GPU strain input to one RGB output and compare it to
    an independent NumPy matrix-log calculation."""
    from elastic_diagnostics import policy_elastic_strain_input

    channels = 1
    hidden = 128
    scale = 0.15
    core = MpmCore(device)
    environment = EnvironmentGPU(device, channels, 32, 32, 1.0, 1.0)
    agents = AgentsGPU(
        device, core, environment, channels, hidden,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        2, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5, scale, True,
    )
    layout = weight_layout(channels, hidden)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    elastic_offset = channels * 3 + 3
    color_start = channels + 5
    for component in range(3):
        weights[layout["fc1w_offset"] + component * layout["in_dim"] + elastic_offset + component] = 1.0
        weights[layout["fc2w_offset"] + (color_start + component) * hidden + component] = 1.0
    agents.load_weights(weights)

    def gpu_color(active_agents: AgentsGPU, f: np.ndarray, fg: np.ndarray, heading: float) -> np.ndarray:
        core.reset_growth_buffers(2)
        core.load_scene(
            np.array([[0.5, 0.5]], dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.asarray(f, dtype=np.float32).reshape(1, 4),
            np.zeros((1, 4), dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )
        device.queue.write_buffer(
            core.rest,
            0,
            _rest_state(np.ones(1), np.ones(1), np.zeros(1), growth_f=np.asarray(fg).reshape(1, 4)),
        )
        environment.reset()
        active_agents.set_active_count(1)
        active_agents.reset_state(29)
        encoder = device.create_command_encoder()
        core.encode_morphology(encoder)
        environment.encode_sense(encoder)
        active_agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
        device.queue.submit([encoder.finish()])
        raw = device.queue.read_buffer(
            active_agents._agent_state_buffer,
            PARTICLE_META_BUFFER_OFFSET,
            active_agents._particle_meta_dtype.itemsize,
        )
        return np.frombuffer(raw, dtype=active_agents._particle_meta_dtype, count=1)["color"][0, :3].copy()

    fg = np.array([[1.25, 0.12], [0.04, 0.92]], dtype=np.float64)
    fe = np.array([[1.11, 0.08], [0.02, 0.90]], dtype=np.float64)
    heading = 0.31
    normalized = policy_elastic_strain_input(
        (fe @ fg)[None], fg[None], np.array([heading]), scale=scale
    )[0]
    # An isolated particle has no channel-7-gradient frame. Volumetric strain is
    # orientation-free; axial and shear perception are therefore suppressed.
    normalized[1:] = 0.0
    expected_color = 1.0 / (1.0 + np.exp(-np.tanh(normalized)))
    actual_color = gpu_color(agents, fe @ fg, fg, heading)
    assert np.allclose(actual_color, expected_color, atol=2e-5), (actual_color, expected_color, normalized)

    theta = -0.72
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    stress_free_color = gpu_color(agents, rotation @ fg, fg, theta)
    assert np.allclose(stress_free_color, 0.5, atol=2e-5), stress_free_color

    disabled_agents = AgentsGPU(
        device, core, environment, channels, hidden,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        2, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5, scale, False,
    )
    disabled_agents.load_weights(weights)
    disabled_color = gpu_color(disabled_agents, fe @ fg, fg, heading)
    assert np.allclose(disabled_color, 0.5, atol=2e-5), disabled_color
    print("[PASS] GPU elastic inputs are correct when enabled and exactly zero in the temporary ablation")


def check_directional_material_fan(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 0.91, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        8, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    parent_c = np.array([[30.0, -10.0, 5.0, 20.0]], dtype=np.float32)
    parent_velocity = np.array([0.25, -0.1], dtype=np.float32)
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        parent_velocity[None, :],
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        parent_c,
        np.ones(1, dtype=np.float32),
    )
    core.reset_growth_buffers(8)
    source_fe = np.array([[1.1, 0.05], [-0.02, 0.95]], dtype=np.float32)
    source_fg = np.eye(2, dtype=np.float32) * np.sqrt(np.float32(7.0))
    source_f = (source_fe @ source_fg).reshape(1, 4)
    source_rest = _rest_state(
        np.ones(1), np.array([7.0]), np.ones(1),
        growth_f=source_fg.reshape(1, 2, 2),
        anisotropy=np.zeros(1), division_bias=np.ones(1),
    )
    device.queue.write_buffer(core.F, 0, source_f)
    device.queue.write_buffer(core.rest, 0, source_rest)
    device.queue.write_buffer(core.C, 0, parent_c)
    device.queue.write_buffer(core.velocities, 0, parent_velocity[None, :])
    core.set_active_count(1)
    environment.reset()
    agents.set_active_count(1)
    agents.reset_state(7)
    # Give this diagnostic a defined +X world-space growth direction without
    # synthetically admitting the already-latched event.
    device.queue.write_buffer(agents._physics_uniform, 80, np.array([0], dtype=np.uint32))
    device.queue.write_buffer(agents._physics_uniform, 84, np.array([0], dtype=np.uint32))
    device.queue.write_buffer(agents._physics_uniform, 88, np.array([1.0, 0.0], dtype=np.float32))
    device.queue.write_buffer(agents._physics_uniform, 96, np.array([0], dtype=np.uint32))
    neutral_weights = np.zeros(agents._total_floats, dtype=np.float32)
    layout = weight_layout(8, 128)
    # Full normalized spread -> a six-sample 360-degree fan at one spacing.
    neutral_weights[layout["fc2b_offset"] + 8 + 1] = 20.0
    agents.load_weights(neutral_weights)

    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    count = agents.read_grown_count()
    core.set_active_count(count)

    positions = core.read_positions()
    velocities = core.read_velocities()
    affine = core.read_affine()
    state = _probe(core, count)
    rest_state = core.read_rest_state()
    assert count == 7
    assert np.allclose(positions[0], [0.5, 0.5], atol=1e-7), positions
    offsets = (positions[1:] - positions[0] + 0.5) % 1.0 - 0.5
    assert np.allclose(np.linalg.norm(offsets, axis=1), 0.01, atol=2e-6), offsets
    actual_angles = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]))
    expected_angles = np.deg2rad([-150, -90, -30, 30, 90, 150])
    assert np.allclose(actual_angles, expected_angles, atol=2e-4), actual_angles
    assert np.allclose(core.read_deformation(), np.repeat(source_fe.reshape(1, 4), count, axis=0), atol=1e-6)
    assert np.allclose(state[:, 0], np.linalg.det(source_fe), atol=1e-5), state
    assert np.isclose(state[:, 1].sum(), float(count), atol=1e-5), state
    assert np.all(state[:, 2] == 0.0), state
    identity = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert np.allclose(rest_state[:, :4], np.repeat(identity[None, :], count, axis=0), atol=1e-6)
    assert np.allclose(rest_state[:, 4], source_rest[0, 4], atol=1e-7)
    assert np.all(rest_state[:, 5] == 0.0)
    assert np.allclose(rest_state[:, 6:10], np.repeat(rest_state[0:1, 6:10], count, axis=0), atol=1e-7)
    assert np.isclose(rest_state[0, 10], 1.0)
    assert np.allclose(rest_state[1:, 10], 0.0)

    # Visual disc area uses the same normalized exponential curve as rest
    # growth: over one of eight macro steps it advances from 0 to 2^(1/8)-1.
    core.set_material(
        0.0, 0.2, 3.0, 0.0,
        growth_duration_macro_steps=8.0,
        substeps_per_macro=4,
        growth_compression_feedback=0.0,
    )
    core.step(4)
    ramped_rest = core.read_rest_state()
    assert np.isclose(ramped_rest[0, 10], 1.0)
    assert np.allclose(ramped_rest[1:, 10], 2.0 ** (1.0 / 8.0) - 1.0, atol=2e-6), ramped_rest[:, 10]
    c_matrix = parent_c.reshape(2, 2)
    all_offsets = np.vstack([np.zeros((1, 2), dtype=np.float32), offsets])
    expected_velocity = parent_velocity + all_offsets @ c_matrix.T
    assert np.allclose(affine, np.repeat(parent_c, count, axis=0), atol=1e-6), affine
    assert np.allclose(velocities, expected_velocity, atol=1e-5), (velocities, expected_velocity)
    assert np.allclose(velocities[0], parent_velocity, atol=1e-7), velocities
    print("[PASS] directional material growth keeps its source and emits a density-spaced six-sample full-circle fan")




def check_channel_seven_gradient_defines_alignment(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, 8, 32, 32, 1.0, 1.0,
        chemical_communication_architecture=PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    )
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        3, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5,
    )
    positions = np.array([[0.42, 0.5], [0.58, 0.5]], dtype=np.float32)
    core.load_scene(
        positions,
        np.zeros((2, 2), dtype=np.float32),
        np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
        np.zeros((2, 4), dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    environment.reset()
    # Channel 7 has a ridge centered at x=0.5, so its gradient points right
    # for the left particle and left for the right particle. All other
    # chemical channels remain flat; morphology is deliberately symmetric.
    field = np.zeros(environment.total_values, dtype=np.float32)
    width = environment.channel_widths[7]
    height = environment.channel_heights[7]
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    ridge = np.exp(-0.5 * ((x - 0.5) / 0.08) ** 2).astype(np.float32)
    offset = environment.channel_offsets[7]
    field[offset:offset + width * height] = np.tile(ridge, height)
    device.queue.write_buffer(environment.buffers[0], 0, field)
    agents.set_active_count(2)
    agents.reset_state(31)
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    agents.load_weights(weights)
    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
    device.queue.submit([encoder.finish()])
    raw = device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        2 * agents._particle_meta_dtype.itemsize,
    )
    alignment = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=2)["alignment"]
    assert alignment[0, 0] > 0.0 and alignment[1, 0] < 0.0, alignment
    assert np.all(np.abs(alignment[:, 1]) < np.abs(alignment[:, 0]) * 0.05), alignment
    assert np.all(np.linalg.norm(alignment, axis=1) <= 1.0 + 1e-6), alignment
    print("[PASS] agent alignment is the L2-clipped chemical-channel-7 gradient")


def _directional_growth_case(
    device: wgpu.GPUDevice,
    growth_direction_bias: float,
    spread_bias: float = -20.0,
    directionality: float = 1.0,
    defined_alignment: bool = True,
    seed: int = 19,
) -> np.ndarray:
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, 8, 256, 256, 0.91, 1.0,
        chemical_communication_architecture=PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    )
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 2.0,
        4, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    origin = np.array([[0.5, 0.5]], dtype=np.float32)
    root2 = np.float32(np.sqrt(2.0))
    core.load_scene(
        origin,
        np.zeros((1, 2), dtype=np.float32),
        np.array([[root2, 0.0, 0.0, root2]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    core.reset_growth_buffers(4)
    device.queue.write_buffer(core.F, 0, np.array([[root2, 0.0, 0.0, root2]], dtype=np.float32))
    stored_direction = np.array(
        [[1.0, 0.0] if growth_direction_bias >= 0.0 else [-1.0, 0.0]],
        dtype=np.float32,
    )
    device.queue.write_buffer(
        core.rest, 0,
        _rest_state(
            np.ones(1), np.array([2.0]), np.ones(1),
            growth_direction=stored_direction,
        ),
    )
    core.set_active_count(1)
    environment.reset()
    if defined_alignment:
        # A +X channel-7 gradient defines the local frame, allowing the test
        # to distinguish local-forward from local-backward growth outputs.
        field = np.zeros(environment.total_values, dtype=np.float32)
        width = environment.channel_widths[7]
        height = environment.channel_heights[7]
        x = (np.arange(width, dtype=np.float32) + 0.5) / width
        wave = (10.0 * np.sin(2.0 * np.pi * (x - 0.5))).astype(np.float32)
        offset = environment.channel_offsets[7]
        field[offset:offset + width * height] = np.tile(wave, height)
        device.queue.write_buffer(environment.buffers[0], 0, field)
    agents.set_active_count(1)
    agents.set_division_directionality(directionality)
    agents.reset_state(seed)
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Angular spread plus desired signed local-space growth direction.
    weights[layout["fc2b_offset"] + 8 + 1] = spread_bias
    weights[layout["fc2b_offset"] + 8 + 2] = growth_direction_bias
    agents.load_weights(weights)
    # Run perception-only evaluations before the growth event.
    for _ in range(16):
        encoder = device.create_command_encoder()
        core.encode_morphology(encoder)
        environment.encode_sense(encoder)
        agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
        device.queue.submit([encoder.finish()])
    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    count = agents.read_grown_count()
    core.set_active_count(count)
    assert count == 2
    return core.read_positions()


def check_growth_direction_is_signed_or_uses_spatial_fallback(device: wgpu.GPUDevice) -> None:
    positive = _directional_growth_case(device, 20.0)
    negative = _directional_growth_case(device, -20.0)
    flat_positive = _directional_growth_case(device, 20.0, defined_alignment=False)
    flat_negative = _directional_growth_case(device, -20.0, defined_alignment=False)
    origin = np.array([0.5, 0.5], dtype=np.float32)
    wrapped = lambda points: (points - origin + 0.5) % 1.0 - 0.5
    positive_offset = wrapped(positive)
    negative_offset = wrapped(negative)
    assert np.linalg.norm(positive_offset[0]) < 2e-6
    assert np.linalg.norm(negative_offset[0]) < 2e-6
    assert np.allclose(positive_offset[1], [0.01, 0.0], atol=2e-6), positive_offset
    assert np.allclose(negative_offset[1], [-0.01, 0.0], atol=2e-6), negative_offset
    assert np.allclose(flat_positive, flat_negative, atol=2e-6), (flat_positive, flat_negative)
    print("[PASS] growth direction is signed and uses a spatial fallback only for an undefined local frame")


def check_morphology_gradient_does_not_override_growth_direction(device: wgpu.GPUDevice) -> None:
    """Ordinary morphology gradients must not override neural growth placement."""
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, 8, 256, 256, 0.91, 1.0,
        chemical_communication_architecture=PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    )
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 2.0,
        6, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    # The morphology gradient is deliberately distinct from the channel-7
    # frame; ordinary policy growth must remain the division axis.
    agents.set_boundary_tangent_min_gradient(1e-6)
    # Particle 0 is just to the right of a small cluster. Its morphology
    # gradient is nonzero, while a zeroed policy would otherwise request a
    # horizontal split. The assertion below reconstructs the exact sampled
    # gradient rather than assuming an ideal direction on the discrete grid.
    positions = np.array([
        [0.53, 0.50],
        [0.50, 0.49],
        [0.50, 0.50],
        [0.50, 0.51],
        [0.505, 0.50],
    ], dtype=np.float32)
    count = len(positions)
    root2 = np.float32(np.sqrt(2.0))
    particle_f = np.tile(np.eye(2, dtype=np.float32).reshape(1, 4), (count, 1))
    particle_f[0] = np.array([root2, 0.0, 0.0, root2], dtype=np.float32)
    core.load_scene(
        positions,
        np.zeros((count, 2), dtype=np.float32),
        particle_f,
        np.zeros((count, 4), dtype=np.float32),
        np.ones(count, dtype=np.float32),
    )
    core.reset_growth_buffers(6)
    growth = np.ones(count, dtype=np.float32)
    growth[0] = 2.0
    cycle = np.zeros(count, dtype=np.float32)
    cycle[0] = 1.0
    device.queue.write_buffer(core.F, 0, particle_f)
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(count), growth, cycle))
    core.set_active_count(count)
    environment.reset()
    # Give channel 7 a vertical gradient that is deliberately distinct from
    # the cluster's mostly horizontal morphology gradient.
    field = np.zeros(environment.total_values, dtype=np.float32)
    width = environment.channel_widths[7]
    height = environment.channel_heights[7]
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    ridge = np.exp(-0.5 * ((y - 0.60) / 0.12) ** 2).astype(np.float32)
    offset = environment.channel_offsets[7]
    field[offset:offset + width * height] = np.repeat(ridge, width)
    device.queue.write_buffer(environment.buffers[0], 0, field)
    agents.set_active_count(count)
    agents.reset_state(29)
    weights = np.zeros(agents._total_floats, dtype=np.float32)
    layout = weight_layout(8, 128)
    weights[layout["fc2b_offset"] + 8 + 1] = -20.0
    agents.load_weights(weights)

    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    grown_count = agents.read_grown_count()
    core.set_active_count(grown_count)
    assert grown_count == count + 1
    raw_meta = device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        agents._particle_meta_dtype.itemsize,
    )
    alignment = np.frombuffer(
        raw_meta, dtype=agents._particle_meta_dtype, count=1,
    )["alignment"][0]
    assert np.linalg.norm(alignment) > 1e-6, alignment
    morphology = core.read_morphology()

    def sample_morphology(field_position: np.ndarray) -> float:
        base = np.floor(field_position).astype(np.int32)
        fraction = field_position - base
        n = morphology.shape[0]

        def load(dx: int, dy: int) -> float:
            return float(morphology[(base[1] + dy) % n, (base[0] + dx) % n])

        a = load(0, 0) * (1.0 - fraction[0]) + load(1, 0) * fraction[0]
        b = load(0, 1) * (1.0 - fraction[0]) + load(1, 1) * fraction[0]
        return a * (1.0 - fraction[1]) + b * fraction[1]

    field_position = positions[0] * morphology.shape[0]
    gradient = 0.5 * np.array([
        sample_morphology(field_position + [1.0, 0.0])
        - sample_morphology(field_position - [1.0, 0.0]),
        sample_morphology(field_position + [0.0, 1.0])
        - sample_morphology(field_position - [0.0, 1.0]),
    ])
    assert np.linalg.norm(gradient) > 1e-6, gradient
    result = core.read_positions()
    daughters = result[[0, count]]
    separation = daughters[1] - daughters[0]
    separation = (separation + 0.5) % 1.0 - 0.5
    assert np.isclose(np.linalg.norm(separation), 0.01, atol=2e-5), daughters
    split_axis = separation / np.linalg.norm(separation)
    heading_axis = alignment / np.linalg.norm(alignment)
    assert np.isclose(abs(np.dot(split_axis, heading_axis)), 1.0, atol=2e-5), (
        split_axis, heading_axis,
    )
    print("[PASS] morphology gradient does not override the neural growth direction")


def check_anisotropic_tensor_split(device: wgpu.GPUDevice) -> None:
    """A det(Fg)=2 sheared rest state must split into two Fe daughters."""
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 0.91, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        4, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    fe = np.array([[1.04, 0.03], [-0.02, 0.98]], dtype=np.float32)
    fg = np.array([[1.6, 0.3], [0.1, 1.26875]], dtype=np.float32)
    assert np.isclose(np.linalg.det(fg), 2.0)
    total_f = fe @ fg
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        total_f.reshape(1, 4),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    core.reset_growth_buffers(4)
    device.queue.write_buffer(core.F, 0, total_f.reshape(1, 4))
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.array([2.0]), np.ones(1), fg[None, :, :]))
    core.set_active_count(1)
    environment.reset()
    agents.set_active_count(1)
    agents.reset_state(73)
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    count = agents.read_grown_count()
    core.set_active_count(count)

    daughter_f = core.read_deformation().reshape(-1, 2, 2)
    daughter_rest = core.read_rest_state()
    identity = np.array([1.0, 0.0, 0.0, 1.0])
    assert count == 2
    assert np.allclose(daughter_f, np.repeat(fe[None, :, :], 2, axis=0), atol=2e-6), daughter_f
    assert np.allclose(daughter_rest[:, :4], identity, atol=1e-7), daughter_rest
    assert np.allclose(daughter_rest[:, 4], 1.0) and np.allclose(daughter_rest[:, 5], 0.0)
    print("[PASS] anisotropic_tensor_split preserves Fe and resets both daughter Fg tensors")


def check_isotropic_increment_preserves_tensor_shape(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(
        0.0, 0.2, 3.0, 1.0, growth_rate=50.0, growth_max=2.0,
        growth_compression_feedback=0.0,
    )
    fg = np.array([[1.15, 0.18], [0.04, 0.92]], dtype=np.float32)
    fe = np.array([[1.02, 0.01], [-0.02, 0.99]], dtype=np.float32)
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        (fe @ fg).reshape(1, 4),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.array([np.linalg.det(fg)]), np.ones(1), fg[None]))
    normalized_before = fg / np.sqrt(np.linalg.det(fg))
    core.step(1)
    fg_after = core.read_rest_state()[0, :4].reshape(2, 2)
    normalized_after = fg_after / np.sqrt(np.linalg.det(fg_after))
    assert np.linalg.det(fg_after) > np.linalg.det(fg)
    assert np.allclose(normalized_after, normalized_before, atol=2e-6), (normalized_before, normalized_after)
    print("[PASS] isotropic tensor increment grows det(Fg) without changing anisotropic shape")


def _directional_increment_case(
    device: wgpu.GPUDevice,
    rotation_angle: float,
    anisotropy: float = 1.0,
    global_anisotropy: float = 1.0,
) -> tuple[np.ndarray, float]:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(
        0.0,
        0.2,
        3.0,
        1.0,
        growth_rate=50.0,
        growth_max=2.0,
        growth_anisotropy=global_anisotropy,
        growth_compression_feedback=0.0,
    )
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    world_direction = rotation @ np.array([1.0, 0.0], dtype=np.float32)
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        rotation.reshape(1, 4),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    device.queue.write_buffer(
        core.rest,
        0,
        _rest_state(
            np.ones(1), np.ones(1), np.ones(1),
            growth_direction=world_direction[None], anisotropy=np.array([anisotropy]),
        ),
    )
    core.step(1)
    return core.read_rest_state()[0, :4].reshape(2, 2), float(np.exp(50.0 * DT))


def check_directional_increment_and_objectivity(device: wgpu.GPUDevice) -> None:
    fg_axis, expected_area = _directional_increment_case(device, 0.0)
    fg_rotated, _ = _directional_increment_case(device, 0.83)
    fg_isotropic_with_axis, _ = _directional_increment_case(device, 0.0, anisotropy=0.0)
    fg_globally_isotropic, _ = _directional_increment_case(device, 0.0, global_anisotropy=0.0)
    # Full-strength direction puts the complete area increment along n;
    # the perpendicular rest stretch stays one. Pulling a rotated world
    # direction back through Re must produce the same intermediate Fg.
    assert np.isclose(np.linalg.det(fg_axis), expected_area, rtol=2e-5)
    assert np.allclose(fg_axis, np.diag([expected_area, 1.0]), atol=3e-6), fg_axis
    assert np.allclose(fg_rotated, fg_axis, atol=3e-6), (fg_axis, fg_rotated)
    assert np.allclose(fg_isotropic_with_axis, np.eye(2) * np.sqrt(expected_area), atol=3e-6), fg_isotropic_with_axis
    assert np.allclose(fg_globally_isotropic, np.eye(2) * np.sqrt(expected_area), atol=3e-6), fg_globally_isotropic
    print("[PASS] directional increment is objective and the global anisotropy multiplier can force isotropic growth")


def _duration_growth_case(device: wgpu.GPUDevice, substeps: int) -> float:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(
        0.0,
        0.2,
        3.0,
        1.0,
        growth_duration_macro_steps=20.0,
        substeps_per_macro=substeps,
        growth_compression_feedback=0.0,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.eye(2, dtype=np.float32).reshape(1, 4),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.ones(1), np.ones(1)))
    core.step(substeps)
    return float(np.linalg.det(core.read_rest_state()[0, :4].reshape(2, 2)))


def check_growth_duration_is_substep_invariant(device: wgpu.GPUDevice) -> None:
    values = np.array([_duration_growth_case(device, s) for s in (1, 16, 64)])
    expected = 2.0 ** (1.0 / 20.0)
    assert np.allclose(values, expected, rtol=2e-5, atol=2e-6), (values, expected)
    print("[PASS] one controller tick advances the same growth at 1/16/64 physics substeps")


def check_persistent_growth_targets_drive_state_not_motion(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(
        device, 8, 256, 256, 0.91, 1.0,
        chemical_communication_architecture=PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    )
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 2.0,
        4, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    initial_velocity = np.array([[0.3, -0.2]], dtype=np.float32)
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        initial_velocity,
        np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    environment.reset()
    # Supply a +Y channel-7 gradient while the NN requests a distinct local
    # diagonal. The stored frame and local angle must remain separate.
    field = np.zeros(environment.total_values, dtype=np.float32)
    width = environment.channel_widths[7]
    height = environment.channel_heights[7]
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    wave = (10.0 * np.sin(2.0 * np.pi * (y - 0.5))).astype(np.float32)
    offset = environment.channel_offsets[7]
    field[offset:offset + width * height] = np.repeat(wave, width)
    device.queue.write_buffer(environment.buffers[0], 0, field)
    agents.set_active_count(1)
    agents.reset_state(11)
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"] + 8] = 2.0
    weights[layout["fc2b_offset"] + 8 + 1] = -2.0
    weights[layout["fc2b_offset"] + 8 + 2] = 1.0
    weights[layout["fc2b_offset"] + 8 + 3] = 2.0
    agents.load_weights(weights)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    state = core.read_rest_state()[0]
    raw_meta = device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        agents._particle_meta_dtype.itemsize,
    )
    alignment = np.frombuffer(
        raw_meta, dtype=agents._particle_meta_dtype, count=1,
    )["alignment"][0]
    expected_frame_angle = np.arctan2(alignment[1], alignment[0])
    expected_spread = 1 / (1 + np.exp(2))
    expected_anisotropy = 1.0 - expected_spread
    assert np.allclose(core.read_velocities(), initial_velocity, atol=1e-7)
    expected_growth_angle = np.arctan2(np.tanh(2.0), np.tanh(1.0))
    assert np.isclose(state[6], expected_growth_angle, atol=2e-6), state
    assert alignment[1] > 0.0 and abs(alignment[0]) < abs(alignment[1]) * 0.05, alignment
    assert np.isclose(state[9], expected_frame_angle, atol=2e-6), (state, alignment)
    assert np.isclose(state[7], expected_anisotropy, atol=2e-6), state
    assert np.isclose(state[8], expected_spread, atol=2e-6), state
    print("[PASS] local growth direction and spread freeze an event's directional-to-isotropic rest growth")


def check_p2g_fixed_point_headroom(device: wgpu.GPUDevice) -> None:
    """A deliberately crowded, fast transfer must retain momentum in i32."""
    core = MpmCore(device)
    count = 4096
    speed = 100.0
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(0.0, 0.2, 3.0, elasticity=0.0, particle_mass=1.0)
    core.load_scene(
        np.full((count, 2), [0.5, 0.5], dtype=np.float32),
        np.full((count, 2), [speed, 0.0], dtype=np.float32),
        np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (count, 1)),
        np.zeros((count, 4), dtype=np.float32),
        np.ones(count, dtype=np.float32),
    )
    core.step(1)
    accum = np.frombuffer(device.queue.read_buffer(core.grid_accum), np.int32).reshape(-1, 3)
    scale = 4096.0
    mass = float(accum[:, 2].astype(np.int64).sum() / scale)
    momentum_x = float(accum[:, 0].astype(np.int64).sum() / scale)
    max_raw = int(np.abs(accum.astype(np.int64)).max())
    assert np.isclose(mass, count * core.particle_mass, rtol=2e-3), mass
    assert np.isclose(momentum_x, count * core.particle_mass * speed, rtol=2e-3), momentum_x
    assert max_raw < np.iinfo(np.int32).max * 0.75, max_raw
    print(f"[PASS] p2g_headroom mass={mass:.1f} momentum={momentum_x:.1f} max_raw={max_raw}")


def check_high_strain_elastic_stability(device: wgpu.GPUDevice) -> None:
    """Regression for the F=1.2I, E=1e4 collapse seen at the old DT."""
    core = MpmCore(device)
    rng = np.random.default_rng(31)
    count = 512
    positions = (np.array([0.5, 0.5]) + rng.uniform(-0.04, 0.04, (count, 2))).astype(np.float32)
    start_radius = float(np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1))))
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(1e4, 0.2, 3.0, elasticity=0.2, growth_rate=0.0)
    core.load_scene(
        positions,
        np.zeros((count, 2), dtype=np.float32),
        np.tile(np.array([1.2, 0, 0, 1.2], dtype=np.float32), (count, 1)),
        np.zeros((count, 4), dtype=np.float32),
        np.ones(count, dtype=np.float32),
    )
    core.step(400)
    final_positions = core.read_positions()
    velocities = core.read_velocities()
    deformation = core.read_deformation().reshape(-1, 2, 2)
    singular = np.linalg.svd(deformation, compute_uv=False)
    final_radius = float(np.sqrt(np.mean(np.sum((final_positions - final_positions.mean(axis=0)) ** 2, axis=1))))
    max_speed = float(np.linalg.norm(velocities, axis=1).max())
    assert np.isfinite(final_positions).all() and np.isfinite(velocities).all() and np.isfinite(singular).all()
    assert final_radius > start_radius * 0.5, (start_radius, final_radius)
    assert max_speed < 100.0, max_speed
    assert singular.min() > 0.75 and singular.max() < 1.3, (singular.min(), singular.max())
    print(
        f"[PASS] high_strain_stability radius={final_radius:.5f}/{start_radius:.5f} "
        f"max_speed={max_speed:.3f} singular=[{singular.min():.3f},{singular.max():.3f}]"
    )


def _cycle_gate_case(
    device: wgpu.GPUDevice,
    cap: int,
    enabled: bool,
    *,
    initial_growth: float = 1.0,
    initial_cycle: float = 0.0,
    runtime_cap: int | None = None,
    commit_lifecycle: bool = True,
    elastic_area: float = 1.0,
) -> tuple[np.ndarray, int]:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 0.91, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        cap, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    core.reset_growth_buffers(cap)
    elastic_root = np.float32(np.sqrt(elastic_area))
    device.queue.write_buffer(
        core.F, 0,
        np.array([[elastic_root, 0.0, 0.0, elastic_root]], dtype=np.float32),
    )
    device.queue.write_buffer(
        core.rest,
        0,
        _rest_state(np.ones(1), np.array([initial_growth]), np.array([initial_cycle])),
    )
    core.set_active_count(1)
    environment.reset()
    agents.set_active_count(1)
    agents.reset_state(19)
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"] + 8 + 4] = 20.0
    agents.load_weights(weights)
    agents.set_growth_enabled(enabled)
    if runtime_cap is not None:
        agents.set_max_active_particles(runtime_cap)

    encoder = device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=commit_lifecycle)
    device.queue.submit([encoder.finish()])
    return _probe(core, 1)[0], agents.read_grown_count()


def check_cycle_start_gates(device: wgpu.GPUDevice) -> None:
    enabled, enabled_count = _cycle_gate_case(device, cap=4, enabled=True)
    assert enabled_count == 1 and enabled[2] == 1.0
    assert _cycle_gate_case(device, cap=4, enabled=False)[0][2] == 0.0
    assert _cycle_gate_case(device, cap=1, enabled=True)[0][2] == 0.0
    assert _cycle_gate_case(device, cap=4, enabled=True, runtime_cap=1)[0][2] == 0.0
    assert _cycle_gate_case(device, cap=4, enabled=True, commit_lifecycle=False)[0][2] == 0.0
    assert _cycle_gate_case(device, cap=4, enabled=True, elastic_area=0.8)[0][2] == 0.0
    compressed_ready, compressed_count = _cycle_gate_case(
        device, cap=4, enabled=True,
        initial_growth=2.0, initial_cycle=1.0, elastic_area=0.8,
    )
    assert compressed_count == 1 and compressed_ready[2] == 1.0 and np.isclose(compressed_ready[1], 2.0)

    # A cycle that began before other particles consumed the remaining
    # slots must be closed at cap without rolling back its accumulated g.
    capped, capped_count = _cycle_gate_case(
        device,
        cap=1,
        enabled=True,
        initial_growth=1.4,
        initial_cycle=1.0,
    )
    assert capped_count == 1 and capped[2] == 0.0, capped
    assert np.isclose(capped[1], 1.4), capped
    print("[PASS] growth emission gates include enable, commit, capacity, and mechanical compression controls")


def check_nearby_intents_are_integrated(device: wgpu.GPUDevice) -> None:
    """Two samples in one region produce one decision, not two events."""
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 1.0, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        8, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    positions = np.array([[0.5000, 0.5], [0.5005, 0.5]], dtype=np.float32)
    core.load_scene(
        positions,
        np.zeros((2, 2), dtype=np.float32),
        np.repeat(np.eye(2, dtype=np.float32).reshape(1, 4), 2, axis=0),
        np.zeros((2, 4), dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    core.reset_growth_buffers(8)
    core.set_active_count(2)
    agents.set_active_count(2)
    agents.reset_state(11)
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"] + 8 + 4] = 20.0
    agents.load_weights(weights)

    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    environment.encode_clear(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=True)
    device.queue.submit([encoder.finish()])
    cycles = _probe(core, 2)[:, 2]
    assert int(np.count_nonzero(cycles > 0.5)) == 1, cycles
    assert agents.read_grown_count() == 2
    print("[PASS] nearby sample intents are averaged into one regional growth event")


def check_directional_wedges_integrate_without_cancelling(device: wgpu.GPUDevice) -> None:
    """Opposing signed votes retain an axial tensor and a signed sample mode."""
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 1.0, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        8, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    positions = np.array([[0.5000, 0.5], [0.5005, 0.5]], dtype=np.float32)
    core.load_scene(
        positions,
        np.zeros((2, 2), dtype=np.float32),
        np.repeat(np.eye(2, dtype=np.float32).reshape(1, 4), 2, axis=0),
        np.zeros((2, 4), dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    core.reset_growth_buffers(8)
    proposal_rest = _rest_state(
        np.ones(2), np.ones(2), np.zeros(2),
        growth_direction=np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        anisotropy=np.ones(2), division_bias=np.zeros(2),
    )
    device.queue.write_buffer(core.rest, 0, proposal_rest)
    core.set_active_count(2)
    agents.set_active_count(2)
    agents.reset_state(23)
    meta = np.zeros(2, dtype=agents._particle_meta_dtype)
    meta["mitosisPropensity"] = 1.0
    device.queue.write_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta.tobytes()
    )

    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    agents.encode_growth_field(encoder)
    device.queue.submit([encoder.finish()])
    state = core.read_rest_state()
    active = np.flatnonzero(state[:, 5] > 0.5)
    assert len(active) == 1, state[:, 5]
    event = state[active[0]]
    assert event[7] > 0.8, event  # opposing directions still imply axial growth
    assert event[8] < 0.2, event  # the selected opposing lobe remains narrow
    signed = np.array([np.cos(event[11]), np.sin(event[11])])
    assert abs(signed[0]) > 0.8, event  # one lobe is selected without vector cancellation
    print("[PASS] opposing directional wedges preserve axial growth and a signed resampling mode")


def check_persistent_growth_hazard(device: wgpu.GPUDevice) -> None:
    """A weak regional drive accumulates on the field and survives gaps."""
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 1.0, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        4, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    agents.set_active_count(1)
    # This deterministic field seed gives the centre region a threshold above
    # three p=.2 increments, so accumulation and pause can be inspected first.
    agents.reset_state(20)
    layout = weight_layout(8, 128)
    inactive_weights = np.zeros(layout["total_floats"], dtype=np.float32)
    drive_weights = inactive_weights.copy()
    growth_probability = 0.2
    drive_weights[
        layout["fc2b_offset"] + 8 + 4
    ] = np.arctanh(growth_probability)
    agents.load_weights(drive_weights)
    def advance(rounds: int) -> None:
        encoder = device.create_command_encoder()
        for _ in range(rounds):
            environment.encode_clear(encoder)
            agents.encode_splat_chemical_state(encoder)
            environment.encode_sense(encoder)
            agents.encode_step(encoder, environment.parity, commit_lifecycle=True)
        device.queue.submit([encoder.finish()])

    cell = (SPATIAL_RANDOM_CELLS // 2) * SPATIAL_RANDOM_CELLS + SPATIAL_RANDOM_CELLS // 2

    def read_credit() -> float:
        offset = (15 * SPATIAL_RANDOM_CELLS * SPATIAL_RANDOM_CELLS + cell) * 4
        raw = device.queue.read_buffer(agents._growth_field, offset, 4)
        return float(np.frombuffer(raw, dtype=np.int32)[0]) / 1048576.0

    advance(3)
    expected = 3.0 * -np.log(1.0 - growth_probability)
    accumulated = read_credit()
    assert np.isclose(accumulated, expected, atol=3e-3), accumulated
    assert _probe(core, 1)[0, 2] == 0.0

    agents.load_weights(inactive_weights)
    advance(2)
    paused = read_credit()
    assert np.isclose(paused, expected, atol=3e-3), paused

    agents.load_weights(drive_weights)
    for _ in range(100):
        advance(1)
        if _probe(core, 1)[0, 2] == 1.0:
            break
    assert _probe(core, 1)[0, 2] == 1.0
    assert agents.read_grown_count() == 1
    assert read_credit() == 0.0
    print(
        "[PASS] persistent regional growth credit "
        f"weak_signal_accumulated={expected:.6f} zero_signal_preserved=yes threshold_admitted=yes"
    )


def check_stateful_private_memory(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0)
    agents = AgentsGPU(
        device, core, environment, 1, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        4, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5,
        policy_architecture=STATEFUL_128_ARCHITECTURE,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    agents.set_active_count(1)
    agents.set_growth_enabled(False)
    agents.reset_state(31)
    layout = weight_layout(1, 128, STATEFUL_128_ARCHITECTURE)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Narrow spread keeps the inheritance assertion to one emitted sample.
    weights[layout["fc2b_offset"] + 2] = -20.0
    # Common motion/growth outputs plus division drive occupy C+7 rows. Drive private channel 0
    # positively and open its gate; all other private channels remain still.
    weights[layout["fc2b_offset"] + 6] = 1.0
    weights[layout["fc2b_offset"] + 14] = 20.0
    agents.load_weights(weights)
    agents.set_communication_timestep(0.25)

    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
    device.queue.submit([encoder.finish()])
    raw = device.queue.read_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, agents._particle_meta_dtype.itemsize
    )
    meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1)[0]
    expected_state = np.tanh(1.0) * 0.25
    assert np.isclose(meta["privateState"][0], expected_state, atol=2e-6), meta
    assert np.allclose(meta["privateState"][1:], 0.0, atol=1e-7), meta
    expected_red = 1.0 / (1.0 + np.exp(-expected_state))
    assert np.isclose(meta["color"][0], expected_red, atol=2e-6), meta
    assert np.allclose(meta["color"][1:3], 0.5, atol=1e-7), meta
    agents.set_internal_state_speed(0.0)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
    device.queue.submit([encoder.finish()])
    frozen_raw = device.queue.read_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, agents._particle_meta_dtype.itemsize
    )
    frozen = np.frombuffer(frozen_raw, dtype=agents._particle_meta_dtype, count=1)[0]
    np.testing.assert_allclose(frozen["privateState"], meta["privateState"], atol=1e-7)
    agents.set_internal_state_speed(1.0)
    root2 = np.float32(np.sqrt(2.0))
    device.queue.write_buffer(core.F, 0, np.array([[root2, 0, 0, root2]], dtype=np.float32))
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.array([2.0]), np.ones(1)))
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=True)
    device.queue.submit([encoder.finish()])
    assert agents.read_grown_count() == 2
    raw = device.queue.read_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, 2 * agents._particle_meta_dtype.itemsize
    )
    samples = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=2)
    np.testing.assert_allclose(samples[0]["privateState"], samples[1]["privateState"], atol=1e-7)
    print("[PASS] recurrent-128 policy updates memory and new material samples inherit private state")


def main() -> None:
    device = pick_device()
    check_morphology_occupancy(device)
    check_single_cell_rollout_seed(device)
    check_supersampled_communication_rounds(device)
    # Event/cycle/daughter regressions above remain as historical executable
    # documentation. The active model's invariants live in the focused suite.
    from continuous_growth_check import (
        check_conservative_resampling,
        check_continuous_volume,
        check_field_normalization_and_opposition,
    )
    check_continuous_volume(device)
    check_field_normalization_and_opposition(device)
    check_conservative_resampling(device)


if __name__ == "__main__":
    main()

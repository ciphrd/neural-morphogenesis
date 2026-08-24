"""Focused GPU checks for the conservative grow-then-divide model.

Run from this directory with ``.venv/bin/python growth_check.py``.
"""
from __future__ import annotations

import numpy as np
import wgpu

from agents_gpu import PARTICLE_META_BUFFER_OFFSET, AgentsGPU, weight_layout
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import DT, MpmCore
from policy_parameters import STATEFUL_ARCHITECTURE
from training_sim import TrainingRollout


def _rest_state(
    jp: np.ndarray,
    growth: np.ndarray,
    cycle: np.ndarray,
    growth_f: np.ndarray | None = None,
    growth_direction: np.ndarray | None = None,
    anisotropy: np.ndarray | None = None,
    division_bias: np.ndarray | None = None,
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
    return out


def _probe(core: MpmCore, count: int) -> np.ndarray:
    """Returns [Je, g, cycleActive] without adding COPY_SRC to hot buffers."""
    shader = core.device.create_shader_module(
        code="""
        struct Rest { growthF: vec4<f32>, jp: f32, cycleActive: f32, growthAngle: f32, growthAnisotropy: f32, divisionBias: f32, growthFrameHeading: f32, }
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
        agents.reset_heading(23)
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
    core.set_material(1e4, 0.2, 3.0, 0.2, 400.0, 2.0, 0.0)
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


def check_compression_inhibition_strength(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    # E=0 isolates the feedback law: the deliberately compressed F cannot
    # elastically rebound above the reference before g2p evaluates Je.
    def grow_once(inhibition: float) -> float:
        core.set_material(
            0.0, 0.2, 3.0, 0.2, 50.0, 2.0, 0.85,
            growth_compression_inhibition=inhibition,
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
        return float(_probe(core, 1)[0, 1])

    inhibited = grow_once(1.0)
    uninhibited = grow_once(0.0)
    expected_full_rate = np.exp(50.0 * DT)
    assert 1.0 < inhibited < uninhibited, (inhibited, uninhibited)
    assert np.isclose(uninhibited, expected_full_rate, atol=2e-6), (uninhibited, expected_full_rate)
    print(f"[PASS] compression_inhibition strength1={inhibited:.6f} strength0={uninhibited:.6f}")


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
        2, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5,
    )
    # Exactly the center of texel (x=8,y=24), avoiding an argmax tie.
    position = np.array([[(8.5 / width), (24.5 / height)]], dtype=np.float32)
    core.load_scene(
        position,
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    agents.set_active_count(1)
    agents.reset_heading(5)
    agents.set_headings(np.array([0.0], dtype=np.float32))
    layout = weight_layout(channels, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Give the first four channels saturated deltas. On the following round
    # their cell-owned levels must splat at the particle, independent of heading.
    for channel in range(4):
        weights[layout["fc2b_offset"] + channel] = 20.0
    env_write_dim = channels
    weights[layout["fc2b_offset"] + env_write_dim + 6] = 20.0
    weights[layout["fc2b_offset"] + env_write_dim + 7] = -20.0
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

    meta = np.frombuffer(
        device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        ),
        dtype=agents._particle_meta_dtype, count=1,
    ).copy()
    np.testing.assert_allclose(meta["chemicalState"][0, :4], 1.0, atol=1e-7)
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
    print("[PASS] cell chemical deltas persist locally; transient Gaussian field rebuild discards old splats")

    meta_raw = device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        agents._particle_meta_dtype.itemsize,
    )
    meta = np.frombuffer(meta_raw, dtype=agents._particle_meta_dtype, count=1)
    assert np.allclose(meta["color"][0, :3], [1.0, 0.0, 0.5], atol=1e-6), meta["color"][0]
    assert np.isclose(meta["color"][0, 3], 1.0), meta["color"][0]
    print("[PASS] sigmoid RGB outputs are stored in particle state")


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
    color_start = channels + 6
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
        active_agents.reset_heading(29)
        active_agents.set_headings(np.array([heading], dtype=np.float32))
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


def check_conservative_split(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 0.91, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True, 2.0,
        4, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
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
    core.reset_growth_buffers(4)
    root2 = np.float32(np.sqrt(2.0))
    device.queue.write_buffer(core.F, 0, np.array([[root2, 0, 0, root2]], dtype=np.float32))
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.array([2.0]), np.ones(1)))
    device.queue.write_buffer(core.C, 0, parent_c)
    device.queue.write_buffer(core.velocities, 0, parent_velocity[None, :])
    core.set_active_count(1)
    environment.reset()
    agents.set_active_count(1)
    agents.reset_heading(7)
    neutral_weights = np.zeros(agents._total_floats, dtype=np.float32)
    layout = weight_layout(8, 128)
    neutral_weights[layout["fc2b_offset"] + 8 + 3] = -20.0
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
    assert count == 2
    assert np.allclose(positions.mean(axis=0), [0.5, 0.5], atol=1e-5), positions
    assert np.allclose(state[:, 0], 1.0, atol=1e-5), state
    assert np.isclose(state[:, 1].sum(), 2.0, atol=1e-5), state
    assert np.all(state[:, 2] == 0.0), state
    assert np.allclose(rest_state[0, 6:8], rest_state[1, 6:8], atol=1e-7), rest_state[:, 6:8]
    offsets = (positions - np.array([0.5, 0.5], dtype=np.float32) + 0.5) % 1.0 - 0.5
    c_matrix = parent_c.reshape(2, 2)
    expected_velocity = parent_velocity + offsets @ c_matrix.T
    assert np.allclose(affine, np.repeat(parent_c, 2, axis=0), atol=1e-6), affine
    assert np.allclose(velocities, expected_velocity, atol=1e-5), (velocities, expected_velocity)
    assert np.allclose(velocities.mean(axis=0), parent_velocity, atol=1e-6), velocities
    print("[PASS] conservative_split count=2 sum_g=2 Fe=identity center_and_apic_preserved")


def check_desired_heading_derives_angular_acceleration(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 32, 32, 1.0, 1.0)
    agents = AgentsGPU(
        device, core, environment, 8, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        2, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5,
    )
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    environment.reset()
    agents.set_active_count(1)
    agents.reset_heading(31)
    agents.set_headings(np.array([0.0], dtype=np.float32))
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Desired local heading points left (+lateral, +pi/2 from forward).
    weights[layout["fc2b_offset"] + 8 + 1] = 20.0
    agents.load_weights(weights)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
    device.queue.submit([encoder.finish()])
    raw = device.queue.read_buffer(
        agents._agent_state_buffer,
        PARTICLE_META_BUFFER_OFFSET,
        agents._particle_meta_dtype.itemsize,
    )
    meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1)
    # The proportional controller exceeds the configured 0.1 turn-rate cap,
    # so one unit communication step lands exactly at that cap.
    assert np.isclose(meta["angularVelocity"][0], 0.1, atol=2e-6), meta["angularVelocity"][0]
    assert np.isclose(meta["heading"][0], 0.1, atol=2e-6), meta["heading"][0]
    print("[PASS] desired heading vector derives bounded angular acceleration and persistent turn state")


def _polarized_split_case(
    device: wgpu.GPUDevice,
    signed_bias: float,
    polarity_bias: float = 20.0,
    directionality: float = 1.0,
) -> np.ndarray:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 8, 256, 256, 0.91, 1.0)
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
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.array([2.0]), np.ones(1)))
    core.set_active_count(1)
    environment.reset()
    agents.set_active_count(1)
    agents.set_division_directionality(directionality)
    agents.reset_heading(19)
    agents.set_headings(np.array([0.0], dtype=np.float32))
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Division-bias target plus desired local growth direction.
    weights[layout["fc2b_offset"] + 8 + 3] = polarity_bias
    weights[layout["fc2b_offset"] + 8 + 4] = signed_bias
    agents.load_weights(weights)
    # Let the persistent growth-angle state settle before the division event.
    for _ in range(16):
        encoder = device.create_command_encoder()
        environment.encode_sense(encoder)
        agents.encode_step(encoder, environment.parity, commit_lifecycle=False)
        device.queue.submit([encoder.finish()])
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    count = agents.read_grown_count()
    core.set_active_count(count)
    assert count == 2
    return core.read_positions()


def check_polarized_division_uses_signed_growth_direction(device: wgpu.GPUDevice) -> None:
    positive = _polarized_split_case(device, 20.0)
    negative = _polarized_split_case(device, -20.0)
    unbiased = _polarized_split_case(device, 20.0, -20.0)
    globally_symmetric = _polarized_split_case(device, 20.0, directionality=0.0)
    expected_positive = np.array([[0.5, 0.5], [0.51, 0.5]], dtype=np.float32)
    expected_negative = np.array([[0.5, 0.5], [0.49, 0.5]], dtype=np.float32)
    assert np.allclose(positive, expected_positive, atol=2e-6), positive
    assert np.allclose(negative, expected_negative, atol=2e-6), negative
    assert np.allclose(unbiased, [[0.495, 0.5], [0.505, 0.5]], atol=2e-6), unbiased
    assert np.allclose(globally_symmetric, [[0.495, 0.5], [0.505, 0.5]], atol=2e-6), globally_symmetric
    assert positive[:, 0].mean() > 0.5 and negative[:, 0].mean() < 0.5
    print("[PASS] signed growth direction places child; global directionality can restore symmetric division")


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
    agents.reset_heading(73)
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
    core.set_material(0.0, 0.2, 3.0, 1.0, growth_rate=50.0, growth_max=2.0, growth_threshold=0.0)
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
        growth_threshold=0.0,
        growth_anisotropy=global_anisotropy,
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
        growth_threshold=0.0,
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
    environment = EnvironmentGPU(device, 8, 256, 256, 0.91, 1.0)
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
    agents.set_active_count(1)
    agents.reset_heading(11)
    agents.set_headings(np.array([0.0], dtype=np.float32))
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"] + 8 + 2] = 2.0
    weights[layout["fc2b_offset"] + 8 + 3] = -2.0
    weights[layout["fc2b_offset"] + 8 + 4] = 1.0
    agents.load_weights(weights)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    state = core.read_rest_state()[0]
    expected_anisotropy = (1 / (1 + np.exp(-2))) * (1 - np.exp(-1))
    assert np.allclose(core.read_velocities(), initial_velocity, atol=1e-7)
    assert np.isclose(state[6], 0.0, atol=2e-6), state
    assert np.isclose(state[7], expected_anisotropy, atol=2e-6), state
    assert np.isclose(state[8], 1 / (1 + np.exp(2)), atol=2e-6), state
    print("[PASS] desired growth vector and anisotropy target smoothly update persistent state")


def check_p2g_fixed_point_headroom(device: wgpu.GPUDevice) -> None:
    """A deliberately crowded, fast transfer must retain momentum in i32."""
    core = MpmCore(device)
    count = 4096
    speed = 100.0
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(0.0, 0.2, 3.0, elasticity=0.0)
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
    assert np.isclose(mass, count, rtol=2e-3), mass
    assert np.isclose(momentum_x, count * speed, rtol=2e-3), momentum_x
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
) -> np.ndarray:
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
    device.queue.write_buffer(
        core.rest,
        0,
        _rest_state(np.ones(1), np.array([initial_growth]), np.array([initial_cycle])),
    )
    core.set_active_count(1)
    environment.reset()
    agents.set_active_count(1)
    agents.reset_heading(19)
    meta = np.frombuffer(
        device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        ),
        dtype=agents._particle_meta_dtype, count=1,
    ).copy()
    meta["chemicalState"][0, 7] = 1.0
    device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta.tobytes())
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))
    agents.set_growth_enabled(enabled)
    if runtime_cap is not None:
        agents.set_max_active_particles(runtime_cap)

    encoder = device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_lifecycle=commit_lifecycle)
    device.queue.submit([encoder.finish()])
    return _probe(core, 1)[0]


def check_cycle_start_gates(device: wgpu.GPUDevice) -> None:
    assert _cycle_gate_case(device, cap=4, enabled=True)[2] == 1.0
    assert _cycle_gate_case(device, cap=4, enabled=False)[2] == 0.0
    assert _cycle_gate_case(device, cap=1, enabled=True)[2] == 0.0
    assert _cycle_gate_case(device, cap=4, enabled=True, runtime_cap=1)[2] == 0.0
    assert _cycle_gate_case(device, cap=4, enabled=True, commit_lifecycle=False)[2] == 0.0

    # A cycle that began before other particles consumed the remaining
    # slots must be closed at cap without rolling back its accumulated g.
    capped = _cycle_gate_case(
        device,
        cap=1,
        enabled=True,
        initial_growth=1.4,
        initial_cycle=1.0,
    )
    assert capped[2] == 0.0, capped
    assert np.isclose(capped[1], 1.4), capped
    print("[PASS] cycle_start_gates final_round_only enabled=yes disabled=no static/runtime_cap=no capped_cycle_closed_g_preserved")


def check_interior_support_admission(device: wgpu.GPUDevice) -> None:
    """The live strength blends chemistry-only and occupancy-weighted hazard."""
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
    agents.reset_heading(41)
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))

    def reset_clock() -> None:
        raw = device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        )
        meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1).copy()
        meta["divisionHazard"][0] = 0.0
        meta["divisionThreshold"][0] = 10.0
        meta["chemicalState"][0, 7] = 0.2
        device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta.tobytes())

    def hazard_once(strength: float) -> float:
        reset_clock()
        agents.set_interior_support_strength(strength)
        encoder = device.create_command_encoder()
        core.encode_morphology(encoder)
        environment.encode_clear(encoder)
        agents.encode_splat_chemical_state(encoder)
        environment.encode_sense(encoder)
        agents.encode_step(encoder, environment.parity, commit_lifecycle=True)
        device.queue.submit([encoder.finish()])
        raw = device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        )
        return float(np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1)["divisionHazard"][0])

    unsupported = hazard_once(0.0)
    supported = hazard_once(1.0)
    morphology = core.read_morphology()
    occupancy = float(morphology[morphology.shape[0] // 2, morphology.shape[1] // 2])
    represented_signal = round(0.2 * 4096.0) / 4096.0
    expected_unsupported = -np.log(1.0 - represented_signal)
    expected_supported = -np.log(1.0 - represented_signal * occupancy)
    assert np.isclose(unsupported, expected_unsupported, atol=2e-6), (unsupported, expected_unsupported)
    assert np.isclose(supported, expected_supported, atol=2e-6), (supported, expected_supported, occupancy)
    assert 0.0 < supported < unsupported, (supported, unsupported)
    print(
        f"[PASS] interior_support occupancy={occupancy:.6f} "
        f"strength0_hazard={unsupported:.6f} strength1_hazard={supported:.6f}"
    )


def check_persistent_division_hazard(device: wgpu.GPUDevice) -> None:
    """Weak growth drive accumulates, survives zero-signal gaps, then admits."""
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
    agents.reset_heading(37)
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))
    meta = np.frombuffer(
        device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        ),
        dtype=agents._particle_meta_dtype, count=1,
    ).copy()
    meta["divisionThreshold"][0] = 10.0
    meta["chemicalState"][0, 7] = 0.2
    device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta.tobytes())

    def advance(rounds: int) -> None:
        encoder = device.create_command_encoder()
        for _ in range(rounds):
            environment.encode_clear(encoder)
            agents.encode_splat_chemical_state(encoder)
            environment.encode_sense(encoder)
            agents.encode_step(encoder, environment.parity, commit_lifecycle=True)
        device.queue.submit([encoder.finish()])

    def read_meta() -> np.ndarray:
        raw = device.queue.read_buffer(
            agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        )
        return np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1).copy()

    advance(3)
    accumulated = read_meta()
    represented_signal = round(0.2 * 4096.0) / 4096.0
    expected = 3.0 * -np.log(1.0 - represented_signal)
    assert np.isclose(accumulated["divisionHazard"][0], expected, atol=2e-6), accumulated
    assert _probe(core, 1)[0, 2] == 0.0

    accumulated["chemicalState"][0, 7] = 0.0
    device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, accumulated.tobytes())
    advance(2)
    paused = read_meta()
    assert np.isclose(paused["divisionHazard"][0], expected, atol=2e-6), paused

    paused["divisionThreshold"][0] = expected + 0.1
    paused["chemicalState"][0, 7] = 0.2
    device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, paused.tobytes())
    advance(1)
    admitted = read_meta()
    assert _probe(core, 1)[0, 2] == 1.0
    assert admitted["divisionHazard"][0] == 0.0
    assert admitted["divisionThreshold"][0] == 0.0
    print(
        "[PASS] persistent_division_hazard "
        f"weak_signal_accumulated={expected:.6f} zero_signal_preserved=yes threshold_admitted=yes"
    )


def check_stateful_private_memory(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0)
    agents = AgentsGPU(
        device, core, environment, 1, 64,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        4, 0.01, 1.0, 1.0, 0.2, 1.0, 0.5, 0.5,
        policy_architecture=STATEFUL_ARCHITECTURE,
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
    agents.reset_heading(31)
    layout = weight_layout(1, 64, STATEFUL_ARCHITECTURE)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Common outputs occupy C+6 rows. Drive private channel 0 positively and
    # open its corresponding gate; all other private channels remain still.
    weights[layout["fc2b_offset"] + 7] = 1.0
    weights[layout["fc2b_offset"] + 15] = 20.0
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
    daughters = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=2)
    np.testing.assert_allclose(daughters[0]["privateState"], daughters[1]["privateState"], atol=1e-7)
    print("[PASS] stateful policy applies speed-scaled gated memory, freezes at 0x, derives RGB, and inherits state at division")


def main() -> None:
    device = pick_device()
    check_morphology_occupancy(device)
    check_single_cell_rollout_seed(device)
    check_supersampled_communication_rounds(device)
    check_growth_without_repulsion(device)
    check_compression_inhibition_strength(device)
    check_transient_cell_chemical_splats(device)
    check_elastic_strain_policy_inputs(device)
    check_conservative_split(device)
    check_desired_heading_derives_angular_acceleration(device)
    check_polarized_division_uses_signed_growth_direction(device)
    check_anisotropic_tensor_split(device)
    check_isotropic_increment_preserves_tensor_shape(device)
    check_directional_increment_and_objectivity(device)
    check_growth_duration_is_substep_invariant(device)
    check_persistent_growth_targets_drive_state_not_motion(device)
    check_p2g_fixed_point_headroom(device)
    check_high_strain_elastic_stability(device)
    check_cycle_start_gates(device)
    check_interior_support_admission(device)
    check_persistent_division_hazard(device)
    check_stateful_private_memory(device)


if __name__ == "__main__":
    main()

"""Focused GPU checks for the conservative grow-then-divide model.

Run from this directory with ``.venv/bin/python growth_check.py``.
"""
from __future__ import annotations

import numpy as np
import wgpu

from agents_gpu import AgentsGPU, weight_layout
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import DT, MpmCore


def _rest_state(
    jp: np.ndarray,
    growth: np.ndarray,
    cycle: np.ndarray,
    growth_f: np.ndarray | None = None,
    growth_direction: np.ndarray | None = None,
) -> np.ndarray:
    count = len(jp)
    out = np.zeros((count, 8), dtype=np.float32)
    if growth_f is None:
        root = np.sqrt(np.asarray(growth, dtype=np.float32))
        out[:, 0] = root
        out[:, 3] = root
    else:
        out[:, :4] = np.asarray(growth_f, dtype=np.float32).reshape(count, 4)
    out[:, 4] = jp
    out[:, 5] = cycle
    if growth_direction is not None:
        out[:, 6:8] = np.asarray(growth_direction, dtype=np.float32)
    return out


def _probe(core: MpmCore, count: int) -> np.ndarray:
    """Returns [Je, g, cycleActive] without adding COPY_SRC to hot buffers."""
    shader = core.device.create_shader_module(
        code="""
        struct Rest { growthF: vec4<f32>, jp: f32, cycleActive: f32, growthDirection: vec2<f32>, }
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


def check_compression_slows_without_stalling(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    # E=0 isolates the feedback law: the deliberately compressed F cannot
    # elastically rebound above the reference before g2p evaluates Je.
    core.set_material(0.0, 0.2, 3.0, 0.2, 50.0, 2.0, 0.85)
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    device.queue.write_buffer(core.rest, 0, _rest_state(np.ones(1), np.ones(1), np.ones(1)))
    core.step(1)
    growth = float(_probe(core, 1)[0, 1])
    assert growth > 1.0, growth
    assert growth < np.exp(50.0 * DT), growth
    print(f"[PASS] compressed_growth_continues g={growth:.6f}")


def check_deposit_is_centered_under_particle(device: wgpu.GPUDevice) -> None:
    """A legacy nonzero deposit distance must not move or duplicate writes."""
    channels = 8
    width = height = 32
    core = MpmCore(device)
    environment = EnvironmentGPU(device, channels, width, height, 0.91, 1.0)
    agents = AgentsGPU(
        device, core, environment, channels, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, True,
        100.0,  # deliberately huge legacy depositDistance: must be ignored
        2, 0.01, 1.0, 1.0, 0.4, 1.0, 0.5, 0.5,
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
    layout = weight_layout(channels, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"]] = 20.0  # saturate channel 0 write
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
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups((count + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])

    scratch = np.frombuffer(device.queue.read_buffer(readback), np.int32).reshape(channels, height, width)
    max_y, max_x = np.unravel_index(np.argmax(scratch[0]), scratch[0].shape)
    assert (max_x, max_y) == (8, 24), (max_x, max_y)
    assert scratch[0, max_y, max_x] > 0
    assert not np.any(scratch[1:]), "one output channel leaked into another"
    print("[PASS] one chemical output deposits directly under the particle; legacy distance is ignored")


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
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))

    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    environment.encode_merge_and_decay(encoder)
    device.queue.submit([encoder.finish()])
    count = agents.read_grown_count()
    core.set_active_count(count)

    positions = core.read_positions()
    velocities = core.read_velocities()
    affine = core.read_affine()
    state = _probe(core, count)
    assert count == 2
    assert np.allclose(positions.mean(axis=0), [0.5, 0.5], atol=1e-5), positions
    assert np.allclose(state[:, 0], 1.0, atol=1e-5), state
    assert np.isclose(state[:, 1].sum(), 2.0, atol=1e-5), state
    assert np.all(state[:, 2] == 0.0), state
    offsets = (positions - np.array([0.5, 0.5], dtype=np.float32) + 0.5) % 1.0 - 0.5
    c_matrix = parent_c.reshape(2, 2)
    expected_velocity = parent_velocity + offsets @ c_matrix.T
    assert np.allclose(affine, np.repeat(parent_c, 2, axis=0), atol=1e-6), affine
    assert np.allclose(velocities, expected_velocity, atol=1e-5), (velocities, expected_velocity)
    assert np.allclose(velocities.mean(axis=0), parent_velocity, atol=1e-6), velocities
    print("[PASS] conservative_split count=2 sum_g=2 Fe=identity center_and_apic_preserved")


def _polarized_split_case(device: wgpu.GPUDevice, signed_bias: float) -> np.ndarray:
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
    agents.reset_heading(19)
    agents.set_headings(np.array([0.0], dtype=np.float32))
    layout = weight_layout(8, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Former local-forward strafe output. +/-20 saturates tanh at +/-1,
    # making the expected fully-polarized geometry exact enough to test.
    weights[layout["fc2b_offset"] + 8 + 3] = signed_bias
    agents.load_weights(weights)
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
    expected_positive = np.array([[0.5, 0.5], [0.51, 0.5]], dtype=np.float32)
    expected_negative = np.array([[0.5, 0.5], [0.49, 0.5]], dtype=np.float32)
    assert np.allclose(positive, expected_positive, atol=2e-6), positive
    assert np.allclose(negative, expected_negative, atol=2e-6), negative
    assert positive[:, 0].mean() > 0.5 and negative[:, 0].mean() < 0.5
    print("[PASS] signed growth direction places child and shifts division center toward +n")


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


def _directional_increment_case(device: wgpu.GPUDevice, rotation_angle: float) -> tuple[np.ndarray, float]:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    core.set_material(0.0, 0.2, 3.0, 1.0, growth_rate=50.0, growth_max=2.0, growth_threshold=0.0)
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
        _rest_state(np.ones(1), np.ones(1), np.ones(1), growth_direction=world_direction[None]),
    )
    core.step(1)
    return core.read_rest_state()[0, :4].reshape(2, 2), float(np.exp(50.0 * DT))


def check_directional_increment_and_objectivity(device: wgpu.GPUDevice) -> None:
    fg_axis, expected_area = _directional_increment_case(device, 0.0)
    fg_rotated, _ = _directional_increment_case(device, 0.83)
    # Full-strength direction puts the complete area increment along n;
    # the perpendicular rest stretch stays one. Pulling a rotated world
    # direction back through Re must produce the same intermediate Fg.
    assert np.isclose(np.linalg.det(fg_axis), expected_area, rtol=2e-5)
    assert np.allclose(fg_axis, np.diag([expected_area, 1.0]), atol=3e-6), fg_axis
    assert np.allclose(fg_rotated, fg_axis, atol=3e-6), (fg_axis, fg_rotated)
    print("[PASS] directional increment preserves det growth and is rotation-objective")


def check_strafe_signal_drives_growth_not_motion(device: wgpu.GPUDevice) -> None:
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
    weights[layout["fc2b_offset"] + 8 + 3] = 1.0
    agents.load_weights(weights)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    direction = core.read_rest_state()[0, 6:8]
    assert np.allclose(core.read_velocities(), initial_velocity, atol=1e-7)
    assert np.allclose(direction, [np.tanh(1.0), 0.0], atol=2e-6), direction
    print("[PASS] former strafe output writes growth direction while maxStrafe=0 leaves velocity unchanged")


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
    # The final substrate channel is the cycle-start probability. Fill it
    # with exactly 1 so the random draw always succeeds when both gates do.
    plane = 256 * 256
    device.queue.write_buffer(
        environment.buffers[0],
        7 * plane * 4,
        np.ones(plane, dtype=np.float32),
    )
    agents.set_active_count(1)
    agents.reset_heading(19)
    agents.load_weights(np.zeros(agents._total_floats, dtype=np.float32))
    agents.set_growth_enabled(enabled)
    if runtime_cap is not None:
        agents.set_max_active_particles(runtime_cap)

    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    return _probe(core, 1)[0]


def check_cycle_start_gates(device: wgpu.GPUDevice) -> None:
    assert _cycle_gate_case(device, cap=4, enabled=True)[2] == 1.0
    assert _cycle_gate_case(device, cap=4, enabled=False)[2] == 0.0
    assert _cycle_gate_case(device, cap=1, enabled=True)[2] == 0.0
    assert _cycle_gate_case(device, cap=4, enabled=True, runtime_cap=1)[2] == 0.0

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
    print("[PASS] cycle_start_gates enabled=yes disabled=no static/runtime_cap=no capped_cycle_closed_g_preserved")


def main() -> None:
    device = pick_device()
    check_growth_without_repulsion(device)
    check_compression_slows_without_stalling(device)
    check_deposit_is_centered_under_particle(device)
    check_conservative_split(device)
    check_polarized_division_uses_signed_growth_direction(device)
    check_anisotropic_tensor_split(device)
    check_isotropic_increment_preserves_tensor_shape(device)
    check_directional_increment_and_objectivity(device)
    check_strafe_signal_drives_growth_not_motion(device)
    check_p2g_fixed_point_headroom(device)
    check_high_strain_elastic_stability(device)
    check_cycle_start_gates(device)


if __name__ == "__main__":
    main()

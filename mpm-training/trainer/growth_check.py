"""Focused GPU checks for the conservative grow-then-divide model.

Run from this directory with ``.venv/bin/python growth_check.py``.
"""
from __future__ import annotations

import numpy as np
import wgpu

from agents_gpu import AgentsGPU
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import DT, MpmCore


def _probe(core: MpmCore, count: int) -> np.ndarray:
    """Returns [Je, g, cycleActive] without adding COPY_SRC to hot buffers."""
    shader = core.device.create_shader_module(
        code="""
        struct Rest { jp: f32, growth: f32, cycleActive: f32, }
        @group(0) @binding(0) var<storage, read> particleF: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read> rest: array<Rest>;
        @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let f = particleF[gid.x];
          let g = max(rest[gid.x].growth, 1e-6);
          out[gid.x] = vec4<f32>(
            (f.x * f.w - f.y * f.z) / g,
            rest[gid.x].growth,
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
    device.queue.write_buffer(core.rest, 0, np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32))
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
    device.queue.write_buffer(core.rest, 0, np.array([[1, 1, 1]], dtype=np.float32))
    core.step(1)
    growth = float(_probe(core, 1)[0, 1])
    assert growth > 1.0, growth
    assert growth < np.exp(50.0 * DT), growth
    print(f"[PASS] compressed_growth_continues g={growth:.6f}")


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
    device.queue.write_buffer(core.rest, 0, np.array([[1, 2, 1]], dtype=np.float32))
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
        np.array([[1.0, initial_growth, initial_cycle]], dtype=np.float32),
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

    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity)
    device.queue.submit([encoder.finish()])
    return _probe(core, 1)[0]


def check_cycle_start_gates(device: wgpu.GPUDevice) -> None:
    assert _cycle_gate_case(device, cap=4, enabled=True)[2] == 1.0
    assert _cycle_gate_case(device, cap=4, enabled=False)[2] == 0.0
    assert _cycle_gate_case(device, cap=1, enabled=True)[2] == 0.0

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
    print("[PASS] cycle_start_gates enabled=yes disabled=no cap=no capped_cycle_closed_g_preserved")


def main() -> None:
    device = pick_device()
    check_growth_without_repulsion(device)
    check_compression_slows_without_stalling(device)
    check_conservative_split(device)
    check_p2g_fixed_point_headroom(device)
    check_high_strain_elastic_stability(device)
    check_cycle_start_gates(device)


if __name__ == "__main__":
    main()

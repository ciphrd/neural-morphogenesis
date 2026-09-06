"""Focused GPU checks for field-integrated growth and resampling.

Run from this directory with ``.venv/bin/python growth_check.py``.
"""
from __future__ import annotations

import numpy as np
import wgpu

from agents_gpu import PARTICLE_META_BUFFER_OFFSET, AgentsGPU, weight_layout
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import DT, GRID_N, REPULSION_FIELD_N, MpmCore, ceil_div
from policy_parameters import PERSISTENT_ENVIRONMENT_ARCHITECTURE, STATEFUL_128_ARCHITECTURE
from training_sim import TrainingRollout

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
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0, chemical_communication_architecture="cell-owned-projection")
    agents = AgentsGPU(device, core, environment, 1, 128, 1.0, 4, 0.01, 1.0, 1.0, 0.5, 0.5, chemical_communication_architecture="cell-owned-projection")
    TrainingRollout(
        core, agents, environment,
        spawn_center=(0.5, 0.5), gravity=0.0, seed=17, initial_particle_count=1,
    )
    assert core.read_positions().shape == (2, 2)
    assert agents.read_sample_count() == 2
    np.testing.assert_allclose(core.read_rest_state()[:, 15], .5)
    # A capacity-limited restart must keep complete seed pairs and must clear
    # the previous rollout's domains before writing the new geometry.
    agents.set_max_active_particles(3)
    TrainingRollout(core, agents, environment, spawn_center=(.4,.4), gravity=0, seed=17, initial_particle_count=1)
    assert core.active_count == agents.read_sample_count() == 2
    np.testing.assert_allclose(core.read_rest_state()[:, 15].sum(),1)
    np.testing.assert_allclose(core.read_positions().mean(axis=0),[.4,.4],atol=1e-7)
    print("[PASS] one seed cell starts as two half-weight triangle samples")

def check_supersampled_communication_rounds(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 64, 64, 0.5, 1.0, chemical_communication_architecture="cell-owned-projection")
    agents = AgentsGPU(device, core, environment, 1, 128, 1.0, 2, 0.01, 1.0, 1.0, 0.5, 0.5, chemical_communication_architecture="cell-owned-projection")
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
        agents.reset_state()
        communication_dt = environment.set_communication_timestep(rounds, 1.0)
        agents.set_communication_timestep(communication_dt)
        encoder = device.create_command_encoder()
        core.encode_morphology(encoder)
        for communication_round in range(rounds):
            environment.encode_clear(encoder)
            agents.encode_splat_chemical_state(encoder)
            environment.encode_sense(encoder)
            agents.encode_step(encoder, environment.parity, commit_growth=communication_round == rounds - 1)
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

def check_persistent_environment_chemistry(device: wgpu.GPUDevice) -> None:
    """Communication rounds exchange chemistry at fixed total chemical time."""
    channels = 1
    width = height = 16
    decay = 0.81
    core = MpmCore(device)
    environment = EnvironmentGPU(device, channels, width, height, decay, 1.0, PERSISTENT_ENVIRONMENT_ARCHITECTURE)
    agents = AgentsGPU(device, core, environment, channels, 128, 1.0, 2, 0.01, 1.0, 0.0, 0.5, 0.5, chemical_communication_architecture=PERSISTENT_ENVIRONMENT_ARCHITECTURE)
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    core.set_active_count(1)
    agents.set_active_count(1)
    agents.reset_state()
    layout = weight_layout(channels, 128)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"]] = 20.0
    agents.load_weights(weights)

    def macro_tick(rounds: int) -> np.ndarray:
        communication_dt = environment.set_communication_timestep(rounds, 1.0)
        agents.set_communication_timestep(communication_dt)
        encoder = device.create_command_encoder()
        for communication_round in range(rounds):
            final_round = communication_round == rounds - 1
            environment.encode_prepare_persistent(encoder, transport=communication_round == 0)
            environment.encode_clear(encoder)
            environment.encode_sense(encoder)
            agents.encode_step(
                encoder, environment.parity, commit_growth=final_round
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

    # Constant secretion has the same integrated mass, but earlier rounds'
    # deposits spread before the next neural evaluation.
    environment.reset()
    reset_cellular_chemistry()
    deposited_four = macro_tick(4)
    np.testing.assert_allclose(deposited_four.sum(), deposited_once.sum(), rtol=2e-5, atol=2e-5)
    assert deposited_four.max() < deposited_once.max()
    environment.reset()
    deposited_three = macro_tick(3)
    np.testing.assert_allclose(deposited_three.sum(), deposited_once.sum(), rtol=2e-5, atol=2e-5)
    environment.reset()
    deposited_four = macro_tick(4)

    agents.load_weights(np.zeros_like(weights))
    reset_cellular_chemistry()
    agents.set_active_count(0)
    aged = macro_tick(4)
    np.testing.assert_allclose(aged.sum(), deposited_four.sum() * decay, rtol=2e-5, atol=2e-5)
    assert aged.max() < deposited_four.max(), (aged.max(), deposited_four.max())
    # Route sensed concentration through a hidden neuron into secretion.
    # Later rounds must respond to earlier deposits within the same macro tick.
    agents.set_active_count(1)
    weights[:] = 0.0
    weights[layout["fc1w_offset"]] = 100.0
    weights[layout["fc2w_offset"]] = 1.0
    weights[layout["fc2b_offset"]] = 0.1
    agents.load_weights(weights)
    environment.reset()
    feedback_once = macro_tick(1)
    environment.reset()
    feedback_four = macro_tick(4)
    assert feedback_four.sum() > feedback_once.sum() * 1.01, (feedback_once.sum(), feedback_four.sum())
    print("[PASS] persistent rounds preserve source/decay time, diffuse deposits, and exchange neural feedback")

def check_persistent_substrate_advection(device: wgpu.GPUDevice) -> None:
    """A uniform MPM velocity translates persistent chemistry one texel."""
    width = height = 16
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, width, height, 1.0, 1.0, PERSISTENT_ENVIRONMENT_ARCHITECTURE, grid_velocity=core.grid_vel)
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
    for _ in range(3):
        environment.encode_prepare_persistent(encoder, transport=False)
    device.queue.submit([encoder.finish()])
    moved = np.frombuffer(
        device.queue.read_buffer(environment.buffers[environment.parity]), np.float32
    ).reshape(height, width)
    assert np.isclose(moved[8, 5], 1.0, atol=2e-5), moved[8]
    assert np.isclose(moved.sum(), 1.0, atol=2e-5), moved.sum()
    print("[PASS] persistent substrate is advected with the MPM velocity field")

def check_stateful_private_memory(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0, chemical_communication_architecture="cell-owned-projection")
    agents = AgentsGPU(device, core, environment, 1, 128, 1.0, 4, 0.01, 1.0, 1.0, 0.5, 0.5, policy_architecture=STATEFUL_128_ARCHITECTURE, internal_state_speed=1.0, chemical_communication_architecture="cell-owned-projection")
    core.load_scene(
        np.array([[0.5, 0.5]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    agents.set_active_count(1)
    agents.set_growth_enabled(False)
    agents.reset_state()
    layout = weight_layout(1, 128, STATEFUL_128_ARCHITECTURE)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    # Resolve logical head offsets instead of the retired motion/division ABI.
    from policy_parameters import policy_heads
    head_offsets = {}
    offset = 0
    for head in policy_heads(1, STATEFUL_128_ARCHITECTURE):
        head_offsets[head.name] = offset
        offset += head.size
    weights[layout["fc2b_offset"] + head_offsets["stateDelta"]] = 1.0
    weights[layout["fc2b_offset"] + head_offsets["stateGate"]] = 20.0
    color_offset = layout["fc2b_offset"] + head_offsets["color"]
    color_logits = np.array([-20.0, 0.7, 20.0], dtype=np.float32)
    weights[color_offset:color_offset + 3] = color_logits
    agents.load_weights(weights)
    agents.set_communication_timestep(0.25)

    encoder = device.create_command_encoder()
    core.encode_morphology(encoder)
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_growth=False)
    device.queue.submit([encoder.finish()])
    raw = device.queue.read_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, agents._particle_meta_dtype.itemsize
    )
    meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1)[0]
    expected_state = np.tanh(1.0) * 0.25
    assert np.isclose(meta["privateState"][0], expected_state, atol=2e-6), meta
    assert np.allclose(meta["privateState"][1:], 0.0, atol=1e-7), meta
    expected_color = 1.0 / (1.0 + np.exp(-color_logits))
    np.testing.assert_allclose(meta["color"][:3], expected_color, atol=2e-6)
    assert np.all((meta["color"][:3] >= 0) & (meta["color"][:3] <= 1))
    agents.set_internal_state_speed(0.0)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_growth=False)
    device.queue.submit([encoder.finish()])
    frozen_raw = device.queue.read_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, agents._particle_meta_dtype.itemsize
    )
    frozen = np.frombuffer(frozen_raw, dtype=agents._particle_meta_dtype, count=1)[0]
    np.testing.assert_allclose(frozen["privateState"], meta["privateState"], atol=1e-7)
    np.testing.assert_allclose(frozen["color"][:3], expected_color, atol=2e-6)
    agents.set_internal_state_speed(1.0)
    root2 = np.float32(np.sqrt(2.0))
    device.queue.write_buffer(core.F, 0, np.array([[root2, 0, 0, root2]], dtype=np.float32))
    from triangle_vertices import vertices_from_edges
    rest = core.read_rest_state()
    rest[:, :4] = [root2, 0, 0, root2]
    rest[:, 8:14] = vertices_from_edges(core.read_positions(), np.array([[.03,0,0,.03]]))
    rest[:, 14] = .03**2 / 4
    device.queue.write_buffer(core.rest, 0, rest)
    encoder = device.create_command_encoder()
    environment.encode_sense(encoder)
    agents.encode_step(encoder, environment.parity, commit_growth=True)
    device.queue.submit([encoder.finish()])
    assert agents.read_sample_count() == 2
    raw = device.queue.read_buffer(
        agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, 2 * agents._particle_meta_dtype.itemsize
    )
    samples = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=2)
    np.testing.assert_allclose(samples[0]["privateState"], samples[1]["privateState"], atol=1e-7)
    print("[PASS] recurrent-128 policy emits bounded RGB independently of memory and new material samples inherit private state")

def check_forced_growth_direction(device):
    """Exercise the shared uniform offsets for retained explicit Lab directions."""
    from continuous_growth_check import make_system, load_samples, read_rest
    core, agents = make_system(device)
    load_samples(core, agents, [[.5, .5]], [[0, 0]])
    agents.load_weights(np.zeros(agents._total_floats, np.float32))
    for direction in [(1., 0.), (0., 1.)]:
        device.queue.write_buffer(agents._physics_uniform, 36, np.array([0, 1], np.uint32))
        device.queue.write_buffer(agents._physics_uniform, 48, np.array(direction, np.float32))
        device.queue.write_buffer(agents._physics_uniform, 56, np.array([0], np.uint32))
        encoder = device.create_command_encoder()
        agents.encode_step(encoder, 0)
        device.queue.submit([encoder.finish()])
        np.testing.assert_allclose(read_rest(core, 1)[0, 5:7], direction, atol=1e-6)
    device.queue.write_buffer(agents._physics_uniform, 36, np.array([0xffffffff, 0], np.uint32))
    encoder = device.create_command_encoder()
    agents.encode_step(encoder, 0)
    device.queue.submit([encoder.finish()])
    np.testing.assert_allclose(read_rest(core, 1)[0, 5:7], [0, 0], atol=1e-6)
    print('[PASS] explicit horizontal/vertical growth and return to policy control')

def main() -> None:
    device = pick_device()
    check_forced_growth_direction(device)
    check_morphology_occupancy(device)
    check_single_cell_rollout_seed(device)
    check_supersampled_communication_rounds(device)
    check_persistent_environment_chemistry(device)
    check_persistent_substrate_advection(device)
    check_stateful_private_memory(device)
    from continuous_growth_check import main as check_domains
    check_domains()

if __name__ == "__main__":
    main()

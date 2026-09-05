"""Focused GPU regressions for continuous field growth and adaptive sampling.

Run from this directory with ``.venv/bin/python continuous_growth_check.py``.
"""
from __future__ import annotations

import numpy as np
import wgpu

from agents_gpu import AgentsGPU
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import DT, MpmCore


def make_system(device: wgpu.GPUDevice, capacity: int = 8) -> tuple[MpmCore, AgentsGPU]:
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0)
    agents = AgentsGPU(
        device, core, environment, 1, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        capacity, 0.01, 0.0, 1.0, 0.4, 1.0, 0.5, 0.5,
    )
    core.set_gravity(0.0)
    core.set_repulsion_strength(0.0, 40.0)
    return core, agents


def load_samples(
    core: MpmCore,
    agents: AgentsGPU,
    positions: np.ndarray,
    vectors: np.ndarray,
    growth_f: np.ndarray | None = None,
) -> None:
    count = len(positions)
    identity = np.tile(np.array([1, 0, 0, 1], np.float32), (count, 1))
    core.load_scene(
        np.asarray(positions, np.float32), np.zeros((count, 2), np.float32),
        identity, np.zeros((count, 4), np.float32), np.ones(count, np.float32),
    )
    rest = np.zeros((count, 12), np.float32)
    rest[:, :4] = identity if growth_f is None else np.asarray(growth_f, np.float32)
    rest[:, 4] = 1.0
    rest[:, 5:7] = np.asarray(vectors, np.float32)
    rest[:, 10] = 1.0
    device = core.device
    device.queue.write_buffer(core.rest, 0, rest)
    core.set_active_count(count)
    agents.set_active_count(count)


def run_growth_field(device: wgpu.GPUDevice, agents: AgentsGPU) -> None:
    encoder = device.create_command_encoder()
    agents.encode_growth_field(encoder)
    device.queue.submit([encoder.finish()])


def read_rest(core: MpmCore, count: int) -> np.ndarray:
    raw = core.device.queue.read_buffer(core.rest, 0, count * 12 * 4)
    return np.frombuffer(raw, np.float32).reshape(count, 12).copy()


def normalized_center_field(core: MpmCore) -> tuple[np.ndarray, np.ndarray]:
    field = np.frombuffer(core.device.queue.read_buffer(core.growth_field), np.int32).reshape(-1, 7)
    populated = field[:, 5] > 0
    assert np.any(populated)
    weight = float(field[populated, 5].astype(np.float64).sum())
    vector = field[populated, :2].astype(np.float64).sum(axis=0) / weight
    tensor = field[populated, 2:5].astype(np.float64).sum(axis=0) / weight
    return vector, tensor


def check_continuous_volume(device: wgpu.GPUDevice) -> None:
    core, agents = make_system(device)
    load_samples(core, agents, np.array([[0.5, 0.5]]), np.array([[0.5, 0.0]]))
    run_growth_field(device, agents)
    rate = 12.0
    core.set_material(0.0, 0.2, 3.0, 0.0, growth_rate=rate, growth_compression_feedback=0.0)
    core.step(1)
    fg = read_rest(core, 1)[0, :4]
    area = float(fg[0] * fg[3] - fg[1] * fg[2])
    expected = float(np.exp(0.5 * rate * DT))
    np.testing.assert_allclose(area, expected, rtol=2e-5, atol=2e-6)
    print(f"[PASS] continuous volume area={area:.8f} expected={expected:.8f}")


def check_field_normalization_and_opposition(device: wgpu.GPUDevice) -> None:
    core, agents = make_system(device)
    p = np.array([[0.5, 0.5]], np.float32)
    load_samples(core, agents, p, np.array([[0.6, 0.0]]))
    run_growth_field(device, agents)
    vector_one, tensor_one = normalized_center_field(core)

    load_samples(core, agents, np.repeat(p, 2, axis=0), np.array([[0.6, 0.0], [0.6, 0.0]]))
    run_growth_field(device, agents)
    vector_two, tensor_two = normalized_center_field(core)
    np.testing.assert_allclose(vector_two, vector_one, atol=3e-6)
    np.testing.assert_allclose(tensor_two, tensor_one, atol=3e-6)

    load_samples(core, agents, np.repeat(p, 2, axis=0), np.array([[0.6, 0.0], [-0.6, 0.0]]))
    run_growth_field(device, agents)
    vector_opposed, tensor_opposed = normalized_center_field(core)
    np.testing.assert_allclose(vector_opposed, 0.0, atol=3e-6)
    np.testing.assert_allclose(tensor_opposed, np.array([0.6, 0.0, 0.0]), atol=1e-3)
    print("[PASS] field is sample-count normalized and opposing vectors retain axial growth")


def check_conservative_resampling(device: wgpu.GPUDevice) -> None:
    core, agents = make_system(device, capacity=4)
    root_two = np.sqrt(2.0).astype(np.float32)
    load_samples(
        core, agents, np.array([[0.5, 0.5]]), np.array([[0.0, 0.0]]),
        np.array([[root_two, 0.0, 0.0, root_two]], np.float32),
    )
    run_growth_field(device, agents)
    assert agents.read_grown_count() == 2
    rest = read_rest(core, 2)
    areas = rest[:, 0] * rest[:, 3] - rest[:, 1] * rest[:, 2]
    np.testing.assert_allclose(areas, np.ones(2), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(float(areas.sum()), 2.0, rtol=2e-5, atol=2e-5)
    print(f"[PASS] resampling conserves represented area: {areas.tolist()}")


def main() -> None:
    device = pick_device()
    check_continuous_volume(device)
    check_field_normalization_and_opposition(device)
    check_conservative_resampling(device)


if __name__ == "__main__":
    main()

"""GPU invariants for area-weighted quadratic chemical transfer."""
from __future__ import annotations

import numpy as np

from agents_gpu import AgentsGPU, PARTICLE_META_BUFFER_OFFSET
from device import pick_device
from capture_policy_inputs import PolicyInputProbe, META_NAMES
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore

def lattice(n: int) -> np.ndarray:
    x, y = np.meshgrid((np.arange(n) + 0.375) / n, (np.arange(n) + 0.625) / n)
    return np.column_stack((x.ravel(), y.ravel())).astype(np.float32)

def main() -> None:
    device = pick_device()
    core = MpmCore(device)
    levels = np.array([0.6, -0.4, 0, 0.2, -0.1, 0.8, -0.7, 0.3], np.float32)
    for resolution in (16, 32, 64):
        env = EnvironmentGPU(device, 8, resolution, resolution, 1.0, 1.0, normalize_deposits_by_local_density=True, chemical_communication_architecture="cell-owned-projection")
        agents = AgentsGPU(device, core, env, 8, 128, 1.0, 32768, 0.01, 1.0, 1.0, 0.5, 0.5, chemical_communication_architecture="cell-owned-projection")

        def project(pos, areas, values=levels, growth=1.0):
            count = len(pos)
            core.load_scene(pos, np.zeros((count, 2), np.float32),
                            np.tile([1, 0, 0, 1], (count, 1)).astype(np.float32),
                            np.zeros((count, 4), np.float32), np.ones(count, np.float32))
            rest = core.read_rest_state()
            rest[:, 14] = areas
            rest[:, 0] = rest[:, 3] = np.sqrt(growth)
            device.queue.write_buffer(core.rest, 0, rest.astype(np.float32))
            agents.set_active_count(count)
            agents.reset_state()
            raw = device.queue.read_buffer(agents._agent_state_buffer,
                                           PARTICLE_META_BUFFER_OFFSET,
                                           count * agents._particle_meta_dtype.itemsize)
            meta = np.frombuffer(raw, dtype=agents._particle_meta_dtype).copy()
            meta['chemicalState'][:] = values
            device.queue.write_buffer(agents._agent_state_buffer,
                                      PARTICLE_META_BUFFER_OFFSET, meta.tobytes())
            env.reset()
            encoder = device.create_command_encoder()
            env.encode_clear(encoder)
            agents.encode_splat_chemical_state(encoder)
            env.encode_sense(encoder)
            device.queue.submit([encoder.finish()])
            return np.frombuffer(device.queue.read_buffer(env.buffers[env.parity]),
                                 np.float32).reshape(8, resolution, resolution).copy()

        # Same fully covered material with different quadrature densities and
        # sub-texel phases. Constant expression must be reproduced everywhere.
        for sampling in (64, 128):
            pos = lattice(sampling)
            field = project(pos, 1 / len(pos))
            np.testing.assert_allclose(field, np.broadcast_to(levels[:, None, None], field.shape), atol=2e-6)
        probe = PolicyInputProbe(device, core, agents, env, 4, .15, True, .045)
        rows = probe.capture(core, env)
        np.testing.assert_allclose(rows[:, len(META_NAMES):len(META_NAMES)+8],
                                   np.broadcast_to(levels, (4, 8)), atol=2e-6)
        pos = lattice(64)
        # Physically compress the full sheet into a half-width strip. Interior
        # expression is unchanged despite doubling represented area density.
        pos[:, 0] = .25 + .5 * pos[:, 0]
        field = project(pos, 1 / len(pos))
        interior = field[:, :, resolution // 4 + 2:3 * resolution // 4 - 2]
        np.testing.assert_allclose(interior, np.broadcast_to(levels[:, None, None], interior.shape), atol=2e-6)
        np.testing.assert_array_equal(field[:, :, :resolution // 4 - 2], 0)

        # Split a point quadrature sample without moving it: signed fields and
        # coverage are invariant. Exercise toroidal wrapping at the same time.
        point = np.array([[.999, .001]], np.float32)
        area = 0.2 / resolution**2
        base = project(point, area)
        split = project(np.repeat(point, 17, axis=0), area / 17)
        np.testing.assert_allclose(split, base, atol=1e-7)
        np.testing.assert_allclose(base.sum(axis=(1, 2)) / resolution**2, area * levels, rtol=2e-6, atol=1e-12)
        grown = project(point, area, growth=2.)
        np.testing.assert_allclose(grown, 2 * base, atol=1e-7)
        assert base[0, 0, 0] > 0 and base[0, -1, -1] > 0

        # Opposing expression cancels in the numerator, without cancelling area.
        cancelled = project(np.repeat(point, 2, axis=0), area,
                            np.stack([levels, -levels]))
        np.testing.assert_allclose(cancelled, 0, atol=1e-7)
        # Small weights previously lost to fixed-point rounding still contribute.
        tiny = project(point, 1e-12)
        np.testing.assert_allclose(tiny.sum(axis=(1, 2)) / resolution**2,
                                   1e-12 * levels, rtol=2e-6, atol=1e-19)
        # Secretion mode retains physical quantity per texel area.
        env.set_deposit_normalization(False)
        env.set_communication_timestep(1, 1.)
        secreted = project(lattice(64), 2 / 64**2)
        np.testing.assert_allclose(secreted, np.broadcast_to(2 * levels[:, None, None], secreted.shape), atol=2e-6)
        print(f'[PASS] {resolution}²: resolution/refinement, compression, coverage, growth, wrapping, cancellation, tiny areas, secretion')

if __name__ == '__main__':
    main()

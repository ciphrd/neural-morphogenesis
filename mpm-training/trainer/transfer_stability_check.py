"""GPU regressions for spurious energy from small-weight grid transfers."""
import numpy as np
from continuous_growth_check import make_system, load_samples, read_rest, p2g_grid
from device import pick_device
from mpm_core import MpmCore

def check_force_free_motion(device):
    core, agents = make_system(device)
    core.set_damping(0, 32)
    for density in (1, 4):
        core.set_material(0, .2, 0, 1, growth_rate=0,
                          particle_mass=10/density, particle_volume=1/density)
        for weight in (1., .01, .001, 1e-7):
            load_samples(core, agents, [[.501, .503]], [[0, 0]],
                         domains=[[.0001, 0, 0, .0001]])
            rest = read_rest(core, 1)
            rest[:, 15] = weight
            device.queue.write_buffer(core.rest, 0, rest)
            velocity = np.array([[1.3, -.7]], np.float32)
            device.queue.write_buffer(core.velocities, 0, velocity)
            initial_grid = p2g_grid(core)
            np.testing.assert_allclose(initial_grid[:, 2].sum(), 10*weight/density, rtol=2e-6)
            for _ in range(32):
                core.step(8)
                actual = core.read_velocities()
                np.testing.assert_allclose(actual, velocity, atol=5e-4, rtol=0)
                affine = np.frombuffer(device.queue.read_buffer(core.C, 0, 16), np.float32)
                assert np.linalg.norm(affine) < .05, (density, weight, affine)
    print('[PASS] 1x/4x force-free motion stays uniform over 256 steps down to q=1e-7; no artificial mass floor')

def check_accumulator_headroom(device):
    core = MpmCore(device)
    core.set_gravity(0)
    core.set_repulsion_strength(0, 40)
    core.set_damping(0, 32)
    core.set_material(0, .2, 0, 1, growth_rate=0, particle_mass=10)
    count = 16384
    velocity = np.array([100., -80.], np.float32)
    core.load_scene(np.tile([.5, .5], (count, 1)).astype(np.float32),
                    np.tile(velocity, (count, 1)),
                    np.tile([1, 0, 0, 1], (count, 1)).astype(np.float32),
                    np.zeros((count, 4), np.float32), np.ones(count, np.float32))
    grid = p2g_grid(core)
    assert np.isfinite(grid).all() and np.all(grid[:, 2] >= 0)
    # Sequential f32 accumulation of 16k contributions has ordinary rounding
    # error (about 4.5e-5 here), unlike the old integer wraparound/sign reversal.
    np.testing.assert_allclose(grid.sum(axis=0), count*10*np.r_[velocity, 1], rtol=1e-4)
    core.step(1)
    np.testing.assert_allclose(core.read_velocities(), np.tile(velocity, (count, 1)), rtol=1e-4)
    print('[PASS] concentrated momentum exceeds old i32 range without wraparound or a velocity spike')

def check_compression(device):
    from triangle_seed import triangulate_seed_cells
    from triangle_vertices import domain_edges
    for density in (1, 4):
        core, agents = make_system(device, capacity=1024)
        spacing = .0027/np.sqrt(density)
        radius = 3.5*np.sqrt(density)
        sites = np.array([(i, j) for i in range(-7, 8) for j in range(-7, 8)
                          if i*i+j*j <= radius*radius])
        centers = .5+sites*spacing
        half_edges = np.tile(np.eye(2)*spacing/2, (len(sites), 1, 1))
        positions, domains, weights = triangulate_seed_cells(centers, half_edges)
        n = len(positions)
        core.load_scene(positions, np.zeros((n, 2), np.float32),
                        np.tile([1, 0, 0, 1], (n, 1)).astype(np.float32),
                        np.zeros((n, 4), np.float32), np.ones(n, np.float32),
                        domains, weights, 'triangle-vertices')
        rest = read_rest(core, n)
        rest[:, 8:14] = .5+.3*(rest[:, 8:14]-.5)
        positions = .5+.3*(positions-.5)
        device.queue.write_buffer(core.rest, 0, rest)
        device.queue.write_buffer(core.positions, 0, positions)
        device.queue.write_buffer(core.F, 0, np.tile([.3, 0, 0, .3], (n, 1)).astype(np.float32))
        device.queue.write_buffer(core.velocities, 0, (-1000*(positions-.5)).astype(np.float32))
        core.set_damping(0, 32)
        core.set_material(10000, .2, 3, .5, growth_rate=0,
                          particle_mass=10/density, particle_volume=1/density)
        peak = 0.
        for _ in range(128):
            core.step(8)
            rest = read_rest(core, n)
            velocity = core.read_velocities()
            assert np.isfinite(rest).all() and np.isfinite(velocity).all()
            assert np.all(np.linalg.det(domain_edges(rest)) > 0)
            peak = max(peak, float(np.linalg.norm(velocity, axis=1).max()))
        assert peak < 50, (density, peak)
        print(f'[PASS] {density}x compressed material survives 1024 steps; peak speed={peak:.3f}')

def main():
    device = pick_device()
    check_force_free_motion(device)
    check_accumulator_headroom(device)
    check_compression(device)

if __name__ == '__main__':
    main()

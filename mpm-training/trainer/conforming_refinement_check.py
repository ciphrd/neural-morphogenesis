"""GPU acceptance checks for staged conforming longest-edge refinement."""
from collections import Counter
import numpy as np
from continuous_growth_check import (make_system, read_rest, run_growth_field,
                                     synchronize_count, SPACING, GRID_N, DX)
from vertex_transport_check import load_triangles, run_g2p
from triangle_vertices import domain_edges
from device import pick_device


def demands(rest):
    v = rest[:, 12:18].astype(float).reshape(-1, 3, 2)
    edges = np.roll(v, -1, axis=1)-v
    edges -= np.floor(edges+.5)
    return np.sum(edges*edges, axis=(1, 2))/(4*np.sqrt(3)*SPACING**2)


def assert_disk(rest):
    """Exact endpoint topology detects hanging vertices, cracks and overlaps."""
    vertices = rest[:, 12:18].copy().view(np.uint32).reshape(-1, 3, 2)
    edges = Counter()
    points = set()
    for tri in vertices:
        keys = [tuple(p) for p in tri]
        points.update(keys)
        for i in range(3):
            edges[tuple(sorted((keys[i], keys[(i+1) % 3])))] += 1
    assert all(n in (1, 2) for n in edges.values())
    assert len(points)-len(edges)+len(rest) == 1, 'nonconforming topology'
    boundary = Counter(p for edge, n in edges.items() if n == 1 for p in edge)
    assert boundary and all(degree == 2 for degree in boundary.values()), 'cracked boundary'
    assert np.all(np.linalg.det(domain_edges(rest)) > 0)


def fixture(core, agents, vertices):
    load_triangles(core, vertices)
    agents.set_active_count(len(vertices))


def check_pair_capacity(device):
    # Only one of the two triangles exceeds the threshold. Their common edge
    # is longest for both, so even the already-resolved neighbor must split.
    a, b, c, d = np.array([[.4, .4], [.407, .4], [.4035, .405], [.4035, .399]])
    triangles = [[a, b, c], [b, a, d]]
    for capacity in (3, 4):
        core, agents = make_system(device, capacity=capacity)
        fixture(core, agents, triangles)
        before = read_rest(core, 2)
        assert demands(before)[0] > 1.75 > demands(before)[1]
        positions = core.read_positions().copy()
        run_growth_field(device, agents)
        n = synchronize_count(core, agents)
        if capacity == 3:
            assert n == 2 and agents.unresolved_samples == 1
            assert agents.capacity_blocked
            np.testing.assert_array_equal(read_rest(core, 2)[:, 8:], before[:, 8:])
            np.testing.assert_array_equal(core.read_positions(), positions)
            assert not np.any(np.frombuffer(device.queue.read_buffer(core.growth_field), np.int32))
        else:
            assert n == 4 and agents.unresolved_samples == 0
            assert_disk(read_rest(core, n))
            np.testing.assert_allclose(read_rest(core, n)[:, 11].sum(), 2)
    print('[PASS] one-sided demand splits both neighbors; one free slot leaves the pair intact and pauses growth')


def check_dependency(device):
    # Triangle zero wants AB; its neighbor wants the longer boundary AD first.
    # First advance the dependency, then bisect AB as a matched pair.
    a, b, c, d = np.array([[.4, .4], [.409, .4], [.4045, .399], [.409, .411]])
    core, agents = make_system(device, capacity=128)
    fixture(core, agents, [[a, c, b], [b, d, a]])
    before = read_rest(core, 2)
    for step in range(20):
        run_growth_field(device, agents)
        n = synchronize_count(core, agents)
        rest = read_rest(core, n)
        assert_disk(rest)
        np.testing.assert_allclose(rest[:, 11].sum(), 2)
        np.testing.assert_allclose(rest[:, 8].sum(), before[:, 8].sum(), rtol=2e-6)
        if step == 0:
            assert n == 3
            np.testing.assert_array_equal(rest[0, 12:18], before[0, 12:18])
    assert np.max(demands(rest)) < 1.75
    print('[PASS] a different neighbor longest edge is refined first; every stage remains conforming and converges')


def check_long_dependency(device):
    # A path longer than one workgroup: every radial edge is longer than the
    # previous one. All requests must reach the one terminal boundary edge.
    count = 257
    angles = np.linspace(0, 1, count+1)
    radius = np.linspace(.008, .012, count+1)
    center = np.array([.4, .4])
    rays = center + radius[:, None]*np.stack((np.cos(angles), np.sin(angles)), axis=1)
    core, agents = make_system(device, capacity=count+1)
    fixture(core, agents, [[center, rays[i], rays[i+1]] for i in range(count)])
    before = read_rest(core, count)
    assert np.all(demands(before) > 1.75)
    run_growth_field(device, agents)
    assert synchronize_count(core, agents) == count+1
    assert agents.unresolved_samples == 0
    after = read_rest(core, count+1)
    np.testing.assert_array_equal(after[:-2, 12:18], before[:-1, 12:18])
    assert_disk(after)
    print('[PASS] 257-triangle dependency path resolves to one boundary operation across workgroups')


def grid_triangles(origin):
    points = np.array([[origin + np.array([i*.004, j*.004]) for j in range(5)]
                       for i in range(5)], np.float32)
    # Determinant-one shear: refining this fixture cannot be explained by area.
    points[..., 0] += 2*(points[..., 1]-origin[1])
    triangles = []
    for i in range(4):
        for j in range(4):
            a, b, c, d = points[i,j], points[i+1,j], points[i,j+1], points[i+1,j+1]
            triangles.extend(([a,b,d], [a,d,c]))
    return triangles


def check_mesh_and_transport(device):
    core, agents = make_system(device, capacity=1024)
    x, y = np.meshgrid(np.arange(GRID_N+1)*DX, np.arange(GRID_N+1)*DX, indexing='ij')
    grid = np.stack((1+.5*np.sin(8*np.pi*y), .4*np.cos(8*np.pi*x)), axis=-1).astype(np.float32)
    for origin in (np.array([.3, .3]), np.array([.995, .995])):
        fixture(core, agents, grid_triangles(origin))
        before = read_rest(core, core.active_count)
        initial_area = .5*np.linalg.det(domain_edges(before)).sum()
        for _ in range(24):
            run_growth_field(device, agents)
            n = synchronize_count(core, agents)
            rest = read_rest(core, n)
            assert_disk(rest)
            np.testing.assert_allclose(rest[:, 11].sum(), 32)
            np.testing.assert_allclose(.5*np.linalg.det(domain_edges(rest)).sum(), initial_area, rtol=5e-5)
        assert np.max(demands(rest)) < 1.75 and n > 32
        # Then continue nonlinear vertex transport and subdivision together.
        device.queue.write_buffer(core.grid_vel, 0, grid.reshape(-1, 2))
        for _ in range(32):
            for _ in range(4):
                run_g2p(core)
            run_growth_field(device, agents)
            synchronize_count(core, agents)
            assert_disk(read_rest(core, core.active_count))
    print('[PASS] sheared mesh converges without hanging edges; exact topology survives refinement and 128 nonlinear steps across seams')


def check_partial_batch(device):
    core, agents = make_system(device, capacity=73)
    fixture(core, agents, grid_triangles(np.array([.3, .3])))
    blocked = False
    for _ in range(16):
        run_growth_field(device, agents)
        n = synchronize_count(core, agents)
        assert n <= 73
        assert_disk(read_rest(core, n))
        np.testing.assert_allclose(read_rest(core, n)[:, 11].sum(), 32)
        blocked |= agents.capacity_blocked
    assert blocked
    print('[PASS] competing boundary and interior groups preserve a complete mesh at an odd capacity')


def check_seed_refinement(device):
    from training_sim import seed_blob
    core, agents = make_system(device, capacity=1024)
    scene = seed_blob(37, (.5, .5), SPACING, 17)
    core.load_scene(*scene)
    agents.set_active_count(core.active_count)
    rest = read_rest(core, core.active_count)
    assert_disk(rest)
    # Apply an identical affine stretch to every duplicate vertex before
    # refining the actual startup mesh, preserving shared coordinate bits.
    vertices = rest[:, 12:18].reshape(-1, 3, 2)
    vertices[..., 0] = .5 + 4*(vertices[..., 0]-.5)
    vertices[..., 1] = .5 + .25*(vertices[..., 1]-.5)
    device.queue.write_buffer(core.rest, 0, rest)
    for _ in range(32):
        run_growth_field(device, agents)
        n = synchronize_count(core, agents)
        rest = read_rest(core, n)
        assert_disk(rest)
        np.testing.assert_allclose(rest[:, 11].sum(), 37)
    assert np.max(demands(rest)) < 1.75 and n > 74
    print('[PASS] actual circular startup mesh stays conforming through isochoric stretch refinement')


def check_metric():
    rng = np.random.default_rng(217)
    for _ in range(1000):
        v = rng.normal(size=(3, 2))
        e = np.roll(v, -1, axis=0)-v
        score = np.sum(e*e)
        covariance_trace = np.sum((v-v.mean(axis=0))**2)/12
        np.testing.assert_allclose(score/36, covariance_trace)
        i = np.argmax(np.sum(e*e, axis=1))
        a, b, c = v[i], v[(i+1)%3], v[(i+2)%3]
        m = (a+b)/2
        for child in (np.array([a,m,c]), np.array([m,b,c])):
            child_edges = np.roll(child, -1, axis=0)-child
            assert np.sum(child_edges*child_edges) <= .75*score*(1+1e-12)
    print('[PASS] squared-edge sum is 36 times centroid second moment; longest-edge children score at most 3/4 of parent')


def main():
    check_metric()
    device = pick_device()
    for check in (check_pair_capacity, check_dependency, check_long_dependency,
                  check_mesh_and_transport, check_partial_batch, check_seed_refinement):
        check(device)


if __name__ == '__main__':
    main()

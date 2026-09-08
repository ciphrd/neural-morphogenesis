"""GPU/CPU acceptance checks for passive triangle-vertex transport.

Run from trainer: .venv/bin/python vertex_transport_check.py
"""
import numpy as np
from continuous_growth_check import make_system, read_rest, DX, GRID_N
from device import pick_device
from mpm_core import DT
from triangle_domain_check import Triangle
from triangle_vertices import domain_edges, vertices_from_edges, unwrap_vertices

def sample_grid(grid, position):
    y = (np.asarray(position) % 1)/DX
    base = np.floor(y-.5).astype(int)
    fx = y-base
    weights = [.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2]
    return sum(weights[i][0]*weights[j][1]*grid[(base[0]+i)%GRID_N,(base[1]+j)%GRID_N]
               for i in range(3) for j in range(3))

def run_g2p(core):
    encoder = core.device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(core.g2p_pipeline)
    compute.set_bind_group(0,core.g2p_bind_group)
    compute.dispatch_workgroups((core.active_count+63)//64)
    compute.end()
    core.device.queue.submit([encoder.finish()])

def load_triangles(core, vertices):
    triangles = [Triangle.from_vertices(v) for v in vertices]
    n = len(triangles)
    core.load_scene(np.array([t.x%1 for t in triangles],np.float32),
                    np.zeros((n,2),np.float32),np.tile([1,0,0,1],(n,1)).astype(np.float32),
                    np.zeros((n,4),np.float32),np.ones(n,np.float32),
                    np.asarray(vertices,np.float32).reshape(n,6)%1,
                    domain_geometry='triangle-vertices')

def actual_triangles(core):
    return [Triangle.from_vertices(v) for v in unwrap_vertices(read_rest(core,core.active_count)[:,8:14])]

def periodic_error(a,b):
    return (np.asarray(a)-np.asarray(b)+.5)%1-.5

def check_nonlinear_shared_vertices(device):
    core,_ = make_system(device)
    coordinates = np.arange(GRID_N+1)*DX
    x,y = np.meshgrid(coordinates,coordinates,indexing='ij')
    grid = np.stack((1.1+.6*np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
                     -.4+.5*np.cos(2*np.pi*x)*np.sin(4*np.pi*y)),axis=-1).astype(np.float32)
    device.queue.write_buffer(core.grid_vel,0,grid.reshape(-1,2))
    max_shared_error = 0.
    centroid_correction = 0.
    for origin in (np.array([.35,.36]),np.array([.95,.97])):
        a,b,c,d = origin+np.array([[0,0],[.18,.01],[.03,.16],[.21,.17]])
        load_triangles(core,[[a,b,c],[b,d,c]])
        for step in range(128):
            before = actual_triangles(core)
            expected_vertices = [t.vertices()+DT*np.array([sample_grid(grid,p) for p in t.vertices()]) for t in before]
            center_velocities = np.array([sample_grid(grid,t.x) for t in before])
            run_g2p(core)
            after = actual_triangles(core)
            for triangle,expected in zip(after,expected_vertices):
                np.testing.assert_allclose(periodic_error(triangle.vertices(),expected),0,atol=1.5e-7)
                np.testing.assert_allclose(periodic_error(triangle.x,expected.mean(axis=0)),0,atol=8e-8)
            # Geometry transport must not replace the existing momentum gather.
            velocities = np.frombuffer(device.queue.read_buffer(core.velocities,0,16),np.float32).reshape(2,2)
            np.testing.assert_allclose(velocities,center_velocities,rtol=2e-6,atol=3e-7)
            shared = periodic_error(after[0].vertices()[[1,2]],after[1].vertices()[[0,2]])
            raw = read_rest(core,2)[:,8:14].reshape(2,3,2)
            np.testing.assert_array_equal(raw[0,[1,2]].view(np.uint32),raw[1,[0,2]].view(np.uint32))
            max_shared_error = max(max_shared_error,float(np.max(np.abs(shared))))
            if step==0:
                correction = expected_vertices[0].mean(axis=0)-(before[0].x+DT*center_velocities[0])
                centroid_correction = max(centroid_correction,float(np.linalg.norm(correction)))
        assert np.max(np.abs(shared))<3e-6, shared
    assert centroid_correction>2e-7, 'Fixture must distinguish vertex transport from centroid extrapolation'
    print(f'[PASS] nonlinear per-vertex CPU/GPU agreement and shared corners over 128 steps, '
          f'including seams: max separation {max_shared_error:.3g}; centroid-path difference {centroid_correction:.3g}')

def check_translation_and_rest(device):
    core,_ = make_system(device)
    load_triangles(core,[[[.999,.998],[1.001,.998],[.999,1.001]]])
    before = actual_triangles(core)[0]
    domain = read_rest(core,1)[0,8:14].copy()
    grid = np.zeros(((GRID_N+1)**2,2),np.float32)
    device.queue.write_buffer(core.grid_vel,0,grid)
    run_g2p(core)
    np.testing.assert_array_equal(read_rest(core,1)[0,8:14],domain)
    np.testing.assert_allclose(periodic_error(core.read_positions()[0],before.x),0,atol=6e-8)
    velocity = np.array([12.,15.],np.float32)
    grid[:] = velocity
    device.queue.write_buffer(core.grid_vel,0,grid)
    for _ in range(16): run_g2p(core)
    after = actual_triangles(core)[0]
    np.testing.assert_allclose(periodic_error(after.vertices(),before.vertices()+16*DT*velocity),0,atol=1e-6)
    np.testing.assert_allclose(after.edges,before.edges,atol=1e-6)
    print('[PASS] zero flow retains geometry exactly; uniform translation preserves shape across both seams')

def check_relocation_momentum_accounting(device):
    # Keeping the point velocity preserves the gathered linear momentum, but
    # the new geometric centroid path can change orbital angular momentum.
    core,_ = make_system(device)
    load_triangles(core,[[[.3,.4],[.6,.42],[.38,.65]]])
    coords = np.arange(GRID_N+1)*DX
    x,y = np.meshgrid(coords,coords,indexing='ij')
    grid = np.stack((1+x*x,.5+y*y),axis=-1).astype(np.float32)
    device.queue.write_buffer(core.grid_vel,0,grid.reshape(-1,2))
    before = actual_triangles(core)[0]
    v = sample_grid(grid,before.x)
    vertex_mean = np.mean([sample_grid(grid,p) for p in before.vertices()],axis=0)
    run_g2p(core)
    after = actual_triangles(core)[0]
    actual_v = np.frombuffer(device.queue.read_buffer(core.velocities,0,8),np.float32)
    np.testing.assert_allclose(actual_v,v,atol=2e-7)
    displacement = after.x-before.x
    expected_shift = DT*(vertex_mean-v)
    np.testing.assert_allclose(displacement-DT*v,expected_shift,atol=5e-8)
    cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
    angular_change_per_mass = cross(displacement,actual_v)
    np.testing.assert_allclose(angular_change_per_mass,cross(expected_shift,v),atol=7e-8)
    print(f'[PASS] point linear-momentum gather retained; geometric relocation angular change '
          f'per unit mass={angular_change_per_mass:.3g} (not guaranteed zero)')

def check_split_and_seed_identity(device):
    from continuous_growth_check import run_growth_field, synchronize_count
    from training_sim import seed_blob
    core,agents=make_system(device,capacity=64)
    # Both incident triangles bisect the same longest edge, in opposite orders.
    a,b,c,d=np.array([[.4,.4],[.41,.4],[.4,.41],[.41,.41]],np.float32)
    load_triangles(core,[[a,b,c],[b,d,c]])
    original=read_rest(core,2)[:,8:14].reshape(-1,2).copy()
    agents.set_active_count(2)
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==4
    vertices=read_rest(core,4)[:,8:14].reshape(-1,2)
    old_bits={tuple(v.view(np.uint32)) for v in original}
    new=[tuple(v.view(np.uint32)) for v in vertices if tuple(v.view(np.uint32)) not in old_bits]
    assert len(new)==4 and len(set(new))==1, 'Independently bisected shared edge must use one exact midpoint'
    for v in original:
        assert np.any(np.all(vertices.view(np.uint32)==v.view(np.uint32),axis=1)), 'A retained endpoint changed during subdivision'
    # Startup cells also share exact coordinates before any advection.
    scene=seed_blob(7,(.9999,.0001),.01,17)
    core.load_scene(*scene)
    vertices=read_rest(core,14)[:,8:14].reshape(-1,2)
    groups={}
    for i,v in enumerate(vertices): groups.setdefault(tuple(v.view(np.uint32)),[]).append(i)
    duplicates=[indices for indices in groups.values() if len(indices)>1]
    assert max(map(len,duplicates))>=6, 'Six triangles should share a lattice corner'
    coords=np.arange(GRID_N+1)*DX
    x,y=np.meshgrid(coords,coords,indexing='ij')
    grid=np.stack((2+np.sin(2*np.pi*y),1+np.cos(2*np.pi*x)),axis=-1).astype(np.float32)
    device.queue.write_buffer(core.grid_vel,0,grid.reshape(-1,2))
    for _ in range(256):
        run_g2p(core)
        bits=read_rest(core,14)[:,8:14].copy().reshape(-1,2).view(np.uint32)
        for indices in duplicates:
            np.testing.assert_array_equal(bits[indices],np.tile(bits[indices[0]],(len(indices),1)))
    print('[PASS] split endpoints preserved, opposite-winding midpoints identical, shared seed corners bit-identical for 256 steps')

if __name__=='__main__':
    device=pick_device()
    check_nonlinear_shared_vertices(device)
    check_translation_and_rest(device)
    check_relocation_momentum_accounting(device)
    check_split_and_seed_identity(device)

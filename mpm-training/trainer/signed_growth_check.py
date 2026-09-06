"""GPU regression checks for domain-integrated, signed boundary growth.

Run: trainer/.venv/bin/python trainer/signed_growth_check.py
"""
from __future__ import annotations
import numpy as np
from continuous_growth_check import (make_system, load_samples, read_rest, DX, GRID_N,
                                     project_growth_field as project)
from device import pick_device
from mpm_core import GROWTH_FIELD_CHANNELS, DT
from vertex_transport_check import load_triangles, run_g2p


def load(core, agents, triangles, vector=(0., 0.), weights=None):
    core.reset_growth_buffers(agents.max_active_particles)
    load_triangles(core, triangles)
    agents.set_active_count(core.active_count)
    rest = core.read_rest_state()
    rest[:, 5:7] = vector
    if weights is not None:
        rest[:, 15] = weights
    core.device.queue.write_buffer(core.rest, 0, rest)
    return rest


def stencil(pos):
    y = (pos % 1.) / DX
    base = np.floor(y-.5).astype(int)
    f = y-base
    w = [0.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2]
    for i in range(3):
        for j in range(3):
            yield ((base[0]+i)%GRID_N)*(GRID_N+1)+(base[1]+j)%GRID_N, w[i][0]*w[j][1]


def dense_domain_weights(triangle):
    """Independent tensor-product Gauss integration under the Duffy mapping."""
    points, weights = np.polynomial.legendre.leggauss(12)
    points, weights = (points+1)/2, weights/2
    a = triangle[0].astype(float)
    edges = triangle[1:].astype(float)-a
    edges -= np.floor(edges+.5)
    result = np.zeros((GRID_N+1)**2)
    for u, wu in zip(points, weights):
        for v, wv in zip(points, weights):
            pos = a + u*edges[0] + (1-u)*v*edges[1]
            for node, w in stencil(pos):
                result[node] += 2*(1-u)*wu*wv*w
    return result


def check_precision_and_domains(device):
    core, agents = make_system(device, capacity=32)
    # This triangle stays within one spline polynomial patch. Seven-point
    # quadrature should match independent integration, unlike its centroid.
    triangle = np.array([[.493,.492], [.506,.493], [.494,.506]], np.float32)
    for weight in (1., 1e-7, 1e6):
        rest = load(core, agents, [triangle], (.0003, -.0002), [weight])
        field = project(core, agents, boundary=False)
        reference = dense_domain_weights(rest[0, 8:14].reshape(3, 2))
        np.testing.assert_allclose(field[:, 5]/weight, reference, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(field[:, :2].sum(0)/weight, [.0003, -.0002], rtol=2e-6, atol=1e-10)
        assert np.all(np.isfinite(field))
        # The old fixed point field loses the entire 1e-7 sample.
        if weight == 1e-7:
            assert np.all(np.round(reference*weight*8192) == 0)
    midpoint = (triangle[0]+triangle[1])/2
    parent = field[:, 5]/1e6
    load(core, agents, [[triangle[0], midpoint, triangle[2]], [midpoint, triangle[1], triangle[2]]], weights=[.5, .5])
    split = project(core, agents, boundary=False)
    np.testing.assert_allclose(split[:, 5], parent, atol=2e-6, rtol=2e-5)
    split_surface = project(core, agents)[:, 8:11]
    load(core, agents, [triangle], weights=[1])
    parent_surface = project(core, agents)[:, 8:11]
    np.testing.assert_allclose(split_surface, parent_surface, atol=3e-8, rtol=2e-5)
    # Exact periodic shift by half a world checks canonical seam handling.
    load(core, agents, [(triangle+.5)%1], weights=[1])
    seam = project(core, agents, boundary=False)[:, 5].reshape(GRID_N+1, GRID_N+1)[:GRID_N,:GRID_N]
    original = parent.reshape(GRID_N+1, GRID_N+1)[:GRID_N,:GRID_N]
    np.testing.assert_allclose(seam, np.roll(original, (GRID_N//2, GRID_N//2), (0,1)), atol=3e-6)
    print('[PASS] tiny weights/weak commands, concentrated headroom, independent domain integration, split invariance and seams')


def square_triangles(center=(.5,.5), cells=8, spacing=DX):
    corner = np.array(center)-cells*spacing/2
    points = np.array([[corner+[i*spacing,j*spacing] for j in range(cells+1)] for i in range(cells+1)], np.float32)
    result = []
    for i in range(cells):
        for j in range(cells):
            a,b,c,d = points[i,j],points[i+1,j],points[i,j+1],points[i+1,j+1]
            result.extend(([a,b,d],[a,d,c]))
    return np.asarray(result)


def check_boundary_sign(device):
    core, agents = make_system(device, capacity=512)
    triangles = square_triangles()
    right = (GRID_N//2+4)*(GRID_N+1)+GRID_N//2
    middle = (GRID_N//2)*(GRID_N+1)+GRID_N//2
    for vector, expected in (((.6,0),(.6,0,0)), ((-.6,0),(-.6,0,0)), ((0,.6),(0,0,.6))):
        load(core,agents,triangles,vector)
        field = project(core,agents)
        np.testing.assert_allclose(field[right,2:5]/field[right,5],expected,atol=2e-6)
        np.testing.assert_allclose(field[right,8:10]/np.linalg.norm(field[right,8:10]),[1,0],atol=2e-6)
        np.testing.assert_array_equal(field[middle,8:11],0)
        assert field[middle,2]+field[middle,4]>0
        # The geometric surface measure is the perimeter, with no internal diagonals.
        np.testing.assert_allclose(field[:,10].sum(),4*8*DX,rtol=2e-6)
    # Shared geometry with opposite proposals cancels before signed conversion.
    load(core,agents,np.concatenate([triangles,triangles]),np.concatenate([np.tile([.6,0],(len(triangles),1)),np.tile([-.6,0],(len(triangles),1))]))
    field = project(core,agents)
    np.testing.assert_allclose(field[:,2:5],0,atol=1e-6)
    # Both geometric normal and tensor must wrap along with a seam-crossing body.
    load(core,agents,triangles,(-.6,0))
    baseline = project(core,agents).reshape(GRID_N+1,GRID_N+1,-1)[:GRID_N,:GRID_N]
    load(core,agents,(triangles+.5)%1,(-.6,0))
    seam = project(core,agents).reshape(GRID_N+1,GRID_N+1,-1)[:GRID_N,:GRID_N]
    np.testing.assert_allclose(seam[:,:,[2,3,4,5,8,9,10]],np.roll(baseline,(32,32),(0,1))[:,:,[2,3,4,5,8,9,10]],atol=2e-6)
    print('[PASS] inward contraction, outward/tangent expansion, interior neutrality, cancellation and periodic boundaries')


def check_boundary_resolution(device):
    core,agents=make_system(device,capacity=1024)
    fields=[]
    mean_log_growth=[]
    for refinement in (1,2):
        # Align cell boundaries with spline knots: exact polynomial quadrature
        # should be invariant here, while knot-crossing triangles are approximate.
        triangles=square_triangles(center=(.5+DX/2,.5+DX/2),cells=8*refinement,spacing=DX/refinement)
        weight=.5/refinement**2
        load(core,agents,triangles,(-.6,0),weights=np.full(len(triangles),weight))
        fields.append(project(core,agents))
        core.set_material(0,.2,0,1,growth_rate=80,growth_compression_feedback=0)
        core.step(1)
        rest=read_rest(core,len(triangles))
        mean_log_growth.append(np.log(np.linalg.det(rest[:,:4].reshape(-1,2,2))).mean())
    # Identical material and actual boundary geometry, with four times as many
    # numerical triangles. This exercises signed normals as well as projection.
    for channels in ([0,1,2,3,4,5],[8,9,10]):
        np.testing.assert_allclose(fields[0][:,channels],fields[1][:,channels],atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(mean_log_growth[0],mean_log_growth[1],atol=3e-7)
    print('[PASS] 4x subdivision of the same boundary preserves signed field and integrated material rate')


def uniform_tensor(core, tensor, budget=0.):
    field = np.zeros(((GRID_N+1)**2, GROWTH_FIELD_CHANNELS),np.float32)
    field[:, 2:5] = tensor
    field[:, 5] = 1
    field[0, 7] = budget
    core.device.queue.write_buffer(core.growth_field,0,field)


def check_signed_integration(device):
    core,agents=make_system(device)
    # Pure contraction, and zero-trace remodeling previously skipped entirely.
    for tensor in ((-.6,0,0),(.4,0,-.4),(-.2,.3,.1)):
        load_samples(core,agents,[[.5,.5]],[[0,0]])
        uniform_tensor(core,tensor)
        core.set_material(0,.2,0,1,growth_rate=80,growth_compression_feedback=0)
        core.step(32)
        rate=np.array([[tensor[0],tensor[1]],[tensor[1],tensor[2]]])
        values,vectors=np.linalg.eigh(rate)
        expected=(vectors*np.exp(values*80*DT*32))@vectors.T
        np.testing.assert_allclose(read_rest(core,1)[0,:4].reshape(2,2),expected,atol=7e-6)
    # Exhausted budget and compression cannot disable the negative eigenvalue.
    load_samples(core,agents,[[.5,.5]],[[0,0]])
    rest=read_rest(core,1);rest[:,7]=1
    core.device.queue.write_buffer(core.rest,0,rest)
    core.device.queue.write_buffer(core.F,0,np.array([[.8,0,0,.8]],np.float32))
    uniform_tensor(core,(.6,0,-.6),budget=1.)
    core.set_material(0,.2,0,1,growth_rate=80,growth_compression_feedback=1,
                      growth_compression_start=.1,growth_compression_stop=.1)
    core.step(32)
    g=read_rest(core,1)[0,:4].reshape(2,2)
    np.testing.assert_allclose(g[0,0],1,atol=2e-6)
    np.testing.assert_allclose(g[1,1],np.exp(-.6*80*DT*32),atol=7e-6)
    # Rotating both the elastic frame and a signed world command leaves the
    # material-frame update unchanged.
    theta=.63
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    load_samples(core,agents,[[.5,.5]],[[0,0]])
    core.device.queue.write_buffer(core.F,0,rotation.astype(np.float32).reshape(1,4))
    rate=rotation@np.diag([.4,-.4])@rotation.T
    uniform_tensor(core,(rate[0,0],rate[0,1],rate[1,1]))
    core.set_material(0,.2,0,1,growth_rate=80,growth_compression_feedback=0)
    core.device.queue.write_buffer(core.grid_vel,0,np.zeros(((GRID_N+1)**2,2),np.float32))
    run_g2p(core)
    np.testing.assert_allclose(read_rest(core,1)[0,:4].reshape(2,2),np.diag(np.exp(np.array([.4,-.4])*80*DT)),atol=3e-7)
    # Stop at the existing rest-area floor with a strong requested contraction.
    load_samples(core,agents,[[.5,.5]],[[0,0]],growth_f=[[.001,0,0,.001]])
    uniform_tensor(core,(-1,0,-1))
    core.set_material(0,.2,0,1,growth_rate=80,growth_compression_feedback=0)
    core.step(32)
    np.testing.assert_allclose(np.linalg.det(read_rest(core,1)[0,:4].reshape(2,2)),1e-6,rtol=1e-5)
    print('[PASS] negative and zero-trace matrix exponentials; contraction survives compression/budget and respects area floor')


def check_boundary_mechanics(device):
    outcomes=[]
    for direction in (-1,0,1):
        core,agents=make_system(device,capacity=512)
        triangles=square_triangles()
        # Only the right boundary layer acts; all other material is passive.
        centers=triangles.mean(1)
        boundary=centers[:,0]>.5+3*DX
        vectors=np.zeros((len(triangles),2));vectors[boundary,0]=direction
        load(core,agents,triangles,vectors,weights=np.full(len(triangles),.5))
        core.set_material(400,.2,0,1,growth_rate=120,growth_compression_feedback=0)
        core.set_damping(0,16)
        start=centers[boundary,0].mean()
        for _ in range(16):
            project(core,agents)
            core.step(16)
        rest=read_rest(core,len(triangles))
        moved=core.read_positions()[boundary,0].mean()-start
        det=np.linalg.det(rest[:,:4].reshape(-1,2,2))
        outcomes.append((moved,det[boundary].mean()))
        assert np.isfinite(rest).all()
        vertices=rest[:,8:14].reshape(-1,3,2)
        assert np.all(np.linalg.det(np.stack([vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],axis=-1))>0)
    assert outcomes[0][0]<outcomes[1][0]<outcomes[2][0],outcomes
    assert outcomes[0][1]<1<outcomes[2][1],outcomes
    print(f'[PASS] actual right-boundary motion and rest area respond inward/passive/outward: {outcomes}')


def check_interior_growth_opposition(device):
    outcomes=[]
    for inward in (False,True):
        core,agents=make_system(device,capacity=512)
        triangles=square_triangles()
        centers=triangles.mean(1)
        boundary=centers[:,0]>.5+3*DX
        vectors=np.tile([.6,0.],(len(triangles),1))
        vectors[boundary,0]=-1 if inward else 0
        load(core,agents,triangles,vectors,weights=np.full(len(triangles),.5))
        core.set_material(400,.2,0,1,growth_rate=120,growth_compression_feedback=0)
        core.set_damping(0,16)
        for _ in range(16):
            project(core,agents)
            core.step(16)
        rest=read_rest(core,len(triangles))
        assert np.isfinite(rest).all()
        outcomes.append(float(core.read_positions()[boundary,0].mean()-centers[boundary,0].mean()))
    assert outcomes[0]>0 and outcomes[1]<outcomes[0],outcomes
    print(f'[PASS] inward boundary opposes growing interior: displacement passive={outcomes[0]:.6g}, inward={outcomes[1]:.6g}')


def main():
    device=pick_device()
    for check in (check_precision_and_domains,check_boundary_sign,check_boundary_resolution,check_signed_integration,check_boundary_mechanics,check_interior_growth_opposition):
        check(device)

if __name__=='__main__':main()

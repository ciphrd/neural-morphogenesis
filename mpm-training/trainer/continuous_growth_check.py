"""GPU acceptance tests for continuous growth and material-domain subdivision.

Run from trainer: .venv/bin/python continuous_growth_check.py
"""
from __future__ import annotations
import time
import numpy as np
import wgpu
from agents_gpu import AgentsGPU
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import DT, GRID_N, MpmCore, REST_FIELDS
from material_domain import Domain, scatter

SPACING = 0.0027
DX = 1/GRID_N


def make_system(device, capacity=32, include_environment=False):
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0)
    agents = AgentsGPU(device, core, environment, 1, 128,
        0.0, 0.0, 1.0, 1.4, 0.8, 0.1, False, 0.0,
        capacity, SPACING, 0.0, 1.0, 0.4, 1.0, 0.5, 0.5)
    core.set_gravity(0)
    core.set_repulsion_strength(0, 40)
    core.set_material(0, 0.2, 0, 1, growth_rate=0)
    return (core, agents, environment) if include_environment else (core, agents)


def load_samples(core, agents, positions, vectors, growth_f=None, domains=None):
    n = len(positions)
    identity = np.tile([1, 0, 0, 1], (n, 1)).astype(np.float32)
    f = identity if growth_f is None else np.asarray(growth_f, np.float32)
    h = f*(SPACING/2) if domains is None else np.asarray(domains, np.float32)
    core.load_scene(np.asarray(positions, np.float32), np.zeros((n, 2), np.float32),
                    f, np.zeros((n, 4), np.float32), np.ones(n, np.float32), h)
    rest = core.read_rest_state()
    rest[:, :4] = f
    rest[:, 5:7] = vectors
    core.device.queue.write_buffer(core.rest, 0, rest)
    agents.set_active_count(n)


def run_growth_field(device, agents):
    encoder = device.create_command_encoder()
    agents.encode_growth_field(encoder)
    device.queue.submit([encoder.finish()])


def read_rest(core, count):
    return np.frombuffer(core.device.queue.read_buffer(core.rest, 0, count*REST_FIELDS*4),
                         np.float32).reshape(count, REST_FIELDS).copy()


def synchronize_count(core, agents):
    n = agents.read_grown_count()
    core.set_active_count(n)
    agents.set_active_count(n)
    return n


def p2g_grid(core):
    encoder = core.device.create_command_encoder()
    for pipeline, group, dispatch in [(core.clear_grid_pipeline, core.clear_grid_bind_group,
                                      ((GRID_N+1)*(GRID_N+1)+63)//64),
                                     (core.p2g_pipeline, core.p2g_bind_group, (core.active_count+63)//64)]:
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, group)
        p.dispatch_workgroups(dispatch)
        p.end()
    core.device.queue.submit([encoder.finish()])
    return np.frombuffer(core.device.queue.read_buffer(core.grid_accum), np.int32).reshape(-1, 3).astype(float)/4096


def check_continuous_growth(device):
    core, agents = make_system(device)
    load_samples(core, agents, [[.5,.5]], [[.5,0]])
    run_growth_field(device, agents)
    core.set_material(0, .2, 0, 1, growth_rate=12, growth_compression_feedback=0)
    core.step(1)
    np.testing.assert_allclose(np.linalg.det(read_rest(core,1)[0,:4].reshape(2,2)),
                               np.exp(.5*12*DT), rtol=2e-5)
    print('[PASS] continuous exponential rest growth')


def check_opposed_field(device):
    core, agents = make_system(device)
    load_samples(core,agents,[[.5,.5],[.5,.5]],[[.6,0],[-.6,0]])
    run_growth_field(device,agents)
    field=np.frombuffer(device.queue.read_buffer(core.growth_field),np.int32).reshape(-1,10)
    np.testing.assert_allclose(field[:,:2].sum(axis=0),0,atol=1)
    np.testing.assert_allclose(field[:,2:5].sum(axis=0)/field[:,5].sum(),[.6,0,0],atol=1e-3)
    print('[PASS] opposing proposals preserve normalized axial growth')


def check_subdivision(device):
    core,agents=make_system(device)
    h=np.array([[SPACING,0],[0,SPACING]],np.float32)
    load_samples(core,agents,[[.5,.5]],[[0,0]],domains=h.reshape(1,4))
    original=read_rest(core,1)[0]
    for count in (2,4):
        run_growth_field(device,agents)
        assert synchronize_count(core,agents)==count
    rest=read_rest(core,4)
    np.testing.assert_allclose(rest[:,11],.25)
    np.testing.assert_allclose(rest[:,12:16],np.tile((h/2).reshape(4),(4,1)))
    np.testing.assert_allclose(rest[:,:4],np.tile(original[:4],(4,1)))
    offsets=core.read_positions()-.5
    np.testing.assert_allclose(offsets.mean(axis=0),0,atol=1e-7)
    cov=sum(.25*(np.outer(d,d)+r[12:16].reshape(2,2)@r[12:16].reshape(2,2).T/3)
            for d,r in zip(offsets,rest))
    np.testing.assert_allclose(cov,h@h.T/3,rtol=5e-5,atol=1e-10)
    print('[PASS] repeated bisection tiles the parent and preserves second moments')


def check_compression_and_passive_stretch(device):
    core,agents=make_system(device)
    load_samples(core,agents,[[.5,.5]],[[0,0]],growth_f=[[3,0,0,3]],
                 domains=[[SPACING/2,0,0,SPACING/2]])
    run_growth_field(device,agents)
    assert agents.read_grown_count()==1, 'large rest area in compressed material is not a spatial deficit'
    load_samples(core,agents,[[.5,.5]],[[0,0]],domains=[[SPACING,0,0,SPACING/2]])
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==2
    np.testing.assert_allclose(read_rest(core,2)[:,11].sum(),1)
    print('[PASS] passive stretch refines; compressed rest growth alone does not')


def check_p2g_conservation(device):
    core,agents=make_system(device)
    h=np.array([[.002,.0003],[.0002,.0015]])
    x=np.array([.5031,.5027])
    c=np.array([[2,-70],[70,-1]],np.float32)
    v=np.array([.2,-.1],np.float32)
    load_samples(core,agents,[x],[[0,0]],domains=h.reshape(1,4))
    device.queue.write_buffer(core.C,0,c.reshape(1,4))
    device.queue.write_buffer(core.velocities,0,v.reshape(1,2))
    core.set_material(0,.2,0,1,growth_rate=0,particle_mass=100)
    before=p2g_grid(core)
    cpu=scatter([Domain(x,h,mass=100,velocity=v,affine=c)],DX)
    expected=np.zeros_like(before)
    for node,row in cpu.items(): expected[node[0]*(GRID_N+1)+node[1]]=[row[1],row[2],row[0]]
    np.testing.assert_allclose(before,expected,atol=.0025,rtol=1e-3)
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==2
    after=p2g_grid(core)
    coords=np.array([(i*DX,j*DX) for i in range(GRID_N+1) for j in range(GRID_N+1)])-x
    def angular(g): return np.sum(coords[:,0]*g[:,1]-coords[:,1]*g[:,0])
    np.testing.assert_allclose(after.sum(axis=0),before.sum(axis=0),atol=.005)
    np.testing.assert_allclose(angular(after),angular(before),atol=2e-5)
    assert np.abs(after[:,2]-before[:,2]).sum()/100 < .01
    print('[PASS] GPU P2G matches CPU; split conserves mass and angular momentum within fixed-point error')


def check_affine_transport(device):
    core,agents=make_system(device)
    h=np.array([[.001,.0003],[0,.001]],np.float32)
    load_samples(core,agents,[[.5,.5]],[[0,0]],domains=h.reshape(1,4))
    l=np.array([[2000,100],[-50,0]],np.float32)
    coords=np.array([(i*DX,j*DX) for i in range(GRID_N+1) for j in range(GRID_N+1)],np.float32)
    velocities=(coords-.5)@l.T
    device.queue.write_buffer(core.grid_vel,0,velocities)
    core.set_material(0,.2,0,0,growth_rate=0)
    encoder=device.create_command_encoder(); p=encoder.begin_compute_pass()
    p.set_pipeline(core.g2p_pipeline);p.set_bind_group(0,core.g2p_bind_group);p.dispatch_workgroups(1);p.end()
    device.queue.submit([encoder.finish()])
    actual=read_rest(core,1)[0,12:16].reshape(2,2)
    np.testing.assert_allclose(actual,(np.eye(2)+DT*l)@h,rtol=3e-5,atol=2e-8)
    c=np.frombuffer(device.queue.read_buffer(core.C,0,16),np.float32).reshape(2,2)
    np.testing.assert_allclose(c,l,rtol=2e-4,atol=.03)
    f=np.frombuffer(device.queue.read_buffer(core.F,0,16),np.float32).reshape(2,2)
    assert abs(f[0,0]-(1+DT*l[0,0])) > .005
    print('[PASS] affine G2P reproduction and geometry transport independent of constitutive clamp')


def check_courant_guard(device):
    core, agents = make_system(device)
    load_samples(core, agents, [[.5, .5]], [[0, 0]])
    # A large but still finite P2G momentum used to pass straight through the
    # grid and could overflow transported domains over subsequent substeps.
    device.queue.write_buffer(core.velocities, 0, np.array([[10000, -10000]], np.float32))
    core.step(1)
    velocity = np.frombuffer(device.queue.read_buffer(core.velocities, 0, 8), np.float32)
    max_grid_speed = .5 * DX / DT
    assert np.isfinite(velocity).all()
    assert np.max(np.abs(velocity)) <= max_grid_speed * 1.001, velocity
    assert np.isfinite(core.read_positions()).all()
    print(f'[PASS] grid CFL guard bounds extreme fixed-point momentum at {max_grid_speed:g}')


def check_capacity(device):
    core,agents=make_system(device,capacity=1)
    load_samples(core,agents,[[.5,.5]],[[1,0]],domains=[[SPACING,0,0,SPACING]])
    before=read_rest(core,1)
    run_growth_field(device,agents)
    assert agents.read_grown_count()==1 and agents.unresolved_samples==1
    np.testing.assert_allclose(read_rest(core,1)[:,[0,1,2,3,8,11,12,13,14,15]],before[:,[0,1,2,3,8,11,12,13,14,15]])
    assert not np.any(np.frombuffer(device.queue.read_buffer(core.growth_field),np.int32))
    print('[PASS] failed capacity allocation preserves state and reports unresolved sampling')


def check_uniform_rollout(device):
    core,agents=make_system(device,capacity=128)
    load_samples(core,agents,[[.5,.5]],[[1,0]])
    agents.set_forced_growth_field_override(True)
    core.set_material(40,.2,0,1,growth_rate=80,growth_anisotropy=0,growth_compression_feedback=0)
    start=time.perf_counter()
    for _ in range(60):
        run_growth_field(device,agents)
        synchronize_count(core,agents)
        core.step(32)
    rest=read_rest(core,core.active_count)
    area=np.sum(rest[:,11]*np.linalg.det(rest[:,:4].reshape(-1,2,2)))
    np.testing.assert_allclose(area,np.exp(80*60*32*DT),rtol=3e-3)
    assert core.active_count>1
    assert np.isfinite(core.read_positions()).all() and np.isfinite(rest).all()
    assert np.all(np.linalg.det(rest[:,12:16].reshape(-1,2,2))>0)
    print(f'[PASS] free uniform growth: area={area:.4f}, samples={core.active_count}, {time.perf_counter()-start:.2f}s')


def check_capacity_rollout(device):
    # An odd cap forces a partially successful split pass. Continue physics
    # after that transition: checking allocation alone misses NaN propagation.
    core, agents = make_system(device, capacity=9)
    load_samples(core, agents, [[.5, .5]], [[1, 0]])
    agents.set_forced_growth_field_override(True)
    core.set_material(10000, .2, 3, .5, growth_rate=80,
                      growth_anisotropy=0, growth_compression_feedback=0)
    capped_growth = None
    capped_steps = 0
    for _ in range(160):
        run_growth_field(device, agents)
        count = synchronize_count(core, agents)
        core.step(32)
        rest = read_rest(core, count)
        assert np.isfinite(core.read_positions()).all()
        assert np.isfinite(rest).all()
        assert np.all(np.linalg.det(rest[:, 12:16].reshape(-1, 2, 2)) > 0)
        if count == 9:
            capped_steps += 1
            if capped_growth is None:
                capped_growth = rest[:, :4].copy()
            np.testing.assert_array_equal(rest[:, :4], capped_growth)
    assert capped_steps >= 80, 'must exercise sustained physics after reaching capacity'
    print(f'[PASS] partial final allocation and {capped_steps * 32} post-cap physics steps stay finite; growth stops')


def check_physical_budget(device):
    for capacity in (8, 32):
        core,agents=make_system(device,capacity=capacity)
        load_samples(core,agents,[[.5,.5]],[[1,0]])
        initial_area=float(read_rest(core,1)[0,8])
        budget=initial_area*1.3
        agents.set_material_area_budget(budget)
        agents.set_forced_growth_field_override(True)
        core.set_material(0,.2,0,1,growth_rate=500,growth_compression_feedback=0)
        for _ in range(4):
            run_growth_field(device,agents)
            synchronize_count(core,agents)
            core.step(64)
        rest=read_rest(core,core.active_count)
        area=np.sum(rest[:,8]*np.linalg.det(rest[:,:4].reshape(-1,2,2)))
        np.testing.assert_allclose(area,budget,rtol=.002)
        assert core.active_count==1
    print('[PASS] world-area growth budget stops independently of numerical sample capacity')


def check_projected_fields_and_state(device):
    from agents_gpu import PARTICLE_META_BUFFER_OFFSET
    from density_gpu_check import _projected_plane
    core,agents,environment=make_system(device,include_environment=True)
    load_samples(core,agents,[[.5025,.5033]],[[0,0]],domains=[[SPACING,0,0,SPACING/2]])
    meta=np.zeros(1,dtype=agents._particle_meta_dtype)
    meta["chemicalState"][:]=.5
    meta["privateState"][0]=np.linspace(-.3,.4,8)
    meta["color"][0]=[.2,.3,.4,1]
    device.queue.write_buffer(agents._agent_state_buffer,PARTICLE_META_BUFFER_OFFSET,meta.tobytes())
    core.set_splat_radius(.004)
    def morphology():
        encoder=device.create_command_encoder();core.encode_morphology(encoder)
        device.queue.submit([encoder.finish()]);return core.read_morphology()
    chemical_before=_projected_plane(environment,agents,0)
    morphology_before=morphology()
    run_growth_field(device,agents);assert synchronize_count(core,agents)==2
    chemical_after=_projected_plane(environment,agents,0)
    morphology_after=morphology()
    raw=device.queue.read_buffer(agents._agent_state_buffer,PARTICLE_META_BUFFER_OFFSET,2*meta.dtype.itemsize)
    children=np.frombuffer(raw,dtype=meta.dtype)
    for field in ("privateState","chemicalState","color"):
        np.testing.assert_allclose(children[field],np.repeat(meta[field],2,axis=0))
    np.testing.assert_allclose(chemical_after.sum(),chemical_before.sum(),rtol=.002,atol=1e-5)
    chemical_error=np.abs(chemical_after-chemical_before).sum()/max(np.abs(chemical_before).sum(),1e-8)
    morphology_error=np.abs(morphology_after-morphology_before).sum()/max(np.abs(morphology_before).sum(),1e-8)
    assert chemical_error < .01, chemical_error
    assert morphology_error < .01, morphology_error
    print(f'[PASS] subdivision inherits chemistry/private state; projection L1 changes chemical={chemical_error:.3g}, morphology={morphology_error:.3g}')


def check_seed_reset(device):
    from training_sim import seed_blob
    core,agents=make_system(device)
    scene=seed_blob(7,(.5,.5),SPACING,17)
    core.reset_growth_buffers(32);core.load_scene(*scene)
    rest=core.read_rest_state()
    np.testing.assert_allclose(rest[:,12:16],scene[-1])
    np.testing.assert_allclose(rest[:,8],4*np.linalg.det(scene[-1].reshape(-1,2,2)))
    # Centers differ by integer translations in the common full-edge basis:
    # these parallelograms form a genuine nonoverlapping lattice partition.
    h=scene[-1][0].reshape(2,2)
    lattice=(scene[0]-scene[0][0])@np.linalg.inv(2*h).T
    np.testing.assert_allclose(lattice,np.round(lattice),atol=3e-5)
    next_scene=seed_blob(1,(.4,.4),SPACING,21)
    core.reset_growth_buffers(32);core.load_scene(*next_scene)
    np.testing.assert_allclose(core.read_rest_state()[:,12:16],next_scene[-1])
    print('[PASS] seed domains tile their lattice and survive rollout reset/load order')


def check_periodic_transfer(device):
    core,agents=make_system(device)
    x=np.array([.0004,.9996])
    h=np.array([[.002,.0003],[.0002,.0015]])
    load_samples(core,agents,[x],[[0,0]],domains=h.reshape(1,4))
    velocity=np.array([.2,-.1],np.float32)
    device.queue.write_buffer(core.velocities,0,velocity.reshape(1,2))
    core.set_material(0,.2,0,1,growth_rate=0,particle_mass=100)
    actual=p2g_grid(core)
    expected=np.zeros_like(actual)
    for node,row in scatter([Domain(x,h,mass=100,velocity=velocity)],DX).items():
        index=(node[0]%GRID_N)*(GRID_N+1)+(node[1]%GRID_N)
        expected[index]+=[row[1],row[2],row[0]]
    np.testing.assert_allclose(actual,expected,atol=.003,rtol=1e-3)
    print('[PASS] domain transfers wrap at both toroidal seams without clipping')


def main():
    device=pick_device()
    for check in (check_continuous_growth,check_opposed_field,check_subdivision,
                  check_compression_and_passive_stretch,check_p2g_conservation,
                  check_affine_transport,check_courant_guard,check_capacity,
                  check_capacity_rollout,check_physical_budget,
                  check_projected_fields_and_state,check_seed_reset,check_periodic_transfer,check_uniform_rollout):
        check(device)

if __name__=='__main__': main()

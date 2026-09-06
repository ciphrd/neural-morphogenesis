"""GPU acceptance tests for continuous growth and material-domain subdivision.

Run from trainer: .venv/bin/python continuous_growth_check.py
"""
from __future__ import annotations
import time
import numpy as np
import wgpu
from agents_gpu import AgentsGPU
from device import pick_device
from simulation_settings import SPLAT_RADIUS
from environment_gpu import EnvironmentGPU
from mpm_core import GROWTH_FIELD_CHANNELS, DT, GRID_N, MpmCore, REST_FIELDS
from triangle_domain_check import Triangle
from triangle_vertices import domain_edges, vertices_from_edges, unwrap_vertices

SPACING = 0.0027
LEG = np.sqrt(2)*SPACING  # Right triangle of area SPACING².
DX = 1/GRID_N

def make_system(device, capacity=32, include_environment=False):
    core = MpmCore(device)
    environment = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0, chemical_communication_architecture="cell-owned-projection")
    agents = AgentsGPU(device, core, environment, 1, 128, 1.0, capacity, SPACING, 1.0, 1.0, 0.5, 0.5, chemical_communication_architecture="cell-owned-projection")
    core.set_gravity(0)
    core.set_repulsion_strength(0, 40)
    core.set_material(0, 0.2, 0, 1, growth_rate=0)
    return (core, agents, environment) if include_environment else (core, agents)

def load_samples(core, agents, positions, vectors, growth_f=None, domains=None):
    n = len(positions)
    identity = np.tile([1, 0, 0, 1], (n, 1)).astype(np.float32)
    f = identity if growth_f is None else np.asarray(growth_f, np.float32)
    h = f*LEG if domains is None else np.asarray(domains, np.float32)
    core.load_scene(np.asarray(positions, np.float32), np.zeros((n, 2), np.float32),
                    f, np.zeros((n, 4), np.float32), np.ones(n, np.float32), vertices_from_edges(positions,h),
                    domain_geometry='triangle-vertices')
    rest = core.read_rest_state()
    rest[:, :4] = f
    rest[:, 5:7] = vectors
    core.device.queue.write_buffer(core.rest, 0, rest)
    agents.set_active_count(n)

def run_growth_field(device, agents):
    encoder = device.create_command_encoder()
    agents.encode_growth_field(encoder)
    device.queue.submit([encoder.finish()])

def project_growth_field(core, agents, boundary=True):
    """Run only projection, before refinement changes this test's domains."""
    encoder = core.device.create_command_encoder()
    # clear field, clear edge index, index edges, scatter domain, scatter boundary, finalize.
    for stage, entry in enumerate(agents._growth_entries):
        if entry == "linkRefinementEdges":
            break
        if entry == "scatterGrowthBoundary" and not boundary:
            continue
        p = encoder.begin_compute_pass()
        p.set_pipeline(agents._growth_pipelines[stage])
        p.set_bind_group(0, agents._growth_bind_groups[stage])
        dispatch = agents._growth_dispatches[stage]
        p.dispatch_workgroups(dispatch if dispatch is not None else (core.active_count+63)//64)
        p.end()
    core.device.queue.submit([encoder.finish()])
    return np.frombuffer(core.device.queue.read_buffer(core.growth_field), np.float32).reshape(-1, GROWTH_FIELD_CHANNELS).copy()


def read_rest(core, count):
    return np.frombuffer(core.device.queue.read_buffer(core.rest, 0, count*REST_FIELDS*4),
                         np.float32).reshape(count, REST_FIELDS).copy()

def synchronize_count(core, agents):
    n = agents.read_sample_count()
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
    return np.frombuffer(core.device.queue.read_buffer(core.grid_accum), np.float32).reshape(-1, 3).astype(float)

def point_p2g(position, velocity, affine, mass):
    """Reference stress-free quadratic MLS-MPM point transfer."""
    result = np.zeros(((GRID_N + 1) ** 2, 3), dtype=float)
    base = np.floor(position / DX - 0.5).astype(int)
    fx = position / DX - base
    weights = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2,
               0.5 * (fx - 0.5) ** 2]
    for i in range(3):
        for j in range(3):
            node = base + [i, j]
            dpos = (np.array([i, j]) - fx) * DX
            weight = weights[i][0] * weights[j][1]
            momentum = weight * mass * (velocity + affine @ dpos)
            index = (node[0] % GRID_N) * (GRID_N + 1) + node[1] % GRID_N
            result[index] += [momentum[0], momentum[1], weight * mass]
    return result

def check_continuous_growth(device):
    core, agents = make_system(device)
    load_samples(core, agents, [[.5,.5]], [[.5,0]])
    project_growth_field(core, agents, boundary=False)
    core.set_material(0, .2, 0, 1, growth_rate=12, growth_compression_feedback=0)
    core.step(1)
    np.testing.assert_allclose(np.linalg.det(read_rest(core,1)[0,:4].reshape(2,2)),
                               np.exp(.5*12*DT), rtol=2e-5)
    print('[PASS] continuous exponential rest growth')

def check_opposed_field(device):
    core, agents = make_system(device)
    load_samples(core,agents,[[.5,.5],[.5,.5]],[[.6,0],[-.6,0]])
    run_growth_field(device,agents)
    field=np.frombuffer(device.queue.read_buffer(core.growth_field),np.float32).reshape(-1,GROWTH_FIELD_CHANNELS)
    np.testing.assert_allclose(field[:,:2].sum(axis=0),0,atol=2e-7)
    np.testing.assert_allclose(field[:,2:5], 0, atol=2e-7)
    core.set_material(0,.2,0,1,growth_rate=12,growth_compression_feedback=0)
    core.step(16)
    np.testing.assert_allclose(read_rest(core,2)[:,:4], [[1,0,0,1]]*2, atol=1e-6)
    print('[PASS] opposing proposals cancel field and physical growth')

def check_vector_blending(device):
    for vectors, expected in (([[.6,0],[.6,0]], [.6,0,0]),
                              ([[.6,0],[0,.6]], [np.sqrt(.18)/2]*3),
                              ([[.6,0],[-.2,0]], [.2,0,0])):
        core, agents = make_system(device)
        load_samples(core,agents,[[.5,.5],[.5,.5]],vectors)
        project_growth_field(core,agents,boundary=False)
        field=np.frombuffer(device.queue.read_buffer(core.growth_field),np.float32).reshape(-1,GROWTH_FIELD_CHANNELS)
        np.testing.assert_allclose(field[:,2:5].sum(axis=0)/field[:,5].sum(), expected, atol=1e-3)
        core.set_material(0,.2,0,1,growth_rate=12,growth_compression_feedback=0)
        core.step(16)
        determinants=np.linalg.det(read_rest(core,2)[:,:4].reshape(-1,2,2))
        np.testing.assert_allclose(determinants, np.exp((expected[0]+expected[2])*12*DT*16),rtol=3e-5)
    print('[PASS] aligned, perpendicular, and unequal opposed vectors blend before tensor conversion')

def check_subdivision(device):
    core,agents=make_system(device)
    h=np.array([[2*LEG,0],[0,2*LEG]],np.float32)
    # Transported area 4*SPACING² produces two successive bisections.
    load_samples(core,agents,[[.5,.5]],[[0,0]],growth_f=[[2,0,0,2]],
                 domains=h.reshape(1,4))
    original=read_rest(core,1)[0]
    expected = [Triangle(np.array([.5,.5]), h)]
    for count in (2,4):
        run_growth_field(device,agents)
        assert synchronize_count(core,agents)==count
        pairs = [t.split() for t in expected]
        expected = [pair[0] for pair in pairs]+[pair[1] for pair in pairs]
    rest=read_rest(core,4)
    np.testing.assert_allclose(rest[:, 15],.25)
    # GPU slot allocation order is not guaranteed; match by centroid.
    for position, row in zip(core.read_positions(), rest):
        target = min(expected, key=lambda t: np.linalg.norm(t.x-position))
        np.testing.assert_allclose(position,target.x,atol=1e-7)
        np.testing.assert_allclose(domain_edges(row),target.edges,atol=1e-7)
        np.testing.assert_allclose(row[14],original[14]/4,rtol=2e-6)
    np.testing.assert_allclose(rest[:,:4],np.tile(original[:4],(4,1)))
    offsets=core.read_positions()-.5
    np.testing.assert_allclose(offsets.mean(axis=0),0,atol=1e-7)
    cov=sum(.25*(np.outer(d,d)+Triangle(np.zeros(2), domain_edges(r)).covariance())
            for d,r in zip(offsets,rest))
    np.testing.assert_allclose(cov,Triangle(np.zeros(2),h).covariance(),rtol=5e-5,atol=1e-10)
    print('[PASS] repeated bisection tiles the parent and preserves second moments')

def check_geometric_refinement_criterion(device):
    core,agents=make_system(device)
    # Rest growth alone does not create a numerical sample until mechanics has
    # actually expanded the transported material domain.
    load_samples(core,agents,[[.5,.5]],[[0,0]],growth_f=[[3,0,0,3]],
                 domains=[[LEG,0,0,LEG]])
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==1
    # A stretched triangle can need several rounds before every child is
    # spatially resolved; each longest-edge split reduces the squared-edge sum.
    load_samples(core,agents,[[.5,.5]],[[0,0]],growth_f=[[2,0,0,2]],domains=[[2*LEG,0,0,LEG]])
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==2
    for _ in range(12):
        run_growth_field(device,agents)
        synchronize_count(core,agents)
    from conforming_refinement_check import demands
    assert np.max(demands(read_rest(core,core.active_count))) < 1.75
    # Isochoric stretch must refine even though current area is unchanged.
    load_samples(core,agents,[[.5,.5]],[[0,0]],domains=[[4*LEG,0,0,LEG/4]])
    before=read_rest(core,1)
    np.testing.assert_allclose(.5*np.linalg.det(domain_edges(before))[0],SPACING**2,rtol=2e-4)
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==2
    print('[PASS] spatial second-moment criterion resolves expansion and isochoric stretch; rest growth alone does not split')

def check_triangle_edges_and_seams(device):
    core,agents=make_system(device)
    base=np.array([[0.,0.],[2.,0.],[.3,.7]])
    for turn in range(3):
        triangle=Triangle.from_vertices(np.roll(base,turn,axis=0))
        edges=triangle.edges*np.sqrt(2*SPACING**2/triangle.signed_area())
        for center in ([.5,.5],[.0001,.9999]):
            parent=Triangle(np.array(center),edges)
            load_samples(core,agents,[center],[[0,0]],domains=edges.reshape(1,4))
            run_growth_field(device,agents)
            assert synchronize_count(core,agents)==2
            for position,row,child in zip(core.read_positions(),read_rest(core,2),parent.split()):
                np.testing.assert_allclose(position,child.x%1,atol=1e-7)
                np.testing.assert_allclose(domain_edges(row),child.edges,atol=1e-7)
                assert np.linalg.det(domain_edges(row)) > 0
    # Untagged old domain arrays must fail before touching live buffers.
    before=core.read_positions().copy()
    scene=(np.array([[.2,.2]],np.float32),np.zeros((1,2),np.float32),
           np.array([[1,0,0,1]],np.float32),np.zeros((1,4),np.float32),np.ones(1,np.float32))
    try:
        core.load_scene(*scene,domain=np.array([[LEG,0,0,LEG]],np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid geometry was accepted')
    np.testing.assert_array_equal(core.read_positions(),before)
    print('[PASS] all three edge choices, periodic triangle splitting and invalid geometry rejection')

def check_point_p2g_and_split_conservation(device):
    core,agents=make_system(device)
    h=np.array([[2*LEG,0],[0,2*LEG]])
    x=np.array([.5031,.5027])
    c=np.array([[2,-70],[70,-1]],np.float32)
    v=np.array([.2,-.1],np.float32)
    growth=np.array([[np.sqrt(2),0,0,np.sqrt(2)]],np.float32)
    load_samples(core,agents,[x],[[0,0]],growth_f=growth,domains=h.reshape(1,4))
    device.queue.write_buffer(core.C,0,c.reshape(1,4))
    device.queue.write_buffer(core.velocities,0,v.reshape(1,2))
    core.set_material(0,.2,0,1,growth_rate=0,particle_mass=100)
    before=p2g_grid(core)
    represented_mass=200  # particleMass * q * det(G)
    expected=point_p2g(x,v,c,represented_mass)
    np.testing.assert_allclose(before,expected,atol=.0025,rtol=1e-3)
    # The transported triangle is refinement geometry only. Changing it
    # must not change an otherwise identical point transfer.
    load_samples(core,agents,[x],[[0,0]],growth_f=growth,
                 domains=(h@np.array([[4.,1.],[0.,.25]])).reshape(1,4))
    device.queue.write_buffer(core.C,0,c.reshape(1,4))
    device.queue.write_buffer(core.velocities,0,v.reshape(1,2))
    np.testing.assert_array_equal(p2g_grid(core),before)
    load_samples(core,agents,[x],[[0,0]],growth_f=growth,domains=h.reshape(1,4))
    device.queue.write_buffer(core.C,0,c.reshape(1,4))
    device.queue.write_buffer(core.velocities,0,v.reshape(1,2))
    run_growth_field(device,agents)
    assert synchronize_count(core,agents)==2
    after=p2g_grid(core)
    coords=np.array([(i*DX,j*DX) for i in range(GRID_N+1) for j in range(GRID_N+1)])-x
    def angular(g): return np.sum(coords[:,0]*g[:,1]-coords[:,1]*g[:,0])
    np.testing.assert_allclose(after.sum(axis=0),before.sum(axis=0),atol=.005)
    np.testing.assert_allclose(angular(after),angular(before),atol=2e-5)
    print('[PASS] point P2G ignores domain geometry; split conserves mass, momentum and angular momentum')

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
    actual=domain_edges(read_rest(core,1)[0])
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
    print(f'[PASS] grid CFL guard bounds extreme momentum at {max_grid_speed:g}')

def check_capacity(device):
    core,agents=make_system(device,capacity=1)
    load_samples(core,agents,[[.5,.5]],[[1,0]],growth_f=[[2,0,0,1]],
                 domains=[[2*LEG,0,0,2*LEG]])
    before=read_rest(core,1)
    before_positions=core.read_positions().copy()
    run_growth_field(device,agents)
    assert agents.read_sample_count()==1 and agents.unresolved_samples==1
    np.testing.assert_allclose(read_rest(core,1)[:,[0,1,2,3,14,15,8,9,10,11,12,13]],before[:,[0,1,2,3,14,15,8,9,10,11,12,13]])
    np.testing.assert_array_equal(core.read_positions(),before_positions)
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
    area=np.sum(rest[:, 15]*np.linalg.det(rest[:,:4].reshape(-1,2,2)))
    np.testing.assert_allclose(area,np.exp(80*60*32*DT),rtol=3e-3)
    assert core.active_count>1
    assert np.isfinite(core.read_positions()).all() and np.isfinite(rest).all()
    assert np.all(np.linalg.det(domain_edges(rest))>0)
    print(f'[PASS] free uniform growth: area={area:.4f}, samples={core.active_count}, {time.perf_counter()-start:.2f}s')

def check_capacity_rollout(device):
    # An odd cap may leave a spare slot when the next operation needs two.
    # Exercise physics after both full-cap and group-capacity safety pauses.
    core, agents = make_system(device, capacity=9)
    load_samples(core, agents, [[.5, .5]], [[1, 0]])
    agents.set_forced_growth_field_override(True)
    core.set_material(10000, .2, 3, .5, growth_rate=80,
                      growth_anisotropy=0, growth_compression_feedback=0)
    capped_steps = 0
    for _ in range(160):
        run_growth_field(device, agents)
        count = synchronize_count(core, agents)
        before_step = read_rest(core, count)
        core.step(32)
        rest = read_rest(core, count)
        assert np.isfinite(core.read_positions()).all()
        assert np.isfinite(rest).all()
        assert np.all(np.linalg.det(domain_edges(rest)) > 0)
        if count == 9 or agents.capacity_blocked:
            capped_steps += 1
            np.testing.assert_array_equal(rest[:, :4], before_step[:, :4])
    assert capped_steps >= 80, 'must exercise sustained physics after reaching capacity'
    print(f'[PASS] partial final allocation and {capped_steps * 32} post-cap physics steps stay finite; growth stops')

def _projected_plane(environment, agents, channel):
    environment.reset()
    encoder = environment.device.create_command_encoder()
    environment.encode_clear(encoder)
    agents.encode_splat_chemical_state(encoder)
    environment.encode_sense(encoder)
    environment.device.queue.submit([encoder.finish()])
    offset = environment.channel_offsets[channel]
    width, height = environment.channel_widths[channel], environment.channel_heights[channel]
    raw = environment.device.queue.read_buffer(environment.buffers[0], offset*4, width*height*4)
    return np.frombuffer(raw, np.float32).reshape(height, width).copy()

def check_projected_fields_and_state(device, scale=1.0):
    from agents_gpu import PARTICLE_META_BUFFER_OFFSET
    core,agents,environment=make_system(device,include_environment=True)
    load_samples(core,agents,[[.5025,.5033]],[[0,0]],growth_f=[[2,0,0,1]],
                 domains=[[2*LEG*scale,0,0,LEG*scale]])
    # Keep the area-to-target ratio fixed while refining the sample geometry.
    agents.set_density_geometry(SPACING*scale)
    meta=np.zeros(1,dtype=agents._particle_meta_dtype)
    meta["chemicalState"][:]=.5
    meta["privateState"][0]=np.linspace(-.3,.4,8)
    meta["color"][0]=[.2,.3,.4,1]
    device.queue.write_buffer(agents._agent_state_buffer,PARTICLE_META_BUFFER_OFFSET,meta.tobytes())
    core.set_splat_radius(SPLAT_RADIUS)
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
    # Check refinement invariance using the configured morphology splat width.
    assert morphology_error < .02, morphology_error
    if scale == 1.0:
        fine_error = check_projected_fields_and_state(device, .5)
        assert fine_error < .5*morphology_error, (fine_error,morphology_error)
    print(f'[PASS] subdivision inherits chemistry/private state; projection L1 changes chemical={chemical_error:.3g}, morphology={morphology_error:.3g}')
    return morphology_error

def check_seed_reset(device):
    from training_sim import seed_blob
    core,agents=make_system(device)
    scene=seed_blob(7,(.5,.5),SPACING,17)
    core.reset_growth_buffers(32);core.load_scene(*scene)
    rest=core.read_rest_state()
    assert len(rest)==14
    np.testing.assert_allclose(rest[:,8:14],scene[5])
    np.testing.assert_allclose(rest[:, 14],.5*np.linalg.det(domain_edges(rest)))
    # Validate the scene's physical weights rather than assuming all seed
    # triangles have equal area (circular disk seeds are area-weighted).
    np.testing.assert_allclose(rest[:, 15],scene[6])
    np.testing.assert_allclose(rest[:, 15].sum(),7,rtol=2e-6)
    assert np.all(rest[:, 15]>0)
    area=.5*np.linalg.det(domain_edges(rest))
    np.testing.assert_allclose(rest[:, 15],7*area/area.sum(),rtol=2e-4)
    next_scene=seed_blob(1,(.4,.4),SPACING,21)
    core.reset_growth_buffers(32);core.load_scene(*next_scene)
    assert core.active_count==2
    np.testing.assert_allclose(core.read_rest_state()[:,8:14],next_scene[5])
    np.testing.assert_allclose(core.read_rest_state()[:, 15],.5)
    print('[PASS] seed geometry and represented weights survive rollout reset/load order')

def check_periodic_transfer(device):
    core,agents=make_system(device)
    x=np.array([.0004,.9996])
    h=np.array([[.002,.0003],[.0002,.0015]])
    load_samples(core,agents,[x],[[0,0]],domains=h.reshape(1,4))
    velocity=np.array([.2,-.1],np.float32)
    device.queue.write_buffer(core.velocities,0,velocity.reshape(1,2))
    core.set_material(0,.2,0,1,growth_rate=0,particle_mass=100)
    actual=p2g_grid(core)
    expected=point_p2g(x,velocity,np.zeros((2,2)),100)
    np.testing.assert_allclose(actual,expected,atol=.003,rtol=1e-3)
    print('[PASS] point transfers wrap at both toroidal seams without clipping')

def main():
    device=pick_device()
    for check in (check_continuous_growth,check_opposed_field,check_vector_blending,check_subdivision,
                  check_geometric_refinement_criterion,check_triangle_edges_and_seams,check_point_p2g_and_split_conservation,
                  check_affine_transport,check_courant_guard,check_capacity,
                  check_capacity_rollout,
                  check_projected_fields_and_state,check_seed_reset,check_periodic_transfer,check_uniform_rollout):
        check(device)

if __name__=='__main__': main()

"""GPU regression checks for the local minimum daughter material weight."""
import numpy as np
from continuous_growth_check import make_system, load_samples, read_rest, run_growth_field, synchronize_count, LEG
from conforming_refinement_check import fixture, assert_disk
from device import pick_device
from config import CONFIG
from agents_gpu import PARTICLE_META_BUFFER_OFFSET
MIN_WEIGHT = CONFIG["simulation"]["MIN_CHILD_WEIGHT"]

def check_grid_span(d):
    dx=1/CONFIG['simulation']['INV_DX']
    a=np.array([.5,.5])
    # Fully overlapping, connected via a middle vertex, and isolated vertex.
    # Last two cases differ at the zero-weight quadratic support endpoint.
    cases=[([[0,0],[1.8,0],[.9,1]],1),
           ([[0,0],[4,0],[2,.25]],1),
           ([[0,0],[4,0],[.25,.25]],0),
           ([[.5,0],[2.5-1/128,0],[.5,.25]],1),
           ([[.5,0],[2.5,0],[.5,.25]],0)]
    for order in ([0,1,2],[1,2,0],[2,0,1]):
        for coordinates,expected in cases:
            for offset in ([0,0],[.5-dx,0],[0,.5-dx]):
                core,agents=make_system(d)
                triangle=(a+np.array(coordinates)*dx+offset)%1
                fixture(core,agents,[triangle[order]])
                r=read_rest(core,1)
                # Valid boundary samples stay unsplit; oversized samples have
                # ample weight, proving refinement cannot rescue them.
                r[:,15]=MIN_WEIGHT if expected else 1
                d.queue.write_buffer(core.rest,0,r)
                run_growth_field(d,agents)
                assert synchronize_count(core,agents)==expected,(order,coordinates,offset)
    print('[PASS] grid support connectivity, indirect links, zero-weight endpoints and periodic seams')

def check_production_seed(d):
    from training_sim import seed_blob
    from simulation_settings import INITIAL_SPACING, INITIAL_PARTICLE_COUNT
    core,agents=make_system(d,capacity=256)
    agents.set_density_geometry(CONFIG['run']['sampleSpacing'])
    for center in ((.5,.5),(.9999,.0001)):
        scene=seed_blob(INITIAL_PARTICLE_COUNT,center,INITIAL_SPACING,0)
        core.load_scene(*scene)
        agents.set_active_count(len(scene[0]))
        for _ in range(3):
            run_growth_field(d,agents)
            n=synchronize_count(core,agents)
            assert n>=len(scene[0])
            np.testing.assert_allclose(read_rest(core,n)[:,15].sum(),scene[6].sum(),rtol=1e-6)
    print('[PASS] production seed retains all material through initial pruning, including at seams')

def check_pruning(d):
    core,agents=make_system(d,capacity=4)
    load_samples(core,agents,[[.3,.3],[.4,.4],[.5,.5],[.6,.6]],[[0,0]]*4)
    r=read_rest(core,4)
    # All connected samples survive, even below the split floor or at zero weight.
    r[:,15]=[MIN_WEIGHT,MIN_WEIGHT,MIN_WEIGHT/2,MIN_WEIGHT-1e-5]
    r[0,0]=.5
    r[2,0]=2
    r[3,15]=0
    d.queue.write_buffer(core.rest,0,r)
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==4
    np.testing.assert_array_equal(read_rest(core,4),r)
    # Disconnect the first and last samples to exercise tail compaction.
    r[[0,3],8:12]=[.3,.3,.5,.3]
    r[[0,3],12:14]=[.3,.301]
    d.queue.write_buffer(core.rest,0,r)
    snapshots=[]
    for buffer,width in [(core.positions,2),(core.velocities,2),(core.C,4),(core.F,4)]:
        values=np.frombuffer(d.queue.read_buffer(buffer,0,4*width*4),np.float32).copy().reshape(4,width)
        if buffer is not core.positions:
            values[:]=np.arange(4*width).reshape(4,width)+1
            d.queue.write_buffer(buffer,0,values)
        snapshots.append((buffer,values))
    meta=np.zeros(4,dtype=agents._particle_meta_dtype)
    meta['chemicalState'][:,0]=[11,22,33,44]
    meta['privateState'][:,0]=[1,2,3,4]
    d.queue.write_buffer(agents._agent_state_buffer,PARTICLE_META_BUFFER_OFFSET,meta)
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==2
    # Tail swap moves sample 2 into slot 0, including its complete state.
    np.testing.assert_array_equal(read_rest(core,2),r[[2,1]])
    for buffer,values in snapshots:
        actual=np.frombuffer(d.queue.read_buffer(buffer,0,2*values.shape[1]*4),np.float32).reshape(2,-1)
        np.testing.assert_array_equal(actual,values[[2,1]])
    actual=np.frombuffer(d.queue.read_buffer(agents._agent_state_buffer,PARTICLE_META_BUFFER_OFFSET,2*meta.dtype.itemsize),meta.dtype)
    np.testing.assert_array_equal(actual,meta[[2,1]])
    # Freed slots can split again, and newly created valid daughters survive.
    r=read_rest(core,2);r[:,15]=2*MIN_WEIGHT
    r[:,0:4]=[1,0,0,1]
    r[:,8:12]=[.4,.4,.412,.4];r[:,12:14]=[.4,.401]
    d.queue.write_buffer(core.rest,0,r)
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==4
    r=read_rest(core,4)
    r[:,8:12]=[.3,.3,.5,.3];r[:,12:14]=[.3,.301]
    d.queue.write_buffer(core.rest,0,r)
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==0
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==0
    core.step(1)
    print('[PASS] small-volume retention, disconnected state compaction, slot reuse and extinction')

def main():
    d=pick_device()
    check_grid_span(d)
    check_production_seed(d)
    check_pruning(d)
    for weight,growth,count in [(2*MIN_WEIGHT,1,2),(2*MIN_WEIGHT-1e-5,1,1),(MIN_WEIGHT,1,1),(MIN_WEIGHT,2,2)]:
        core,agents=make_system(d)
        load_samples(core,agents,[[.5,.5]],[[1,0]],domains=[[4*LEG,0,0,LEG/4]])
        r=read_rest(core,1);r[0, 15]=weight;r[0,0]=growth
        d.queue.write_buffer(core.rest,0,r)
        run_growth_field(d,agents)
        assert synchronize_count(core,agents)==count
        after=read_rest(core,count)
        np.testing.assert_allclose(after[:, 15].sum(),weight)
        np.testing.assert_allclose(after[:, 14].sum(),r[:, 14].sum())
        assert not agents.capacity_blocked
    print('[PASS] inclusive child threshold, conservation, and renewed eligibility after growth')
    a,b,c,e=np.array([[.4,.4],[.407,.4],[.4035,.405],[.4035,.399]])
    for weights in ([1,MIN_WEIGHT],[MIN_WEIGHT,1],[2*MIN_WEIGHT,2*MIN_WEIGHT]):
        core,agents=make_system(d)
        fixture(core,agents,[[a,b,c],[b,a,e]])
        r=read_rest(core,2);r[:, 15]=weights;d.queue.write_buffer(core.rest,0,r)
        before=core.read_positions().copy()
        run_growth_field(d,agents);n=synchronize_count(core,agents)
        assert n==(4 if min(weights)==2*MIN_WEIGHT else 2)
        if n==2: np.testing.assert_array_equal(core.read_positions(),before)
        assert_disk(read_rest(core,n))
    print('[PASS] insufficient weight on either side preserves the complete shared-edge pair')
    core,agents=make_system(d,capacity=256)
    load_samples(core,agents,[[.5,.5]],[[0,0]],domains=[[.3,0,0,.00003]])
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==0
    print('[PASS] extreme stretched triangle is deleted before it can refine')

if __name__=='__main__':main()

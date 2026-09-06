"""GPU regression checks for the local minimum daughter material weight."""
import numpy as np
from continuous_growth_check import make_system, load_samples, read_rest, run_growth_field, synchronize_count, LEG
from conforming_refinement_check import fixture, assert_disk
from device import pick_device
from config import CONFIG
MIN_WEIGHT = CONFIG["simulation"]["MIN_CHILD_WEIGHT"]

def main():
    d=pick_device()
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
    for _ in range(40):
        run_growth_field(d,agents);n=synchronize_count(core,agents)
        assert n<=int(1/MIN_WEIGHT)
    r=read_rest(core,n)
    assert 1<n<=int(1/MIN_WEIGHT)
    assert np.all(r[:, 15] / 2 < MIN_WEIGHT)
    run_growth_field(d,agents)
    assert synchronize_count(core,agents)==n
    np.testing.assert_allclose(r[:, 15].sum(),1)
    assert np.all(r[:, 15]>=MIN_WEIGHT)
    print(f'[PASS] extreme length refinement stops at {n} descendants (bound {int(1/MIN_WEIGHT)}) without deleting material')

if __name__=='__main__':main()

"""GPU regressions for stable capacity allocation and fresh-rollout reset."""
import numpy as np
from conforming_refinement_check import fixture
from continuous_growth_check import make_system, read_rest, run_growth_field, synchronize_count
from device import pick_device


def main():
    device=pick_device()
    count=32;capacity=count+17
    core,agents=make_system(device,capacity=capacity)
    triangles=[]
    for i in range(count):
        origin=np.array([.2+.025*(i%8),.2+.025*(i//8)])
        triangles.append(origin+np.array([[0.,0.],[.01,0.],[0.,.01]]))
    baseline=None
    for repeat in range(8):
        device.queue.write_buffer(core.grid_vel,0,np.ones(core.grid_vel.size//4,np.float32))
        core.reset_growth_buffers(capacity)
        assert not np.any(np.frombuffer(device.queue.read_buffer(core.grid_vel),np.float32))
        agents.reset_state()
        fixture(core,agents,triangles)
        before=read_rest(core,count)
        run_growth_field(device,agents)
        assert synchronize_count(core,agents)==capacity
        after=read_rest(core,capacity)
        # Ascending root IDs win all 17 remaining slots. Later triangles are
        # untouched; each child's third vertex is inherited from its parent.
        np.testing.assert_array_equal(after[17:count,8:],before[17:,8:])
        np.testing.assert_array_equal(after[count:,12:14],before[:17,8:10])
        if baseline is None:baseline=after.copy()
        else:np.testing.assert_array_equal(after,baseline)
        assert agents.capacity_blocked
        np.testing.assert_allclose(after[:,15].sum(),before[:,15].sum())
    print('[PASS] Eight identical capacity-contention outcomes, ascending child IDs, mass conservation, cleared grid velocity')


if __name__=='__main__':main()

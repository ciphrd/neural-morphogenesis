"""Exercise the real rollout loop with deterministic simulated measurements."""
from unittest.mock import MagicMock, patch
import numpy as np
import evolve
from domain_fitness import DomainEvaluation, MatchMetrics, target_mask
from targets import load_target

def run_case(*, bad_steps=(), blocked=False, horizon=20, cutoff=None, enabled=True,
             reach_capacity_at=None):
    args=evolve.build_arg_parser().parse_args(['--macro-steps',str(horizon),
        '--shape-check-interval','2','--shape-confirmations','2','--shape-settle-steps','5'])
    args.growth_steps=cutoff;args.stable_stop=enabled
    target=load_target(args.target)
    core=MagicMock();core.active_count=10
    agents=MagicMock();agents.particle_capacity=args.particles
    agents.max_active_particles=args.particles
    agents.capacity_blocked=blocked;agents.unresolved_samples=0
    flags=[]
    class Sim:
        def __init__(self,*a,**kw): pass
        def macro_step(self,*a,growth_enabled):
            flags.append(growth_enabled)
            if reach_capacity_at == len(flags): core.active_count=agents.max_active_particles
        def positions(self): return np.zeros((10,2))
    def score(*a):
        bad=len(flags) in bad_steps
        return DomainEvaluation(.5 if bad else .01,None,None,MatchMetrics(.5 if bad else .01,.01,.01))
    with patch.object(evolve,'TrainingRollout',Sim),patch.object(evolve,'score_domains',score):
        fitness,positions=evolve.rollout(np.zeros(1),target,target_mask(target,64),None,args,0,
            core,agents,MagicMock(),return_positions=True)
    assert positions.shape==(10,2) and np.isfinite(fitness)
    return flags,core.rollout_diagnostics

def main():
    flags,d=run_case()
    assert d['stableMatch'] and d['steps']==9
    assert flags==[True]*4+[False]*5
    flags,d=run_case(bad_steps=(6,))
    assert d['stableMatch'] and d['steps']==15
    assert flags[:7]==[True]*4+[False]*2+[True]
    flags,d=run_case(blocked=True)
    assert not d['stableMatch'] and len(flags)==1 and d['stopReason']=='capacity'
    flags,d=run_case(reach_capacity_at=3)
    assert len(flags)==3 and d['steps']==3 and d['atCapacity'] and d['stopReason']=='capacity'
    flags,d=run_case(horizon=7)
    assert not d['stableMatch'] and d['settling'] and len(flags)==7
    flags,d=run_case(cutoff=3,enabled=False)
    assert flags==[True]*3+[False]*17 and not d['stableMatch']
    print('[PASS] Actual rollout control: stable match, immediate capacity stop, horizon, explicit growth cutoff')

if __name__=='__main__':main()

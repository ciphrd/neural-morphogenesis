"""Exercise the real rollout loop with deterministic simulated measurements."""
from unittest.mock import MagicMock, patch
import numpy as np
import evolve
from domain_fitness import DomainEvaluation, MatchMetrics, target_mask
from targets import load_target

def run_case(*, bad_steps=(), blocked=False, horizon=20, cutoff=None, enabled=True,
             reach_capacity_at=None, samples_at_200=10, initial_count=10,
             step_scores=None, snapshot=False):
    args=evolve.build_arg_parser().parse_args(['--macro-steps',str(horizon),
        '--shape-check-interval','2','--shape-confirmations','2','--shape-settle-steps','5'])
    args.growth_steps=cutoff;args.stable_stop=enabled
    target=load_target(args.target)
    core=MagicMock();core.active_count=initial_count
    agents=MagicMock();agents.particle_capacity=args.particles
    agents.max_active_particles=args.particles
    agents.capacity_blocked=blocked;agents.unresolved_samples=0
    flags=[]
    scored=[]
    class Sim:
        def __init__(self,*a,**kw): pass
        def macro_step(self,*a,growth_enabled):
            flags.append(growth_enabled)
            if len(flags) == 200: core.active_count=samples_at_200
            if reach_capacity_at == len(flags): core.active_count=agents.max_active_particles
        def positions(self): return np.full((10,2),len(flags),dtype=float)
    def score(*a):
        scored.append(len(flags))
        bad=len(flags) in bad_steps
        value = (step_scores or {}).get(len(flags), .5 if bad else .01)
        return DomainEvaluation(value,np.full((2,2),len(flags)),None,MatchMetrics(value,.01,.01))
    with patch.object(evolve,'TrainingRollout',Sim),patch.object(evolve,'score_domains',score):
        result=evolve.rollout(np.zeros(1),target,target_mask(target,64),None,args,0,
            core,agents,MagicMock(),return_positions=True,return_snapshot=snapshot)
    if snapshot:
        return result, scored
    fitness,positions=result
    assert positions.shape==(10,2) and np.isfinite(fitness)
    return flags,core.rollout_diagnostics,scored

def main():
    flags,d,scored=run_case(horizon=100)
    assert flags==[True]*100 and not d['stableMatch'] and not d['settling']
    assert scored==sorted({max(1,round(100*(1-offset))) for offset in evolve.CAPTURE_OFFSETS})
    assert not any(step < 90 for step in scored)
    flags,d,scored=run_case(blocked=True)
    assert len(flags)==1 and d['stopReason']=='capacity' and scored==[1]
    flags,d,scored=run_case(reach_capacity_at=3)
    assert len(flags)==3 and d['atCapacity'] and scored==[3]
    flags,d,scored=run_case(cutoff=3)
    assert flags==[True]*3+[False]*17
    flags,d,scored=run_case(horizon=300)
    assert len(flags)==200 and d['stopReason']=='low-growth' and scored==[200]
    flags,d,scored=run_case(horizon=220,samples_at_200=11)
    assert len(flags)==220 and d['stopReason']=='horizon'
    flags,d,scored=run_case(horizon=220,reach_capacity_at=200)
    assert len(flags)==200 and d['stopReason']=='capacity'
    flags,d,scored=run_case(initial_count=evolve.build_arg_parser().parse_args([]).particles)
    assert not flags and d['stopReason']=='capacity' and scored==[0]
    result, scored = run_case(horizon=100, snapshot=True,
        step_scores={90: .5, 92: .4, 95: .02, 98: .3, 100: .7})
    assert scored == [90, 92, 95, 98, 100]
    assert result.fitness == result.evaluation.total == .02
    assert result.diagnostics['scoreStep'] == 95 and result.diagnostics['steps'] == 100
    assert result.diagnostics['missing'] == .02
    np.testing.assert_array_equal(result.positions, np.full((10,2),95))
    np.testing.assert_array_equal(result.evaluation.raster, np.full((2,2),95))
    assert result.diagnostics['scoredSteps'] == scored
    # Rounding on short horizons samples each available checkpoint only once.
    result, scored = run_case(horizon=3, snapshot=True)
    assert scored == [3] and result.diagnostics['scoreStep'] == 3
    # Early capacity uses the best of already captured poses and the stop pose.
    result, scored = run_case(horizon=100, reach_capacity_at=96, snapshot=True,
                             step_scores={90:.1, 92:.2, 95:.3, 96:.4})
    assert scored == [90, 92, 95, 96] and result.fitness == .1
    assert result.diagnostics['scoreStep'] == 90 and result.diagnostics['steps'] == 96
    result, scored = run_case(horizon=100, snapshot=True,
                             step_scores={90:float('nan'), 92:float('inf'), 95:.03})
    assert result.fitness == .01 and result.diagnostics['scoreStep'] == 98
    assert evolve._aggregate_scores([.5,.02,.7], None) == .02
    assert np.isinf(evolve._aggregate_scores([float('nan'),float('inf')]))
    print('[PASS] No stable checks; minimum late-window scoring, winning pose, short horizons, capacity, low growth and growth cutoff')

if __name__=='__main__':main()

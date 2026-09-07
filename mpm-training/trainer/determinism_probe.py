"""Compare identical rollouts and optional ordered GPU reference kernels.

Ordered stages are a diagnostic reference, not the production algorithm: one
invocation visits samples in index order, retaining the existing float math.
This distinguishes scheduling effects from seeded randomness and stale state.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


from deterministic_reference import install_ordered_stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--architecture',default='stateful-128')
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--sample-every',type=int,default=1)
    parser.add_argument('--max-seconds',type=float,default=300.,help='wall-time budget per repeat; checked between macro steps')
    parser.add_argument('--reuse',action='store_true',help='reuse GPU buffers and reset the rollout between repeats')
    parser.add_argument('--initial-condition',default='none')
    parser.add_argument('--ordered-stages',nargs='*',default=[],choices=[
        'p2g','agentStep','splatChemicalState','indexRefinementEdges',
        'propagateRefinement','reserveRefinement'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.steps,args.repeats,args.sample_every)<1:
        parser.error('steps, repeats and sample interval must be positive')
    if not np.isfinite(args.max_seconds) or args.max_seconds<=0:
        parser.error('max-seconds must be finite and positive')
    install_ordered_stages(args.ordered_stages)
    import single_weight_sensitivity as setup
    setup.POLICY_ARCHITECTURE=args.architecture
    weights=np.load(args.weights)
    args.output.mkdir(parents=True,exist_ok=False)
    from config import CONFIG
    root=Path(__file__).resolve().parents[1]
    source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder,pattern in [('core','*.wgsl'),('trainer','*.py'),('core','config.json')]
        for p in (root/folder).glob(pattern)}
    baseline={};differences=[];hashes=[];times=[];terminal=[]
    def save_report(complete=False):
        report=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            config=CONFIG,source_sha256=source_hashes,weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            complete=complete,seconds=times,terminal=terminal,differences=differences,hashes=hashes)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
    save_report()
    for repeat in range(args.repeats):
        start=perf_counter()
        if not args.reuse or repeat==0:
            core,sim=setup._build_sim(weights,args.seed)
        from training_sim import TrainingRollout
        density=setup._density()
        sim=TrainingRollout(core,sim.agents,sim.environment,spawn_center=(.5,.5),gravity=0.,
            seed=args.seed,mpm_enabled=setup.MPM_ENABLED,initial_particle_count=density.initial_particles,
            initial_spacing=density.initial_spacing,initial_condition=args.initial_condition)
        def snapshot(label):
            state=setup._capture(core,sim.agents)
            for i,buffer in enumerate(sim.environment.buffers):
                state[f'environment_{i}']=np.frombuffer(core.device.queue.read_buffer(buffer),dtype=np.float32).copy()
            state['grid_velocity']=np.frombuffer(core.device.queue.read_buffer(core.grid_vel),dtype=np.float32).copy()
            hashes.append(dict(repeat=repeat,stage=label,sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()}))
            if repeat==0:
                baseline[label]=state
            else:
                if label not in baseline:
                    differences.append(dict(repeat=repeat,stage=label,field='stage_presence',reference_missing=True))
                    return
                for key,value in state.items():
                    ref=baseline[label][key]
                    if value.shape != ref.shape or value.tobytes()!=ref.tobytes():
                        differences.append(dict(repeat=repeat,stage=label,field=key,shape=list(value.shape),
                            reference_shape=list(ref.shape),max_abs=float(np.max(np.abs(value.astype(float)-ref))) if value.shape==ref.shape and value.size else None))
        snapshot('initial')
        original_step=core.step
        step=0
        def physics(*a,**kw):
            if step%args.sample_every==0 or step==1:
                snapshot(f'{step}:before_physics')
            return original_step(*a,**kw)
        core.step=physics
        timed_out=False
        for step in range(1,args.steps+1):
            sim.macro_step(setup.DEFAULT_SUBSTEPS_PER_MACRO)
            if step%args.sample_every==0 or step==1:
                snapshot(f'{step}:after_physics')
            if core.active_count>=sim.agents.max_active_particles or sim.agents.capacity_blocked:
                snapshot('terminal');break
            if step%50==0:
                save_report()
                print(f'repeat {repeat}: step {step}, samples {core.active_count}',flush=True)
            if perf_counter()-start>args.max_seconds:
                timed_out=True
                break
        core.step=original_step
        terminal.append(dict(step=step,samples=core.active_count,timed_out=timed_out))
        if repeat and terminal[-1]!=terminal[0]:
            differences.append(dict(repeat=repeat,stage='terminal',field='control',
                                    reference=terminal[0],actual=terminal[-1]))
        times.append(perf_counter()-start)
        print(f'repeat {repeat}: {times[-1]:.3f}s; differences={len(differences)}',flush=True)
        # Keep the reference alive until readback, then explicitly release GPU
        # resources between independent devices (fresh-state test).
        if not args.reuse or repeat==args.repeats-1:core.device.destroy()
        save_report()
        if timed_out:
            if args.reuse and repeat!=args.repeats-1:core.device.destroy()
            print('Time budget reached; incomplete comparison, not a determinism pass.',flush=True)
            return
    save_report(complete=True)
    print(json.dumps(dict(difference_count=len(differences),first_differences=differences[:12])),flush=True)


if __name__=='__main__':main()

"""CPU-only checks for unavailable adapters and GPU report aggregation."""
from types import SimpleNamespace
from gpu_timing import GpuTimings
from timing import aggregate_rollouts

def main():
    gpu=GpuTimings(SimpleNamespace(features=set()))
    gpu.begin_sample()
    with gpu.measure(None,'gpuPhysics'):pass
    gpu.finish_sample()
    assert not gpu.report()['supported'] and gpu.report()['samples']==0
    gpu.reset()
    assert gpu.steps==0
    gpu.supported=True
    gpu.device=SimpleNamespace(create_query_set=lambda **kwargs: SimpleNamespace(destroy=lambda: None))
    sampled=[]
    for step in range(1,46):
        gpu.begin_sample()
        if gpu.active:sampled.append(step)
    assert sampled==[1,21,41]
    from unittest.mock import MagicMock
    encoder=MagicMock()
    gpu.active=True
    gpu.pending=[]
    gpu.begin_compute_pass(encoder,'fine')
    writes=encoder.begin_compute_pass.call_args.kwargs['timestamp_writes']
    assert writes['beginning_of_pass_write_index']==0 and writes['end_of_pass_write_index']==1
    gpu.pending=['full']*(gpu.QUERY_COUNT//2)
    gpu.begin_compute_pass(encoder,'overflow')
    assert gpu.dropped==1 and encoder.begin_compute_pass.call_args.kwargs=={}
    gpu.interval=0
    gpu.begin_sample()
    assert not gpu.active
    report={'seconds':10,'stages':{},'gpu':{'supported':True,'samples':2,'seconds':.4,
        'stages':{'gpuPhysics':{'seconds':.4,'count':2,'maxSeconds':.3}},'droppedIntervals':0}}
    result=aggregate_rollouts([report,report])
    assert result['gpu']['samples']==4 and result['gpu']['seconds']==.8
    assert result['gpu']['stages']['gpuPhysics']['count']==4
    assert result['gpu']['stages']['gpuPhysics']['maxSeconds']==.3
    assert 'gpu' not in aggregate_rollouts([{'seconds':1,'stages':{}}])
    print('[PASS] Unsupported GPU fallback, reset, sampled GPU aggregation and older timing records')

def native_check():
    import numpy as np
    import evolve
    from targets import load_target
    from domain_fitness import target_mask
    from update_rule import UpdateRule
    from simulation_settings import CHEM_CHANNELS
    from parallel_workers import build_pool
    args=evolve.build_arg_parser().parse_args(['--target','lizard-64','--cell-memory','recurrent',
        '--macro-steps','45','--particles','64','--population','2','--elites','1'])
    evolve.finalize_policy_configuration(args)
    evolve.finalize_density_configuration(args)
    target=load_target(args.target)
    mask=target_mask(target,args.raster_resolution)
    population=[evolve.get_weights(UpdateRule(CHEM_CHANNELS,args.policy_architecture)) for _ in range(2)]
    with build_pool(2,args.particle_capacity,target,mask,None,args) as pool:
        result=evolve.run_generation(population,args,np.random.default_rng(7),pool,return_snapshot=True)
    gpu=result[6].generation_timings['rollouts']['gpu']
    assert gpu['samples']==6 and gpu['droppedIntervals']==0, gpu
    for name in ('gpuMorphologyBlurHorizontal','gpuMorphologyBlurVertical','gpuChemistryClear','gpuChemistryGradient','gpuNeural',
                 'gpuGrowth:commitResample','gpuPhysicsP2G','gpuPhysicsG2P','gpuPhysicsGridUpdate','gpuPhysicsGridClear'):
        assert gpu['stages'][name]['count']>=6 and gpu['stages'][name]['seconds']>0, gpu
    assert not {'gpuPhysics','gpuChemistry','gpuGrowth'} & gpu['stages'].keys(), gpu
    print('[PASS] Native two-worker GPU timestamps: repeated samples, all stages, no omitted intervals')

if __name__=='__main__':
    import sys
    main()
    if '--native' in sys.argv:native_check()

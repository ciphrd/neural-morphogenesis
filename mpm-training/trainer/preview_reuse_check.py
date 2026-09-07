"""Verify representative snapshot selection and PNG reuse without GPU or replay."""
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import evolve
import train_server
from seed_schedule_check import RecordingPool
from rollout_snapshot import RolloutSnapshot
from domain_fitness import DomainEvaluation, MatchMetrics
from raster import RasterFitnessBreakdown
from targets import TargetShape

def main():
    args=SimpleNamespace(seeds_per_candidate=3,population=3,elites=3,mutation_sigma=.05,
        policy_architecture='stateless-128',particle_densities=[.5,1.],density_aggregation='worst')
    population=[np.array([i]) for i in range(3)]
    calls=[]
    def worker(weights,seed,density,return_snapshot=False):
        assert return_snapshot
        score=float((2-int(weights[0]))*100+seed%97+density)
        result=RolloutSnapshot(score,None,None,dict(candidate=int(weights[0]),seed=seed,density=density))
        calls.append(result)
        return result
    with patch.object(evolve,'worker_rollout',worker):
        result=evolve.run_generation(population,args,np.random.default_rng(123),RecordingPool(),return_snapshot=True)
    assert len(calls)==18 and int(result[0][0][0])==2
    snapshot=result[6]
    assert snapshot.diagnostics==dict(candidate=2,seed=result[2],density=result[3])
    assert snapshot is max(calls[12:],key=lambda r:r.fitness)

    n=8
    mask=np.ones((n,n));rgb=np.zeros((n,n,3));rgb[...,0]=.8
    candidate=rgb.copy();candidate[...,1]=.25
    evaluation=DomainEvaluation(.2,mask,RasterFitnessBreakdown(.2,0,0,0,0,0,.2),MatchMetrics(0,0,0),candidate)
    snapshot=pickle.loads(pickle.dumps(RolloutSnapshot(.3,evaluation,np.array([[.5,.5]]),{'steps':12})))
    target=TargetShape.from_wire(dict(mask=mask.ravel(),rgb=rgb.ravel(),resolution=n))
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(train_server,'IMAGES_DIR',Path(tmp)), \
         patch.object(train_server,'target',target), \
         patch.object(train_server,'target_raster',mask), \
         patch.object(train_server,'args',SimpleNamespace(raster_resolution=n)), \
         patch.object(evolve,'rollout',side_effect=AssertionError('preview must not replay')), \
         patch('domain_fitness.score_domains',side_effect=AssertionError('preview must not rescore')):
        diagnostics=train_server._save_generation_images(7,snapshot)
        assert diagnostics['total']==.2 and diagnostics['rollout']=={'steps':12}
        for kind,expected in [('agents',candidate),('target',rgb),('diff',np.abs(candidate-rgb))]:
            pixels=np.asarray(Image.open(Path(tmp)/f'gen_00007_{kind}.png'))
            np.testing.assert_array_equal(pixels,(expected[::-1]*255).astype(np.uint8))
        assert (Path(tmp)/'gen_00007_grown.png').exists()
    print('[PASS] Correct winner/seed/density snapshot, pickling and exact PNG reuse without replay or rescoring')

if __name__=='__main__':main()

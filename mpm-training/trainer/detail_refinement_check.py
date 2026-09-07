"""CPU checks for warm starts, mutation scales and exact colored alignment."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np

import evolve
from domain_fitness import rasterize_triangles, score_domains
from detail_alignment import rotate_vertices
from targets import TargetShape
from seed_schedule_check import RecordingPool


def main():
    args = evolve.build_arg_parser().parse_args(['--population','6','--elites','2',
        '--mutation-factors','1','.1','.01','--fitness-alignment','geometry'])
    evolve.finalize_policy_configuration(args)
    evolve.validate_fitness_configuration(args)
    with tempfile.TemporaryDirectory() as tmp:
        weights = evolve.initial_population(args,np.random.default_rng(0))[0]
        path = Path(tmp)/'parent.npy';np.save(path,weights);args.initial_weights=path
        a=evolve.initial_population(args,np.random.default_rng(71))
        b=evolve.initial_population(args,np.random.default_rng(71))
        np.testing.assert_array_equal(a[0],weights)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        assert not np.shares_memory(a[0],a[1])
        sigmas=[]
        def mutation(parent,sigma,rng,architecture):
            sigmas.append(sigma);return parent.copy()
        with patch.object(evolve,'mutate',mutation), patch.object(evolve,'worker_rollout',lambda *a:1.):
            result=evolve.run_generation(a,args,np.random.default_rng(0),RecordingPool())
        np.testing.assert_allclose(sigmas,[.05,.005,.0005,.05])
        np.testing.assert_array_equal(result[0][0],a[0])
        np.testing.assert_array_equal(result[0][1],a[1])
        for bad in (np.zeros(3), np.full(weights.shape,np.nan)):
            np.save(path,bad)
            try: evolve.initial_population(args,np.random.default_rng(0))
            except ValueError: pass
            else: raise AssertionError('invalid warm start accepted')
    # An asymmetric colored rectangle: the pose must optimize RGB as well as
    # silhouette, and a real color mistake must retain a nonzero penalty.
    triangles=np.array([[[.3,.4],[.7,.4],[.7,.6]],[[.3,.4],[.7,.6],[.3,.6]]])
    colors=np.array([[1.,0.,0.],[0.,0.,1.]])
    mask,rgb=rasterize_triangles(triangles,64,colors)
    target=TargetShape.from_wire(dict(mask=mask.ravel(),rgb=rgb.ravel(),resolution=64))
    rotated=rotate_vertices(triangles,np.deg2rad(13.7),target.center)
    result=score_domains(rotated,target,mask,args,colors)
    assert result.total < 1e-7,result.total
    assert score_domains(rotated,target,mask,args,np.full_like(colors,.5)).total > .05
    assert np.isinf(score_domains(np.full_like(rotated,np.nan),target,mask,args,colors).total)
    print('[PASS] Warm start validation/reproducibility, elite preservation, mutation ladder, precise RGB and invalid geometry')


if __name__ == '__main__': main()

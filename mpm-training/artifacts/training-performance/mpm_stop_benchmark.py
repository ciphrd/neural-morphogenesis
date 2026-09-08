import sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'trainer'))
import numpy as np
from domain_fitness import evaluate_domains,rasterize_triangles,rotate_density,centered_triangles,target_mask,MatchMetrics
from targets import load_target
from training_sim import seed_blob

def match_only(vertices,target,mask):
    triangles,area=centered_triangles(vertices,target.center);n=len(mask);mass=mask.sum()
    density=rasterize_triangles(triangles,n);material_pixels=area*n*n
    best=MatchMetrics(float('inf'),float('inf'),float('inf'));best_angle=0.
    def evaluate(angle):
        nonlocal best,best_angle
        d=density if angle==0 else rotate_density(density,angle,target.center)
        p=np.clip(d,0,1);escaped=max(0.,material_pixels-float(d.sum()))/mass
        match=MatchMetrics(float(np.maximum(mask-p,0).sum()/mass),float(np.maximum(p-mask,0).sum()/mass)+escaped,float(np.maximum(d-1,0).sum()/mass))
        if match.error<best.error-1e-10:best=match;best_angle=angle
    for i in range(16):evaluate(2*np.pi*i/16)
    step=2*np.pi/16
    for _ in range(2):
        step/=3;center=best_angle
        evaluate(center-step);evaluate(center+step)
    return best

target=load_target('lizard-64');mask=target_mask(target,256)
for count in (20,1500):
    vertices=seed_blob(count,(.5,.5),.015,0)[5];colors=np.random.default_rng(7).random((len(vertices),3))
    full=evaluate_domains(vertices,target,mask,colors=colors)
    cheap=match_only(vertices,target,mask)
    np.testing.assert_allclose([cheap.missing,cheap.spill,cheap.overlap],[full.match.missing,full.match.spill,full.match.overlap],atol=1e-12)
    for label,fn in [('full',lambda:evaluate_domains(vertices,target,mask,colors=colors)),('match_only',lambda:match_only(vertices,target,mask))]:
        times=[]
        for _ in range(3):
            start=time.perf_counter();fn();times.append(time.perf_counter()-start)
        print(len(vertices),label,np.median(times),flush=True)

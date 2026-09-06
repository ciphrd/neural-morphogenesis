import sys, time, cProfile,pstats
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'trainer'))
import numpy as np
from targets import load_target
from domain_fitness import evaluate_domains, target_mask
from training_sim import seed_blob
target=load_target('lizard-64')
for count in (20,250,1500):
    scene=seed_blob(count,(.5,.5),.015,0); vertices=scene[5]
    colors=np.random.default_rng(7).random((len(vertices),3))
    for resolution in (128,256):
        mask=target_mask(target,resolution);target.color_raster(resolution)
        for color in (False,True):
            times=[]
            for _ in range(3):
                t=time.perf_counter();evaluate_domains(vertices,target,mask,colors=colors if color else None); times.append(time.perf_counter()-t)
            print(len(vertices),resolution,color,'median_seconds',np.median(times),flush=True)
p=cProfile.Profile();p.enable();evaluate_domains(vertices,target,mask,colors=colors);p.disable()
pstats.Stats(p).strip_dirs().sort_stats('cumulative').print_stats(20)

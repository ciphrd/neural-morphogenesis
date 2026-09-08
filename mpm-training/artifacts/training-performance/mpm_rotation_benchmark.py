import sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'trainer'))
import numpy as np
from domain_fitness import rotate_density
n=256
rng=np.random.default_rng(1);field=rng.random((n,n,4)); center=(.5,.5);angles=[2*np.pi*i/16 for i in range(1,16)]
def coordinates(angle):
    c,s=np.cos(angle),np.sin(angle)
    y,x=np.indices((n,n));oy,ox=np.array(center)[::-1]*n-.5
    sy=c*(y-oy)+s*(x-ox)+oy;sx=-s*(y-oy)+c*(x-ox)+ox
    iy=np.floor(sy).astype(int);ix=np.floor(sx).astype(int)
    return iy,ix,sy-iy,sx-ix
coords=[coordinates(a) for a in angles]
def rotate(field,coord):
    iy,ix,fy,fx=coord; out=np.zeros_like(field)
    for dy,dx,w in ((0,0,(1-fy)*(1-fx)),(0,1,(1-fy)*fx),(1,0,fy*(1-fx)),(1,1,fy*fx)):
        y=iy+dy;x=ix+dx;valid=(y>=0)&(y<n)&(x>=0)&(x<n)
        out+=field[np.clip(y,0,n-1),np.clip(x,0,n-1)]*(w*valid)[...,None]
    return out
for a,coord in zip(angles,coords):
    expected=np.stack([rotate_density(field[...,i],a,center) for i in range(4)],-1)
    np.testing.assert_allclose(rotate(field,coord),expected,atol=1e-12)
for label,fn in [('scipy',lambda a,c:np.stack([rotate_density(field[...,i],a,center) for i in range(4)],-1)),('shared-coordinates',lambda a,c:rotate(field,c))]:
    times=[]
    for _ in range(3):
        t=time.perf_counter()
        for a,c in zip(angles,coords):fn(a,c)
        times.append(time.perf_counter()-t)
    print(label,np.median(times))

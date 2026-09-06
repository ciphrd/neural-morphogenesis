"""Behavioral checks for material coverage, stopping, and browser parity."""
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from domain_fitness import (rasterize_triangles, target_mask, evaluate_domains,
                            StableMatchStop, MatchMetrics)
from targets import TargetShape, load_target

ROOT = Path(__file__).resolve().parents[1]

def tiled_target(target):
    half = target.texel_size()/2
    triangles = []
    for x, y in target.points.astype(float):
        a, b, c, d = [[x-half,y-half],[x+half,y-half],[x+half,y+half],[x-half,y+half]]
        triangles.extend([[a,b,c],[a,c,d]])
    return np.array(triangles)

def subdivide(t):
    a,b,c = t[:,0],t[:,1],t[:,2]
    m = (a+b)/2
    return np.concatenate([np.stack([a,m,c],1),np.stack([m,b,c],1)])

def check_geometry():
    rng = np.random.default_rng(71)
    t = rng.uniform(.1,.9,(80,3,2))
    for n in (32,64):
        r = rasterize_triangles(t,n)
        expected = np.abs(np.linalg.det(t[:,1:]-t[:,:1])).sum()/2
        np.testing.assert_allclose(r.sum()/n**2,expected,atol=1e-12)
        np.testing.assert_allclose(r,rasterize_triangles(subdivide(t),n),atol=1e-11)
    for name in ('circle','donut','legs','line2'):
        target=load_target(name); triangles=tiled_target(target); mask=target_mask(target,64)
        perfect=evaluate_domains(triangles,target,mask)
        assert perfect.total < 1e-7,(name,perfect)
        assert perfect.match.error < 2e-6
        refined=evaluate_domains(subdivide(triangles),target,mask)
        np.testing.assert_allclose(perfect.total,refined.total,atol=1e-10)
        wrapped=evaluate_domains((triangles+np.array([.48,.49]))%1,target,mask)
        assert wrapped.total < 1e-7,(name,wrapped.total)
        smaller=evaluate_domains((triangles-.5)*.65+.5,target,mask)
        larger=evaluate_domains((triangles-.5)*1.2+.5,target,mask)
        doubled=evaluate_domains(np.concatenate([triangles,triangles]),target,mask)
        assert smaller.match.missing > .2
        assert larger.match.spill > .1
        assert doubled.match.overlap > .6
        assert min(smaller.total,larger.total,doubled.total)>perfect.total
        np.testing.assert_allclose(mask.sum()/64**2,target.filled_area(),atol=1e-7)
    t=TargetShape.from_export({'nx':4,'ny':4,'pixels':[{'x':1,'y':1},{'x':1,'y':1}]})
    assert t.filled_area()==t.texel_size()**2
    assert np.isinf(evaluate_domains(np.full((1,3,2),np.nan),t,target_mask(t,64)).total)
    # Lost material outside the image must remain a spill penalty.
    target=TargetShape(np.array([[.5,.5]]),(4,4)); tri=tiled_target(target)
    extra=np.concatenate([tri,tri+np.array([.4,.4])])
    assert evaluate_domains(extra,target,target_mask(target,64)).match.error > 1
    print('[PASS] Exact area, subdivision, periodic translation, holes, missing/spill/overlap, invalid geometry')

def check_stop():
    good=MatchMetrics(.01,.01,.01); bad=MatchMetrics(.2,0,0)
    s=StableMatchStop(interval=5,confirmations=2,settle_steps=7)
    assert not s.due(4) and s.due(5)
    assert not s.observe(5,good,.1)
    assert not s.observe(10,bad,.2) and s.streak==0
    assert not s.observe(15,good,.1)
    assert not s.observe(20,good,.1) and not s.growth_enabled
    assert not s.due(26) and s.due(27)
    assert not s.observe(25,bad,.2) and s.growth_enabled
    for step in (30,35): s.observe(step,good,.1)
    assert not s.observe(40,good,.1)
    assert s.observe(42,good,.1) and s.complete and not s.growth_enabled
    for blocked in (True,):
        s=StableMatchStop(confirmations=1)
        assert not s.observe(10,good,.1,blocked) and s.growth_enabled
    assert not StableMatchStop(enabled=False).due(10)
    print('[PASS] Confirmation reset, failed settling resumes growth, full settling duration, capacity failure, disabled stop')

def check_browser():
    fixtures=[]
    for name in ('circle','donut','legs'):
        t=load_target(name)
        tri=tiled_target(t)
        for vertices in (tri,subdivide(tri),(.78*(tri-.5)+.5+np.array([.48,.49]))%1,np.concatenate([tri,tri])):
            fixtures.append({'vertices':vertices.reshape(-1).tolist(),
                'target':{'points':t.points.tolist(),'texelSize':t.texel_size()},
                'expected':evaluate_domains(vertices,t,target_mask(t,64)).match})
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([str(ROOT/'viewer/node_modules/.bin/tsc'),str(ROOT/'viewer/src/gpu/shapeMatch.ts'),
            '--target','ES2020','--module','commonjs','--outDir',tmp,'--skipLibCheck','--resolveJsonModule','--esModuleInterop'],check=True)
        script=Path(tmp)/'check.cjs'
        script.write_text('''const {matchDomains,targetMask,StableMatchStop}=require('./viewer/src/gpu/shapeMatch.js');
const data=JSON.parse(require('fs').readFileSync(0,'utf8'));
const results=data.map(f=>matchDomains(f.vertices,f.target,targetMask(f.target,64)));
const s=new StableMatchStop({stableStop:true,shapeTarget:data[0].target,shapeCheckInterval:5,shapeConfirmations:2,shapeSettleSteps:7});
const g={missing:.01,spill:.01,overlap:.01},b={missing:.2,spill:0,overlap:0};
const states=[5,10,15,20,25,30,35,40,42].map(step=>{s.observe(step,[10,25].includes(step)?b:g);return [s.complete,s.growthEnabled];});
process.stdout.write(JSON.stringify({results,states}));''')
        r=subprocess.run(['node',str(script)],input=json.dumps([{k:v for k,v in f.items() if k!='expected'} for f in fixtures]),text=True,capture_output=True,check=True)
        actual=json.loads(r.stdout)
        for f,a in zip(fixtures,actual['results']):
            np.testing.assert_allclose(list(a.values()),[f['expected'].missing,f['expected'].spill,f['expected'].overlap],atol=1e-7)
        s=StableMatchStop(interval=5,confirmations=2,settle_steps=7);states=[]
        for step in (5,10,15,20,25,30,35,40,42):
            s.observe(step,MatchMetrics(.2,0,0) if step in (10,25) else MatchMetrics(.01,.01,.01),.1)
            states.append([s.complete,s.growth_enabled])
        assert states==actual['states']
    print('[PASS] Python/browser geometry metrics and stopping state parity')

if __name__=='__main__':
    check_geometry();check_stop();check_browser()

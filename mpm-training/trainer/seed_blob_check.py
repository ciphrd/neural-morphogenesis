"""Triangle seed coverage, material weights and trainer/viewer parity checks."""
from __future__ import annotations

import numpy as np
import json
from pathlib import Path
import subprocess
import tempfile

from density import INITIAL_PACKING_SPACING_SCALE
from training_sim import seed_blob
from triangle_vertices import unwrap_vertices

def check_seed_partition():
    from collections import Counter
    for count in (1, 2, 5, 7, 13, 37, 100, 256, 1000):
        for center in ((.5,.5), (.9999,.0001)):
            positions, _, _, _, _, domains, weights, tag = seed_blob(count, center, .01, seed=17)
            assert tag == 'triangle-vertices' and len(positions) == 2*count
            np.testing.assert_allclose(weights.sum(), count, rtol=2e-7)
            vertices = unwrap_vertices(domains)
            edges = vertices[:,1:]-vertices[:,0:1]
            areas = .5*np.linalg.det(edges.transpose(0,2,1))
            assert np.all(areas > 0)
            expected_area = count*(.01*INITIAL_PACKING_SPACING_SCALE)**2*np.sqrt(3)/2
            np.testing.assert_allclose(areas.sum(), expected_area, rtol=1e-5)
            np.testing.assert_allclose(weights/areas, count/areas.sum(), rtol=2e-7)
            # Shared edges must have identical endpoints and opposite winding.
            raw = domains.reshape(-1,3,2)
            directed = Counter((tuple(t[j]),tuple(t[(j+1)%3])) for t in raw for j in range(3))
            assert max(directed.values()) == 1
            boundary = [(a,b) for a,b in directed if (b,a) not in directed]
            degrees = Counter(a for a,b in boundary)
            assert all(n == 1 for n in degrees.values())
            assert set(degrees) == {b for a,b in boundary}
            vertex_count = len({tuple(v) for t in raw for v in t})
            edge_count = (len(directed)+len(boundary))//2
            assert vertex_count-edge_count+2*count == 1
            # All exposed edges form one regular polygon on the intended circle.
            offsets = (np.asarray([a for a,b in boundary])-center+.5)%1-.5
            radii = np.linalg.norm(offsets,axis=1)
            np.testing.assert_allclose(radii,radii.mean(),atol=8e-8)
            angles = np.sort(np.arctan2(offsets[:,1],offsets[:,0]))
            gaps = np.diff(np.r_[angles,angles[0]+2*np.pi])
            np.testing.assert_allclose(gaps,2*np.pi/len(boundary),atol=3e-5)
            local_centers = (positions.astype(float)-center+.5)%1-.5
            np.testing.assert_allclose((local_centers*areas[:,None]).sum(axis=0)/areas.sum(),0,atol=8e-8)
            if count >= 37:
                assert 1-np.cos(np.pi/len(boundary)) < .02
    print('[PASS] circular boundary, conforming mesh, positive areas, uniform material density and seam wrapping')

def check_viewer_parity():
    root = Path(__file__).resolve().parents[1]
    esbuild = root/'viewer/node_modules/.bin/esbuild'
    cases = [dict(count=n, centerX=x, centerY=y, spacing=.01, seed=seed)
             for n,x,y,seed in [(1,.5,.5,0),(2,.5,.5,1),(5,.5,.5,17),(7,.5,.5,17),
                                (37,.5,.5,17),(100,.5,.5,23),(13,.9999,.0001,42),
                                (256,.9999,.0001,42)]]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)/'rng.cjs'
        subprocess.run([str(esbuild), str(root/'viewer/src/gpu/rng.ts'), '--bundle',
                        '--platform=node', '--format=cjs', f'--outfile={bundle}'],
                       check=True, capture_output=True, text=True)
        script = """
const {seedBlob, seedRows} = require(process.argv[1]);
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
function plain(scene) {
  return Object.fromEntries(Object.entries(scene).map(([k,v])=>[k, ArrayBuffer.isView(v)?Array.from(v):v]));
}
console.log(JSON.stringify({blobs:input.map(c=>plain(seedBlob(c))),
  rows:plain(seedRows({rows:2,columns:3,centerX:.5,centerY:.5,spacing:.01}))}));
"""
        result = subprocess.run(['node','-e',script,str(bundle)],input=json.dumps(cases),
                                check=True,capture_output=True,text=True)
    output = json.loads(result.stdout)
    for config, actual in zip(cases, output['blobs']):
        expected = seed_blob(config['count'], (config['centerX'],config['centerY']),
                             config['spacing'],config['seed'])
        assert actual['count']==2*config['count'] and actual['domainGeometry']=='triangle-vertices'
        for name, array in zip(('positions','velocities','F','C','Jp','domain','quadratureWeights'),expected[:7]):
            np.testing.assert_allclose(actual[name], array.reshape(-1), atol=6e-8, rtol=2e-6)
    rows = output['rows']
    assert rows['count']==12 and rows['domainGeometry']=='triangle-vertices'
    np.testing.assert_allclose(rows['quadratureWeights'],.5)
    v = unwrap_vertices(rows['domain'])
    area = .5*np.linalg.det((v[:,1:]-v[:,0:1]).transpose(0,2,1))
    # Explicit float32 world coordinates quantize each endpoint, unlike the
    # former small local edge vectors. Allow that endpoint rounding in area.
    np.testing.assert_allclose(area.sum(),6*.01**2,rtol=5e-6)
    centers = np.array(rows['positions']).reshape(-1,2,2).mean(axis=1)
    np.testing.assert_allclose(centers,[[.49,.495],[.5,.495],[.51,.495],
                                      [.49,.505],[.5,.505],[.51,.505]],atol=6e-8)
    print('[PASS] TypeScript/Python triangle seed parity and row count/area/ordering')

if __name__ == "__main__":
    check_seed_partition()
    check_viewer_parity()

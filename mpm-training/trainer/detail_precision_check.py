"""CPU oracle: separate geometry capacity, target sampling and pose error."""
from dataclasses import asdict
import json
from pathlib import Path
import argparse

import numpy as np

from targets import load_target, TargetShape
from domain_fitness import evaluate_domains
from domain_fitness_check import tiled_target, subdivide
from detail_alignment import evaluate_precise, rotate_vertices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/lizard-detail/oracle'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = load_target('lizard-64')
    ys, xs = np.nonzero(source.occupancy >= .5)
    # A binary oracle removes fractional-alpha ambiguity: every target texel
    # is exactly two triangles, independently of the learned growth process.
    target = TargetShape.from_export(dict(nx=80, ny=80, pixels=[dict(x=int(x),y=int(y)) for x,y in zip(xs,ys)]))
    triangles = tiled_target(target)
    mask = target.mask(256)
    rows=[]
    for degrees in (0, 1.25, 7., 13.7, 33.):
        vertices = rotate_vertices(triangles, np.deg2rad(degrees), target.center)
        legacy = evaluate_domains(vertices, target, mask)
        precise = evaluate_precise(vertices, target, mask)
        row = dict(degrees=degrees, legacy=legacy.total, precise=precise.total,
                   legacy_match=asdict(legacy.match), precise_match=asdict(precise.match))
        rows.append(row)
        print(json.dumps(row), flush=True)
        assert precise.total < 1e-5, row
    perfect = evaluate_precise(triangles, target, mask)
    refined = evaluate_precise(subdivide(triangles), target, mask)
    assert abs(perfect.total-refined.total) < 1e-8
    eroded = evaluate_precise((triangles-target.center)*.97+target.center, target, mask)
    assert eroded.total > .001
    source_masks = {str(n):load_target(f'lizard-{n}').mask(256) for n in (64,128,256)}
    target_sampling = {n:float(np.abs(m-source_masks['256']).sum()/source_masks['256'].sum()) for n,m in source_masks.items()}
    report = dict(oracle_triangles=len(triangles), rotation=rows, target_sampling_l1=target_sampling,
                  three_percent_shrink_score=eroded.total,
                  caveat='Binary mesh oracle at 80% scale avoids periodic seam ambiguity under rotation. It bypasses neural growth and physics; it only tests representation and scoring.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,n in zip(axes[0],('64','128','256')):
        ax.imshow(source_masks[n],origin='lower',cmap='gray',vmin=0,vmax=1)
        ax.set_title(f'Target source: {n} px');ax.axis('off')
    vertices=rotate_vertices(triangles,np.deg2rad(13.7),target.center)
    legacy=evaluate_domains(vertices,target,mask)
    precise=evaluate_precise(vertices,target,mask)
    for ax,data,title in zip(axes[1],(mask,np.abs(mask-legacy.raster),np.abs(mask-precise.raster)),
                             ('Binary mesh oracle','Current scoring error (13.7°)','Geometry alignment error')):
        ax.imshow(data,origin='lower',cmap='magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(args.output/'precision.png',dpi=150);plt.close(fig)
    print('[PASS] Exact rotated lizard, subdivision invariance, genuine shrink penalty')


if __name__ == '__main__':
    main()

"""CPU regression checks and repeatable SVG alignment accuracy/runtime benchmark."""
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from targets import TargetShape
from svg_target import render_svg
from svg_fitness import evaluate_svg
from domain_fitness import evaluate_domains, score_domains
from detail_alignment import evaluate_precise, rotate_vertices
from domain_fitness_check import subdivide
from types import SimpleNamespace

SOURCE = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<path fill="#e04020" d="M25 25H45V55H75V75H25Z"/></svg>'''


def fixture(path):
    path.write_text(SOURCE)
    target = TargetShape.from_svg(path)
    triangles = []
    for y in range(25, 75, 5):
        for x in range(25, 75, 5):
            if x >= 45 and y < 55:
                continue
            a, b, c, d = np.array([[x,y],[x+5,y],[x+5,y+5],[x,y+5]], float)/100
            triangles.extend([[a,b,c],[a,c,d]])
    cx, cy, scale = target.svg_transform
    triangles = (np.asarray(triangles)-[cx, cy])*scale+.5
    triangles[..., 1] = 1-triangles[..., 1]
    return target, triangles


def main(report_path=None):
    with tempfile.TemporaryDirectory() as directory:
        target, triangles = fixture(Path(directory)/'test.svg')
        n = 128
        mask = target.mask(n)
        restored = TargetShape.from_wire(target.wire(n))
        np.testing.assert_array_equal(restored.mask(n), mask)
        np.testing.assert_array_equal(restored.color_raster(n), target.color_raster(n))
        assert restored.svg_source == SOURCE
        assert target.overlay_points(32).shape[0] > 0
        colors = np.tile(np.array([224, 64, 32])/255, (len(triangles), 1))
        rows = []
        for degrees in (0, 1.25, 7, 13.7, 33, 359.3):
            vertices = rotate_vertices(triangles, np.deg2rad(degrees), target.center)
            started = perf_counter()
            precise = evaluate_svg(vertices, target, mask, colors=colors)
            elapsed = perf_counter()-started
            baseline = evaluate_domains(vertices, target, mask, colors=colors)
            rows.append(dict(degrees=degrees, svg=precise.total, raster=baseline.total, seconds=elapsed))
            assert precise.total < .002, rows[-1]
            assert precise.match.error < .025, (degrees, precise.match)
            assert abs((precise.breakdown.angle+np.deg2rad(degrees)+np.pi) % (2*np.pi)-np.pi) < .003
        original = evaluate_svg(triangles, target, mask, colors=colors)
        refined = evaluate_svg(subdivide(triangles), target, mask, colors=np.tile(colors, (2, 1)))
        np.testing.assert_allclose(original.total, refined.total, atol=1e-10)
        shrunk = evaluate_svg((triangles-.5)*.8+.5, target, mask, colors=colors)
        wrong_color = evaluate_svg(triangles, target, mask, colors=1-colors)
        doubled = evaluate_svg(np.concatenate([triangles, triangles]), target, mask)
        assert shrunk.total > original.total+.02
        assert wrong_color.total > original.total+.05
        assert doubled.match.overlap > .8
        assert np.isinf(evaluate_svg(triangles*np.nan, target, mask).total)
        assert len(target._svg_pose_cache) == 16
        args = SimpleNamespace(fitness_coverage_weight=1., fitness_spill_weight=1.,
            fitness_boundary_weight=.1, fitness_crowding_weight=1., outside_weight=1.,
            fitness_color_weight=1., fitness_alignment='raster')
        assert score_domains(triangles, target, mask, args, colors).target_raster is not None
        # Curves, holes, partial alpha and RGB survive rotation and rendering.
        curved = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path fill="blue" opacity=".5" fill-rule="evenodd" d="M90 50A40 40 0 1 0 10 50A40 40 0 1 0 90 50 M60 50A10 10 0 1 0 40 50A10 10 0 1 0 60 50"/></svg>'
        a, rgb = render_svg(curved, 64)
        assert a[32,32] == 0 and .49 < a.max() < .51
        np.testing.assert_array_equal(rgb[...,2], a)
        # Benchmark all three scorers on the same geometry and target raster.
        vertices = rotate_vertices(triangles, np.deg2rad(13.7), target.center)
        timing = {}
        for name, evaluator in [('raster', evaluate_domains), ('geometry', evaluate_precise), ('svg', evaluate_svg)]:
            samples = []
            for _ in range(3):
                start = perf_counter(); evaluator(vertices, target, mask, colors=colors)
                samples.append(perf_counter()-start)
            timing[name] = float(np.median(samples))
        # Subdivision increases simulation mesh size without changing the target.
        dense = vertices
        dense_colors = colors
        for _ in range(5):
            dense = subdivide(dense)
            dense_colors = np.tile(dense_colors, (2, 1))
        dense_timing = {}
        for name, evaluator in [('raster', evaluate_domains), ('geometry', evaluate_precise), ('svg', evaluate_svg)]:
            samples = []
            for _ in range(3):
                start = perf_counter(); evaluator(dense, target, mask, colors=dense_colors)
                samples.append(perf_counter()-start)
            dense_timing[name] = float(np.median(samples))
        report = dict(triangles=len(triangles), resolution=n, rotation=rows, median_seconds=timing,
                      dense_triangles=len(dense), dense_median_seconds=dense_timing)
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        print('[PASS] Continuous SVG pose, colors, geometry defects, subdivision, checkpoint, cache, curves and holes')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    main(parser.parse_args().report)

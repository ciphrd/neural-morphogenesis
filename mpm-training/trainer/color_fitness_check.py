"""Color integration, alpha mapping, subdivision, and aligned scoring checks."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from targets import TargetShape
from domain_fitness import evaluate_domains, rasterize_triangles
from domain_fitness_check import subdivide


def check_multiple_batches():
    rng = np.random.default_rng(7)
    small = np.array([[0., 0.], [.02, 0.], [0., .02]])
    triangles = small + rng.uniform(.05, .9, (300, 1, 2))
    colors = rng.random((300, 3))
    # Exercise both the 256-triangle limit and the pixel-work limit.
    large = np.tile(np.array([[[.05,.05],[.95,.05],[.05,.95]]]), (3,1,1))
    # A fully cropped first batch must still advance the color index.
    cropped = np.concatenate([triangles[:256]+2, triangles[256:]])
    for geometry, rgb_values, resolution in (
        (triangles, colors, 32), (large, colors[:3], 256), (cropped, colors, 32)
    ):
        expected_mask = np.zeros((resolution, resolution))
        expected_rgb = np.zeros((resolution, resolution, 3))
        for triangle, color in zip(geometry, rgb_values):
            single = rasterize_triangles(triangle[None], resolution)
            expected_mask += single
            expected_rgb += single[..., None]*color
        actual_mask, actual_rgb = rasterize_triangles(geometry, resolution, rgb_values)
        np.testing.assert_allclose(actual_mask, expected_mask, atol=1e-12)
        np.testing.assert_allclose(actual_rgb, expected_rgb, atol=1e-12)
    print('[PASS] RGB correspondence across triangle-limit, pixel-limit and fully cropped batches')


def main():
    check_multiple_batches()
    triangles = np.array([[[.25,.25],[.75,.25],[.75,.75]],
                          [[.25,.25],[.75,.75],[.25,.75]]])
    colors = np.array([[1.,0.,0.], [0.,0.,1.]])
    mask, rgb = rasterize_triangles(triangles, 32, colors)
    refined_mask, refined_rgb = rasterize_triangles(subdivide(triangles), 32, np.concatenate([colors, colors]))
    np.testing.assert_allclose(mask, refined_mask, atol=1e-12)
    np.testing.assert_allclose(rgb, refined_rgb, atol=1e-12)
    target = TargetShape.from_wire(dict(mask=mask.ravel().tolist(), rgb=rgb.ravel().tolist(), resolution=32))
    exact = evaluate_domains(triangles, target, mask, colors=colors)
    wrong = evaluate_domains(triangles, target, mask, colors=np.ones((2,3))*.5)
    assert exact.total < 1e-12 and exact.breakdown.color < 1e-12
    assert wrong.total > .05 and wrong.breakdown.color > .05
    ignored = evaluate_domains(triangles, target, mask, colors=np.ones((2,3))*.5, color_weight=0)
    assert ignored.total < 1e-12
    overlap_mask, overlap_rgb = rasterize_triangles(np.concatenate([triangles,triangles]),32,np.concatenate([colors,colors]))
    np.testing.assert_allclose(overlap_rgb/np.maximum(overlap_mask[...,None],1),rgb)
    rotated = (triangles-.5) @ np.array([[0.,-1.],[1.,0.]]) + .5
    aligned = evaluate_domains(rotated, target, mask, colors=colors)
    assert aligned.total < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        rgba = np.zeros((8,8,4), np.uint8)
        rgba[2:6,2:6] = [255,0,0,128]
        rgba[0,0,:3] = [0,255,0]  # Invisible RGB must not leak into training.
        path = Path(tmp)/'target.png'
        Image.fromarray(rgba).save(path)
        source = TargetShape.from_png(path)
        rgb = source.color_raster(32)
        np.testing.assert_allclose(rgb[...,0],source.mask(32))
        assert not rgb[...,1:].any()
        restored = TargetShape.from_wire(source.wire(32))
        np.testing.assert_allclose(restored.color_raster(32),rgb)
    print('[PASS] Exact RGB areas, subdivision invariance, color loss, rotation, soft alpha and checkpoint round trip')

if __name__ == '__main__': main()

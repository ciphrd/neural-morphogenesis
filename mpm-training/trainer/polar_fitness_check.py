"""Check FFT loss against exhaustive pixel differences, geometry and server output."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pickle
import tempfile
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from polar_fitness import match_polar, sharpen, evaluate_polar, opaque_image, shape_priority_terms
from domain_fitness import rasterize_triangles, score_domains
from targets import TargetShape


def check_opaque_alpha():
    # Alpha depends exclusively on geometry, including dark and translucent
    # source material. Hidden RGB in void must not enter the objective.
    coverage = np.array([[0., .01, .5, 1., 2.]])
    straight = np.array([.2, .5, .8])
    premultiplied = coverage[..., None] * straight
    premultiplied[0, 0] = [1., 0., 1.]
    image = opaque_image(coverage, premultiplied)
    np.testing.assert_array_equal(image[..., 3], [[0., 1., 1., 1., 1.]])
    np.testing.assert_allclose(image[0, 1:, :3], np.tile(straight, (4, 1)))
    np.testing.assert_array_equal(image[0, 0], [0., 0., 0., 0.])
    black = opaque_image(coverage, np.zeros_like(premultiplied))
    np.testing.assert_array_equal(black[..., 3], image[..., 3])
    assert not black[..., :3].any()

    n = 48
    vertices = np.array([[[.3,.3],[.7,.3],[.7,.7]],
                         [[.3,.3],[.7,.7],[.3,.7]]])
    mask = rasterize_triangles(vertices, n)
    colors = np.zeros((2, 3))
    target = TargetShape.from_wire(dict(mask=mask.ravel(), rgb=np.zeros((n,n,3)).ravel(),
                                       resolution=n, center=[.5,.5]))
    exact = evaluate_polar(vertices, target, mask, colors)
    assert exact.total < 1e-9, exact.total
    assert set(np.unique(exact.raster)) == {0., 1.}
    # Removing black triangles must hurt even though every RGB pixel stays zero.
    missing = evaluate_polar(vertices[:1], target, mask, colors[:1])
    assert missing.total > .01, missing.total
    # Adding black material outside the target must hurt too.
    excess = evaluate_polar((vertices-.5)*1.15+.5, target, mask, colors)
    assert excess.total > .01, excess.total
    assert missing.breakdown.color == 0. and excess.breakdown.color == 0.
    # All occupied target pixels, including partial alpha edges/interiors, are
    # made opaque with the same straight RGB as opaque candidate triangles.
    translucent_mask = mask * .2
    target = TargetShape.from_wire(dict(mask=translucent_mask.ravel(),
        rgb=(translucent_mask[...,None]*straight).ravel(), resolution=n, center=[.5,.5]))
    match = evaluate_polar(vertices, target, translucent_mask, np.tile(straight, (2,1)))
    assert match.total < 1e-9, match.total
    # Subdivision cannot turn fractional shared edges into void or alter RGB.
    from domain_fitness_check import subdivide
    refined = evaluate_polar(subdivide(vertices), target, translucent_mask, np.tile(straight, (4,1)))
    assert refined.total < 1e-9, refined.total
    # Exercise the actual PNG loader with translucent black foreground and
    # invisible colored background. Occupancy must not depend on either RGB.
    with tempfile.TemporaryDirectory() as tmp:
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[2:6, 2:6] = [0, 0, 0, 64]
        rgba[0, 0, :3] = [255, 0, 255]
        path = Path(tmp) / 'black.png'
        Image.fromarray(rgba).save(path)
        png = TargetShape.from_png(path)
        opaque = opaque_image(png.mask(n), png.color_raster(n))
        assert not opaque[..., :3].any()
        assert set(np.unique(opaque[..., 3])) == {0., 1.}
        assert opaque[..., 3].sum() > 0
    print('[PASS] Binary alpha, black material versus void, black spill/missing penalties, opaque target parity, PNG and subdivision')


def check_shape_priority():
    # Shape has priority even when the better shape has very wrong colors.
    shape, color = shape_priority_terms(np.array([[100,100,100,0], [0,0,0,.2]]), 1.)
    assert (shape+color)[0] < (shape+color)[1]
    # No gate exploitation: worsening geometry never lowers the score for
    # fixed color error, including extremely large errors and near-zero shape.
    for rgb_error in (0., .01, 1., 100., 1e8):
        errors = np.zeros((1000,4)); errors[:,:3] = rgb_error
        errors[:,-1] = np.r_[0., np.geomspace(1e-8, 100., 999)]
        shape, color = shape_priority_terms(errors, 1.)
        assert np.all(np.diff(shape+color) > 0)
        assert np.all(color <= .1)
        if rgb_error:
            assert np.all(color > 0) and color[0] > color[-1]
    assert shape_priority_terms(np.array([1.,1.,1.,.5]), 1., 0.)[1] == 0.
    half = shape_priority_terms(np.array([1.,1.,1.,.5]), 1., .5)[1]
    full = shape_priority_terms(np.array([1.,1.,1.,.5]), 1., 1.)[1]
    np.testing.assert_allclose(half, full/2)
    print('[PASS] Shape-first ranking, continuous nonzero color influence, monotonic shape penalty and color multiplier')


def main():
    check_opaque_alpha()
    check_shape_priority()
    rng = np.random.default_rng(19)
    for n in (31, 32, 101):
        target = rng.normal(size=(7, n, 4))
        candidate = rng.normal(size=target.shape)
        curves, k, mirrored, aligned = match_polar(candidate, target)
        brute = np.array([[np.square(candidate-np.roll(t, i, axis=1)).sum(axis=1).mean()
                           for i in range(n)] for t in (target, target[:, (-np.arange(n)) % n])])
        np.testing.assert_allclose(curves, brute, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(curves[int(mirrored), k], np.square(candidate-aligned).sum(axis=1).mean())
        # Independent brute-force weighted objective across every pose.
        energy = np.square(target[..., -1]).sum(axis=1).mean()
        def objective(reference):
            e = np.square(candidate-reference).sum(axis=1).mean(axis=0)/energy
            shape, color = e[-1], e[:-1].mean()
            return shape + .1*(.1+.9/(1+shape/.25))*color/(1+color)
        weighted = np.array([[objective(np.roll(t,i,axis=1)) for i in range(n)]
                             for t in (target,target[:,(-np.arange(n))%n])])
        curves, _, _, _ = match_polar(candidate,target,shape_priority=True)
        np.testing.assert_allclose(curves, weighted, atol=1e-12)
        for reflect in (False, True):
            t = target[:, (-np.arange(n)) % n] if reflect else target
            curves, k, mirrored, aligned = match_polar(np.roll(t, 7, axis=1), target)
            assert (k, mirrored) == (7, reflect)
            assert curves[int(mirrored), k] < 1e-10
            np.testing.assert_array_equal(aligned, np.roll(t, 7, axis=1))
    # Independent separable 5x5 Gaussian, including reflected border pixels.
    image = rng.uniform(size=(12, 11, 4))
    g = np.exp(-np.arange(-2, 3)**2/2); g /= g.sum()
    blurred = np.stack([convolve(image[..., c], g[:, None]*g[None, :], mode='mirror') for c in range(4)], axis=-1)
    np.testing.assert_allclose(sharpen(image), 3*image-2*blurred, atol=1e-14)

    n=64
    # Asymmetric multicolored L, with internal subdivision and exact area raster.
    vertices=np.array([[[.35,.3],[.43,.3],[.43,.7]],[[.35,.3],[.43,.7],[.35,.7]],
                       [[.43,.3],[.65,.3],[.65,.38]],[[.43,.3],[.65,.38],[.43,.38]]])
    colors=np.array([[.1,.7,.2],[.1,.7,.2],[.8,.2,.1],[.8,.2,.1]])
    mask,rgb=rasterize_triangles(vertices,n,colors)
    areas=np.abs(np.linalg.det(vertices[:,1:]-vertices[:,:1]))/2
    center=np.average(vertices.mean(axis=1),axis=0,weights=areas)
    target=TargetShape.from_wire(dict(mask=mask.ravel(),rgb=rgb.ravel(),resolution=n,center=center))
    original=evaluate_polar(vertices,target,mask,colors)
    assert original.total < 1e-9, original.total
    assert not original.polar.reflected
    scores=[]
    for reflected in (False, True):
        for angle in (0., .43, np.pi/2, 2.37):
            c,s=np.cos(angle),np.sin(angle)
            v=vertices-target.center
            if reflected: v=v*np.array([1,-1])
            v=v@np.array([[c,s],[-s,c]])+target.center
            evaluation=evaluate_polar(v,target,mask,colors)
            scores.append(evaluation.total)
            assert evaluation.polar.reflected == reflected, (angle,reflected,evaluation.polar)
            delta=np.angle(np.exp(1j*(evaluation.polar.angle-angle)))
            # Binary edges can move the best pose by one angular bin.
            assert abs(delta)<=2*np.pi/evaluation.polar.target.shape[1]+1e-12, (angle,delta)
            assert evaluation.match.missing < .12, evaluation.match
            wrong_color = evaluate_polar(v,target,mask,1-colors)
            assert evaluation.total < wrong_color.total
            np.testing.assert_allclose(evaluation.total,
                evaluation.breakdown.coverage + evaluation.breakdown.color, atol=1e-12)
    bad=evaluate_polar(vertices,target,mask,1-colors)
    assert bad.total > original.total
    assert bad.breakdown.color > 0
    translated=evaluate_polar(vertices+np.array([.11,-.08]),target,mask,colors)
    assert translated.total<1e-9,translated.total
    assert score_domains(vertices,target,mask,SimpleNamespace(fitness_function='polar'),colors).total<1e-9
    shape_only=score_domains(vertices,target,mask,SimpleNamespace(fitness_function='polar',fitness_color_weight=0),colors)
    assert shape_only.polar.target.shape[-1] == 1
    assert shape_only.total < 1e-9
    # SVG must also use the new pixel objective rather than its legacy scorer.
    from targets import load_target
    svg=load_target('vector-l')
    svg_result=score_domains(vertices,svg,svg.mask(n),SimpleNamespace(fitness_function='polar'),colors)
    assert svg_result.polar is not None and np.isfinite(svg_result.total)
    wrong=vertices.copy();wrong[-2:, :, 0]+=.04
    assert evaluate_polar(wrong,target,mask,colors).total > original.total
    assert np.isinf(evaluate_polar(np.empty((0,3,2)),target,mask).total)

    from rollout_snapshot import RolloutSnapshot
    import train_server
    snapshot=pickle.loads(pickle.dumps(RolloutSnapshot(original.total,original,np.array([[.5,.5]]),{})))
    with tempfile.TemporaryDirectory() as tmp, patch.object(train_server,'IMAGES_DIR',Path(tmp)), \
         patch.object(train_server,'target',target), patch.object(train_server,'target_raster',mask), \
         patch.object(train_server,'args',SimpleNamespace(raster_resolution=n)):
        metadata=train_server._save_generation_images(1,snapshot)
        p=metadata['polar']
        assert p['shift']==0 and not p['reflected']
        saved=np.load(Path(tmp)/'gen_00001_polar.npz')
        np.testing.assert_array_equal(saved['candidate'],original.polar.candidate)
        objective = json.loads(str(saved['objective_json']))
        assert objective == p['objective']
        np.testing.assert_allclose(objective['shapeLoss']+objective['colorContribution'],original.total,atol=1e-10)
        expected=np.rint(np.clip((original.polar.target[...,:3]+2)/5,0,1)*255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(Image.open(Path(tmp)/'gen_00001_polar_target.png')),expected)
        # Reconstruct a nonzero score from exported arrays and saved settings,
        # independently of the implementation helper and current config.
        weighted_snapshot = RolloutSnapshot(bad.total,bad,np.array([[.5,.5]]),{})
        train_server._save_generation_images(3,weighted_snapshot)
        weighted = np.load(Path(tmp)/'gen_00003_polar.npz')
        settings = json.loads(str(weighted['objective_json']))
        errors = weighted['squared_difference'].sum(axis=1).mean(axis=0)/settings['shapeEnergy']
        shape_error, color_error = errors[-1], errors[:-1].mean()
        gate = settings['colorFloor']+(1-settings['colorFloor'])/(1+shape_error/settings['shapeScale'])
        expected_score = shape_error + settings['maxColorContribution']*settings['colorWeight']*gate*color_error/(1+color_error)
        np.testing.assert_allclose(expected_score,bad.total,atol=1e-12)
        snapshot.evaluation=shape_only
        train_server._save_generation_images(2,snapshot)
        actual=np.asarray(Image.open(Path(tmp)/'gen_00002_target.png'))
        expected=(np.clip(shape_only.target_raster[::-1],0,1)*255).astype(np.uint8)
        np.testing.assert_array_equal(actual,expected)
        with patch.object(train_server,'_images_dir_for_run',return_value=Path(tmp)):
            assert Path(train_server.run_image('current','gen_00001_polar_target.png').path).is_file()
            assert Path(train_server.run_image('current','gen_00001_polar.npz').path).is_file()
    print('[PASS] exhaustive FFT/reflection equivalence, sharpening, rotated/color/translated geometry, snapshot export and routes')
    print('Rotated same-shape losses:',np.round(scores,6),'wrong-color loss:',round(bad.total,6))


if __name__=='__main__': main()

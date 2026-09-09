"""Check FFT loss against exhaustive pixel differences, geometry and server output."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pickle
import tempfile
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from polar_fitness import match_polar, sharpen, evaluate_polar
from domain_fitness import rasterize_triangles, score_domains
from targets import TargetShape


def main():
    rng = np.random.default_rng(19)
    for n in (31, 32, 101):
        target = rng.normal(size=(7, n, 4))
        candidate = rng.normal(size=target.shape)
        curves, k, mirrored, aligned = match_polar(candidate, target)
        brute = np.array([[np.square(candidate-np.roll(t, i, axis=1)).sum(axis=1).mean()
                           for i in range(n)] for t in (target, target[:, (-np.arange(n)) % n])])
        np.testing.assert_allclose(curves, brute, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(curves[int(mirrored), k], np.square(candidate-aligned).sum(axis=1).mean())
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
            assert abs(delta)<2*np.pi/evaluation.polar.target.shape[1], (angle,delta)
            assert evaluation.match.missing < .12, evaluation.match
    bad=evaluate_polar(vertices,target,mask,1-colors)
    assert max(scores)<bad.total*.1,(scores,bad.total)
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
        np.testing.assert_allclose(saved['squared_difference'].sum(axis=1).mean(),original.total,atol=1e-10)
        expected=np.rint(np.clip((original.polar.target[...,:3]+2)/5,0,1)*255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(Image.open(Path(tmp)/'gen_00001_polar_target.png')),expected)
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

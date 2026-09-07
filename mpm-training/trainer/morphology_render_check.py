"""Native density parity and warmed morphology benchmark; no training mutations."""
import json
from time import perf_counter
import numpy as np
from device import pick_device
from mpm_core import MpmCore, REPULSION_FIELD_N, CONSTANTS


def main():
    device = pick_device()
    core = MpmCore(device)
    assert core.density_render_enabled, 'This comparison requires float32-blendable'
    rng = np.random.default_rng(19)
    n = REPULSION_FIELD_N

    def upload(count, clustered=False):
        pos = rng.uniform(-1, 2, (count, 2)).astype(np.float32)
        if clustered: pos[:] = rng.normal(.5, .008, (count, 2))
        if count >= 8:
            pos[:8] = [[0,0],[1,1],[0,.5],[.5,0],[.99999,.99999],[-.001,1.001],[.25,.75],[.75,.25]]
        rest = np.zeros((count,16), np.float32)
        rest[:,0] = rng.uniform(.5,2,count); rest[:,3] = rng.uniform(.5,2,count)
        rest[:,15] = rng.uniform(.1,3,count)
        device.queue.write_buffer(core.positions,0,pos)
        device.queue.write_buffer(core.rest,0,rest)
        core.set_active_count(count)

    def run(render):
        core.density_render_enabled = render
        encoder = device.create_command_encoder()
        core.encode_morphology(encoder)
        device.queue.submit([encoder.finish()])
        return core.read_morphology()

    worst = 0.0
    for count, clustered in [(8,False),(400,False),(400,True),(4000,True)]:
        upload(count,clustered)
        for splat in [0,.016,.1]:
            core.set_splat_radius(splat)
            for blur in [0,.01,.1]:
                core.set_morphology(blur,1.0)
                compute = run(False)
                raw = device.queue.read_texture({'texture':core.density_texture},
                    {'bytes_per_row':n*4,'rows_per_image':n},(n,n,1))
                density = np.frombuffer(raw,np.float32).reshape(n,n).copy()
                sigma = blur*n
                radius = min(int(np.ceil(3*sigma)),CONSTANTS["MORPHOLOGY_MAX_RADIUS"])
                weights = np.exp(-.5*np.arange(-radius,radius+1)**2/max(sigma*sigma,1e-8))
                weights /= weights.sum()
                reference = density.astype(np.float64)
                for axis in [1,0]:
                    reference = sum(w*np.roll(reference,offset,axis) for w,offset in zip(weights,range(-radius,radius+1)))
                reference /= reference+1
                np.testing.assert_allclose(compute,reference,atol=3e-7,rtol=3e-6)
                render = run(True)
                error = float(np.max(np.abs(compute-render))); worst=max(worst,error)
                np.testing.assert_allclose(render,compute,atol=5e-5,rtol=5e-5)
    print('[PASS] Boundary wrapping, particle weights, clustered overlap, splat/blur radii; max occupancy error',worst,flush=True)
    core.set_splat_radius(.016); core.set_morphology(.01,1)
    results=[]
    for count in [40,400,4000]:
        upload(count,True)
        for render in [False,True]:
            for _ in range(3):run(render)
        times={False:[],True:[]}
        for repetition in range(6):
            for render in ([False,True] if repetition%2==0 else [True,False]):
                core.density_render_enabled=render
                start=perf_counter()
                encoder=device.create_command_encoder()
                for _ in range(20):core.encode_morphology(encoder)
                device.queue.submit([encoder.finish()])
                core.read_morphology()
                times[render].append((perf_counter()-start)/20)
        compute=float(np.median(times[False])); render=float(np.median(times[True]))
        results.append({'particles':count,'compute_ms':compute*1000,'quads_ms':render*1000,'speedup':compute/render})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()

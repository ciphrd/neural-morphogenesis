"""Preset invariants, browser parity, and GPU reset/first-tick integration."""
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from initial_conditions import InitialCondition, PRESETS, validate_initial_condition
from training_sim import seed_blob, TrainingRollout
from density import INITIAL_PACKING_SPACING_SCALE

def radius(count=37):
    return np.sqrt(count*(.01*INITIAL_PACKING_SPACING_SCALE)**2*np.sqrt(3)/(2*np.pi))

def check_cpu():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)/'initial.cjs'
        subprocess.run([str(root/'viewer/node_modules/.bin/esbuild'),
                        str(root/'viewer/src/gpu/initialConditions.ts'), '--bundle',
                        '--platform=node', '--format=cjs', f'--outfile={bundle}'],
                       check=True, capture_output=True)
        for center in ((.5,.5), (.9999,.0001)):
            for preset in PRESETS:
                for strength in (0., .3, 1.):
                    scene = seed_blob(37, center, .01, 17)
                    original = [v.copy() if isinstance(v,np.ndarray) else v for v in scene]
                    initial = InitialCondition(preset,strength,3,17,center,radius())
                    source = dict(count=len(scene[0]), positions=scene[0].ravel().tolist(), F=scene[2].ravel().tolist(), domain=scene[5].ravel().tolist())
                    initial.deform(scene)
                    chemistry, private = initial.states(scene[0],9)
                    assert np.isfinite(chemistry).all() and np.isfinite(scene[2]).all()
                    vertices = scene[5].reshape(-1,3,2).astype(float)
                    d = (vertices[:,1:]-vertices[:,0:1]+.5)%1-.5
                    areas = np.linalg.det(d)/2
                    assert np.all(areas > 0)
                    np.testing.assert_allclose(areas.sum(), np.pi*radius()**2, rtol=3e-5)
                    np.testing.assert_allclose(np.linalg.det(scene[2].reshape(-1,2,2)), 1, atol=2e-7)
                    if preset == 'none' or strength == 0:
                        for a,b in zip(scene[:7],original[:7]): np.testing.assert_array_equal(a,b)
                        assert not chemistry.any() and not private.any()
                    script = '''const {InitialCondition}=require(process.argv[1]);
const data=JSON.parse(require('fs').readFileSync(0,'utf8'));
const scene=data.scene; for(const k of ['positions','F','domain'])scene[k]=Float32Array.from(scene[k]);
const p=new InitialCondition(data.preset,data.strength,3,17,data.center,data.radius);
p.deform(scene); const state=p.states(scene.positions,9);
console.log(JSON.stringify([scene.positions,scene.F,scene.domain,state.chemistry,state.privateState].map(x=>Array.from(x))));'''
                    result = subprocess.run(['node','-e',script,str(bundle)],input=json.dumps(dict(scene=source,preset=preset,strength=strength,center=center,radius=radius())),text=True,capture_output=True,check=True)
                    for actual,expected in zip(json.loads(result.stdout),(scene[0],scene[2],scene[5],chemistry,private)):
                        np.testing.assert_allclose(actual,expected.ravel(),atol=1e-7,rtol=1e-6)
    for args in [('invalid',.3,0,9,True), ('chemical-pole',float('nan'),0,9,True), ('internal-state',.3,0,9,False), ('handed-chemistry',.3,0,1,True)]:
        try: validate_initial_condition(*args)
        except ValueError: pass
        else: raise AssertionError(args)
    print('[PASS] All presets: material area, positive deformation, seam wrapping, zero-strength baseline, validation, Python/browser parity')

def check_gpu():
    from device import pick_device
    from mpm_core import MpmCore
    from environment_gpu import EnvironmentGPU
    from agents_gpu import AgentsGPU, PARTICLE_META_BUFFER_OFFSET
    device = pick_device()
    core = MpmCore(device)
    for architecture in ('cell-owned-projection','persistent-environment'):
        env = EnvironmentGPU(device, 9, 64, 64, 1.0, 1.0, chemical_communication_architecture=architecture)
        agents = AgentsGPU(device, core, env, 9, 128, 1.0, 128, 0.01, 1.0, 1.0, 0.5, 0.5, policy_architecture='stateful-128', chemical_communication_architecture=architecture)
        for preset in PRESETS:
            rollout = TrainingRollout(core,agents,env,(.5,.5),.01,0.,17,
                                      initial_particle_count=37, initial_condition=preset,
                                      initial_condition_channel=3)
            def read_meta():
                return np.frombuffer(device.queue.read_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET,74*agents._particle_meta_dtype.itemsize),dtype=agents._particle_meta_dtype).copy()
            meta = read_meta()
            chemical = preset.startswith('chemical-') or preset == 'handed-chemistry'
            assert bool(meta['chemicalState'].any()) == chemical
            assert bool(meta['privateState'].any()) == (preset == 'internal-state')
            if chemical and architecture == 'persistent-environment':
                assert np.frombuffer(device.queue.read_buffer(env.buffers[0]),np.float32).any()
            if chemical and architecture == 'cell-owned-projection':
                encoder = device.create_command_encoder()
                env.encode_clear(encoder); agents.encode_splat_chemical_state(encoder); env.encode_sense(encoder)
                device.queue.submit([encoder.finish()])
                assert np.frombuffer(device.queue.read_buffer(env.buffers[env.parity]),np.float32).any()
            rollout.macro_step(1)
            assert np.isfinite(core.read_positions()).all()
            TrainingRollout(core,agents,env,(.5,.5),.01,0.,17,initial_particle_count=37)
            reset = read_meta()
            assert not reset['chemicalState'].any() and not reset['privateState'].any()
            assert not np.frombuffer(device.queue.read_buffer(env.buffers[0]),np.float32).any()
    print('[PASS] GPU: both chemistry architectures, initial state upload, first tick, reset clears all cues')

if __name__ == '__main__':
    check_cpu()
    check_gpu()

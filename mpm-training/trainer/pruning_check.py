"""Compare parallel pruning against the original serial GPU kernel, byte for byte."""
import argparse
import json
from pathlib import Path

import numpy as np
import wgpu

from agents_gpu import AgentsGPU, PARTICLE_META_BUFFER_OFFSET
from device import pick_device
from environment_gpu import EnvironmentGPU
from gpu_timing import GpuTimings
from mpm_core import MpmCore, GRID_N, INV_DX, REPULSION_FIELD_N
from shader_template import load_core_shader


REFERENCE = '''
@compute @workgroup_size(1)
fn pruneReference() {
  var count = atomicLoad(&agentState.sampleCount);
  var pi = 0u;
  loop {
    if (pi >= count) { break; }
    if (!disconnectedFromGrid(pi)) { pi++; continue; }
    count--;
    if (pi != count) {
      positions[pi] = positions[count];
      velocities[pi] = velocities[count];
      particleC[pi] = particleC[count];
      particleF[pi] = particleF[count];
      particleRest[pi] = particleRest[count];
      agentState.particleMeta[pi] = agentState.particleMeta[count];
    }
  }
  atomicStore(&agentState.sampleCount, count);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path)
    args = parser.parse_args()
    device = pick_device()
    capacity = 8192
    core = MpmCore(device)
    env = EnvironmentGPU(device, 9, 64, 64, .91, 4.)
    agents = AgentsGPU(device, core, env, 9, 128, 1., capacity, .0027, 1., 1., .5, .5)
    shader = load_core_shader('growthField.wgsl', {
        'CHANNELS': 9, 'GRID_N': GRID_N, 'INV_DX': INV_DX,
        'MORPHOLOGY_FIELD_N': REPULSION_FIELD_N, 'REFINE_CAPACITY': capacity,
        'REFINE_HASH_SIZE': 1 << (6 * capacity - 1).bit_length(),
    })
    pipeline = device.create_compute_pipeline(layout='auto', compute={
        'module': device.create_shader_module(code=shader + REFERENCE), 'entry_point': 'pruneReference'})
    buffers = {0: core.positions, 2: core.rest, 3: agents._agent_state_buffer,
               4: core.C, 5: core.velocities, 6: core.F}
    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
        {'binding': key, 'resource': {'buffer': buf}} for key, buf in buffers.items()])
    rng = np.random.default_rng(419)

    def fixture(count, invalid, host_count=None):
        agents.set_active_count(count if host_count is None else host_count)
        rest = rng.uniform(.1, .9, (count, 16)).astype(np.float32)
        # Includes connected triangles crossing the periodic domain boundary.
        origin = rng.random((count, 1, 2), dtype=np.float32)
        vertices = origin + np.array([[0, 0], [.006, 0], [0, .006]], np.float32)
        vertices[invalid] = origin[invalid] + np.array([[0, 0], [.15, 0], [0, .15]], np.float32)
        rest[:, 8:14] = (vertices % 1).reshape(count, 6)
        data = {0: rng.random((count, 2), dtype=np.float32).tobytes(), 2: rest.tobytes(),
                4: rng.random((count, 4), dtype=np.float32).tobytes(),
                5: rng.random((count, 2), dtype=np.float32).tobytes(),
                6: rng.random((count, 4), dtype=np.float32).tobytes()}
        meta = rng.integers(0, 256, PARTICLE_META_BUFFER_OFFSET + count * agents._particle_meta_dtype.itemsize,
                            dtype=np.uint8)
        meta[:4] = np.frombuffer(np.uint32(count).tobytes(), np.uint8)
        data[3] = meta.tobytes()
        return data

    def restore(data):
        for key, value in data.items():
            if value:
                device.queue.write_buffer(buffers[key], 0, value)

    def encode(encoder, original, timer=None):
        if original:
            p = timer.begin_compute_pass(encoder, 'reference') if timer else encoder.begin_compute_pass()
            p.set_pipeline(pipeline)
            p.set_bind_group(0, group)
            p.dispatch_workgroups(1)
            p.end()
        else:
            for name in ('clearRefinement', 'classifyPruning', 'pruneMaterial'):
                index = agents._growth_entries.index(name)
                p = timer.begin_compute_pass(encoder, name) if timer else encoder.begin_compute_pass()
                p.set_pipeline(agents._growth_pipelines[index])
                p.set_bind_group(0, agents._growth_bind_groups[index])
                dispatch = agents._growth_dispatches[index]
                p.dispatch_workgroups(agents._dispatch if dispatch is None else dispatch)
                p.end()

    def snapshot(data):
        return {key: bytes(device.queue.read_buffer(buffers[key], 0, len(value))) if value else b''
                for key, value in data.items()}

    cases = [0, 1, 63, 64, 65, 129, 257, 4000]
    for count in cases:
        for pattern in ('none', 'all', 'alternating', 'random', 'tail', 'head'):
            invalid = {'none': np.zeros(count, bool), 'all': np.ones(count, bool),
                       'alternating': np.arange(count) % 2 == 0, 'random': rng.random(count) < .15,
                       'tail': np.arange(count) >= count // 2,
                       'head': np.arange(count) < count // 2}[pattern]
            # Simulate the post-split GPU count being larger than the host dispatch.
            data = fixture(count, invalid, max(1, (count + 1) // 2))
            results = []
            for original in (True, False):
                restore(data)
                encoder = device.create_command_encoder()
                encode(encoder, original)
                device.queue.submit([encoder.finish()])
                results.append(snapshot(data))
            assert results[0] == results[1], (count, pattern)
            actual_count = int.from_bytes(results[1][3][:4], 'little')
            assert actual_count == count - int(invalid.sum()), (count, pattern, actual_count)
    print('[PASS] 48 cases: exact full-buffer state/order, no/all/mixed deletion, periodic seams, stale scratch reset, appended daughters')

    if args.benchmark:
        timer = GpuTimings(device, interval=1)
        assert timer.supported, 'GPU timestamps required'
        report = {'adapter': dict(device.adapter.info), 'samples': 15,
                  'notes': 'Median GPU microseconds; parallel total includes the full existing refinement clear (conservative).',
                  'cases': []}
        for count in (400, 4000, 8000):
            for fraction in (0., .01, .5, 1.):
                invalid = rng.random(count) < fraction
                data = fixture(count, invalid)
                values = {True: [], False: []}
                for sample in range(18):
                    for original in rng.permutation([True, False]):
                        restore(data)
                        timer.reset()
                        timer.begin_sample()
                        encoder = device.create_command_encoder()
                        encode(encoder, original, timer)
                        device.queue.submit([encoder.finish()])
                        timer.finish_sample()
                        if sample >= 3:
                            values[bool(original)].append(timer.report()['seconds'] * 1e6)
                row = {'particles': count, 'invalid': int(invalid.sum()),
                       'reference_us': float(np.median(values[True])),
                       'parallel_us': float(np.median(values[False]))}
                report['cases'].append(row)
                print(row, flush=True)
        args.benchmark.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Standalone deposit investigation; production shaders/settings are not modified.

Run from the repository root with trainer/.venv/bin/python
trainer/chemical_deposit_bench.py --output artifacts/chemical-deposit/report.json.
Synthetic frozen snapshots isolate spatial contention. An optional saved triangle
snapshot contributes real particle centroids, not a restored live rollout.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import wgpu

import agents_gpu
from agents_gpu import AgentsGPU, PARTICLE_META_BUFFER_OFFSET
from chemical_channels import default_channel_profiles
from device import pick_device
from environment_gpu import EnvironmentGPU
from gpu_timing import GpuTimings
from mpm_core import MpmCore


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--batch', type=int, default=8,
                        help='Repeated clears/kernels per submission to amortize host overhead')
    parser.add_argument('--counts', type=int, nargs='+', default=[400, 4000, 8000])
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    if args.samples < 5 or min(args.counts) <= 0 or not 1 <= args.batch <= 128:
        parser.error('use at least 5 samples and positive particle counts')
    rng = np.random.default_rng(731)
    scenes = []
    for count in args.counts:
        base = rng.random((count, 2), dtype=np.float32) - .5
        for name, width in [('uniform', 1.), ('cluster', .15), ('tight', .015)]:
            scenes.append((f'{name}-{count}', .5 + base * width))
    if args.snapshot:
        with np.load(args.snapshot) as data:
            scenes.append(('saved-centroids', data['vertices'].reshape(-1, 3, 2).mean(axis=1).astype(np.float32)))

    device = pick_device()
    timer = GpuTimings(device, interval=1)
    if not timer.supported:
        raise RuntimeError('Native GPU timestamps are required for this benchmark')
    core = MpmCore(device)
    channels = 9
    env = EnvironmentGPU(device, channels, 64, 64, .91, 4.,
                         chemical_communication_architecture='persistent-environment',
                         channel_profiles=default_channel_profiles(channels))
    total = env.total_values
    shards = 16
    # Only benchmark bind groups use this scratch; no environment passes run.
    env.deposit_scratch = device.create_buffer(size=total * 2 * shards * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    captured = {}
    loader = agents_gpu.load_core_shader
    group_entries = {}
    create_group = device.create_bind_group

    def capture(name, constants):
        source = loader(name, constants)
        captured[name] = source
        return source

    def capture_group(**kwargs):
        group = create_group(**kwargs)
        group_entries[id(group)] = kwargs['entries']
        return group

    with patch.object(agents_gpu, 'load_core_shader', capture), patch.object(device, 'create_bind_group', capture_group):
        agents = AgentsGPU(device, core, env, channels, 128, 1.,
                           max(len(p) for _, p in scenes), .01, 1., 1., .5, .5,
                           policy_architecture='stateless-128',
                           chemical_communication_architecture='persistent-environment')
    weights = rng.normal(0, .12, agents._total_floats).astype(np.float32)
    agents.load_weights(weights)
    source = captured['agents.wgsl']
    owners = [next(j for j in range(c + 1)
                   if (env.channel_widths[j], env.channel_heights[j]) ==
                      (env.channel_widths[c], env.channel_heights[c])) for c in range(channels)]
    owner_array = 'array<u32, 9>(' + ', '.join(f'{c}u' for c in owners) + ')'
    shared = replace_once(source,
        'addDepositFloat(FIELD_TOTAL + index, weightedArea);',
        f'if ({owner_array}[c] == c) {{ addDepositFloat(FIELD_TOTAL + index, weightedArea); }}')
    def shard_source(code):
        code = replace_once(code, 'rest: ParticleRest) {', 'rest: ParticleRest, shard: u32) {')
        code = replace_once(code, 'let index = fieldIndex(c, k.ys[y], k.xs[x]);',
            'let index = fieldIndex(c, k.ys[y], k.xs[x]) + shard * FIELD_TOTAL * 2u;')
        code = code.replace('positions[pi], particleRest[pi]);',
                            f'positions[pi], particleRest[pi], pi % {shards}u);')
        return code.replace('result.envWrite, pos, particleRest[pi]);',
                             f'result.envWrite, pos, particleRest[pi], pi % {shards}u);')
    gated = replace_once(source, 'depositMaterialSample(result.envWrite, pos, particleRest[pi]);',
        'if (physics.growthEnabled > 0.5) { depositMaterialSample(result.envWrite, pos, particleRest[pi]); }')

    def compile_pipeline(code, entry, entries):
        pipeline = device.create_compute_pipeline(layout='auto',
            compute={'module': device.create_shader_module(code=code), 'entry_point': entry})
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        return pipeline, group

    full_entries = group_entries[id(agents._communication_bind_groups[0])]
    splat_entries = group_entries[id(agents._splat_bind_group)]
    variants = {}
    retry_buffer = device.create_buffer(size=max(len(p) for _, p in scenes) * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    for name, code, count in [('baseline', source, 1), ('shared-area', shared, 1),
                               ('shards16', shard_source(source), shards),
                               ('shared-area-shards16', shard_source(shared), shards)]:
        instrumented = code + '''
var<private> casRetries: u32 = 0u;
@group(0) @binding(15) var<storage, read_write> retryCount: array<u32>;
'''
        instrumented = replace_once(instrumented, 'previous = result.old_value;',
                                    'casRetries += 1u; previous = result.old_value;')
        call = ('depositMaterialSample(levels, positions[pi], particleRest[pi]);' if count == 1 else
                f'depositMaterialSample(levels, positions[pi], particleRest[pi], pi % {shards}u);')
        instrumented = replace_once(instrumented, call, call + '\n retryCount[pi] = casRetries;')
        variants[name] = {
            'full': compile_pipeline(code, 'agentStep', full_entries),
            'splat': compile_pipeline(code, 'splatChemicalState', splat_entries),
            'shards': count,
            'retries': compile_pipeline(instrumented, 'splatChemicalState', splat_entries +
                        [{'binding': 15, 'resource': {'buffer': retry_buffer}}]),
        }
    gated_pipeline = compile_pipeline(gated, 'agentStep', full_entries)
    clear_pipelines = {}
    for count in (1, shards):
        clear_pipelines[count] = compile_pipeline(f'''
@group(0) @binding(5) var<storage, read_write> scratch: array<u32>;
@compute @workgroup_size(256) fn clear(@builtin(global_invocation_id) id: vec3<u32>) {{
  if (id.x < {total * 2 * count}u) {{ scratch[id.x] = 0u; }}
}}''', 'clear', [{'binding': 5, 'resource': {'buffer': env.deposit_scratch}}])
    reduced = device.create_buffer(size=total * 2 * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    reduce_pipeline = device.create_compute_pipeline(layout='auto', compute={
        'module': device.create_shader_module(code=f'''
@group(0) @binding(0) var<storage, read> scratch: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(256) fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {{
  if (id.x >= {total * 2}u) {{ return; }}
  var value = 0.0;
  for (var s = 0u; s < {shards}u; s++) {{ value += scratch[id.x + s * {total * 2}u]; }}
  result[id.x] = value;
}}'''), 'entry_point': 'reduce'})
    reduce_group = device.create_bind_group(layout=reduce_pipeline.get_bind_group_layout(0), entries=[
        {'binding': 0, 'resource': {'buffer': env.deposit_scratch}},
        {'binding': 1, 'resource': {'buffer': reduced}}])

    def dispatch(encoder, label, pipeline, group, groups, timed):
        p = timer.begin_compute_pass(encoder, label) if timed else encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, group)
        p.dispatch_workgroups(groups)
        p.end()

    def run(pipeline, kind, count, timed=True):
        if timed:
            timer.begin_sample()
        encoder = device.create_command_encoder()
        batch = args.batch if timed else 1
        for _ in range(batch):
            dispatch(encoder, 'clear', *clear_pipelines[count],
                     (total * 2 * count + 255) // 256, timed)
            dispatch(encoder, 'kernel', *pipeline, agents._dispatch, timed)
            if count > 1:
                dispatch(encoder, 'reduce', reduce_pipeline, reduce_group, (total * 2 + 255) // 256, timed)
        command = encoder.finish()
        started = time.perf_counter()
        device.queue.submit([command])
        # This installation's queue-completion callback has a native ABI mismatch.
        # A four-byte readback provides the same completion fence (included in wall time).
        device.queue.read_buffer(reduced if count > 1 else env.deposit_scratch, 0, 4)
        wall_us = (time.perf_counter() - started) * 1e6 / batch
        if timed:
            timer.finish_sample()
            return {**{k: v['seconds'] * 1e6 / batch for k, v in timer.report()['stages'].items()},
                    'wall': wall_us}

    def scratch_result(name):
        count = variants[name]['shards']
        raw = np.frombuffer(device.queue.read_buffer(reduced if count > 1 else env.deposit_scratch,
                            0, total * 2 * 4), np.float32).reshape(2, total).copy()
        if name.startswith('shared-area'):
            for c, owner in enumerate(owners):
                offset, start = env.channel_offsets[c], env.channel_offsets[owner]
                size = env.channel_widths[c] * env.channel_heights[c]
                raw[1, offset:offset + size] = raw[1, start:start + size]
        return raw

    report = {'adapter': dict(device.adapter.info), 'samples': args.samples, 'batch': args.batch,
              'grid_widths': env.channel_widths, 'area_owners': owners,
              'notes': ['Frozen stateless-128 random weights; zero chemical/morphology fields; identity strain.',
                        'Particle area is 0.0001 via spacing=0.01 and unit quadrature weight; no physics advances.',
                        'The disabled-deposit control uses a runtime branch in the same gated shader.',
                        'Sharded totals include scratch clear and a GPU reduction; no chemistry merge or physics.',
                        'Shared area is expanded on CPU only for validation; a production consumer would index the owner.',
                        'Wall time covers submission and a four-byte completion readback, divided by batch size; excludes encoding.',
                        'Retry counters are collected in separate instrumented splat runs and are not used for timings.',
                        'Saved centroids use synthetic material area and weights, not restored rollout state.'],
              'scenes': []}
    for scene_name, positions in scenes:
        count = len(positions)
        core.load_scene(positions, np.zeros((count, 2), np.float32),
                        np.tile([1, 0, 0, 1], (count, 1)).astype(np.float32),
                        np.zeros((count, 4), np.float32), np.ones(count, np.float32))
        agents.set_active_count(count)
        agents.reset_state()
        meta = np.zeros(count, agents._particle_meta_dtype)
        meta['chemicalState'] = rng.uniform(-1, 1, (count, channels))
        device.queue.write_buffer(agents._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, meta)
        cases = [(name, kind, item[kind], item['shards'], True)
                 for name, item in variants.items() for kind in ('full', 'splat')]
        cases += [('gated-on', 'full', gated_pipeline, 1, True),
                  ('gated-off', 'full', gated_pipeline, 1, False)]
        samples = {f'{name}/{kind}': [] for name, kind, *_ in cases}
        for round_index in range(args.samples + 3):
            for case_index in rng.permutation(len(cases)):
                name, kind, pipeline, shard_count, enabled = cases[case_index]
                agents.set_growth_enabled(enabled)
                timer.reset()
                values = run(pipeline, kind, shard_count, timed=round_index >= 3)
                if values:
                    samples[f'{name}/{kind}'].append(values)
        # Validate signed numerator and positive area, including GPU shard reduction.
        errors = {}
        retries = {}
        agents.set_growth_enabled(True)
        for kind in ('full', 'splat'):
            reference = None
            for name, item in variants.items():
                run(item[kind], kind, item['shards'], timed=False)
                result = scratch_result(name)
                if reference is None:
                    reference = result
                # Cancellation calls for absolute error relative to deposited area.
                tolerance = 2e-5 * np.maximum(reference[1], 1e-12)
                assert np.all(np.abs(result - reference) <= tolerance), (scene_name, kind, name)
                errors[f'{name}/{kind}'] = float(np.max(np.abs(result - reference)))
        for name, item in variants.items():
            run(item['retries'], 'splat', item['shards'], timed=False)
            values = np.frombuffer(device.queue.read_buffer(retry_buffer, 0, count * 4), np.uint32)
            retries[name] = {'total_failed_cas': int(values.sum()),
                             'failed_cas_per_particle': float(values.mean()),
                             'max_failed_cas_per_particle': int(values.max())}
        stats = {}
        for name, rows in samples.items():
            stats[name] = {key + '_us': float(np.median([r[key] for r in rows])) for key in rows[0]}
            totals = [sum(v for k, v in r.items() if k != 'wall') for r in rows]
            stats[name].update(total_us=float(np.median(totals)),
                               total_p10_us=float(np.percentile(totals, 10)),
                               total_p90_us=float(np.percentile(totals, 90)))
            stats[name].update(wall_p10_us=float(np.percentile([r['wall'] for r in rows], 10)),
                               wall_p90_us=float(np.percentile([r['wall'] for r in rows], 90)))
        report['scenes'].append({'name': scene_name, 'particles': count, 'timings': stats,
                                 'max_absolute_error': errors, 'instrumented_retries': retries})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(scene_name, json.dumps({k: {'gpu': round(v['total_us'], 2), 'wall': round(v['wall_us'], 2)}
                                     for k, v in stats.items()}), flush=True)


if __name__ == '__main__':
    main()

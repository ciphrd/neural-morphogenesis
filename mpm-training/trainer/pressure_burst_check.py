"""Reproducible pressure-burst experiment; writes CSV traces and frozen replays.

Run: .venv/bin/python pressure_burst_check.py --output /tmp/pressure-burst
This diagnostic does not alter production material defaults or shader files.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from agents_gpu import AgentsGPU
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore, DT, GRID_N
from training_sim import seed_blob
from pressure_diagnostics import read_snapshot, measure

SPACING = .0027

def build(device, density, divisor, capacity, damping):
    core = MpmCore(device, physics_dt=DT/divisor)
    env = EnvironmentGPU(device, 1, 32, 32, 0.5, 1.0, chemical_communication_architecture="cell-owned-projection")
    agents = AgentsGPU(device, core, env, 1, 128, 1.0, capacity, SPACING / np.sqrt(density), 1.0, 1.0, 0.5, 0.5, chemical_communication_architecture="cell-owned-projection")
    core.set_gravity(0)
    core.set_repulsion_strength(0, 40)
    core.set_damping(damping, 16)
    core.set_material(10000, .2, 3, .5, growth_rate=120, growth_anisotropy=1,
                      growth_compression_feedback=1, particle_mass=10/density,
                      particle_volume=1/density, )
    core.load_scene(*seed_blob(5*density, (.5, .5), SPACING/np.sqrt(density), 17))
    agents.set_active_count(core.active_count)
    rest = read_snapshot(core)['rest']
    rest[:, 5:7] = [1, 0]
    device.queue.write_buffer(core.rest, 0, rest)
    return core, agents, env

def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def grow(device, density, divisor, args):
    core, agents, env = build(device, density, divisor, args.capacity*density, args.damping)
    snapshot = read_snapshot(core)
    rows = [dict(t=0., macro=0, phase='initial', capacity_blocked=0, **measure(snapshot, density=density))]
    checkpoint = snapshot
    biggest_jump = -1.
    checkpoint_time = 0.
    first_inversion = None
    first_capacity = None
    start = time.perf_counter()
    for macro in range(args.macros):
        encoder = device.create_command_encoder()
        agents.encode_growth_field(encoder)
        device.queue.submit([encoder.finish()])
        count = agents.read_sample_count()
        core.set_active_count(count)
        agents.set_active_count(count)
        after = read_snapshot(core)
        rows.append(dict(t=macro*32*DT, macro=macro, phase='refine',
                         capacity_blocked=int(agents.capacity_blocked), **measure(after, density=density)))
        snapshot = after
        for tick in range(0, 32, args.sample_ticks):
            previous = rows[-1]
            core.step(args.sample_ticks*divisor)
            after = read_snapshot(core)
            metrics = measure(after, density=density)
            t = (macro*32+tick+args.sample_ticks)*DT
            row = dict(t=t, macro=macro, phase='physics',
                       capacity_blocked=int(agents.capacity_blocked), **metrics)
            rows.append(row)
            jump = metrics['kinetic_energy']+metrics['affine_kinetic_energy']-previous['kinetic_energy']-previous['affine_kinetic_energy']
            if first_inversion is None and jump > biggest_jump:
                biggest_jump, checkpoint, checkpoint_time = jump, snapshot, previous['t']
            if first_inversion is None and metrics['inverted_triangles']:
                first_inversion = t
            if first_capacity is None and (count >= args.capacity*density or agents.capacity_blocked):
                first_capacity = t
            snapshot = after
        if first_inversion is not None and (macro+1)*32*DT >= first_inversion+4*32*DT:
            break
    name = f'density-{density}_dt-{divisor}'
    write_csv(args.output/f'{name}.csv', rows)
    np.savez_compressed(args.output/f'{name}-pre-burst.npz', **checkpoint)
    summary = dict(name=name, density=density, dt=core.dt, first_inversion_time=first_inversion,
                   first_capacity_time=first_capacity, checkpoint_time=checkpoint_time,
                   max_kinetic_jump_before_inversion=biggest_jump,
                   peak_speed=max(r['max_speed'] for r in rows), samples=rows[-1]['samples'],
                   elapsed_seconds=time.perf_counter()-start)
    print(json.dumps(summary), flush=True)
    return summary, checkpoint

def replay(device, snapshot, density, divisor, args):
    core = MpmCore(device, physics_dt=DT/divisor)
    core.set_gravity(0)
    core.set_repulsion_strength(0, 40)
    # Remove external growth and damping. Keep the actual plastic/hardening law.
    core.set_damping(0, 16)
    core.set_material(10000, .2, 3, .5, growth_rate=0, particle_mass=10/density,
                      particle_volume=1/density, )
    rest = snapshot['rest']
    core.load_scene(snapshot['positions'], snapshot['velocities'], snapshot['deformation'],
                    snapshot['affine'], rest[:, 4], rest[:, 8:14], rest[:, 15], 'triangle-vertices')
    device.queue.write_buffer(core.rest, 0, rest)
    rows = [dict(t=0., **measure(snapshot, density=density))]
    for tick in range(32):
        core.step(divisor)
        rows.append(dict(t=(tick+1)*DT, **measure(read_snapshot(core), density=density)))
    write_csv(args.output/f'replay-density-{density}_dt-{divisor}.csv', rows)
    summary = dict(density=density, divisor=divisor,
                   relative_energy_change=rows[-1]['total_energy']/rows[0]['total_energy']-1,
                   max_relative_energy=max(r['total_energy'] for r in rows)/rows[0]['total_energy']-1,
                   kinetic_change=rows[-1]['kinetic_energy']+rows[-1]['affine_kinetic_energy']-rows[0]['kinetic_energy']-rows[0]['affine_kinetic_energy'],
                   elastic_change=rows[-1]['elastic_energy']-rows[0]['elastic_energy'],
                   inverted_triangles=rows[-1]['inverted_triangles'])
    print('replay '+json.dumps(summary), flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/pressure-burst'))
    parser.add_argument('--densities', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--divisors', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--macros', type=int, default=120)
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--sample-ticks', type=int, choices=[1, 2, 4, 8, 16, 32], default=4)
    parser.add_argument('--damping', type=float, default=0.)
    args = parser.parse_args()
    if min(args.divisors+args.densities) < 1 or args.capacity < 10 or args.macros < 1:
        parser.error('positive densities/divisors/macros and capacity >=10 required')
    args.output.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    summaries, replays = [], []
    for density in args.densities:
        baseline = None
        for divisor in args.divisors:
            summary, snapshot = grow(device, density, divisor, args)
            summaries.append(summary)
            if divisor == 1:
                baseline = snapshot
        if baseline is not None:
            for divisor in (1, 2, 4, 8):
                replays.append(replay(device, baseline, density, divisor, args))
    metadata = dict(settings={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    reference_dt=DT, grid_n=GRID_N, seed=17, base_ticks_per_macro=32,
                    base_spacing=SPACING, initial_seed_cells_per_density=5, material=dict(E=10000, nu=.2, hardening=3, elasticity=.5), summaries=summaries, replays=replays)
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')

if __name__ == '__main__':
    main()

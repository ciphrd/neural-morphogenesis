"""Paired local-search experiment under CURRENT simulation settings.

Loads weights only, never resumes or modifies an archived/live run. Every
mutation scale uses the same Gaussian directions and rollout seeds. Results
include failures, terminal geometry and a held-out seed for the selected child.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from config import CONFIG
from evolve import (build_arg_parser, finalize_policy_configuration,
                    finalize_density_configuration, validate_fitness_configuration, mutate)
from parallel_workers import build_pool, worker_rollout
from targets import load_target
from domain_fitness import target_mask
from raster import build_target_distance_field


def write_report(path, report):
    def finite(value):
        if isinstance(value, dict):
            return {k:finite(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(finite(report), indent=2, allow_nan=False))


def capture(weights, seed):
    import parallel_workers as workers
    result = worker_rollout(weights, seed, return_snapshot=True)
    return result, workers._core.read_rest_state()[:, 8:14], workers._agents.read_colors(workers._core.active_count)


def main():
    parser = build_arg_parser()
    parser.add_argument('--probe-weights', type=Path, required=True)
    parser.add_argument('--probe-output', type=Path, required=True)
    parser.add_argument('--probe-sigmas', type=float, nargs='+', default=[.02, .002, .0002])
    parser.add_argument('--probe-directions', type=int, default=2)
    parser.add_argument('--probe-repeats', type=int, default=1,
                        help='independent repeats of identical weights/seeds to measure GPU rollout noise')
    parser.add_argument('--probe-seeds', type=int, nargs='+', default=[12345, 67890])
    parser.add_argument('--probe-holdout', type=int, default=24680)
    args = parser.parse_args()
    if args.probe_directions < 0 or any(not np.isfinite(s) or s <= 0 for s in args.probe_sigmas):
        parser.error('directions must be nonnegative and finite sigmas must be positive')
    if args.probe_holdout in args.probe_seeds:
        parser.error('holdout must differ from selection seeds')
    if args.probe_repeats < 1 or len(set(args.probe_seeds)) != len(args.probe_seeds):
        parser.error('repeats must be positive and selection seeds unique')
    finalize_policy_configuration(args)
    finalize_density_configuration(args)
    validate_fitness_configuration(args)
    out = args.probe_output
    parent = np.load(args.probe_weights).astype(np.float32)
    # Validate weight layout before creating output or initializing a GPU.
    mutate(parent, 0., np.random.default_rng(0), args.policy_architecture)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out/'parent.npy', parent)
    target = load_target(args.target)
    mask = target_mask(target, args.raster_resolution)
    candidates = [('parent', parent, 0.)]
    for sigma in args.probe_sigmas:
        for direction in range(args.probe_directions):
            delta = mutate(parent, sigma, np.random.default_rng(args.seed+direction), args.policy_architecture)-parent
            for sign in (-1, 1):
                candidates.append((f's{sigma:g}_d{direction}_{sign:+d}', parent+sign*delta, sigma))
    report = dict(config=CONFIG, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  weights_sha256=hashlib.sha256(parent.tobytes()).hexdigest(), candidates=[])
    root = Path(__file__).resolve().parents[1]
    report['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder,pattern in [('trainer','*.py'),('core','*.wgsl'),('core','config.json')]
        for p in (root/folder).glob(pattern)}
    write_report(out/'report.json', report)
    with build_pool(args.workers or 2, args.particle_capacity, target, mask,
                    build_target_distance_field(mask), args) as pool:
        jobs = [(label, weights, sigma, seed, repeat) for label,weights,sigma in candidates
                for seed in args.probe_seeds for repeat in range(args.probe_repeats)]
        results = pool.map(capture, [j[1] for j in jobs], [j[3] for j in jobs])
        for (label, weights, sigma, seed, repeat), (snapshot, vertices, colors) in zip(jobs, results):
            e = snapshot.evaluation
            row = dict(candidate=label, sigma=sigma, seed=seed, repeat=repeat, fitness=snapshot.fitness,
                       terminal_fitness=e.total, breakdown=asdict(e.breakdown) if e.breakdown else None,
                       diagnostics=snapshot.diagnostics)
            report['candidates'].append(row)
            np.savez_compressed(out/f'{label}_{seed}_{repeat}.npz', vertices=vertices, colors=colors,
                                raster=e.raster, rgb=e.color_raster)
            write_report(out/'report.json', report)
            print(json.dumps(row), flush=True)
        means = {label:np.mean([r['fitness'] for r in report['candidates'] if r['candidate']==label]) for label,_,_ in candidates}
        winner = min(candidates, key=lambda c:means[c[0]])
        np.save(out/'selected.npy', winner[1])
        report['means'] = means
        report['selected'] = winner[0]
        report['holdout'] = []
        for label,weights in [('parent',parent), ('selected',winner[1])]:
            snapshot, vertices, colors = pool.submit(capture, weights, args.probe_holdout).result()
            report['holdout'].append(dict(candidate=label, fitness=snapshot.fitness, diagnostics=snapshot.diagnostics))
            np.savez_compressed(out/f'holdout_{label}.npz', vertices=vertices, colors=colors,
                                raster=snapshot.evaluation.raster, rgb=snapshot.evaluation.color_raster)
        write_report(out/'report.json', report)
        print(json.dumps(dict(means=means, selected=winner[0], holdout=report['holdout'])), flush=True)


if __name__ == '__main__':
    main()

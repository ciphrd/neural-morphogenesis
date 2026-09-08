# Training performance investigation — 2026-09-06

## Scope and limits

Inspected the current run settings, generation history, worker processes, simulation orchestration, raster scoring, and generation-image publication. Ran a bounded native-GPU profile and CPU raster microbenchmarks. Production code and the running training process were not modified.

The live run had 16 workers active during the investigation. Measurements therefore include background contention; they are indicative local timings, not isolated throughput benchmarks. The GPU profile covered early growth (40–50 samples), not a complete 3,000-sample rollout. No worker-count speedup is claimed.

Current run: recurrent 128-unit policy, persistent environment, lizard-64, 16 candidates, one seed/density, cap 3,000, horizon 2,000 macro steps, 16 physics substeps and eight neural updates per macro, raster 256, stable checks every 10 macro steps. Generation 0's representative winner stopped at step 982 with 2,999 samples and capacity-blocked status.

## Measurements

A 100-step native GPU rollout took 7.68 s, excluding 1.43 s initialization. Profiled macro-step orchestration accounted for approximately 4.59 s; 13 full raster evaluations accounted for 2.84 s. Native synchronization and dispatch overhead were prominent. Native callback interactions make individual nested cProfile attribution less reliable than these coarse sections.

Median of three CPU evaluations on deterministic synthetic triangle disks (same target, with prewarmed target RGB):

| Samples | Raster | Shape only | Shape + RGB |
|---:|---:|---:|---:|
| 40 | 128 | 21 ms | 52 ms |
| 40 | 256 | 76 ms | 214 ms |
| 500 | 128 | 30 ms | 62 ms |
| 500 | 256 | 97 ms | 234 ms |
| 3,000 | 128 | 71 ms | 108 ms |
| 3,000 | 256 | 204 ms | 340 ms |

At 3,000 samples/256, exact triangle clipping consumed about 127 ms; rotation resampling about 141 ms of a 340 ms evaluation. RGB adds three rotated channels; a normal score performs 76 scalar image rotations across its 20-angle search. This overhead exists even for very small organisms.

A standalone prototype retained the exact existing missing/spill/overlap calculation and angle refinement, but omitted RGB and fitness terms during a stopping-only check:

| Samples | Full score | Match-only check | Speedup per check |
|---:|---:|---:|---:|
| 40 | 215 ms | 46 ms | 4.7× |
| 3,000 | 337 ms | 167 ms | 2.0× |

Its three stopping metrics matched full evaluation within 1e-12 on these fixtures. This is not yet production regression coverage or an end-to-end training speedup.

A shared-coordinate NumPy bilinear rotation prototype matched SciPy within 1e-12 but was slower: 148 ms versus 113 ms for 15 four-channel rotations. Do not adopt that implementation.

## Recommended changes, in priority order

1. **Separate stopping checks from fitness scoring.** `evolve.rollout` currently invokes full `score_domains` every ten steps. Before the final capture window, a failed stable-match check only needs geometry metrics to reset its confirmation state. Use a match-only path there; still compute full RGB/shape fitness at scoring checkpoints, terminal states, and successful settling checks. Share the exact geometry and pose-search implementation rather than duplicating it. Preserve full scoring when a stable-match check succeeds, because settling scores contribute to fitness. The measured 2–4.7× improvement applies only to these checks, not the whole rollout.

2. **Reuse the winner's evaluated terminal state for previews.** `train_server` waits for all candidates, then `_save_generation_images` calls `rollout` again and scores its terminal state again before the generation is broadcast. This adds a serial complete rollout to every generation. Have each worker return compact terminal geometry/colors/positions and diagnostics with its scalar fitness, select the representative winner using the existing seed/density rules, then render that saved state. Avoid transmitting a full state trajectory or per-step arrays. Preserve the distinction between selection fitness (temporal/seed/density aggregation) and the representative final snapshot.

3. **Investigate the extra physics synchronization barrier.** Every macro step reads sample status and `MpmCore.step` also forces a four-byte sync at its end. At 2,000 steps this means roughly 4,000 synchronous readbacks before fitness reads. The status read is needed to size subsequent dispatches. The physics barrier exists for a documented Metal command-buffer limit, so do not simply remove all barriers. Test whether the final small physics chunk can rely on the next required status/fitness read while retaining barriers between long chunks. Verify long-rollout stability and exact results before adopting.

4. **Benchmark worker counts in a quiet run.** The current explicit `--workers 16` shares one GPU. More processes may increase CPU/GPU contention. Compare fixed candidate weights and seeds at 4, 8, 12 and 16 workers, after initialization, recording generation wall time. Current process CPU utilization alone cannot identify the optimum. Worker startup is amortized and not the main steady-state issue.

5. **Optional quality/speed tradeoff:** `--raster-resolution 128` made these RGB scores 3.1–4.1× faster. It changes geometry and color fidelity and can change fitness rankings and stopping behavior. Prefer the first two improvements, which can preserve the objective. Increasing `--shape-check-interval` also reduces work but changes stop timing. Reducing physics or neural update counts changes simulation behavior and should not be the first optimization.

## Reproducing the CPU experiments

From the repository root:

```sh
trainer/.venv/bin/python artifacts/training-performance/mpm_profile_raster.py
trainer/.venv/bin/python artifacts/training-performance/mpm_stop_benchmark.py
trainer/.venv/bin/python artifacts/training-performance/mpm_rotation_benchmark.py
```

The raster profiler reports 128/256 shape/RGB timings and a cumulative CPU profile. The stopping prototype asserts metric agreement before timing. The rotation prototype asserts numerical agreement before comparing timings. All operate on synthetic fixtures; none change or start a training run.

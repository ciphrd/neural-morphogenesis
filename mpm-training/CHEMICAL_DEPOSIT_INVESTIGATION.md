# Chemical deposit contention — 2026-09-07

Chemical deposition is a measurable contributor to `gpuNeural`. A standalone
prototype reduces the full agent pass plus scratch clearing/reduction by **19%**
on saved lizard particle centroids and **58–67%** on dense 8,000-particle synthetic
clusters. These are frozen-snapshot microbenchmarks on the Apple M2 Max/Metal,
not end-to-end training speedups. Production shaders and simulation settings
were not changed by this investigation.

## What is contending

`core/agents.wgsl:depositMaterialSample` deposits a signed chemical numerator and
positive material area into a 3×3 stencil for each channel. `addDepositFloat`
implements float32 addition with an integer compare-and-swap retry loop.

The nine default channels use three 16×16, three 32×32, and three 64×64 grids.
Every particle therefore makes up to 162 successful atomic additions before
counting retries. Material area is independent of channel identity, so channels
with identical dimensions can share their area accumulator. This reduces the
successful-addition count to 108, but does not resolve numerator contention.

The strongest prototype distributes particles over 16 scratch copies using
`particleIndex % 16`, then sums those copies with a GPU reduction. It retains
float32 accumulation, avoiding the small-contribution loss associated with
fixed-point deposition. Sharing area is an additional, smaller improvement.

## Measurements

Canonical results: `artifacts/chemical-deposit/validated-report.json`.
Medians of 15 randomized measurement rounds, with 16 updates per submission.
Times below include scratch clearing and, for the combined prototype, reduction.
The network remains in the measured full pass.

| Particle distribution | Current GPU time | Shared area + 16 copies | Time reduction |
|---|---:|---:|---:|
| 400, uniform domain | 1.418 ms | 1.410 ms | negligible |
| 400, 0.15-wide square | 1.550 ms | 1.410 ms | 9% |
| 400, 0.015-wide square | 1.812 ms | 1.433 ms | 21% |
| 8,000, uniform domain | 1.665 ms | 1.551 ms | 7% |
| 8,000, 0.15-wide square | 3.943 ms | 1.674 ms | 58% |
| 8,000, 0.015-wide square | 5.462 ms | 1.780 ms | 67% |
| 7,997, saved lizard centroids | 1.909 ms | 1.550 ms | 19% |

The saved-centroid full-pass 10th–90th percentile ranges were 1.903–1.917 ms
for the baseline and 1.543–1.563 ms for the combined prototype. Synchronized
wall time, including an amortized four-byte completion readback, decreased from
1.974 to 1.632 ms. It excludes Python command encoding.

On that same geometry:

- Disabling deposition through a runtime branch reduced the gated full pass
  from 1.905 to 1.422 ms: approximately 25% marginal deposit cost.
- Sharing area alone reduced the original full pass to 1.835 ms, about 4%.
- Using 16 copies without sharing area reduced it to 1.582 ms, about 17%.
- An isolated deposit pass using random signed chemical values decreased from
  0.553 to 0.171 ms with both changes, about 69%.

Separate instrumented deposit runs recorded 277 failed compare-and-swap attempts
per particle on the saved centroids, versus 46 with 16 copies and 39 with both
changes. In the tight 8,000-particle cluster the counts were 1,566 versus 266
and 235. Instrumentation changes scheduling, so these counts explain the
mechanism rather than predict uninstrumented timing exactly.

## Validation and limits

The benchmark builds variants from the actual templated agent shader, using
identical buffers and weights. Both full inference deposits and isolated signed
deposits are checked against the original kernel for every scene. After GPU
reduction and expansion of shared area, all numerator and area errors were
within `2e-5 * max(reference_area, 1e-12)` per grid entry. The largest absolute
error was 5.22e-7 in the tight cluster and 1.49e-8 on saved centroids. Accumulation
order changes; bitwise equivalence is not expected.

The benchmark uses a stateless 33→128→14 network with seeded synthetic weights,
zero chemical/morphology input fields, identity deformation, and material area
0.0001 per particle. It does not advance growth, chemistry, or mechanics.
The lizard case takes only triangle centroids from
`artifacts/lizard-detail/capacity-8000/parent_12345.npz`; it does not restore that
rollout's weights, chemical fields, or material areas. Recurrent policies and
live multi-worker training have not been benchmarked here.

Initial single-update measurements occasionally returned zero-duration neural
timestamps. `report.json` is that exploratory run and should not be used for
speedup claims. `batched-report.json` is an intermediate eight-update run with
larger variability. The canonical sixteen-update run also checks synchronized
wall time and retains percentile ranges in its JSON.

## Recommended implementation experiment

Prioritize scratch partitioning; shared area alone is insufficient for dense
scenes. The current sparse-layout prototype expands scratch from 126 KiB to
1.97 MiB per simulation, plus a 126 KiB reduction output. A production version
could sum copies directly in the existing environment merge/materialization
consumer, avoiding the separate reduction output and pass. Both persistent
environment and cell-owned projection consumers must read the shared area
owner, and trainer/viewer buffer layouts must stay aligned.

Before enabling this by default, compare 4/8/16 copies in complete rollouts and
check chemistry conservation, tiny areas, opposing signed expression, and
rollout/fitness drift. Low particle counts and sparse scenes benefit much less;
16 is a useful prototype setting, not a tuned universal choice. The remaining
~1.4 ms sensing/network pass on the saved geometry is a separate optimization
target.

Reproduce the canonical run from the repository root:

```sh
trainer/.venv/bin/python trainer/chemical_deposit_bench.py \
  --samples 15 --batch 16 --counts 400 8000 \
  --snapshot artifacts/lizard-detail/capacity-8000/parent_12345.npz \
  --output artifacts/chemical-deposit/validated-report.json
```

Native Metal access is required; the filesystem sandbox exposes no GPU adapter.

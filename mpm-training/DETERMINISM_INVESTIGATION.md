# Determinism audit

The observed variation is GPU execution-order dependence, not an unseeded
random-number call inside the growth policy. Controlled ablations isolated
physics accumulation, chemical accumulation, and refinement allocation. A
separate reused-worker test exposed stale grid velocity at rollout reset.

The immediate target is **identical active simulation state and fitness for
identical inputs on the same GPU/backend/build**. This does not promise bitwise
agreement between Metal and browser backends, GPU vendors, or library versions.
WGSL permits implementation variation in floating-point accuracy, fusion,
rounding, and built-in functions; atomic access also does not impose an
application-chosen order between invocations. See the
[WGSL execution and floating-point specification](https://www.w3.org/TR/WGSL/#execution).

## Measured isolation

The recurrent lizard parent is the copied policy in
`artifacts/lizard-detail/probe-gpu/parent.npy`. Tests compare bytes, not approximate
floating-point tolerances. The trace captures positions, velocities,
deformation, affine state, rest geometry/growth, alignment, color, private and
chemical state, and both chemical field buffers. Newer traces additionally
capture grid velocity. Scratch hash-table layouts are not part of that state.

| Intervention | First divergence | Evidence |
| --- | --- | --- |
| Original parallel kernels, fresh GPU objects | First macro step, after physics | Velocity difference up to 1.24e-6; rest state up to 1.13e-6 |
| Order P2G only | Step 2, chemical field before physics | Field difference 2.33e-10, subsequently amplified |
| Order P2G and chemical deposition | Step 26, refinement before physics | Sample arrays are reordered; indexwise position difference 0.0512 is not itself a physical displacement |
| Also order allocation | No differences through 80 steps | All captured active-state bytes identical |
| Order the above, reuse buffers with `chemical-pole`, before reset fix | Initial grid velocity; then step 1 chemistry | Initial stale velocity up to 2.706; chemical field difference 0.001085 |
| Same reuse test after clearing grid velocity | No differences in three 30-step repeats | Includes both fields, grid velocity and active particle/neural state |

The allocator/permutation finding matters even before capacity: summing in
sample-index order is only deterministic if sample indices themselves are
deterministic. An array permutation can subsequently change floating-point
reduction results. At capacity, scheduling also chooses which valid refinements
receive the last slots.

Artifacts and full traces: [artifacts/determinism](artifacts/determinism).
The first ablation reports were generated before the two production fixes.
Later probe versions record source/config/weight hashes. Readback tracing adds
synchronization, so the actual multiprocessing training path is tested
separately using `detail_probe.py --deterministic-reference`.

That training-path check returned exactly **1.3311658798217574** for all five
80-step evaluations (three repeats plus two validation executions, two worker
processes). Each ended at 80 steps and 116 samples. This confirms repeatability
in that tested mode; it is not a good lizard fit. A fully serial 2,000-step
comparison was stopped for excessive runtime before completing its first
repeat. **Full-horizon determinism is therefore not yet validated.** The probe
now saves progress and enforces a per-repeat time budget between macro steps.

## Inventory of sources and controls

| Code path | Classification | Required control / finding |
| --- | --- | --- |
| `core/p2g.wgsl:addGridFloat` | **Confirmed nondeterministic float reduction** | CAS protects updates from being lost; it does not fix summation order. Replace with fixed-order reduction or an exact reproducible accumulator. |
| `core/agents.wgsl:addDepositFloat` | **Confirmed nondeterministic float reduction** | Both persistent-environment output deposition and cell-owned splatting call it. Order contributions before summing. |
| `core/growthField.wgsl:reserveRefinement` | **Confirmed scheduling-dependent sample IDs and capacity arbitration** | Fixed: ascending root/sample allocation in one small ordered pass. A future parallel implementation must preserve those decisions. |
| Grid velocity retained by `reset_growth_buffers` / `resetGrowthBuffers` | **Confirmed stale-state leak for seeded chemistry** | Fixed in Python and browser: zero grid velocity before the new rollout. It was read by first-step chemical transport before physics overwrote it. This is not the explanation for the original `none`-preset divergence. |
| `indexRefinementEdges` hash insertion | Scheduling-dependent scratch layout | Hash slot placement varies. Completed lookups count/find exact edges and should yield the same boundary/neighbor semantics. Ordering this pass was unnecessary in the 80-step isolation. Do not hash scratch layout as physical state. |
| `propagateRefinement` in-place pointer jumping | Scheduling-dependent intermediate scratch values | A reader may see an already advanced parent. With valid terminating dependency chains and sufficient rounds, final roots converge. A 257-triangle dependency regression passes. Ping-pong old/new root arrays would make each round explicitly reproducible and simplify the proof; not changed here. |
| Integer growth-field / compute-density / repulsion accumulation | Order-independent integer sums | Preserve bounded conversion/accumulation. Overflow/invalid values are correctness risks; integer atomic addition is not the float-CAS problem. |
| Additive density-quad rendering | Backend-dependent numeric path | Draw/primitive order and selected feature path must remain fixed. Do not assume its rounding matches the integer compute path. It was not needed as an additional ordering intervention in the completed short traces. |
| Neural inference, diffusion, gradients, G2P, blur | Fixed per-output formulas / fixed pass ordering | Each output has an owner or reads a prior-pass field. No random sampling found. Floating-point operations still depend on backend/compiler rules. |
| Capacity / low-growth / late-window stopping | Deterministic thresholds that **amplify** upstream differences | Different splits alter stop time and the terminal snapshot. These rules are not RNG sources. Test identical stop reason, step and count as well as fitness. |
| NumPy evolution RNG, PyTorch policy initialization | Intentional seeded randomness | Both training entry points seed their RNGs. Preserve RNG state and population to resume an identical evolutionary run; loading best weights starts a new run, not the original RNG trajectory. |
| Initial-condition hashes and perturbations | Intentional seeded randomness | Fixed hash-derived values. `none` ignores seed for geometry/state; different seed numbers then repeat the same initial condition. |
| Viewer policy randomization / interactive mutation | Intentional external randomness | `randomPolicySeed()` uses crypto/Math.random; mutation can default to Math.random. Export exact weights and record the chosen seed. These UI actions do not execute during a fixed-weight Python rollout. Python/JS random initialization algorithms are not interchangeable. |
| Worker completion order | Controlled | `pool.map` preserves submitted order; selection uses stable sorting. Timing values do not enter fitness. More workers can change GPU scheduling, exposing the float reductions above. |
| Queue waits / GPU profiling | Controlled ordering, potentially different scheduling | Pass boundaries enforce dependencies. Extra waits can change which float atomic wins first, but do not create a separate RNG. More waits alone are not a determinism fix. |
| Source/configuration/target/backend changes | Different inputs, not pseudorandomness | Freeze weights, target, resolved density, settings, source/build and GPU feature path. A seed alone is insufficient. |

Reset audit: initial active particle states and metadata are overwritten;
claimable deformation/rest/velocity slots are reset; chemical buffers and parity
are reset; chemical scratch/gradients, morphology and growth/refinement scratch
are rebuilt before consumption. Grid velocity was the exception found and
experimentally isolated. Inactive position tails are not part of the active
state and must never be read before a child writes its slot.

## Implemented controls

1. Stable refinement allocation in the shared WGSL, used by Python and browser.
   Only allocation is serialized; neural work and mesh splitting remain parallel.
   Eight repeated contention fixtures produce exactly the same child IDs and
   geometry, including an odd capacity, and conserve represented weight.
2. Clear old grid velocity on Python/browser rollout reset. Three ordered,
   seeded-chemistry reuse runs match byte for byte after the fix.
3. `--deterministic-reference` in the native trainer/worker path. It serializes
   the callers of float accumulation in increasing sample order while retaining
   the existing floating-point formulas. It is intentionally a slow reference,
   especially because it also serializes neural inference in `agentStep`.
   Default physics/chemical reductions remain parallel. Checkpoint metadata and
   server settings identify this mode. Browser replay does not implement the
   native reference mode; shared allocation/reset fixes do apply there.
4. `trainer/determinism_probe.py` provides staged ablations, fresh-device and
   reused-buffer tests, sampled state hashes, and first-divergence diagnostics.
   `trainer/deterministic_kernels_check.py` tests allocation and reset behavior.

Example diagnostic from the repository root:

```sh
trainer/.venv/bin/python trainer/determinism_probe.py \
  --weights artifacts/lizard-detail/probe-gpu/parent.npy \
  --steps 80 --repeats 3 --reuse --initial-condition chemical-pole \
  --ordered-stages p2g agentStep splatChemicalState \
  --output /tmp/mpm-determinism-check
```

The output directory must be new. Omit `--ordered-stages` to measure the normal
parallel path. Add `--deterministic-reference` to `evolve.py`, `train_server.py`,
or `detail_probe.py` to use the reference in actual native worker rollouts.
The trace probe's default budget is 300 seconds per repeat; `--max-seconds`
overrides it. A timed-out report is explicitly incomplete and is not a pass.

Checks completed: the eight-repeat allocator/reset regression, the conforming
refinement suite (including 257-link propagation, odd capacity and transported
meshes), seeded-state reuse traces, shared-seed selection regression, physics
submission/barrier regression, actual worker-path reference repeats, viewer
production build, and whitespace/diff checks.

## Efficient deterministic design

The preferred production solution preserves parallel contribution computation
and makes only the **reduction order** explicit:

- **P2G:** each particle writes its nine node contributions into fixed slots.
  Deterministically group records by `(node ID, particle ID, stencil slot)`;
  reduce each node in a fixed tree/order. A spatial bin structure may accelerate
  this, but append order inside bins must not become the new summation order.
- **Chemistry:** split neural inference from deposition. The NN writes per-agent
  channel outputs once; another pass generates contributions. Group by
  `(field/channel/texel, particle ID, stencil slot)` and reduce in fixed order.
  This avoids serializing the expensive NN and covers both chemical architectures.
- **Allocation:** replace the small ordered loop with deterministic prefix-based
  allocation only if profiling justifies it. Preserve the exact rule that a
  two-slot operation can be skipped when only one slot remains while a later
  one-slot operation may still fit. A naive prefix cutoff is not equivalent.
- **Root propagation:** separate read/write buffers per round, or compute roots
  from immutable links. Keep the pass count fixed and validate long chains and
  nonmanifold rejection.

An alternative is an exact/binned integer superaccumulator for float deposits.
It must handle signed values, cancellation, range and overflow explicitly.
Simply rounding everything into a single fixed-point integer scale would trade
the nondeterminism problem for a precision/range problem—particularly undesirable
for the fine-detail objective. Neither parallel design is implemented here.

Acceptance for the efficient version: bit-identical **within-mode** repeated
states, terminal stop and loss under fresh/reused workers and one/multiple
workers; paired-capacity/conservation tests; both chemical architectures;
long rollouts and perturbation presets; finite-state checks; and measured
runtime/memory cost. A different fixed reduction tree need not reproduce the
serial reference's last bits, but its numeric differences must be bounded and
physical behavior assessed. Cross-device bit parity would require a separately
specified arithmetic model and is not established by these tests.

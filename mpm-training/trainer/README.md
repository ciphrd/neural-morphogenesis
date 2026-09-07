# Python trainer

Headless WebGPU orchestration for the shared MPM tissue-growth shaders. Defaults come from `../core/config.json` through `config.py`; `simulation_settings.py` exposes typed and derived values without maintaining a second set of defaults.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python evolve.py --help
.venv/bin/python train_server.py --target circle
```

Use `.venv/bin/python train_server.py --serve-only` to serve the saved current
run and archives without starting training or initializing a GPU. This mode does
not rotate or modify checkpoint files.

`evolve.py` evaluates and mutates policy populations. `parallel_workers.py` owns worker-local GPU systems. `training_sim.py` seeds and advances rollouts. `mpm_core.py`, `agents_gpu.py`, and `environment_gpu.py` allocate buffers and schedule the shared shaders. `domain_fitness.py` scores the transported material geometry and computes shape/color losses. `train_server.py` publishes run settings, generation records, weights and diagnostic images.

For refinement experiments, `--initial-weights PATH.npy` starts a new run from
a parent policy (select its matching `--cell-memory` architecture), and
`--mutation-factors 1 .1 .01 .001` cycles offspring through multiples of
`--mutation-sigma`. Elites remain intact and scores are reevaluated. The default
factor is `[1]`. `--fitness-alignment geometry` opts into slower, exact triangle
integration while optimizing rotation; the default `raster` mode is unchanged.
CLI checkpoints and server settings record these choices. The geometry mode
changes training scores; browser shape stopping still uses raster alignment.
See [the lizard investigation](../LIZARD_DETAIL_INVESTIGATION.md) for measured
precision limits, repeated GPU experiments, costs, and unproven next steps.

`--deterministic-reference` enables a slow native-GPU diagnostic that orders
physics and chemical floating-point accumulation by sample index. It retains
the existing formulas but also serializes neural inference; it is not intended
as a fast production mode or a cross-device bit-parity guarantee. Workers and
checkpoint/server metadata carry the selection; browser replay retains parallel
float reductions. Shared refinement allocation now assigns slots in ascending
sample order, and Python/browser rollout resets clear old grid velocity before
transporting seeded chemistry. See [the determinism audit](../DETERMINISM_INVESTIGATION.md)
for isolated sources, tested limits, and a proposed efficient reduction design.

## PNG targets

Place square RGBA PNGs in `targets/` and select them by filename stem, for
example `targets/tree.png` with `--target tree`. Alpha is the desired material
occupancy (`0` empty, `255` filled); fractional alpha is preserved as soft edge
coverage. RGB provides color supervision. Source resolution is arbitrary,
the image is converted from PNG's y-down coordinates to the simulation's y-up
coordinates, and the alpha-weighted shape is centered while its full canvas
spans the complete simulation domain. Transparent padding therefore remains part of
the target's scale. PNGs must contain an alpha channel and must not share a stem
with a legacy JSON target.

Growth and numerical refinement are separate: continuous stress-free tensor growth creates material; conforming triangle subdivision conserves it. Sample counts are synchronized after refinement to size subsequent GPU dispatches. Native GPU submission is chunked where necessary to avoid excessive outstanding work.

Run focused checks with `.venv/bin/python <name>_check.py`. See the root README for the main regression suites. A short end-to-end smoke run is:

```sh
.venv/bin/python evolve.py --generations 1 --population 2 --elites 1 \
  --macro-steps 4 --particles 64 --workers 1 --checkpoint-every 1 \
  --checkpoint-dir /tmp/mpm-smoke
```

Run metadata and weights must match the current model schema; there is no checkpoint migration layer. Archived research snapshots retain their original meaning and are not acceptance baselines for the current layout.

Training rollouts stop immediately when sample capacity is reached or refinement reports capacity blocked. At macro step 200, rollouts with fewer than 10% more samples than their actual initial seed stop with `low-growth`; the terminal state is scored normally. Exactly 10% growth passes this check.

PNG targets train RGB as well as alpha coverage. Each sample colors its triangle with its NN RGB output; exact pixel intersections integrate these colors, and overlapping colors are area-averaged. The aligned premultiplied RGB mean squared error is normalized by target alpha area and added to the existing shape loss. `--fitness-color-weight` defaults to 1; set it to 0 for shape-only training. Legacy JSON targets remain shape-only. RGB is embedded in checkpoints and shown in target/agent generation PNGs, with an absolute RGB difference in `gen_XXXXX_diff.png`; fitness diagnostics include a separate `color` term. Fitness model version is now 4.

Stable-match polling and settling are disabled in training. Only the five late-window fitness checkpoints and early terminal states are scored. The legacy `--stable-stop` option is accepted for command compatibility but has no effect; new replay settings export `stableStop: false`.

The training server requests terminal snapshots from workers, then selects the winning candidate's existing worst-seed/density snapshot. It writes the exact scored rasters to preview PNGs without rerunning the winner or rescoring it. The CLI keeps scalar worker results when previews are unnecessary. Run `rollout_control_check.py`, `seed_schedule_check.py`, and `preview_reuse_check.py` to verify these paths.

The frontend's **Training timing** panel shows per-generation wall time and a choice of all-worker totals or the representative winner's rollout. Stages include setup, neural/growth command submission, growth status synchronization, physics, geometry/color/position readback, and fitness scoring. Each stage reports total seconds, call count, mean and maximum duration. Host timers use `perf_counter`; existing queue waits are charged to the host operation that waits, so these are not GPU kernel timings. The status readback is labeled **GPU completion / status readback**, with CPU sample-count updates measured separately. Parallel worker totals can exceed generation elapsed time.

Generation records carry an optional `timing` object over WebSocket and in history JSONL, including worker-pool, selection, preview, checkpoint and other preparation time. Generation elapsed ends before history writing and WebSocket delivery; pool time includes first-generation worker startup and result transfers. Measurements appear when a generation completes, and older records show an unavailable state. Run `timing_check.py` for deterministic aggregation checks.

Training omits the explicit synchronization after the final physics chunk. The next required growth-status readback or terminal fitness readback synchronizes that work; intermediate physics chunks retain their barriers. Direct `MpmCore.step()` callers remain synchronous unless they explicitly pass `wait_for_completion=False`. In the timing panel, some physics waiting can consequently appear under growth synchronization or geometry readback; compare generation elapsed time when evaluating the change.

Timing history now retains compact measurements for every generation, independently of the 500-generation weight-history limit. Both live and archived history responses include `timings`; reconnecting backfills missed records. The frontend shows stacked stage bars for each measured generation, while the donut and duration statistics show arithmetic means across the full measured run (startup included). Missing timing records are excluded; missing stages within a measured generation contribute zero. Per-call averages use total time divided by total calls; maxima remain explicitly labeled observed maxima. Changing the selected replay generation does not change run averages.

GPU timing views use native timestamp queries to separate physics, neural inference, chemical fields, morphology, and growth/refinement. By default one macro step in 20 is sampled (steps 1, 21, 41, …); `--gpu-timing-interval 0` disables sampling. Supported adapters must expose both timestamp-query features. Timestamp ticks are converted using the native queue timestamp period. Unsupported adapters and older history remain usable with host timings.

GPU stacked bars show average measured GPU intervals per sampled macro step within each generation; the GPU donut pools samples across all measured generations. These are sampled intervals, not extrapolated whole-generation GPU totals or exclusive GPU busy time: scheduling gaps within an interval can contribute. Host and GPU measurements overlap and must not be added or subtracted to infer overhead. Sampling adds an end-of-step timestamp resolution/readback; its host cost and wait appear separately as **GPU profiling readback**. Omitted timestamp intervals are reported as an incomplete breakdown.

The detailed GPU view separates physics grid clearing, particles-to-grid, grid update, grid-to-particles, and four repulsion passes; chemistry clearing, materialization, gradients, transport/diffusion/decay, cell deposits, and merging; and every growth/refinement shader pass. Repeated passes are summed per sampled macro step, with per-call averages available in the details. Transport, diffusion and decay share a shader and remain one measurement. Fine intervals use compute-pass boundary timestamps. Older coarse GPU records retain their original labels; coarse totals are not emitted alongside their detailed replacements. Profiling adds timestamp overhead on sampled steps but no per-pass CPU waits.

Morphology now uses one instanced additive `r32float` render pass for density on adapters exposing `float32-blendable`, followed by horizontal and vertical blur. Unsupported adapters retain clear/deposit/convert compute passes. Nine periodic images per particle preserve wrapped kernels, and fragment contributions retain the compute path’s fixed-point rounding before floating-point blending. Small accumulation differences remain possible. Repulsion retains its existing compute implementation. Browser replay uses the same render shader and feature fallback.

Blur weights are normalized and uploaded only when morphology settings change, removing repeated Gaussian exponentials from each texel. Resolution and blur width are unchanged. GPU charts distinguish quad/compute deposition, clearing/conversion, and the two blur passes. Run `trainer/.venv/bin/python trainer/morphology_render_check.py` from the project root for native parity checks and an alternating warmed benchmark. On the M2 Max, the initial morphology-only benchmark measured 1.36–1.49× speedup for quads versus compute at 40–4,000 clustered particles (both using precomputed blur weights). These measurements include CPU encoding and terminal synchronization, and do not establish a whole-generation speedup.

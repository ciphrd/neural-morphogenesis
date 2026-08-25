# Density feature implementation plan

This plan implements the model in `DENSITY_MODEL.md` while preserving today's
simulation exactly at density multiplier `q=1`.

## Implementation status

Implemented through the first usable end-to-end version:

- shared versioned resolver and Python golden tests;
- runtime particle mass, volume, geometry, repulsion, and policy-gradient scale;
- multi-density worker evaluation with mean-per-density and worst/mean aggregation;
- backward-compatible checkpoint/run metadata and replay tooling;
- density-weighted debug rasters;
- viewer density selector with derived counts and advanced overrides;
- CPU, TypeScript, WebGPU shader, growth-regression, density-switching, and
  one-generation multiprocessing smoke tests.

The supported range is conservatively declared as `0.5×`–`2×`. Longer learned-
policy calibration runs and morphology tolerance data remain empirical follow-up
work; the runtime and training feature itself is operational.

## Product contract

- `q` means particle sampling density relative to the current preset, not
  material mass density.
- Existing `--particles` and `--initial-particles` values become reference
  counts at `q=1`.
- A single-density run defaults to `q=1`; old commands and checkpoints behave as
  before.
- Multi-density training evaluates every candidate against the same `(seed, q)`
  cases and aggregates robustly across densities.
- The browser exposes density as a rollout-restarting control. Derived values
  are read-only in the normal UI; existing low-level sliders become advanced
  overrides.
- Resolved values and a density-model version are recorded in run/checkpoint
  metadata.

## Phase 0: lock the reference behavior

Before changing runtime values:

1. Add deterministic `q=1` snapshots for seed placement, resolved settings,
   static chemical/morphology fields, a short elastic rollout, and Chamfer/debug
   raster scores.
2. Capture the current policy-input report at `q=1` for a fixed checkpoint and
   seed.
3. Record performance for a representative rollout.

Gate: after the resolver is introduced, all `q=1` snapshots and scores must be
unchanged within existing floating-point tolerances. GPU pipeline or buffer
layout changes may alter bytes only where an existing test already permits
backend-dependent rounding.

Primary files:

- `trainer/growth_check.py`
- `trainer/elastic_diagnostics_check.py`
- new `trainer/density_check.py`
- `trainer/capture_policy_inputs.py`

## Phase 1: shared density contract and pure resolvers

Add a shared data file, for example `core/density.json`, containing:

- model/schema version;
- reference spacing `0.0027`;
- reference chemical and repulsion radius ratios;
- supported/default density multipliers;
- minimum/maximum safety bounds.

Implement immutable resolvers in Python and TypeScript:

```text
resolve(reference settings, q) -> resolved density settings
```

The result includes spacing, actual initial count/cap, particle mass/volume,
deposit sigma in field texels, repulsion radius/strength/cap, and any
density-adjusted input scale. It also retains the reference values for metadata
and UI display.

Suggested files:

- new `trainer/density.py`
- new `viewer/src/gpu/density.ts`
- `core/density.json`
- shared golden cases in `core/density_cases.json`
- `trainer/density_check.py`
- a viewer unit test for the same cases

Use explicit rounding rules in both languages. Reject non-finite/non-positive
`q`, counts below one, resolved caps above allocation capacity, and values
outside the declared safe interval.

Gate: Python and TypeScript produce identical golden outputs; `q=1` resolves to
the current constants exactly.

## Phase 2: configuration, CLI, and metadata plumbing

Add CLI settings:

```text
--particle-densities 1.0
--density-aggregation worst
```

`--particle-densities` accepts one or more positive multipliers. Keep
`--particles` and `--initial-particles` as reference counts to preserve existing
commands. Validate the maximum resolved cap before creating workers.

Extend checkpoint/run settings with additive, backward-compatible fields:

- `densityModelVersion` / `density_model_version`;
- `referenceParticleDensity` or `referenceParticleSpacing`;
- `trainingDensityMultipliers`;
- reference cap and reference initial count;
- representative/debug replay density;
- resolved per-density settings used by the run.

Legacy metadata with no density fields resolves as `q=1`. Do not reinterpret its
stored `particles`, `initial_particle_count`, `split_displacement`, or
`deposit_sigma`.

Primary files:

- `trainer/evolve.py`
- `trainer/train_server.py`
- `trainer/render_rollout.py`
- `trainer/capture_policy_inputs.py`
- `trainer/parallel_workers.py`
- `viewer/src/gpu/types.ts`
- `viewer/src/net/settingsStorage.ts`

Gate: an old checkpoint replays unchanged; a new `q=1` checkpoint can be read by
the browser; invalid density configurations fail before worker startup.

## Phase 3: runtime-resolved mechanics and sensing

### Particle mass and volume

Make base particle mass and volume runtime material values so one persistent
worker can execute different `q` values without rebuilding pipelines. The
existing `Material` uniform is already 48 bytes: its ten meaningful floats are
followed by two padding floats. Reuse those final two slots as
`particleMass` and `particleVolume`, retaining the buffer size.

Update the identical `Material` declarations in `core/p2g.wgsl` and
`core/g2p.wgsl`, and extend `set_material()` in Python/TypeScript. Replace the
compile-time uses in P2G. Thread the live mass into field diagnostics and Python
elastic diagnostics instead of importing a global constant.

Primary files:

- `core/p2g.wgsl`
- `core/g2p.wgsl`
- `trainer/mpm_core.py`
- `viewer/src/gpu/mpmCore.ts`
- `viewer/src/gpu/fieldDiagnostics.wgsl`
- `viewer/src/gpu/render.ts`
- `trainer/elastic_diagnostics.py`

### Particle geometry and fields

At rollout reset, apply the resolved:

- split displacement;
- chemical deposit sigma;
- repulsion splat radius;
- repulsion strength and max delta;
- actual initial count and active-particle cap;
- particle mass and volume.

`AgentsGPU` currently sizes its metadata buffer from constructor
`max_active_particles` but also has a runtime cap setter. Rename/separate these
concepts as `particle_capacity` and `max_active_particles`; allocate workers at
the maximum cap across training densities and change only the runtime cap per
rollout. Reset per-slot state through capacity, not the current cap, so later
higher-density rollouts cannot inherit stale data.

Primary files:

- `trainer/agents_gpu.py`
- `trainer/training_sim.py`
- `trainer/parallel_workers.py`
- `trainer/evolve.py`
- `viewer/src/gpu/agents.ts`
- `viewer/src/gpu/simulation.ts`

### Policy input normalization

Make chemical-gradient normalization runtime-adjustable only after the static
field measurements confirm the proposed `1/s` rule. Add it to the agent physics
uniform (expanding the 64-byte buffer to the next 16-byte-aligned size) rather
than recompiling the policy shader for every rollout. Keep chemical value and
morphology-gradient scales unchanged initially.

Gate: deterministic static patches at `q={0.5,1,2}` have matching normalized
chemical values/gradients and morphology occupancy distributions within declared
tolerances. Elastic patches have matching grid mass, acceleration, and strain.

## Phase 4: density-aware scoring and diagnostics

The live Chamfer fitness already averages nearest-neighbor distances and should
remain unchanged initially. Add a baseline that samples the same analytic/static
shape at each `q` and measures its residual density bias.

Add `particle_weight=1/q` to `rasterize_points_sum()` and thread it through debug
image generation. Default the argument to `1.0` for callers and legacy
checkpoints. Do not change target raster construction.

Extend density diagnostics to record:

- weighted occupancy and aligned Chamfer distance;
- occupied area, components, principal axes, and perimeter;
- `active_count/q` over time;
- nearest-neighbor distance divided by resolved spacing;
- policy-input quantiles;
- morphology interior-density quantiles;
- grid mass, elastic energy per represented area, and kinetic energy;
- runtime and peak active count.

Primary files:

- `trainer/raster.py`
- `trainer/debug_images.py`
- `trainer/train_server.py`
- new `trainer/density_check.py`

Gate: a fixed physical patch produces comparable diagnostics and images across
densities. If Chamfer bias exceeds the accepted tolerance, design a weighted
occupancy fitness as a separate, measured change rather than modifying Chamfer
inside this phase.

## Phase 5: multi-density evolution

Change the evaluation unit from `seed` to `(seed, q)`:

1. Generate one ordered case matrix per generation.
2. Evaluate every candidate on the identical matrix.
3. Average across seeds within each density.
4. Aggregate density scores using the configured rule; default to worst-density
   fitness for robustness.
5. Store per-density scores for the winner and generation summary.

The worker task must carry `q` explicitly. Before constructing each rollout it
resolves and applies all runtime settings. Persistent GPU objects remain reused;
only uniform writes, count resets, and scene loading occur between cases.

Start with `{0.5, 1.0}`. Hold out `2.0` for transfer evaluation, then add it to
training after behavior and cost are acceptable.

Primary files:

- `trainer/evolve.py`
- `trainer/parallel_workers.py`
- `trainer/train_server.py`

Gate: candidate ordering is independent of task scheduling; all candidates see
identical cases; a one-value list `[1.0]` reproduces current selection; density
breakdowns survive checkpoint/resume.

## Phase 6: browser playback and UI

Add a `Particle density` multiplier control with conservative presets. Changing
it restarts the rollout because seed count, spacing, mass, and cap all change.
It should not mutate the trained reference settings.

Show a read-only resolved summary:

- spacing;
- initial/maximum particle count;
- particle mass/volume;
- chemical and repulsion radii;
- whether advanced overrides are active.

Move split displacement, deposit sigma, splat radius, and density-derived
repulsion controls under an `Advanced overrides` section. Resetting overrides
returns to resolver outputs. Disable density selections whose resolved cap
exceeds allocated capacity, or rebuild capacity deliberately with a warning
about cost.

Primary files:

- `viewer/src/ui/PhysicsPanel.tsx` or a new density panel
- `viewer/src/gpu/types.ts`
- `viewer/src/gpu/simulation.ts`
- `viewer/src/render/GridCanvas.tsx`
- `viewer/src/net/settingsStorage.ts`

Gate: reference playback remains unchanged; switching density restarts cleanly;
returning to `q=1` reproduces the reference seed; reloading a saved run restores
the density selection and override state.

## Phase 7: calibration and supported-range declaration

Run the three deterministic scenes from the model document and paired learned
policy rollouts. Calibrate only settings whose measured distributions drift:

- chemical-gradient input normalization;
- morphology density references;
- nonzero repulsion scaling;

Record tolerances and the supported density range in `core/density.json`. Values
outside that range require an explicit unsafe/experimental override and are not
used for training.

Gate: across the supported interval, normalized policy inputs, continuum
mechanics, and weighted morphology pass their tolerances; performance gains at
coarse density are documented.

## Test matrix

Every phase should preserve these cases:

| Case | Purpose |
| --- | --- |
| legacy checkpoint, no density metadata | backward compatibility |
| new checkpoint at `q=1` | exact reference behavior |
| static patch at `q=0.5,1,2` | field and sensing invariance |
| elastic patch at `q=0.5,1,2` | material invariance |
| uniform deterministic growth | count/area/division invariance |
| learned policy paired rollout | end-to-end morphology |
| maximum supported `q` | capacity and memory safety |
| invalid/unsafe `q` | early validation |

Run Python checks, viewer typecheck/tests, shader compilation checks, and a short
headless GPU rollout before merging each runtime phase.

## Recommended pull-request boundaries

1. Reference snapshots plus pure resolver and parity tests.
2. Metadata/CLI plumbing with `q=1` only.
3. Runtime mass, volume, geometry, capacity, and shader changes.
4. Density diagnostics and debug-raster weighting.
5. Multi-density worker evaluation and aggregation.
6. Browser density control and advanced overrides.
7. Calibration data, supported-range limits, and documentation.

This ordering keeps each change reviewable and makes regressions attributable.
Multi-density training is deliberately not enabled until the single-rollout
runtime is demonstrably density-normalized.

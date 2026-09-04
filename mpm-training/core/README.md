# core

Shared WGSL, byte-for-byte identical between `../trainer/` (headless,
via Python `wgpu`) and `../viewer/` (browser WebGPU) — the physics
compute shaders below, plus `agents.wgsl`/`environment.wgsl` (the
evolved policy's forward pass and its GPU-resident chemical field),
which moved in here for the same single-source-of-truth reason once the
Python trainer's own hot training loop needed to run them on its own
wgpu device too, not just the browser — see `../trainer/training_sim.py`'s
own module docstring.

## What's here

- `clearGrid.wgsl` — zeroes the per-substep grid accumulator.
- `p2g.wgsl` — particle-to-grid transfer (APIC + MLS-MPM stress, corotated
  elasticity).
- `gridUpdate.wgsl` — momentum → velocity, gravity, damping, sticky
  boundary.
- `g2p.wgsl` — grid-to-particle transfer, F/Jp update, SVD-based
  plasticity clamp.
- `repulsion.wgsl` — particle-particle repulsion via a density field
  (`clearDensity`/`splatDensity`/`densityToTexture`/`applyRepulsion`),
  **always on, every substep** — a real, standard part of the simulation
  here, not an optional extra. It's what keeps particles from overlapping
  when new ones get placed right next to existing ones.
- `agents.wgsl` — the evolved policy's forward pass
  (specialized as stateless `Dense(128)` or eight-state `Dense(64)`, followed
  by concatenated logical heads), with its local frame rebuilt each step from
  the L2-clipped gradient of chemical channel 3 (gradient-based steering).
  Not part of the physics passes above — a training-loop concern, not
  MLS-MPM itself — but shares this directory so both consumers load the
  exact same shader.
- `environment.wgsl` — the GPU-resident chemical field `agents.wgsl`
  senses/writes. It contains both selectable lifecycles: transient
  materialization from cell-owned chemistry, and persistent ping-pong
  blur/decay followed by density-normalized addition of signed neural deltas. Both use the same Sobel sensing and
  toroidal domain.
- `constants.json` — the numeric constants (`GRID_N`, `DX`, `INV_DX`,
  `DT`, `PARTICLE_MASS`, `VOL`, `MAX_PARTICLES`, `FIELD_N`,
  `DEFAULT_SPLAT_RADIUS`, `DEFAULT_REPULSION_STRENGTH`) every consumer
  needs for template substitution and buffer sizing.
- `policy_parameters.json` — shared logical-head initialization priors,
  Xavier gains, and mutation-scale buckets used by Python training and browser
  randomization. The GPU ABI remains one concatenated output matrix.

## Uniform surface

Reduced from the sandbox's own: `Material` (mu0, lambda0, hardening,
yieldLow, yieldHigh), `activeCount`, `gravity`, `damping`, `SplatParams`
(sigma), `RepulsionParams` (strength). **No `Mouse` uniform** — there is
no interactive tool here, `gridUpdate.wgsl` only ever applies gravity +
damping + the sticky boundary.

## Relationship to `mls-mpm/`

This is an **independent copy-and-strip**, not a shared/refactored source
— `mls-mpm/src/gpu/*.wgsl` is untouched by this project and keeps its own
mouse-interaction (Move/Force/Attract-to-Point) and field-visualization
diagnostic channels, which this core has no use for. WGSL has no
`#include`; small duplication across self-contained shader files is this
project's own established convention (see `mls-mpm/src/gpu/p2g.wgsl` and
`g2p.wgsl`'s own header comments), not an oversight here either.

**Kept in sync by hand.** `constants.json`'s values, and every WGSL edit
made relative to the sandbox's own shaders, are derived from
`mls-mpm/src/gpu/mpm.ts` — that file is the numeric source of truth. If
`mls-mpm`'s own `GRID_N`/`DT`/`DEFAULT_SPLAT_RADIUS`/etc. ever change,
this folder needs a matching manual update; nothing here detects drift
automatically.

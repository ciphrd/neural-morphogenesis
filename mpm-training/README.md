# Neural morphogenesis with MLS-MPM

This project evolves particle-based organisms toward arbitrary target shapes.
Particles are simulated as an elastic material with MLS-MPM, carry a small
neural policy, sense the values and gradients of multiple chemical substrate
channels, and support two selectable chemical-memory architectures: a
diffusing/decaying persistent world field, or persistent cell-owned chemistry
projected into a transient world field before each brain invocation.

Growth treats MPM particles as material samples rather than cells. Each policy
emits one local 2-D growth vector, which is volume-weighted and smoothly splatted
onto the MPM grid. Its magnitude grows stress-free material continuously; new
samples are added only when needed to resolve the enlarged material.

See [`GROWTH_MODEL.md`](GROWTH_MODEL.md) for the design rationale and the
contrast with the former division-based abstraction.

The target training shape does not directly affect the simulation and is not
an input to the neural policy. It is used only to evaluate morphology during
evolution. The policy must discover chemical signaling and directional growth
behaviors that produce a good target match.

## Fitness

Selection uses a pose-invariant bounded raster score at 256×256 resolution.
Weighted Gaussian particle density is converted to occupancy with
`1 - exp(-density / reference)` and compared to the target at four scales. The
score separately measures missing coverage, outside spill, silhouette edges,
and crowding. A coarse-to-fine rotation search reaches roughly 2.5-degree pose
precision. Five snapshots across the final 10% of each rollout are combined as
70% mean and 30% worst-snapshot fitness, so stable detail wins without reducing
the entire late trajectory to one frame.

## Particle-density normalization

Particle count is a numerical sampling choice. Use `--particle-densities` to
train the same policy across relative sampling densities without hand-tuning
split spacing, chemical radius, repulsion radius, or particle mass:

```bash
cd trainer
.venv/bin/python evolve.py \
  --particles 400 --initial-particles 5 \
  --particle-densities 0.5 1.0 \
  --density-aggregation worst
```

`--particles` and `--initial-particles` are the reference counts at `1×`.
For multiplier `q`, spacing scales as `1/sqrt(q)`, counts scale as `q`, and
per-particle mass/rest area scale as `1/q`. Chemical and repulsion settings are
resolved from the same preset. The chemical kernel and gradient normalization
remain fixed in field-texel space; each particle contributes `1 / density` so
the fixed-resolution field observes a comparable continuum concentration.
A rollout-seeded world-space random field supplies lifecycle thresholds and
flat-heading division fallbacks, so changing q no longer changes stochastic forcing merely
because numerical particle slots were added or removed.
Workers allocate once for the largest requested cap and switch density through
runtime uniforms between rollouts.

The default remains `--particle-densities 1.0`, which resolves exactly to the
historical settings. The calibrated range is currently `0.5×`–`2×`; values
outside it require `--allow-unsafe-density`. With multiple densities, each
candidate sees the identical density/seed matrix and selection uses the worst
mean-per-density fitness by default. The viewer replays the representative
worst-density rollout and exposes a single `Particle density` selector; the old
individual controls remain under advanced physics overrides.

See `DENSITY_MODEL.md` for the scaling rationale and
`DENSITY_IMPLEMENTATION_PLAN.md` for validation details.

## What drives growth

Growth is deliberately split across four layers:

1. **Policies propose a local 2-D vector.** Magnitude is requested growth rate;
   direction is the preferred growth orientation.
2. **The MPM grid integrates the proposal.** The same quadratic B-spline kernel
   used by mechanics accumulates a represented-volume-normalized vector and
   positive growth tensor. More numerical samples therefore do not create more
   material.
3. **Continuum volume grows first.** G2P continuously exponentiates the smooth
   tensor into each sample's rest deformation `Fg`.
4. **Resampling restores precision.** Each sample carries a quadrature weight
   `q` and a transported material domain `H`. When a domain edge becomes too
   long, geometric bisection produces two child domains with half the weight.
   Mechanics, growth-field integration, chemistry and morphology integrate over
   the domains. APIC uses their full moment matrices. See `GROWTH_MODEL.md` for
   the numerical quadrature limits and independent physical area budget.

The policy therefore controls where and how strongly tissue wants to grow,
while the integration layer decides how that request is represented numerically.
Heading is
not policy state: the local frame always follows the gradient of chemical
channel 3. The morphology-gradient input lanes remain separate observations in
that channel-3-relative frame.

The current eight-channel policy has 30 inputs: 24 chemical value/gradient
components, morphology occupancy and its two heading-frame gradient
components, plus heading-frame elastic Hencky volume, axial, and shear
strain. Its shared 128-unit tanh trunk feeds chemical deltas, a two-component
local growth vector, and either RGB or recurrent state outputs. The heads remain concatenated into one
matrix for GPU inference, but use head-specific
initialization and mutation scales from `core/policy_parameters.json`.

The CLI `--mutation-sigma` remains the global evolution step size. Each output
head multiplies it by a fixed sensitivity scale:

| Parameter bucket | Initial bias prior | Mutation multiplier |
| --- | --- | ---: |
| shared trunk | zero | 1.00 |
| chemical delta rates | neutral | 0.50 |
| growth vector | zero-centered local vector | 0.20 |
| cell color | sigmoid = 0.50 | 0.50 |
| private-state residual (`recurrent`) | neutral | 0.20 |
| private-state gate (`recurrent`) | sigmoid ≈ 0.12 | 0.15 |

Small head-specific bias jitter prevents freshly initialized policies from
being identical at zero input. Lower mutation multipliers on persistent
direction and growth controls prevent a single mutation from causing a much
larger behavioral jump than an equally sized trunk mutation.

## Cell memory

Training supports two explicit behaviors selected with `--cell-memory`. Both
new variants keep the same 128-unit hidden layer so memory does not silently
change network capacity:

| Variant | Inputs | Hidden width | Outputs | Parameters at C=9 |
| --- | ---: | ---: | ---: | ---: |
| `none` | 33 | 128 | 14 | 6,158 |
| `recurrent` | 41 | 128 | 27 | 8,859 |

The recurrent controller adds eight private values to every particle. Each
communication round senses `tanh(state)` and emits eight residual candidates
and eight sigmoid gates. The update is `state += gate * tanh(delta) * dt`,
clamped to `[-4,4]`. A daughter inherits the updated state of its parent, while
a rollout reset zeros every state channel. Particle RGB is no longer an output
head in this variant: it is `sigmoid(state[0:3])`, making neural color a visible
projection of memory dynamics.

Structural chirality has deliberately not been added to these channels yet.
Under the existing optional mirror-average pass they are treated as ordinary
reflection-invariant scalars; no channel is assigned even/odd parity or paired
with another state channel.

New settings and checkpoints store the topology as `hiddenLayers: [128]`
(`hidden_layers` in Python metadata). `hiddenDim` remains as a compatibility
alias for the current one-hidden-layer evaluator. Legacy `stateful-64`
checkpoints retain their recorded width and replay unchanged.

The viewer exposes `Cell memory` and the current single hidden-layer width as
exploration controls. A shape change never reinterprets checkpoint weights:
it creates a fresh deterministic brain from the active rollout seed, rebuilds
the GPU policy, and labels the source `Seeded random`. Reset restores the
selected generation's trained brain.

Run a controlled paired experiment with identical evolution arguments and
seed using:

```bash
cd trainer
.venv/bin/python compare_policy_architectures.py --output comparisons/puddle -- \
  --target puddle --generations 50 --population 16 --workers 4
```

This writes isolated checkpoints and logs for both variants, plus
`summary.json` and a directly viewable `report.html`.

## Chemical communication architecture

Chemical memory is an independent run-level architecture selected with
`--chemical-communication-architecture`:

| Variant | Memory owner | Chemical-head meaning | Field lifecycle |
| --- | --- | --- | --- |
| `persistent-environment` | spatial environment | signed chemical delta rate | hold the field fixed through all neural rounds, then diffuse/decay once and add the final delta |
| `cell-owned-projection` | each cell | signed chemical delta rate | integrate per-cell chemistry, then clear and rebuild the sensed field from cell state every round |

The cell-memory axis is selected independently with `--cell-memory`.
For example, the earlier stateful persistent-field model is:

```bash
cd trainer
.venv/bin/python train_server.py \
  --cell-memory recurrent \
  --chemical-communication-architecture persistent-environment
```

The cell-owned model is the compatibility default. Both architecture
identifiers are stored in run settings and checkpoint metadata, and the viewer
rebuilds the matching GPU pipelines automatically when a run is loaded. Its
Rollout panel also offers a playback-only chemical-architecture selector for
comparing the same weights under either communication model.
Untagged legacy checkpoints are inferred from their recorded decay: positive
decay selects `persistent-environment`, while zero decay selects
`cell-owned-projection`.
Older recorded runs remain replayable with their recorded channel count and
weight shapes. A policy must be retrained to move from the former eight-channel
layout to the current nine-channel layout.

### Multi-scale channel layout

The production nine-channel system uses a data-driven `3 / 3 / 3` transport
layout from `core/chemical_channels.json`:

| Channels | Scale | Native grid | Cell delta timescale | Field delta timescale | Base-decay exponent | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0–2 | global | 64² | 24 | 24 | 0.05 | organism-wide coordination |
| 3–5 | regional | 256² | 10 | 10 | 0.50 | body regions and repeated structures |
| 6–7 | local | 512² | 3 | 3 | 1.00 | short-range signaling |
| 8 | local | 512² | 3 | 3 | 1.00 | local signaling |

Channels are packed into common field, gradient, and deposit buffers, so this
does not add agent-shader storage bindings. Each recorded channel profile owns
its resolution scale, cell-delta timescale, field-delta timescale, decay exponent,
diffusion multiplier, and deposit-sigma multiplier. Both temporal scales divide
the corresponding signed delta rate, letting slow channels accumulate gradually
instead of turning their output into a concentration target. The shader derives indexing and transport from generated arrays;
adding a scale or moving a channel between scales is therefore a configuration
change rather than another GPU architecture.

When local-density limiting is enabled, every texel accumulates a matched
chemical numerator `N` and represented-material density `D` from exactly the
same kernel weights. Its effective source is `N / max(D, D_capacity)`: raw
Gaussian deposition is preserved below capacity, while additional overlapping
material cannot amplify the source above the density-weighted mean. Persistent
decay and forcing are integrated with the exact constant-source leaky-system
factor `(1-r)/(-log(r))`, where `r` is that channel's retention for the tick.
The integer atomic scale is derived from the live particle-capacity and
projection bounds each round, retaining deterministic accumulation while
avoiding the former fixed 1/4096 quantization.

The live `decay` setting remains the local-channel retention. A channel with
exponent `a` retains `decay^a` per unit communication time, so coarse global
channels can be long-lived without removing the existing control. Sobel
gradients are converted back to the finest-grid convention before entering the
policy, preventing coarse channels from receiving artificially larger neural
inputs. New run/checkpoint metadata records the complete expanded profiles.
Configurations without profiles retain the old homogeneous spatial layout;
chemical-head outputs retain signed additive-delta semantics.

The three elastic lanes can be ablated without changing checkpoint dimensions
through `ELASTIC_STRAIN_INPUTS_ENABLED`; it is currently enabled.

## One macro step

New rollouts begin with one seeded particle (`INITIAL_PARTICLE_COUNT = 1`);
all additional particles must arise through policy-controlled division.

In simplified pseudocode:

```text
morphology = blur_and_normalize(particle_density)

communication_dt = communication_speed / neural_updates_per_macro

repeat neural_updates_per_macro communication rounds:
  if cell-owned projection:
    clear the transient substrate
    for each active particle:
      growth-deformed gaussian-splat particle.chemical_state into every substrate channel
  compute substrate gradients

  for each active particle:
    inputs = []
    alignment = channel_3_gradient / max(length(channel_3_gradient), 1)

    for each substrate channel:
        inputs += value at particle
        inputs += dot(gradient, alignment)
        inputs += dot(gradient, perpendicular(alignment))

    inputs += morphology occupancy at particle
    inputs += dot(morphology_gradient, alignment)
    inputs += dot(morphology_gradient, perpendicular(alignment))

    elastic_F = deformation_F * inverse(stress_free_growth_Fg)
    elastic_H = 0.5 * matrix_log(elastic_F * transpose(elastic_F))
    elastic_H_local = rotate_tensor_into_channel_3_gradient_frame(elastic_H)
    inputs += tanh(trace(elastic_H_local) / elastic_strain_scale)
    inputs += tanh((elastic_H_local.xx - elastic_H_local.yy) / elastic_strain_scale)
    inputs += tanh((2 * elastic_H_local.xy) / elastic_strain_scale)

    outputs = neural_policy(inputs)

    chemical_delta = clamp(chemical_output, -1, 1)
    if cell-owned chemistry:
      particle.chemical_state += chemical_delta * communication_dt / channel_delta_timescale
      particle.chemical_state = clamp(particle.chemical_state, -1, 1)
    particle.color = sigmoid(red_output, green_output, blue_output)

    if this is the final communication round:
      if persistent environment:
        domain-integrate a fixed-world kernel for the final signed chemical delta
      particle.world_growth_vector = rotate_to_world(tanh(local_growth_vector))

volume-weighted B-spline splat vectors and outer-product tensors to MPM nodes
normalize the field by represented material volume
when a transported domain edge becomes under-resolved:
  atomically allocate a sample slot
  bisect the longest material edge and use the two child centers
  copy material/policy state and halve original area and quadrature weight
  retain the smaller domains for moment-consistent APIC transfers

if persistent environment:
  diffuse and decay the frozen substrate once
  add the final neural round's deposits

propagate the expanded material-sample count to the simulation

repeat MLS-MPM physics substeps:
    splat the particle density field
    apply bounded short-range repulsion
    particle-to-grid transfer, including elastic stress
    update grid velocities
    grid-to-particle transfer
    update position, velocity, and deformation
    continuously exponentiate the gathered growth tensor into Fg
```

`neural_updates_per_macro` controls agent-state deliberation resolution, not
raw communication speed. `communication_speed` controls how much chemical,
memory, orientation, and persistent-field time elapses before one mechanical
tick. In persistent-environment mode every neural round sees the same frozen
spatial field and only the final output is deposited; in cell-owned mode the
projected chemistry can evolve between rounds. Raising the round count never
multiplies the total integration time.

A sample created during the agent pass participates after the host publishes
the updated active count. It begins running its own neural policy on the
following macro step.

## Historical per-sample growth geometry (superseded)

> The direction/spread scheme below documents the previous experiment. It is
> no longer used, and its policy checkpoints have a different output shape; see
> [`GROWTH_MODEL.md`](GROWTH_MODEL.md) for the active field-integrated model.

Policy growth uses the `F = Fe Fg` tensor machinery to physically expand the
continuum before adding numerical samples. The target area is determined by the
fan size, and the following emission step partitions that grown rest material.

Each particle stores a full 2D growth tensor `Fg` and uses the multiplicative
decomposition

```text
F = Fe Fg
Fe = F inverse(Fg)
```

where:

- `F` is the total current deformation;
- `Fg` is the stress-free growth deformation;
- `Fe` is the elastic deformation.

Only `Fe` is passed to the constitutive model. Consequently, elasticity
resists displacement from the particle's **grown** rest configuration instead
of continually trying to restore its original size.

Let

```text
g  = determinant(Fg)   # stress-free area multiplier
Je = determinant(Fe)   # elastic area multiplier
```

For a sample preparing an admitted growth event:

```text
log_area_per_substep = ln(2) / (growth_duration * substeps_per_macro)
compression = max(0, -ln(Je))
pressure_gate = 1 - smoothstep(compression_start, compression_stop, compression)
effective_log_area = log_area_per_substep * mix(1, pressure_gate, feedback_strength)

new_g = min(
    g * exp(effective_log_area),
    1 + emitted_sample_count
)

area_factor = new_g / g
strength = (1 - normalized_spread) * global_anisotropy
```

`growth_duration` is measured in macro/controller updates, not physics time.
Consequently, changing `substeps_per_macro` for numerical stability does not
change how many opportunities agents have to sense and communicate before
emission. A duration of 48 gives an uncompressed particle approximately
48 policy evaluations per doubling; zero disables growth. Its trained value is
the backend constant `GROWTH_DURATION_MACRO_STEPS` in
`trainer/simulation_settings.py`. The backend sends that initial value to the
viewer, whose Growth-panel slider is a playback-only override.

The default contact-inhibition thresholds are both `0.10`, selecting a hard
cutoff at 10% elastic areal compression. Setting the start below the stop
restores a smooth slowdown interval. The same gate suppresses new growth
hazard and prevents final emission while compressed. It never rolls back
accumulated `Fg`: a quiescent sample continues from the same state after pressure
release. `feedback_strength=0` is the exact pressure-independent compatibility
and ablation mode. Since `Je` is a continuum deformation ratio rather than a
raw neighbor count, this feedback remains meaningful across particle-density
multipliers.

With no directional signal, the rest deformation grows isotropically:

```text
Fg = sqrt(area_factor) * Fg
```

With a nonzero direction `n`, the incremental rest stretches are

```text
parallel      = area_factor ^ ((1 + strength) / 2)
perpendicular = area_factor ^ ((1 - strength) / 2)
```

At `strength = 0`, this is isotropic. At `strength = 1`, the entire area
increment is placed along the selected axis. The local axis is first rotated by
the channel-3-gradient heading, then transformed through the elastic rotation
before applying the material update.

The existing global anisotropy multiplier remains a playback compatibility
control: 0 forces isotropic rest growth, while 1 lets spread range continuously
from directional growth for a ray to isotropic growth for a full circle.

The tensor increment treats `n` and `-n` identically because an axial stretch
depends on `n n^T`. The sign is retained for one-sided sample placement.

Increasing `Fg` while initially holding `F` fixed makes `Fe` temporarily
compressive. The MLS-MPM constitutive force then expands the material toward
the new rest state. This is the mechanism that changes the organism's physical
shape before a new particle is inserted.

## Legacy directional fan emission (inactive)

The current neural direction is a signed vector transformed from agent-local
space through the channel-3-gradient frame. Unlike the old axial model, `v`
and `-v` grow into opposite regions. If that local frame is undefined, a
rollout-seeded spatial direction supplies an unbiased fallback.

For radial sample spacing `d`, normalized spread `s`, and the playback cap `m`:

```text
spread = s * m * 2*pi
angular_spacing = pi/3
sample_count = max(1, ceil(spread / angular_spacing))

for j in 0 .. sample_count:
    t = (j + 0.5) / sample_count - 0.5
    angle = direction_angle + spread * t
    new_position = source_position + d * [cos(angle), sin(angle)]
```

The `pi/3` step follows from chord spacing at radius `d`: six points cover a
full circle with neighboring points one nominal spacing apart. Using stratum
centers avoids duplicating the two endpoints at 360 degrees. Narrow cones emit
one point; a half-circle emits three; a full circle emits six. Positions wrap
around the toroidal domain.

Each event atomically reserves a contiguous block with a capped
compare/exchange loop. Concurrent fans can be partially truncated at capacity,
but the published active count never exceeds initialized storage.

Before emission, the source grows to stress-free area `1 + sample_count` using
the existing `F = Fe Fg` mechanics. Spread controls the rest-growth shape:
narrow cones expand strongly along their direction, while a full-circle fan is
isotropic. At the target area, that continuum material is partitioned into the
source plus the emitted baseline samples. Every sample receives `Fg = identity`
and `F = Fe`, preserving stress while conserving total rest area and mass.

The source position is not displaced by the partition. New samples inherit its
plastic state, APIC affine field, locally sampled velocity, chemical state,
color, and recurrent neural memory. If capacity truncates a fan, the source
retains the remaining un-sampled rest area. Only rendering starts new samples
at zero area and fades them to full size over `growth_duration`. Both source and
emitted samples receive the growth cooldown.


## What determines the final morphology

The visible organism is an emergent result of:

- **neural growth field:** where the integrated vector magnitude requests
  continuous rest-volume growth;
- **chemical feedback:** where particles write signals and how neighbors react;
- **directional material growth:** the tensor formed by neighboring growth
  vectors, with a global isotropy/anisotropy material control;
- **adaptive resampling:** where extra quadrature points are introduced as
  represented volume grows;
- **elastic relaxation:** how neighboring material accommodates new rest area;
- **plasticity:** which sufficiently large elastic deformations become
  permanent;
- **repulsion:** bounded local separation of overlapping particles;
- **damping and friction:** removal of kinetic energy during expansion.

There is no global rest-shape mesh. Each particle owns a local `Fg`, while
particles are coupled through the MLS-MPM grid. Neighboring particles can
therefore retain residual elastic stress when their growth tensors or local
rest configurations are incompatible.

The growth vector is recomputed every macro step, while `Fg` continuously
accumulates the integrated history. Refinement placement prefers lower local
morphology density and uses the signed vector only as a fallback axis.

## Manual particle insertion

The front-end **Add particle** tool is a debugging interaction, not part of the
conservative biological growth law. It inserts a particle directly at the
clicked domain position with

```text
velocity = 0
F = identity
Fg = identity
C = 0
Jp = 1
cell_cycle_active = false
```

This creates mass immediately and only respects the viewer's playback particle
cap.

## Relevant implementation files

- [`core/agents.wgsl`](core/agents.wgsl) — sensing, neural policy, chemical
  writes, and local-to-world growth-vector publication.
- [`core/growthField.wgsl`](core/growthField.wgsl) — MPM-grid integration and
  conservative adaptive resampling.
- [`core/g2p.wgsl`](core/g2p.wgsl) — deformation update, plastic clamp, and
  tensor-valued `Fg` growth law.
- [`core/p2g.wgsl`](core/p2g.wgsl) — effective grown mass/volume and elastic
  stress evaluated from `Fe`.
- [`viewer/src/gpu/simulation.ts`](viewer/src/gpu/simulation.ts) — browser
  macro-step ordering and growth-count propagation.
- [`trainer/training_sim.py`](trainer/training_sim.py) — headless training
  rollout with the same ordering as the browser.
- [`trainer/simulation_settings.py`](trainer/simulation_settings.py) — current
  policy, material, growth, damping, and repulsion defaults.
- [`trainer/continuous_growth_check.py`](trainer/continuous_growth_check.py) —
  focused GPU checks for continuous growth, conservative resampling, and
  continued physics after a partially successful final allocation reaches capacity.
- [`viewer/capacity-check.html`](viewer/capacity-check.html) — open
  `/capacity-check.html` on the viewer dev server to run the browser controller,
  chemistry, and physics through 8,943 samples and verify finite positions during
  continued playback at capacity. The page reports PASS or the first failure.
- [`trainer/capture_policy_inputs.py`](trainer/capture_policy_inputs.py) —
  records exact raw policy inputs for stable particle slots during a headless
  rollout and writes an offline HTML dashboard plus its source JSON.

## Inspecting policy inputs over time

From `trainer/`, replay a checkpoint and track the first five stable particle
slots with:

```bash
.venv/bin/python capture_policy_inputs.py \
  --weights checkpoints/best.npy \
  --meta checkpoints/best_meta.json \
  --steps 500 \
  --sample-every 2 \
  --tracked 5
```

By default, the tool first tries freshly randomized policies for up to 400
macro steps each. It logs progress, keeps searching until one splits, saves
that policy beside the report as `policy_input_report.weights.npy`, then resets
and measures the successful rollout from step zero. Pass
`--no-search-for-split` to measure the supplied checkpoint directly instead.
Use `--initial-particles 5` to seed more cells during both the randomized
search and its measured replay; split detection then waits for the population
to rise above five. Without this option, `initial_particle_count` continues to
come from the checkpoint metadata, falling back to the shared
`INITIAL_PARTICLE_COUNT` in `core/constants.json` (currently 5) for metadata
that predates the field. Multiple initial cells are arranged as a
compact hexagonally packed disk with nearest-neighbor spacing equal to
`split_displacement`; partial outer shells are distributed around the boundary
rather than clumped on one side.

This writes `policy_input_report.html`, which opens directly in a browser, and
`policy_input_report.json`, which contains the same raw data for later scripts.
The dashboard provides the agent population at every macro step, per-particle
raw traces, a time-by-feature heatmap, and distribution statistics. Particle slots are stable: an unborn daughter is
shown as inactive until its slot is claimed, then its samples remain attached
to that same slot. Values are captured with the same bilinear field sampling,
heading-frame rotations, morphology blur, and elastic Hencky-strain transform
used by the live policy.

## Policy input normalization

The policy receives bounded, zero-preserving inputs calibrated from the
2,000-step live capture in `trainer/policy_input_report.json` (5,005 tracked
particle samples):

```text
chemical value[c]       = tanh(raw_value[c] / 0.17)
chemical gradient[c]    = tanh(raw_gradient[c] / 0.045)
morphology occupancy    = clamp(2 * raw_occupancy - 1, -1, 1)
morphology gradient     = tanh(raw_gradient / 0.018)
elastic strain          = unchanged (already tanh-normalized)
```

All chemical channels use the same value scale, and every forward/lateral
chemical gradient uses the same gradient scale. This preserves channel-
permutation as well as rotation symmetry instead of encoding the accidental
channel roles of the random policy used for measurement. The scales live in
`core/constants.json`; they are the rounded absolute P95 of the distributions
pooled across all channels (and both directions for gradients). A value equal
to its scale maps to approximately `0.762`, while large outliers approach `±1`
smoothly.

Chemical inputs influence growth only through the network; no chemical channel
has a hard-coded growth role.

New policy-input reports store both `raw_inputs` and normalized `inputs`; the
HTML dashboard's **Input space** selector switches every trace, heatmap, and
distribution table between them.

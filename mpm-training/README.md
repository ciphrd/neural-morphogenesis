# Neural morphogenesis with MLS-MPM

This project evolves particle-based organisms toward arbitrary target shapes.
Particles are simulated as an elastic material with MLS-MPM, carry a small
neural policy, sense the values and gradients of multiple chemical substrate
channels, and support two selectable chemical-memory architectures: a
diffusing/decaying persistent world field, or persistent cell-owned chemistry
projected into a transient world field before each brain invocation.

Growth uses a conservative morphoelastic **grow-then-divide** model. It does
not insert overlapping particles and rely on repulsion to create space.
Instead, a particle first accumulates stress-free rest area, elasticity moves
the material toward that enlarged rest configuration, and the fully grown
particle is finally replaced by two baseline particles.

The target training shape does not directly affect the simulation and is not
an input to the neural policy. It is used only to evaluate morphology during
evolution. The policy must discover chemical signaling and directional growth
behaviors that produce a good target match.

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
A rollout-seeded world-space random field also supplies initial headings and
lifecycle thresholds, so changing q no longer changes stochastic forcing merely
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

Three signals have distinct responsibilities:

1. **A dedicated neural output starts growth.** Its signed `[-1, 1]` value is
   clamped to `[0, 1]`; positive values are the per-macro-step probability of
   entering a cell cycle, while zero and negative values inhibit admission.
2. **Neural targets control persistent growth geometry.** Two bounded outputs
   propose a local growth direction, while a sigmoid proposes anisotropy; the
   stored angle and anisotropy relax smoothly toward them. Another sigmoid
   selects how strongly growth-directed division is polarized versus centered.
3. **The morphoelastic law supplies the amount of growth.** Once a cell cycle
   is active, a configured duration determines approximately how many mechanical
   macro steps it takes to double stress-free area. Elastic compression
   smoothly lengthens this duration and fully pauses growth above the
   configured arrest threshold; releasing compression resumes the cycle.

The policy directly controls cycle admission but not the amount of growth in an
active cycle. It may use any substrate values and gradients to choose its signed
division drive, heading, growth direction, anisotropy, and division polarity.

The current eight-channel policy has 30 inputs: 24 chemical value/gradient
components, morphology occupancy and its two heading-relative gradient
components, plus heading-relative elastic Hencky volume, axial, and shear
strain. Its shared 128-unit tanh trunk feeds seven logical output heads: nine
desired chemical-expression levels, a two-component desired heading, growth anisotropy,
division bias, a two-component desired growth direction, a signed division
drive, and three sigmoid RGB cell-color outputs (19 outputs total). The heads remain concatenated into one
matrix for GPU inference and checkpoint compatibility, but use head-specific
initialization and mutation scales from `core/policy_parameters.json`.

The CLI `--mutation-sigma` remains the global evolution step size. Each output
head multiplies it by a fixed sensitivity scale:

| Parameter bucket | Initial bias prior | Mutation multiplier |
| --- | --- | ---: |
| shared trunk | zero-centered, ±0.5 random bias | 1.00 |
| chemical expression targets | neutral | 0.50 |
| desired heading | local-forward | 0.20 |
| growth anisotropy | sigmoid ≈ 0.20 | 0.15 |
| division bias | sigmoid = 0.50 | 0.25 |
| growth direction | local-forward | 0.20 |
| division drive | neutral | 0.25 |
| cell color | sigmoid = 0.50 | 0.50 |
| private-state residual (`recurrent`) | neutral | 0.20 |
| private-state gate (`recurrent`) | sigmoid ≈ 0.12 | 0.15 |

Random trunk bias and head-specific weight gains make freshly initialized
policies behaviorally distinct even at neutral input, while the head bias
centers retain sensible population-wide defaults. Lower mutation multipliers
on persistent direction and growth controls prevent a single mutation from
causing a much larger behavioral jump than an equally sized trunk mutation.

## Cell memory

Training supports two explicit behaviors selected with `--cell-memory`. Both
new variants keep the same 128-unit hidden layer so memory does not silently
change network capacity:

| Variant | Inputs | Hidden width | Outputs | Parameters at C=8 |
| --- | ---: | ---: | ---: | ---: |
| `none` | 30 | 128 | 17 | 6,161 |
| `recurrent` | 38 | 128 | 30 | 8,862 |

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
| `persistent-environment` | spatial environment | desired expression level | hold the field fixed through all neural rounds, then diffuse/decay once and relax toward the final density-weighted target |
| `cell-owned-projection` | each cell | desired expression level | relax per-cell chemistry toward the target, then clear and rebuild the sensed field from cell state every round |

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

| Channels | Scale | Native grid | Cell relaxation time | Field response time | Base-decay exponent | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0–2 | global | 64² | 24 | 24 | 0.05 | organism-wide coordination |
| 3–5 | regional | 256² | 10 | 10 | 0.50 | body regions and repeated structures |
| 6–7 | local | 512² | 3 | 3 | 1.00 | short-range signaling |
| 8 | local | 512² | 3 | 3 | 1.00 | local signaling |

Channels are packed into common field, gradient, and deposit buffers, so this
does not add agent-shader storage bindings. Each recorded channel profile owns
its resolution scale, cell-relaxation time, field-response time, decay exponent,
diffusion multiplier, and deposit-sigma multiplier. The two temporal scales
control how quickly cellular expression and persistent substrate approach the
desired level. The shader derives indexing and transport from generated arrays;
adding a scale or moving a channel between scales is therefore a configuration
change rather than another GPU architecture.

When local-density limiting is enabled, every texel accumulates a matched
chemical numerator `N` and represented-material density `D` from exactly the
same kernel weights. Cell-owned projection uses `N / max(D, D_capacity)`.
Persistent-environment mode instead recovers the local desired level `N/D` and
mixes the transported field toward it; `D/(D+D_capacity)` controls how strongly
sparse kernel tails pull. This bounded target update can add or subtract and
cannot drift through repeated accumulation.
The integer atomic scale is derived from the live particle-capacity and
projection bounds each round, retaining deterministic accumulation while
avoiding the former fixed 1/4096 quantization.

The live `decay` setting remains the local-channel retention. A channel with
exponent `a` retains `decay^a` per unit communication time, so coarse global
channels can be long-lived without removing the existing control. Sobel
gradients are converted back to the finest-grid convention before entering the
policy, preventing coarse channels from receiving artificially larger neural
inputs. New run/checkpoint metadata records the complete expanded profiles.
Configurations without profiles retain the old homogeneous spatial layout.

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

    for each substrate channel:
        inputs += value at particle
        inputs += gradient along particle heading
        inputs += gradient perpendicular to heading

    inputs += morphology occupancy at particle
    inputs += morphology gradient along particle heading
    inputs += morphology gradient perpendicular to heading

    elastic_F = deformation_F * inverse(stress_free_growth_Fg)
    elastic_H = 0.5 * matrix_log(elastic_F * transpose(elastic_F))
    elastic_H_local = rotate_tensor_into_heading_frame(elastic_H)
    inputs += tanh(trace(elastic_H_local) / elastic_strain_scale)
    inputs += tanh((elastic_H_local.xx - elastic_H_local.yy) / elastic_strain_scale)
    inputs += tanh((2 * elastic_H_local.xy) / elastic_strain_scale)

    outputs = neural_policy(inputs)

    chemical_target = clamp(chemical_output, -1, 1)
    particle.chemical_state exponentially approaches chemical_target
      using communication_dt and the channel relaxation time
    if the local heading angle changed since the previous neural update:
      rotate the persistent world heading target by that angular change
    converge heading toward the world target with a first-order response,
      capped by the maximum turn rate

    raw_growth_direction = tanh(direction_x, direction_y)
    local_growth_direction = normalize_or_zero(raw_growth_direction)
    growth_anisotropy = sigmoid(anisotropy_output)
    division_bias = sigmoid(polarity_output)
    world_growth_direction = rotate(local_growth_direction, heading)
    particle.growth_direction = world_growth_direction
    particle.growth_anisotropy = growth_anisotropy
    particle.division_bias = division_bias
    particle.color = sigmoid(red_output, green_output, blue_output)

    if this is the final communication round:
      if persistent environment:
        growth-deformed gaussian-splat the final relaxed expression target
      growth_probability = clamp(remap(division_drive, division_chance_boost), 0, 1)
      decrement division cooldown

      if growth is enabled
       and population is below the particle cap
       and this particle is not already growing
       and its cooldown is finished
       and division_hazard crosses its persistent random threshold:
          particle.cell_cycle_active = true

if persistent environment:
  diffuse and decay the frozen substrate once
  move each supported texel toward the final density-weighted target

propagate any newly divided particle count to the simulation

repeat MLS-MPM physics substeps:
    splat the particle density field
    apply bounded short-range repulsion
    particle-to-grid transfer, including elastic stress
    update grid velocities
    grid-to-particle transfer
    update position, velocity, deformation, and stress-free growth
```

`neural_updates_per_macro` controls agent-state deliberation resolution, not
raw communication speed. `communication_speed` controls how much chemical,
memory, orientation, and persistent-field time elapses before one mechanical
tick. In persistent-environment mode every neural round sees the same frozen
spatial field and only the final relaxed target is applied; in cell-owned mode the
projected chemistry can evolve between rounds. Raising the round count never
multiplies the total integration time.

A daughter created during the agent pass participates in that macro step's
physics. It begins running its own neural policy on the following macro step.

Growth admission uses a persistent stochastic clock rather than discarding a
new Bernoulli draw every macro step. Before updating it, the bounded final-channel
signal advances the clock directly:

```text
p = clamp(last_chemical, 0, 1)
division_hazard += -log(1 - p)
threshold = exponential_random(mean=1)  # drawn once per prospective cycle

if division_hazard >= threshold:
    begin_cell_cycle()
    division_hazard = 0
    threshold = unset
```

Thus zero signal never advances the clock, weak or intermittent signal retains
its accumulated contribution, and saturated signal preserves immediate
admission. Parent and daughter clocks reset independently after division.

## Morphoelastic deformation model

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

For a particle in an active cell cycle:

```text
log_area_per_substep = ln(2) / (growth_duration * substeps_per_macro)
compression = max(0, -ln(Je))
pressure_gate = 1 - smoothstep(compression_start, compression_stop, compression)
effective_log_area = log_area_per_substep * mix(1, pressure_gate, feedback_strength)

new_g = min(
    g * exp(effective_log_area),
    division_area
)

area_factor = new_g / g
strength = neural_growth_anisotropy * global_anisotropy
```

`growth_duration` is measured in macro/controller updates, not physics time.
Consequently, changing `substeps_per_macro` for numerical stability does not
change how many opportunities agents have to sense, communicate, and reorient
before division. A duration of 48 gives an uncompressed particle approximately
48 policy evaluations per doubling; zero disables growth. Its trained value is
the backend constant `GROWTH_DURATION_MACRO_STEPS` in
`trainer/simulation_settings.py`. The backend sends that initial value to the
viewer, whose Growth-panel slider is a playback-only override.

The default contact-inhibition thresholds are both `0.10`, selecting a hard
cutoff at 10% elastic areal compression. Setting the start below the stop
restores a smooth slowdown interval. The same gate suppresses new
cycle hazard and prevents final mitosis while compressed. It never rolls back
accumulated `Fg`: a quiescent cell continues from the same state after pressure
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
increment is placed along the selected axis. The axis is transformed through
the elastic rotation so that rotating the organism rotates the growth response
without changing the material law.

The viewer's Growth panel exposes `global_anisotropy` from 0 to 1. It is a
playback-only multiplier: 0 forces isotropic, blob-favoring rest growth, while
1 preserves the neural policy's full per-particle anisotropy.

The tensor increment treats `n` and `-n` identically because an axial stretch
depends on `n n^T`. The sign is retained and used during division.

Increasing `Fg` while initially holding `F` fixed makes `Fe` temporarily
compressive. The MLS-MPM constitutive force then expands the material toward
the new rest state. This is the mechanism that changes the organism's physical
shape before a new particle is inserted.

## Conservative division and daughter placement

Division occurs when the parent's stress-free area reaches

```text
determinant(Fg) = 2
```

Daughter placement uses the persistent world-space direction selected by the
NN growth head. For split distance `d`:

```text
n = world_growth_direction
bias = division_bias * division_directionality

half_offset = n * d / 2
center_shift = bias * half_offset

parent_position   = old_position - half_offset + center_shift
daughter_position = old_position + half_offset + center_shift
```

With zero bias, the split is symmetric along the neural growth axis. With full
bias, the parent remains at the old position and the daughter is placed one
split distance toward the neural growth direction. Intermediate values smoothly
interpolate between those cases. Positions wrap around the toroidal simulation
domain. The same neural direction controls the stress-free growth tensor,
division placement, and optional strafe acceleration.

The viewer's Growth panel exposes `division_directionality` from 0 to 1 as a
playback-only polarization cap: 0 makes each growth-directed split
center-preserving and symmetric, while 1 permits the policy's division bias to
keep the parent in place and put the daughter the full split distance toward
the NN output.

Division conserves mass and rest area:

```text
before: one parent with det(Fg) = 2 and mass = 2 * base_mass
after:  two particles with det(Fg) = 1 and mass = base_mass each
```

Both daughters return to `Fg = identity`, while their total deformation is
set to

```text
daughter_F = parent_F * inverse(parent_Fg)
```

This preserves the parent's elastic deformation `Fe`, so stress does not jump
at division. The daughters also inherit the plastic state, APIC affine field,
heading, persistent world heading target, previous local heading target, and centered momentum. Both receive a division
cooldown and independent random-number state.

Visually, the existing daughter remains full-sized while the newly created
daughter emerges from zero size. Its visible area follows the same exponential
curve as stress-free volume growth, reaching full
size after one `growth_duration`. The renderer applies the square root of that
area fraction to particle radius. This appearance ramp does not alter physical
mass, deformation, stress, chemistry, or the conservative split described
above. Seeded and manually placed particles start at full size.

A cell's transient chemical substrate footprint is deformed by its stress-free
growth tensor `Fg`. Its projected area therefore expands continuously with the
represented material (and along the same axis for anisotropic growth). At
division the area-2 parent projection becomes two area-1 daughter projections;
the rendering-only newborn fade does not temporarily remove the daughter's
chemistry. Persistent internal chemical state remains unscaled.


## What determines the final morphology

The visible organism is an emergent result of:

- **neural growth admission:** where the policy produces positive division drive;
  the viewer's division-chance boost can continuously remap its signed range
  toward a full `[0,1]` probability range;
- **chemical feedback:** where particles write signals and how neighbors react;
- **directional rest growth:** the accumulated tensor `Fg` of each particle;
- **growth-directed division polarity:** whether the parent stays fixed or the
  daughter pair remains centered;
- **elastic relaxation:** how neighboring material accommodates new rest area;
- **plasticity:** which sufficiently large elastic deformations become
  permanent;
- **repulsion:** bounded local separation of overlapping particles;
- **damping and friction:** removal of kinetic energy after growth events.

There is no global rest-shape mesh. Each particle owns a local `Fg`, while
particles are coupled through the MLS-MPM grid. Neighboring particles can
therefore retain residual elastic stress when their growth tensors or local
rest configurations are incompatible.

The growth direction is recomputed every macro step. A particle can grow along
different axes during one cell cycle, and `Fg` accumulates that history.
Daughter placement is independent: it always uses the rear of the parent's
current heading.

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
  writes, cell-cycle admission, growth-directed division, and daughter initialization.
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
- [`trainer/growth_check.py`](trainer/growth_check.py) — analytical and GPU
  regression checks for growth and conservative division.
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

Chemical inputs influence division only through the network. A separate signed
policy output drives persistent division hazard, so no chemical channel has a
hard-coded growth role.

New policy-input reports store both `raw_inputs` and normalized `inputs`; the
HTML dashboard's **Input space** selector switches every trace, heatmap, and
distribution table between them.

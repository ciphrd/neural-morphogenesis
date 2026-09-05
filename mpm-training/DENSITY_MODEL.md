# Particle-density model

## Goal

Make one policy and one semantic simulation preset produce comparable
world-space morphology over a useful range of particle counts. Training should
be able to use a coarse particle sampling and playback should be able to use a
finer sampling without retuning an unrelated collection of physics and sensing
parameters.

This is a sampling-density problem, not a request to change the material's
physical mass density. Those two concepts need separate names in code.

## Primary control

Use a dimensionless density multiplier `q` as the public control:

```text
q = particle_number_density / reference_particle_number_density
s = particle_spacing / reference_particle_spacing = 1 / sqrt(q)
```

The current preset is the reference:

```text
reference_growth_sample_spacing h0 = 0.0027
reference density multiplier q = 1
```

For the approximately hexagonal packing used by `seed_blob`, the corresponding
absolute number density and area represented by one resting particle are:

```text
number_density(q) = q * 2 / (sqrt(3) * h0^2)
particle_area(q)  = 1 / number_density(q)
                  = (sqrt(3) / 2) * (h0^2 / q)
```

Absolute number density is useful in diagnostics, but `q` is a better run
setting: `0.25x`, `0.5x`, `1x`, and `2x` communicate the training cost directly
and do not expose a large domain-dependent number to the user.

`SPLIT_DISPLACEMENT` then becomes a derived value:

```text
growth_sample_spacing(q) = h0 / sqrt(q) = h0 * s
```

The particle cap and seeded population should describe the same initial and
maximum physical areas at every density:

```text
particle_cap(q)          = round(reference_particle_cap * q)
initial_particle_count(q) = max(1, round(reference_initial_count * q))
```

Rounding should be deterministic and the resolved values should be stored in
checkpoint metadata.

## Scaling categories

Every setting should declare one of these semantics. This is preferable to a
large formula that silently scales everything.

### 1. Particle-scale geometry

These quantities are lengths measured in cell spacings. They scale with `s`.

| Quantity | Proposed resolution | Current reference ratio |
| --- | --- | --- |
| emitted-sample radial spacing | `h0 * s` | `1.0 h` |
| repulsion splat sigma | `repulsion_radius_in_cells * h` world units | `1.481 h` |
| directional deposit offset, if restored | `deposit_offset_in_cells * h` | current `DEPOSIT_DISTANCE` is inactive |

The active chemical parameter is `DEPOSIT_SIGMA`, not `DEPOSIT_DISTANCE`.
`DEPOSIT_DISTANCE` is currently a legacy uniform slot and deposits are centered
under each particle. `DEPOSIT_SIGMA` is expressed in chemical-field texels.

Density model v1 scaled this already-sub-texel kernel with `h`: 0.458 texels at
q=0.5, 0.324 at q=1, and 0.229 at q=2. The fixed 256² observation grid cannot
represent that continuous shrinking-support argument; increasingly many cells
alias onto the same texels at high q. Model v2 instead treats chemistry as a
fixed-grid continuum observation:

```text
deposit_sigma(q)                  = reference_deposit_sigma
chemical_projection_weight(q)    = 1 / q
chemical_gradient_input_scale(q) = reference_chemical_gradient_input_scale
```

Thus q times as many particles publish at 1/q amplitude through the same
grid-resolved kernel. The GPU density regression verifies projected mass,
particle-sampled signal, represented population N/q, and controlled spatial
spread across q={0.5,1,2}.

Density model v3 also removes particle-slot identity from the branching
process. Growth thresholds and the fallback emission direction used only where
the channel-3 gradient is flat sample a rollout-seeded 128x128 world-space random
field. Nearby numerical samples therefore share the same macroscopic stochastic
forcing at every q. After material emission, the source and new samples
advance a lineage-generation counter so the next threshold is a
new spatial-field layer rather than a child-slot hash.

This intentionally changes stochastic q=1 trajectories relative to model v2;
the mechanical and chemical q=1 constants remain unchanged. Sample sidecars
record `firstCapStep`, toroidal envelope width/height/area, RMS radius, and p95
radius so learned-policy comparisons measure spatial convergence, not only
final particle count.

### 2. Per-particle extensive quantities

One particle represents an area proportional to `h^2`, so these values scale
with `s^2 = 1/q`:

```text
particle_mass(q)   = reference_particle_mass / q
particle_volume(q) = reference_particle_volume / q
```

Mass and volume must scale together. Their ratio is the material mass density,
which should remain fixed. In P2G this also keeps the elastic stress weight,
grid mass, and momentum sampling consistent as particle resolution changes.
The current compile-time values are both `1.0`; keeping them fixed makes a
twice-as-dense sample contain twice as much physical material.

Any future per-particle source integrated over world area should use the same
`1/q` weight. Chemical projection now follows this rule explicitly because its
support is fixed on the numerical observation grid.

### 3. Continuum/world-scale quantities

These describe the same material or the same macroscopic observation at every
sampling density and should not scale:

- material `E`, `nu`, hardening, and elasticity;
- gravity, physical damping time, and communication time;
- growth area ratio, growth duration, and compression threshold;
- morphology blur sigma when it is intended to observe the same world-space
  boundary thickness;
- chemical field and MPM grid resolutions during the first implementation;
- policy morphology-gradient input scale when morphology blur remains a fixed
  world-space length.

Keeping the morphology blur fixed makes it an anti-aliasing/macroscopic sensor:
low-density particles look grainier before the blur, but the policy observes
approximately the same boundary scale.

### 4. Dimensionless quantities

Probabilities, blend fractions, friction retention factors, division
directionality, chirality, and normalized neural response rates remain fixed.

### 5. Numerical regularizers

Repulsion is not a continuum material law in the current implementation; it is
a density-gradient regularizer. With an unnormalized Gaussian whose sigma
scales with `h`, its gradient scales as `1/h`. Holding relative displacement per
physics step fixed suggests:

```text
repulsion_strength(q)  = reference_repulsion_strength * s^2
repulsion_max_delta(q) = reference_repulsion_max_delta * s
```

The current reference strength is zero, so this needs a nonzero calibration
experiment before becoming a guaranteed rule.

These rules should live in an explicitly named numerical-regularizer section of
the resolver, not be confused with the material model.

## Density-normalized scoring and diagnostics

The live evolutionary fitness uses the bounded weighted-occupancy raster in
`raster.py`. Per-particle weight is `1/q`; its summed density is calibrated from
the target mass and expected represented particle count before being saturated
with `1 - exp(-weighted_density / reference)`. Static cross-density baselines
must still verify residual sampling error.

Candidate particles carry a per-particle represented-area weight:

```text
candidate_weight = 1 / q
```

and use `candidate_weight * kernel` in the sum scatter. The bounded occupancy
and aligned debug raster use the same weighting. The outside-shape penalty is
already a mean over particles and remains comparable across densities.

Selection compares that bounded field at multiple scales and scores missing
coverage, spill, boundary disagreement, and crowding separately.

## Morphology-density sensing

The raw morphology source is also an unnormalized particle count field. Under
the proposed `splat_radius proportional to h` rule, both particles per area and
splat area cancel, so the interior raw-density distribution and
`MORPHOLOGY_DENSITY_REFERENCE` should be approximately invariant. Do not scale
the reference preemptively.

Instead, measure the median and quantiles of blurred interior density for the
same packed patch at each `q`. If residual discretization causes drift, resolve
the reference from that measured calibration curve.

## Resolution limits

The density multiplier cannot be arbitrary. A valid range must satisfy all of
these constraints:

- enough particles per occupied MPM grid cell for stable continuum sampling;
- growth-sample spacing and repulsion sigma resolved by the repulsion field;
- chemical sigma resolved by the chemical field and not dominated by its
  one-texel minimum footprint;
- bounded Gaussian kernels not materially truncated;
- enough initial particles to represent the seed without a qualitative topology
  change;
- particle cap remains within GPU and training budgets.

At the reference preset:

```text
MPM cell width / h             = 0.0078125 / 0.0027 = 2.89
repulsion texel width / h      = (1 / 512) / 0.0027 = 0.72
chemical texel width / h       = (1 / 256) / 0.0027 = 1.45
morphology blur sigma / h      = 0.01 / 0.0027 = 3.70
```

The chemical deposit sigma is only `0.324` texel at the reference density and
already uses a minimum one-texel-radius kernel. This is likely to set the coarse
and fine ends of the useful range before the abstract formulas do. The first
target range should therefore be conservative: `q in {0.5, 1.0, 2.0}`, adding
`0.25` only after the field-resolution tests pass.

## Proposed configuration API

Keep one reference preset and resolve it once at run construction:

```python
DensityPreset(
    multiplier=0.5,
    reference_spacing=0.0027,
    reference_particle_cap=150,
    reference_initial_particles=5,
    chemical_radius_in_cells=0.46875,
    repulsion_radius_in_cells=1.481481,
)

resolved = resolve_density_settings(reference_settings, density_preset)
```

`resolved` should contain every actual shader/host value, including the derived
particle cap, initial count, mass, volume, field-texel chemical sigma, and input
normalization. Training, viewer playback, rollout rendering, and checkpoint
metadata must consume the same resolved object. The viewer can show the density
multiplier as the normal control and put resolved values in a read-only
diagnostics section. Existing sliders can remain temporarily as explicit
advanced overrides, with a visible indication that an override breaks the
density preset.

The policy should not receive `q` as an input initially. If the normalized model
works, exposing density lets the network specialize unnecessarily and hides
remaining simulator coupling.

## Validation harness

Use paired rollouts with identical policy weights and randomness. For each
`q`, scale initial count and particle cap, then compare against the `q=1`
reference after converting particles to weighted occupancy.

Record at least:

1. weighted final occupancy raster and rotation-aligned raster distance;
2. occupied area, centroid, principal axes, perimeter, and component count;
3. active count divided by `q` over macro time;
4. distributions of chemical values/gradients and morphology inputs seen by
   the policy;
5. elastic strain, kinetic energy per represented area, and grid mass;
6. nearest-neighbor distance divided by `h`;
7. blurred interior morphology-density quantiles;
8. wall time and GPU memory.

Three diagnostic scenes should precede learned-policy tests:

- static hexagonal patches at each density, to validate chemical and morphology
  fields without growth or mechanics;
- an elastic patch under the same world-space load, to validate mass/volume
  scaling;
- deterministic uniform growth, to validate count, occupied area, and division
  geometry before policy feedback complicates the result.

## Training strategy

Once paired deterministic checks pass, sample `q` per rollout during evolution
instead of training one checkpoint at one density. Start with `{0.5, 1.0}` and
score weighted occupancy identically. Reserve `q=2.0` as a transfer test until
the policy is stable, then include it if affordable. Fitness aggregation should
penalize the worst density or a high quantile, not only average the densities;
otherwise evolution can sacrifice the expensive case.

The checkpoint should store the reference preset, supported density interval,
and all resolved settings used during training. A high-resolution replay is then
a declared sampling change, not a hand-tuned physics variant.

## Implementation sequence

1. Add a pure density resolver and checkpoint metadata, initially resolving
   exactly to today's `q=1` constants. Add unit tests for the formulas.
2. Add the validation harness and baseline snapshots before changing behavior.
3. Route split displacement, chemical sigma, repulsion sigma, initial count, and
   cap through the resolver.
4. Make particle mass and volume density-resolved in both Python and browser
   shader construction.
5. Add density weight to candidate fitness rasterization.
6. Calibrate chemical-gradient normalization and morphology density reference
   from static patches.
7. Calibrate nonzero repulsion with deterministic mechanics
   scenes.
8. Train across densities and only then replace the ordinary viewer sliders
   with the density control plus advanced overrides.

The key acceptance criterion is not bitwise trajectory equality. Particle
systems at different samplings will diverge microscopically. Success means the
weighted world-space morphology, policy-input distributions, and continuum
mechanics remain within declared tolerances over the supported density range.

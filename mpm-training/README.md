# Neural morphogenesis with MLS-MPM

This project evolves particle-based organisms toward arbitrary target shapes.
Particles are simulated as an elastic material with MLS-MPM, carry a small
neural policy, sense the values and gradients of multiple chemical substrate
channels, and write back into those channels directly under their current
positions.

Growth uses a conservative morphoelastic **grow-then-divide** model. It does
not insert overlapping particles and rely on repulsion to create space.
Instead, a particle first accumulates stress-free rest area, elasticity moves
the material toward that enlarged rest configuration, and the fully grown
particle is finally replaced by two baseline particles.

The target training shape does not directly affect the simulation and is not
an input to the neural policy. It is used only to evaluate morphology during
evolution. The policy must discover chemical signaling and directional growth
behaviors that produce a good target match.

## What drives growth

Three signals have distinct responsibilities:

1. **The last substrate channel starts growth.** Its value at a particle is
   clamped to `[0, 1]` and treated as the per-macro-step probability of
   entering a cell cycle.
2. **The former neural strafe outputs direct growth.** Their two bounded values
   form a local growth vector, which is rotated by the particle heading into
   world space. Its direction selects the growth axis; its magnitude selects
   how anisotropic growth is.
3. **The morphoelastic law supplies the amount of growth.** Once a cell cycle
   is active, a configured exponential growth rate increases stress-free area.
   Elastic compression slows this rate continuously, without a hard mechanical
   cutoff.

The policy therefore does not directly output a scalar growth amount. It
controls growth indirectly by writing the growth substrate, reacting to all
substrate values and gradients, changing its heading, and selecting a growth
direction.

The current eight-channel policy has 13 outputs: eight chemical writes, one
turning output, two retained but unused acceleration outputs, and two growth-
direction outputs. Checkpoints from the previous four-direction deposit model
had 37 outputs and are not shape-compatible; training must be restarted for
this architecture.

## One macro step

In simplified pseudocode:

```text
for each active particle:
    inputs = []

    for each substrate channel:
        inputs += value at particle
        inputs += gradient along particle heading
        inputs += gradient perpendicular to heading

    inputs += wrapped position relative to the rollout spawn center

    outputs = neural_policy(inputs)

    deposit one chemical output per channel under the particle
    update heading from the turning output

    local_growth_direction = tanh(former_strafe_x, former_strafe_y)
    world_growth_direction = rotate(local_growth_direction, heading)
    particle.growth_direction = world_growth_direction

    growth_probability = clamp(last_substrate_value, 0, 1)
    decrement division cooldown

    if growth is enabled
       and population is below the particle cap
       and this particle is not already growing
       and its cooldown is finished
       and random_uniform() < growth_probability:
        particle.cell_cycle_active = true

merge deposits into the substrate field
diffuse/decay the substrate field

propagate any newly divided particle count to the simulation

repeat MLS-MPM physics substeps:
    apply bounded particle repulsion
    particle-to-grid transfer, including elastic stress
    update grid velocities
    grid-to-particle transfer
    update position, velocity, deformation, and stress-free growth
```

A daughter created during the agent pass participates in that macro step's
physics. It begins running its own neural policy on the following macro step.

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
compression_scale = clamp(Je / compression_reference, 0, 1)

new_g = min(
    g * exp(growth_rate * compression_scale * dt),
    division_area
)

area_factor = new_g / g
strength = clamp(length(growth_direction), 0, 1)
```

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

The most recent signed neural growth direction determines placement. For split
distance `d`:

```text
q = world_growth_direction
strength = clamp(length(q), 0, 1)

if length(q) is nonzero:
    n = normalize(q)
else:
    n = random_unit_direction()

half_offset = n * d / 2
center_shift = strength * half_offset

parent_position   = old_position - half_offset + center_shift
daughter_position = old_position + half_offset + center_shift
```

With zero direction strength, the split is symmetric around the old position
and uses a random axis. With full strength, the parent remains at the old
position and the daughter is placed one split distance along `+n`.
Intermediate values smoothly interpolate between those cases. Positions wrap
around the toroidal simulation domain.

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
heading, angular velocity, and centered momentum. Both receive a division
cooldown and independent random-number state.

## What determines the final morphology

The visible organism is an emergent result of:

- **spatial growth admission:** which particles encounter growth substrate;
- **chemical feedback:** where particles write signals and how neighbors react;
- **directional rest growth:** the accumulated tensor `Fg` of each particle;
- **signed division polarity:** where daughters are placed;
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
different axes during one cell cycle; `Fg` accumulates that history, while the
eventual daughter placement uses the latest signed direction.

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
  writes, cell-cycle admission, signed division, and daughter initialization.
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

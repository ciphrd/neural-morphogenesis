# Material-domain growth (model version 2)

The material grows continuously; numerical samples are added by subdividing
transported material domains. `GROWTH_REDESIGN.md` records the pre-implementation
review and derivation. `DOMAIN_IMPLEMENTATION_PLAN.md` records the implementation
and validation scope.

## Material and numerical state

Each sample carries `F`, growth tensor `G`, affine velocity `C`, numerical weight
`q`, original world area `A0`, and current half edges `H=[h1 h2]`. Its current
material domain is `x + H[-1,1]^2`. `F` is the constitutive deformation used with
`Fe=F inverse(G)`; plasticity/fluidity can modify it. `H` follows the actual grid
velocity gradient and is never changed by a constitutive clamp.

The shared ParticleRest record has 16 floats (64 bytes). Existing offsets are
retained: 0–3 are `G`, 4 is plastic `jp`, 5–6 are the policy growth vector,
7 caches the macro interval's initial `det(G)`, 8 stores `A0`, 9 is reserved,
10 is marker appearance, 11 is `q`, and 12–15 are row-major `H`. Several scalar
WGSL field names remain historical ABI names. Diagnostic readers accept old
12-float stored snapshots; live GPU records use 16 floats everywhere.

Grown rest area is `A0 det(G)` in world units. Mechanical mass and stress volume
remain `particleMass q det(G)` and `particleVolume q det(G)` in the existing
simulation normalization. Density presets scale mass and volume together.

Seed blobs retain their rotated hexagonal lattice centers. The corresponding
rotated lattice parallelograms tile without overlap. Rows use square domains.
Reset buffers **before** loading seed geometry. External legacy scenes with no
provided domains retain point transfers until the first growth-field pass
initializes a domain from `F` and target spacing; arbitrary input point clouds
are not thereby guaranteed to tile.

## Growth law

The policy proposes a world vector `u`; `r=|u|`, `d=u/r`, and `T=r d d^T` (zero
at zero rate). Domain-integrated quadratic B-spline projection computes the
represented-grown-volume-weighted mean tensor at each MPM node. Opposed vectors
retain their common expansion axis. There is no upper clamp on a lineage's
`det(G)` in this average.

G2P gathers the tensor over the domain, blends its anisotropy, applies contact
inhibition, and rotates it into the elastic frame before advancing
`G <- exp(dt Lg) G`. Its trace sets logarithmic area production. Hardwired
positive-tension redirection has been removed: sampling does not change the
material's growth command. Plasticity and fluidity retain their existing
constitutive behavior, independently of domain geometry.

## Subdivision

Refine when `4 max(|h1|², |h2|²) / targetSpacing² >= 1.75`. `splitDisplacement`
is the legacy settings name for target spacing; it is no longer an insertion
separation. Select the longest material edge. Splitting edge 1 gives:

```
x_minus = x - h1/2       x_plus = x + h1/2
H_minus = H_plus = [h1/2, h2]
q_minus = q_plus = q/2
A0_minus = A0_plus = A0/2
v_minus = v - C h1/2     v_plus = v + C h1/2
```

Copy `F`, `G`, `C`, chemistry, appearance and private policy state. Child domains
exactly partition the parent. The next split remembers which edge was already
halved; biaxial expansion therefore refines in two dimensions without a spatial
hash or morphology search. Passive stretching triggers the same process even
with zero growth command. Compressed grown material need not gain samples until
its spatial domain expands.

One bisection per existing sample is allowed per macro interval. New slots do
not execute the same commit pass. Atomic allocation enforces the hard capacity;
there is no region-ownership texture or candidate-placement arbitration.

## Transfers

All domain integrals initially use a shared 3×3 Gauss-Legendre rule. For a
quadratic grid basis `Ni`, the integrated normalized weight is `Wi=<Ni>_domain`
and its gradient is `<grad Ni>_domain`.

P2G deposits mass `m Wi` and affine momentum
`m Wi [v+C(xi-x)]`. Elastic force uses `-Vg (Pe Fe^T) <grad Ni>`. G2P fits its
affine state using

```
D = dx² I/4 + H H^T/3
v = sum_i Wi vi
C = (sum_i Wi vi (xi-x)^T) inverse(D)
L = sum_i vi <grad Ni>^T
H <- (I + dt L) H
```

The kernel moment `dx² I/4` plus domain covariance replaces the old constant
point inverse moment. The child covariance plus child-center separation equals
the parent covariance. Consequently subdivision does not add the uncompensated
APIC angular momentum of a separated point pair.

Gauss quadrature integrates these low-order moments exactly in real arithmetic.
The grid basis is piecewise polynomial, so subdivision does **not** preserve
every nodal value exactly across spline knots. Refinement improves that
integration error. GPU fixed-point atomics add rounding error. Basis gradients
are integrated directly for force and kinematics; the old MLS surrogate
`4/dx² * Wi * (xi-x)` is not reused for extended domains.

Chemical deposition integrates a fixed-world kernel over `H`, weighted by
represented grown area. It no longer enlarges that kernel with `G` as well.
Morphology/repulsion density and viewer mechanical diagnostics use domain
quadrature too. Rendered particle markers remain user-sized sample glyphs, not
literal outlines of material domains.

## Physical and numerical limits

`materialAreaBudget` is an optional maximum **grown rest area in world units**;
zero disables it. It is available in the viewer Growth panel and shared run
defaults, and is recorded in training metadata. `set_material_area_budget` and
`setMaterialAreaBudget` expose the same runtime control.

Each macro interval sums `A0 det(G)` and distributes remaining area in
proportion to the interval's starting material area. Each sample's exponential
increment is clipped to that allowance. Subdivision copies the starting growth
and halves `A0`, preserving the allowance. This conservative allocation can
underuse the budget for heterogeneous rates during one interval; it reallocates
next interval. The global area accumulator has 1e-8 world-area resolution and
signed-i32 headroom of approximately 21 world-area units. Budget tests allow
for that quantization; this is not arbitrary-precision enforcement.

Numerical capacity remains a **safety pause**, independent of the optional
physical budget. At capacity, the growth field is cleared before integration;
elastic motion continues. Failed bisections preserve the source and increment
`unresolvedSamples` in the counter buffer. The viewer reports that growth is
paused at the sampling limit. `AgentsGPU.unresolved_samples` and
`GpuSimulation.samplingStatus` expose it programmatically. A capacity-limited
rollout must not be interpreted as a resolution-converged physical endpoint.

## Validation and remaining limits

Run from `trainer/`:

```
.venv/bin/python material_domain.py
.venv/bin/python growth_resampling_math_check.py
.venv/bin/python growth_check.py
.venv/bin/python elastic_diagnostics_check.py
.venv/bin/python density_gpu_check.py
```

The domain suite covers split moments, CPU/GPU P2G agreement, affine G2P,
independent geometry transport, passive stretching versus compressed growth,
capacity, independent physical budgets, chemical/morphology projection,
material/policy-state inheritance, seed/reset behavior, and a free-growth run.
The diagnostic suite also compiles the viewer's rendering and field shaders.
Build playback with `npm run build` in `viewer/`.

On the implementation run (Apple M2 Max), the CPU nodal mass L1 error versus
24×24 quadrature fell from 0.129% to 0.0416% after bisection in the tested patch.
The GPU split changed projected chemistry by 0.0477% and morphology by 0.179%
in its smooth-field test. A 60×32-substep isotropic rollout reached rest area
11.0205 from 1, with 16 samples and positive finite domains, in approximately
1.3 seconds including host synchronization. These are specific benchmarks,
not general error bounds or a comparison against the previous runtime.

Independent affine domains may still develop gaps/overlaps under nonuniform
motion. Severe shear, coarse domains, fast motion relative to the macro
interval, and very small quadrature weights need further convergence studies.
Geometry uses the existing explicit time-step family and requires a suitable
CFL limit. There is no new fracture model, global remapping or coarsening.
The density smoke test checks its existing static/zero-command scenarios;
learned-policy morphology convergence across densities is not established.

This changes physical discretization and sample trajectories. New runs record
`growthModelVersion=2`; previous weights remain loadable, but old trajectory
snapshots are historical evidence rather than expected exact replay results.

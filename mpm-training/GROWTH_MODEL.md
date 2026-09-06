# Point-transfer material growth (model version 12)

The material grows continuously; numerical samples are added by subdividing
transported material domains. `GROWTH_REDESIGN.md` records the pre-implementation
review and derivation. `DOMAIN_IMPLEMENTATION_PLAN.md` records the implementation
and validation scope.

## Material and numerical state

Each sample carries `F`, growth tensor `G`, affine velocity `C`, numerical weight
`q`, original world area `A0`, and three explicit world-space vertices A/B/C.
Vertices are stored canonically in [0,1)² and advected directly. Position x is
derived from their centroid; vertices are never reconstructed from x or edges.
For geometry calculations only, E=[B-A, C-A] uses shortest periodic differences,
and current area is abs(det(E))/2. F is constitutive deformation, with
Fe=F inverse(G); plasticity/fluidity can modify it without changing vertices.

The shared ParticleRest record now has 20 floats (80 bytes). Offsets 0–11 retain
their previous meaning: G(4), jp, growth vector(2), initial det(G), A0, reserved,
appearance, q. Floats 12–13 hold A, 14–15 B, 16–17 C, and 18–19 are alignment
padding. WGSL packs A/B in the historical vec4 field `domain`, followed by
`vertexC` and `domainPadding`. All live shaders and buffers use the new stride;
diagnostic readers still accept historical 12/16-float snapshots as well.

Grown rest area is `A0 det(G)` in world units. Mechanical mass and stress volume
remain `particleMass q det(G)` and `particleVolume q det(G)` in the existing
simulation normalization. Density presets scale mass and volume together.

Seed blobs partition each cell of the original rotated hexagonal lattice into
two triangles along its shorter diagonal, producing equilateral triangles.
Row layouts divide squares into right triangles. The two samples are placed at
triangle centroids, with q=0.5 each; their total area and mechanical material
equal the original cell within float32 endpoint quantization. A lattice-corner
cache creates each shared corner once; all incident triangles copy those exact
coordinate bits. Geometry and positions use the same rotation.

The legacy `initialParticleCount` / `--initial-particles` setting now denotes
seed cells: N cells emit 2N triangle samples, ordered as pairs (2i,2i+1).
Capacity remains an actual sample limit. Startup clamps seed cells to
floor(capacity/2), retaining complete pairs; capacity below two is invalid for
rollout seeding. CLI validation requires the requested cells to fit. Fixed lab
layouts require sufficient capacity and report an error if they do not fit.
Density scaling still acts on seed cells, target spacing and mass/volume; the
extra factor of two in initial sampling is offset by half weights.

Seed spacing is not independently configurable. It is twice the resolved
refinement spacing, so new triangles begin near the steady-state post-split
scale at every sampling density.

Reset buffers **before** loading seed geometry and weights. Explicit scene
domains must carry `domain_geometry='triangle-vertices'` (Python) or
`domainGeometry:'triangle-vertices'` (TypeScript); untagged/legacy domain arrays are
rejected. Scene domain arrays contain six floats per triangle, [Ax,Ay,Bx,By,Cx,Cy],
all canonical finite positions. Loaders accept positive per-sample quadrature weights and initialize
A0 as `0.5*det(E)/det(F)`, rejecting nonfinite or nonpositive determinants.
Legacy point-only scenes gain a right triangle with undeformed area spacing²,
transformed by F, at the first growth-field pass. Arbitrary point clouds and
interactive additions are not thereby guaranteed to tile.

## Growth law

The policy proposes a world vector `u`, capped to unit length. Point-based
quadratic B-spline projection computes the represented-grown-volume-weighted
mean signed vector at each MPM node. Only then is it converted to
`T=r d d^T`, where `r=min(|mean(u)|,1)` and `d` is its unit direction (zero
tensor at zero rate). Equal opposite requests cancel; perpendicular requests
produce weaker diagonal growth. Additional identical samples do not amplify
the rate. There is no upper clamp on a lineage's `det(G)` in this average.

G2P gathers the tensor at the particle center, blends its anisotropy, applies contact
inhibition, and rotates it into the elastic frame before advancing
`G <- exp(dt Lg) G`. Its trace sets logarithmic area production. Hardwired
positive-tension redirection has been removed: sampling does not change the
material's growth command. Plasticity and fluidity retain their existing
constitutive behavior, independently of domain geometry.

## Subdivision

The refinement demand is

```
S = |AB|² + |BC|² + |CA|²
A_effective = S / (4 sqrt(3))
demand = A_effective / splitDisplacement²
```

Request refinement when `demand >= 1.75`. For an equilateral triangle,
`A_effective` equals its area. More generally `S/36` is the mean squared distance
from the centroid over the uniform triangle. The trigger therefore measures
spatial extent, including isochoric stretch, without reacting to rigid rotation.
It scales quadratically with size; `splitDisplacement` remains target sample
spacing in world units. This is a spatial-resolution criterion, not a rigorous
velocity interpolation error estimator or a constitutive strain limit.

Select the longest edge, using canonical endpoint ordering for identical edge
length calculations on both sides. Exact length ties use a global lexicographic
endpoint order, preventing cycles among three or more equally long edges.
For either child, longest-edge bisection reduces S to at most 3/4 of its parent's
value. Repeated refinement resolves a fixed stretched triangle; sustained shear
can continue to consume samples even without material growth.

Cyclically relabel its endpoints A,B and the opposite vertex C. With M=(A+B)/2, the two children
are (A,M,C) and (M,B,C), with preserved winding. A/B/C are copied unchanged;
midpoints use lexicographically ordered endpoints so both orientations of a
shared edge compute identical bits. Centroids are derived from the children:


```
x_minus = x - (B-A)/6       x_plus = x + (B-A)/6
vertices_minus = [A, M, C]
vertices_plus  = [M, B, C]
q_minus = q_plus = q/2
A0_minus = A0_plus = A0/2
v_minus = v               v_plus = v
```

Copy `F`, `G`, `C`, chemistry, appearance and private policy state. Child domains
exactly partition the parent. Since each child's `q` is half the parent's, every
bisection conserves the represented material. Edge matching uses a hash of
exact endpoint coordinates; morphology does not participate in refinement. The transported domain controls daughter placement, but
does not widen or otherwise modify particle-grid coupling.

Refinement is **conforming and staged**. Each macro interval:

1. Rebuild a GPU half-edge hash from the exact canonical endpoint coordinates.
   Match an edge only by exact endpoint equality; hash collisions are resolved
   by probing, never by treating nearby points as identical.
2. For each triangle, follow its longest edge to its neighbor. If that neighbor
   prefers another edge, follow the dependency. A path ends at a boundary edge
   or an edge longest for both incident triangles. Atomic pointer jumping takes
   at most ceil(log2(active samples)) dispatches to find these terminal groups.
3. Route all refinement requests to their terminal group. Reserve one slot for
   a boundary bisection or two slots for a matched interior pair, in one atomic
   capacity transaction. If the whole operation cannot fit, leave both sources
   unchanged. Independent groups can still succeed in that interval.
4. Commit all reserved groups. Both sides of an interior edge receive the same
   midpoint before any transport resumes. No T-junction is introduced.

Only terminal operations run in an interval; upstream requests are reevaluated
on the next interval. This advances longest-edge propagation in bounded stages,
with a conforming mesh after **every** stage, rather than allowing temporary
hanging edges while finishing a full recursive closure. A request can take
several macro intervals to resolve. Every original sample splits at most once
per interval; newborns start participating in the next interval.

The scratch buffer has a power-of-two edge hash with at least six slots per
allocated sample, five per-sample words, and one blocked flag (about 12.4 MB at
200,000 capacity). No CPU topology readback or quadratic neighbor search is
needed. Multiple requests sharing a terminal operation are coalesced.
Non-manifold edges with more than two incident triangles are ambiguous and
reported as unresolved rather than partially split. Matching coincident edges
uses geometric identity, not persistent material/vertex IDs; arbitrary overlapping
scene domains are not a supported manifold mesh.

## Minimum split weight

Every proposed daughter must have effective material weight
`0.5 * quadratureWeight * max(det(growthF), 1e-6) >= 1/32`.
This local condition applies to area and length splits, including both members
of a conforming shared-edge operation. An ineligible pair stays intact; its mass,
chemistry and domains are unchanged. Weight-limited refinement does not report a
capacity failure or disable growth. Genuine rest growth can restore eligibility.
Density presets scale base mass separately, so this weight is in preset-relative
material units. Existing samples already below the floor are not merged or removed.

Without growth, a weight-1 lineage can produce at most 32 descendants; a weight-0.5
lineage at most 16. Conformity can stop earlier. The initial threshold is an
experimental sampling choice in `MIN_CHILD_WEIGHT`, not a physical fracture law.
Run `cd trainer && .venv/bin/python split_weight_check.py` for boundary values,
conservation, shared-edge eligibility, and sustained extreme-stretch checks.

## Transfers

P2G mass and momentum now accumulate as f32 values using integer atomic
compare/exchange on their bit patterns. The buffer remains three 32-bit words
per node; grid update and the density renderer decode those words as floats.
No optional floating-point atomic feature is required. Diagnostic and growth
field accumulators retain their own independent fixed-point encodings.

The previous 1/4096 mass/momentum quantization produced artificial velocity
gradients at small quadrature weights: a force-free particle with q=0.001 at
4x density accelerated from speed 1 to about 9 over 256 substeps. The floating
transfer retains unit speed within 6e-6 in that reproduction. P2G also uses the
actual positive q, without manufacturing a minimum mass at q<1e-6. The dedicated
transfer regression covers 1x/4x sampling down to q=1e-7 and concentrated
momentum beyond the previous signed-i32 range. Float summation still has ordinary
order-dependent roundoff; compare/exchange contention can cost more than integer
addition under concentration. The existing velocity CFL guard remains a final
safety bound, not a guarantee of elastic stability or valid triangle geometry.


Physics uses the ordinary quadratic point-based MLS-MPM transfer. Each particle
evaluates one 3×3 grid stencil at its center. Effective mass and stress volume
remain weighted by `q det(G)`, so subdivision preserves their totals. G2P
reconstructs the standard APIC affine matrix with `D inverse = 4/dx² I` and
uses it to transport constitutive state:

```
v = sum_i Ni(xp) vi
C = 4/dx² sum_i Ni(xp) vi (xi-xp)^T
F <- (I + dt C) F
```

Geometry uses three additional quadratic 3×3 velocity gathers, one at each
stored vertex, from the same grid velocity buffer:

```
va = sum_i Ni(A) vi     vb = sum_i Ni(B) vi     vc = sum_i Ni(C) vi
A' = fract(A + dt va)  B' = fract(B + dt vb)  C' = fract(C + dt vc)
x' = periodic_centroid(A', B', C')
```

Each stored coordinate undergoes the same independent arithmetic regardless of
its local vertex index. The GPU tests require bit-identical shared corners over
128 nonlinear-flow steps and across all incident seed triangles over 256 steps,
including periodic crossings. Split tests require unchanged endpoint bits and
identical midpoints when the same edge is bisected in opposite directions.

This removes centroid/edge reconstruction drift; it does not remove float32
quantization of world positions themselves. Extremely small displacements can
round away. The periodic centroid, area and renderer assume local triangles:
each coordinate span is less than half the world period. Larger domains would
need explicit winding information to disambiguate their edges on the torus.

Particle velocity still uses the centroid-based point gather, not the average
corner velocity. This retains its gathered linear momentum, but means centroid
displacement is not generally dt times stored velocity in non-affine flow.
Relative to point advection, the position correction is
`delta = dt*(mean(va,vb,vc)-particleVelocity)` and changes orbital angular
momentum by `mass * cross(delta, particleVelocity)` with unchanged APIC C.
Exact angular-momentum conservation of the complete transport is therefore
not claimed; a full domain-aware transfer is separate work. Constitutive F
likewise remains based on the centroid gradient, not reconstructed from E.

The domain `E` is not integrated into P2G, the momentum/constitutive part of
G2P, growth projection, chemical deposition, morphology/repulsion density, or
viewer field diagnostics. Those paths use one weighted sample at `xp`.
Children copy the parent velocity and affine
matrix. Symmetric placement and halved weights preserve global mass, linear
momentum, and APIC angular momentum, although the nodal field can change at the
instant of refinement. Rendered markers remain user-sized sample glyphs.

In both Training and Lab, Rendering → Shape → Domain fully rasterizes the
actual stored triangles using the selected particle Color and Alpha. The vertex
shader unwraps B/C around A and emits only the four potentially visible
periodic copies for seam crossings. It reads GPU domains directly, with no CPU
readback or CPU-generated geometry. “Overlay particle domains” independently
draws crisp cyan one-device-pixel edges after bloom for every shape selection,
including Domain.

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
physical budget. At capacity, or when a complete requested pair cannot fit in
the remaining slots, the growth field is cleared before integration; elastic
motion continues. A spare slot may therefore remain unused. Failed operations
preserve their sources and increment
`unresolvedSamples` by the number of requests blocked at that terminal group.
Header word 2 reports `capacityBlocked`, independently of the sample count.
Deferred requests whose terminal dependency made progress are not counted as
capacity failures. The viewer reports that growth is
paused at the sampling limit. `AgentsGPU.unresolved_samples` and
`GpuSimulation.samplingStatus` expose it programmatically. A capacity-limited
rollout must not be interpreted as a resolution-converged physical endpoint.

## Validation and remaining limits

Run from `trainer/`:

```
.venv/bin/python transfer_stability_check.py
.venv/bin/python triangle_domain_check.py
.venv/bin/python vertex_transport_check.py
.venv/bin/python conforming_refinement_check.py
.venv/bin/python seed_blob_check.py
.venv/bin/python growth_resampling_math_check.py
.venv/bin/python growth_check.py
.venv/bin/python elastic_diagnostics_check.py
.venv/bin/python domain_render_check.py
.venv/bin/python density_gpu_check.py
```

The domain suite covers split geometry, point-transfer domain independence,
CPU/GPU P2G agreement, affine G2P,
independent geometry transport, growth-driven sampling versus passive deformation,
capacity, independent physical budgets, chemical/morphology projection,
material/policy-state inheritance, seed/reset behavior, and a free-growth run.
The diagnostic suite also compiles the viewer's rendering and field shaders.
Build playback with `npm run build` in `viewer/`.

On the earlier model-6 implementation run (Apple M2 Max), the GPU split changed
projected chemistry by 0.00458% and morphology by 1.29% in its elongated-triangle
smooth-field test. Halving triangle dimensions and target spacing reduced the
morphology change to 0.466%. A
60×32-substep isotropic rollout reached rest area 11.0205 from 1, with 8 samples
and positive finite domains. P2G remains one 3×3 stencil. G2P now evaluates four
3×3 stencils (one centroid and three vertices). Paired startup seeds also double
initial sample count compared with the original parallelogram model. No stable
wall-clock performance benchmark is claimed for the new model.

Shared vertices retain duplicate bit-identical coordinates, and neighbor splits
are coordinated using a temporary edge graph. This preserves an initially
conforming mesh; it does not repair cracks in historical scenes. Curved material
boundaries remain approximated by straight triangle edges. Direct velocity
sampling also uses the existing grid as-is, including zero velocity at empty
nodes; no velocity extrapolation or additional material support is introduced.
A sustained anisotropic-growth pressure probe still produces inverted thin
triangles after exhausting refinement capacity. Floating-point grid transfer
fixes an independent artificial-energy mechanism; it does not make this
capacity-limited geometry stress test pass or establish that every reported
explosion has the same cause.
Severe shear, coarse sampling, fast motion relative to the macro
interval, and very small represented-material weights need further convergence studies.
Geometry uses the existing explicit time-step family and requires a suitable
CFL limit. There is no new fracture model, global remapping or coarsening.
The density smoke test checks its existing static/zero-command scenarios;
learned-policy morphology convergence across densities is not established.

This changes physical discretization and sample trajectories. New runs record
`growthModelVersion=12` and `domainGeometry=triangle`; previous policy weights
remain loadable, but old trajectory
snapshots are historical evidence rather than expected exact replay results.

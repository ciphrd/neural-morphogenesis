# Triangular material domains

Status: model version 7 stores three explicit vertices per triangle and updates
them directly from grid velocities. The 80-byte rest ABI replaces model 5's
centroid/edge representation. Startup uses identical cached corner coordinates;
splitting copies existing endpoints and computes orientation-independent
midpoints. Squared-edge-sum demand now drives staged conforming neighbor
refinement (superseding the historical area-only scope below). `GROWTH_MODEL.md` documents the current implementation. The remainder
of this document retains the earlier investigation and design rationale.

## Recommendation

Implement triangles as an alternate refinement geometry while retaining the
current point-based MLS-MPM solver. This requires four geometry floats per
sample, just like the existing model, and no connectivity buffer. Agreed scope:
retain the existing area trigger. Length-triggered refinement below is background
for a possible separate experiment, not part of the triangle implementation.

Triangles do not by themselves provide shared vertices, a conforming mesh, or
gap-free transport under nonuniform motion. Model 5 samples velocity directly at
each corner, removing the independent centroid-extrapolation discrepancy, but
does not coordinate newly introduced subdivision midpoints with neighbors.
A connected triangle mesh would be a different design,
requiring vertex ownership, connectivity, compatible motion and refinement.

## Original representation and transport (models 4–5)

Keep the particle position x at the triangle centroid. Store a row-major matrix
E = [u v], where u = B-A and v = C-A, in ParticleRest.domain (floats 12–15).
Reconstruct unwrapped vertices locally:

```
A = x - (u+v)/3
B = A + u
C = A + v
area = abs(det(E))/2
```

Model 4 used `E <- (I + dt C_apic) E`. Model 5 instead samples quadratic grid
velocity separately at A/B/C, advects each vertex and rebuilds the centroid
and edge matrix. Constitutive F and growth G remain independent. The position
update changes, while the point-based particle velocity/C gather is retained;
see `GROWTH_MODEL.md` for the resulting angular-momentum limitation.
Preserve positive winding at initialization; diagnose collapsed,
inverted and nonfinite domains rather than treating an absolute determinant as
evidence that the geometry is healthy.

The binary record remains 64 bytes, but its meaning changes: never reinterpret
an old half-edge matrix as triangle edges. Add explicit model/geometry metadata
to scene and snapshot inputs and increment the growth model version for new
triangle runs. A build-wide geometry choice can avoid per-particle tags; a
mixed-geometry population would require a discriminator.

## Exact bisection

Compare squared lengths of u, v-u and -v. Cyclically relabel the endpoints of
the selected longest edge as A,B and the opposite vertex as C. Set M=(A+B)/2.
Replace the parent with ordered triangles (A,M,C) and (M,B,C). Cyclic relabeling
and this vertex order preserve winding. Tie-breaking should be deterministic
and identical in CPU/WGSL implementations.

The two centroids are `x-(B-A)/6` and `x+(B-A)/6`. Each child has half the area;
their union is exactly the parent and their interiors are disjoint. Recompute
each child's edge matrix from its ordered vertices: unlike parallelogram
bisection, the two children generally have DIFFERENT matrices.

Halve q and A0; copy F, G, C_apic, particle velocity, chemistry, appearance and
private policy state. Keep the current allocation, lineage RNG and capacity
handling. With copied velocity/C_apic and symmetric centroid positions, the
current point APIC mass, first moment, linear/angular momentum and quadratic
kinetic-energy accounting are preserved in exact arithmetic. Nodal transfers
can still change. Do not substitute velocity +/- C_apic*offset: with the current
point APIC moment this adds orbital angular momentum (see the existing
growth_resampling_math_check.py counterexample).

Compute geometry in the parent's local unwrapped coordinates; wrap only the
final child centroids. Independently wrapping vertices corrupts seam-crossing
edge lengths and areas. At capacity, leave all parent state intact.

## What should trigger subdivision?

The existing code triggers on current area, not edge strain:
`4*abs(det(H))/spacing² >= 1.75`. The direct triangle equivalent is
`0.5*abs(det(E))/spacing² >= 1.75`. Longest-edge selection controls the cut,
not whether subdivision happens. Area-preserving elongation stays unsplit.

For literal length-sensitive refinement, evaluate
`area/targetArea >= 1.75 OR maxEdge > maximumEdgeLength`. Calibrate length
against the actual seed shape: an equilateral triangle of area targetArea has
side length sqrt(4*targetArea/sqrt(3)), whereas diagonal halves of the current
seed cells have a different shape/area. Do not reuse 1.75 as a length threshold
without an explicit calibration.

This optional trigger changes a deliberate existing safeguard: the GPU suite
currently asserts that folded, long/thin domains do not amplify the population.
Length refinement can consume capacity under shear without material growth.
One cut does not necessarily lower both child maximum lengths; even an
equilateral parent leaves an original full-length side in each child. Keep the
one-bisection-per-macro-interval limit and measure unresolved length demand,
small weights, degeneracy and capacity use before enabling it by default.
An aspect-ratio-only criterion is unsuitable as the sole stopping rule because
uniformly shrinking a bad shape does not change its aspect ratio.

## Initialization is the main integration decision

Implemented footprint-preserving baseline: split every existing seeded parallelogram
along its shorter diagonal into two triangles, placing particles at their centroids.
For hex-lattice rhombi this produces equilateral triangles.
Each receives half the original q and A0; total material and initial footprint
are preserved. This doubles initial particle count and changes point locations,
so it also changes projected fields and initial numerical resolution. Account
for capacity, density settings, deterministic lab indexing and seed count
semantics explicitly. Retaining spacing² as target area makes each triangle
start at half its source cell area and delays its first area-triggered split.

Scene loaders now accept explicit per-sample weights, and both seeders supply
q=0.5. Merely changing seed positions/domain arrays without those weights would
double mechanical material. Genuinely new single particles retain q=1; their
fallback triangle has area spacing² before deformation.

For a matched-count comparison, build a triangular tiling with approximately
one target area per triangle instead. That is a new seeder, with changed centers
and boundary selection; preserving every existing lattice center and footprint
is not achieved by replacing each cell with an arbitrary triangle. Legacy point
clouds without geometry can receive a canonical triangle of the prescribed area
transformed by F, but this does not establish a tiling.

## Implementation map

1. core/growthField.wgsl: replace determinant factor, fallback geometry,
   longest-edge selection and both daughter matrices. Preserve allocation and
   material/policy inheritance. The fallback A0 calculation also changes.
2. trainer/training_sim.py and viewer/src/gpu/rng.ts: matching triangle seeders,
   winding, weights, count semantics and deterministic orientation.
3. trainer/mpm_core.py and viewer/src/gpu/mpmCore.ts: scene weights and triangle
   A0 initialization, currently using `4*det(H)/det(F)`; define validation for
   singular/inverted scene inputs. Audit single-particle append/reset paths and
   viewer/src/gpu/types.ts SceneData. Initialization must not erase seed domains.
4. core/g2p.wgsl: direct vertex velocity sampling and consistent centroid/edge
   transport. P2G, growth projection, chemistry and morphology remain point transfers.
5. trainer/simulation_settings.py and run/scene serialization: version/tag the
   changed geometry and explicitly reject or convert old domain data. Previous
   policy weights may remain loadable, but trajectories will differ.
6. viewer/src/gpu/render.wgsl and render.ts: true-domain wireframe checkbox,
   drawing reconstructed vertices with toroidal handling, separately from the
   marker glyphs. Outlines remain crisp after bloom and follow camera zoom.
7. trainer/continuous_growth_check.py and seed/reset checks: replace square-domain
   fixtures; verify GPU/CPU split and trainer/viewer seed parity, material area
   budget, all inherited state, periodic seams, capacity and passive elongation.

## Validation completed and next gate

Run from the repository root:

```
trainer/.venv/bin/python trainer/triangle_domain_check.py
```

Passed 600 CPU bisections over randomized triangles and all three edge choices:
exact child vertices, signed half-area, centroid, covariance decomposition and
point APIC totals. Also checked affine transport, parallelogram conversion,
selection of the third edge, and the distinction between length/area demand.
These checks establish the geometry/algebra, not live shader correctness.

Before adopting the model, run GPU acceptance tests and viewer builds, then
compare both geometries at matched material area and sampling resolution under
isotropic expansion, directional growth, rotation and area-preserving shear.
Measure sample count, field jumps at splitting, domain quality, gaps/overlaps,
mass/momentum, runtime and capacity pauses. Triangle geometry is feasible; an
improvement in morphology or stability remains to be demonstrated.

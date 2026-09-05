# Growth and adaptive material sampling — review and discussion proposal

Review date: 2026-09-05. This describes the current working tree, including its
uncommitted changes. It proposes a model; it does not change the simulator.

**Recommendation:** retain continuous tensor growth and give every numerical
sample an explicit material domain. Add samples by subdividing that domain.
Make spatial field and momentum transfer consistent with the domain model.
Changing only the insertion direction will leave the underlying ambiguity.

## 1. What is implemented

The shared WGSL drives both training and playback. The relevant path is
`agents.wgsl` → `growthField.wgsl` → `p2g.wgsl` / `g2p.wgsl`.

For notation below, `q_p` means the shader's `quadratureWeight`, not the public
sampling-density multiplier also called `q` in `DENSITY_MODEL.md`.

| Stage | Current behavior | Location |
| --- | --- | --- |
| Proposal | Policy emits a local vector, rotated into world coordinates; magnitude sets rate, direction sets an expansion axis. | `core/agents.wgsl:1019` |
| Integration | Quadratic B-spline scatter of a positive semidefinite tensor, normalized by represented grown rest volume at each node; G2P gathers the nodal average. | `core/growthField.wgsl:362`, `core/g2p.wgsl:279` |
| Material growth | `F = Fe Fg`; stress depends on `Fe`. Update `Fg` by a matrix exponential after pulling the world growth tensor into the elastic rotation frame. | `core/p2g.wgsl:249`, `core/g2p.wgsl:385` |
| Mechanical feedback | Compression gates rate. Positive elastic Hencky strain redirects growth while retaining its trace. | `core/g2p.wgsl:389`, `core/g2p.wgsl:413` |
| Mass and stress weighting | Effective mass and grown rest volume are proportional to `q_p det(Fg)`. Growth adds mass at constant grown-rest density. | `core/p2g.wgsl:258` |
| Refinement trigger | Split when `max(q_p det(Fg), q_p sigma_max(F)^2) >= 1.75`. | `core/growthField.wgsl:158` |
| Placement | Blend growth and deformation directions; isotropic cases use a hash; directional cases search morphology coverage and continuation. | `core/growthField.wgsl:223` |
| Split | Centered pair at fixed separation `splitDisplacement`; halve weights, copy material state, set velocities to `v ± C offset`. | `core/growthField.wgsl:462` |
| Capacity | Reaching the numerical sample cap clears physical growth globally. | `core/growthField.wgsl:506` |

More explicitly, for a bounded vector `u_p`, define

\[
r_p=\|u_p\|,\qquad d_p=u_p/r_p,\qquad T_p=r_p d_p d_p^T,
\]

with `T_p = 0` at zero rate. The field is approximately

\[
\bar T_i=\frac{\sum_p w_{ip}a_pT_p}{\sum_p w_{ip}a_p},
\qquad T(x_p)=\sum_i w_{ip}\bar T_i,
\qquad a_p=q_p\det F_{g,p}.
\]

The scatter actually clamps `det(Fg)` to 8 before weighting, whereas P2G does
not apply that upper clamp. Under substantial accumulated growth, different
lineages can therefore have different relative weights in sensing growth and
in mechanics. Refinement halves `q_p` but does not reduce `det(Fg)`.

The grid is fixed. It carries the growth-rate field and mediates mechanical
motion; it is the material's rest configuration and spatial extent that grow.
The growth field is built before refinement and held over the macro step's
physics substeps. This is a temporal approximation that needs convergence checks.

The central choices are sensible: opposing vectors retain axial growth, local
averaging avoids multiplying the rate by sample count, growth is continuous,
and refinement no longer creates physical mass.

## 2. What remains mathematically incomplete

**A scalar area does not specify a spatial domain.** After halving `q_p`, the
trigger treats both principal squared lengths as halved. A true directional
subdivision halves one length and leaves the perpendicular length unchanged.
The code has no state in which to remember that distinction. Repeated sampling
therefore depends on hashes and coverage searches instead of subdivision geometry.

**The displacement is unrelated to the represented domain size.** Two samples
with different deformation and weight use the same separation. Neither child
is guaranteed to lie inside the material its parent represented. Region claims
arbitrate candidates; they do not define a partition of material or establish
nonoverlap. The claim is for the positive child location, not both moved sites.

**Conservation of totals is weaker than preservation of fields.** For a parent
at `x`, weight `a`, and split offset `δ`, the current point measure changes by

\[
a\,\delta_x\quad\longrightarrow\quad
\tfrac a2\delta_{x-\delta}+\tfrac a2\delta_{x+\delta}.
\]

Its total and centroid agree, but its second moment increases by
`a δ δᵀ`. For a smooth sensing kernel `K`, its projected field changes by
`(a/2) δᵀ ∇²K δ + O(|δ|⁴)`. Thus morphology, chemical projection, nodal mass,
and nodal stress contributions need not remain unchanged. Identical material
state and conserved total stress weight do not imply identical grid forces.

**The affine velocity correction has a conservation consequence.** The current
`v± = v ± Cδ` preserves linear momentum and the local affine velocity field.
It adds orbital angular momentum

\[
\Delta L=m\,\delta\times(C\delta).
\]

With the current quadratic APIC stencil, its internal moment matrix is
`D = Δx² I/4`. Copying `C` to two half-mass particles retains the old summed
internal APIC angular momentum, so it does not compensate for this addition.
For rigid rotation at angular rate `ω`, the increase is `m ω |δ|²`.
Translational kinetic energy also increases by `m |Cδ|²/2`; the unchanged
internal affine contribution cannot offset it. These are algebraic observations
about the split, not a measurement that they dominate current rollout failures.

**The stored `F` is not a pure record of transported geometry.** G2P first
advects it, then changes it through fluidity relaxation and singular-value
clamping. Using it as the complete spatial footprint loses deformation that
has been treated as plastic. A domain must follow actual motion independently.

**Two physical choices are mixed with numerical repair.** Tensile redirection
changes the material law, even if the area rate is unchanged. Likewise, a sample
cap changes the amount of physical material that can grow. Neither should be
needed to make a resampler work. They can be deliberate model choices, but
should have separate controls and meanings.

The forced inward-radial Lab field also applies an **isotropic** growth tensor
(`enforceGrowthField`), despite displaying radial arrows. It does not test pure
radial anisotropic growth. Incompatible growth can legitimately generate stress;
it does not establish that a seam must appear. Fracture requires its own model,
and loss of MPM coupling can instead produce numerical tearing.

## 3. Physical model to retain and make explicit

Start with a two-dimensional growing elastic continuum. Treat plasticity and
fluidity as subsequent constitutive extensions so the geometry can be validated
without their state bookkeeping obscuring it.

Let `X` be a material label, `x=φ(X,t)` its position, `F=∇X φ` its total
deformation, and `G` its growth tensor:

\[
F=F_eG,\qquad J_g=\det G>0,\qquad J_e=\det F_e>0.
\]

For a material patch of original area `A₀`,

\[
A_g=A_0J_g,\qquad A=A_0\det F,\qquad
m=\rho_g A_g,\qquad \rho=\frac mA=\frac{\rho_g}{J_e}.
\]

`A_g` is grown stress-free area; `A` is occupied spatial area. Compression can
make them different. `ρ_g` is an explicit constant density per grown rest area.
This chooses actual material production, rather than swelling at fixed mass.
Mass and world-area units must be calibrated together; the present nominal
`VOL` should not simply be interpreted as geometric world area.

Retain the normalized tensor field. Decide whether its average is per grown
rest area (matching the current intent), current spatial area, or mass. Use
grown rest area initially, without a lineage-dependent determinant cap.

Let `B(x,t)` be the resulting world-axis tensor with units of inverse time,
after the isotropy blend and any explicitly selected inhibition. With `R_e`
the elastic polar rotation, define the constitutive orientation convention

\[
L_g=R_e^TB R_e,\qquad \dot G=L_gG,\qquad
G^{n+1}=\exp(\Delta tL_g)G^n.
\]

This is a rotation-based growth law, not a full tensor pullback through elastic
stretch. It is a reasonable objective convention and matches current intent.
It guarantees

\[
\frac{d}{dt}\log J_g=\operatorname{tr}B,\qquad
\dot m=m\operatorname{tr}B.
\]

For uniform areal rate `γ`, isotropic growth gives `G=e^(γt/2)I` and
`A_g=A₀e^(γt)`; uniaxial growth gives `G=diag(e^(γt),1)` with the same area.
Actual positions follow momentum balance, not a direct growth displacement.

For the mass source `s=ρ tr(B)`, explicitly choose new material to arrive with
the local velocity. Then

\[
\partial_t\rho+\nabla\cdot(\rho v)=s,
\qquad
\partial_t(\rho v)+\nabla\cdot(\rho v\otimes v)
=\nabla\cdot\sigma+\rho b+s v.
\]

This explains why increasing mass while retaining velocity is appropriate for
this source model. A nutrient reservoir is optional future physics; the initial
model assumes an available supply. For a material-owned concentration `c`,
retaining `c` while mass grows assumes the added material inherits that
concentration. Fixed chemical amount would instead require dilution.

Use an elastic energy per grown rest area,
`E=∫_{Ω₀} J_g ψ(Fe) dA₀`, integrated over reference material.
During a mechanical step with growth held
fixed, this gives the familiar grown-volume-weighted elastic force. Growth
supplies mass and can supply energy; refinement supplies neither.

## 4. A geometric rule for adding samples

Represent each sample by an affine material domain

\[
\Omega_p=\{x_p+H_p\xi:\xi\in[-1,1]^2\},\qquad
H_p=[h_1\ h_2].
\]

Store its original area `A₀,p`, current half-edge matrix `H_p`, velocity and
affine velocity gradient, growth/constitutive state, and material-owned policy
state. In the purely elastic case, initialize `4 det(H_p)=A₀,p det(F_p)`.
The existing weight becomes the derived quantity `q_p=A₀,p/A_baseline`.

Transport the domain with the actual velocity gradient `L=∇v`:

\[
\dot H_p=L_p H_p.
\]

Do not advance `H` directly with growth, and do not shrink it when elastic
strains are clamped. Growth changes the rest state; mechanical motion changes
the spatial domain. Positivity-preserving integration or time-step control is
required, just as for other deformation state.

To bisect along the first material coordinate, use the parent's old `h₁`:

\[
x_\pm=x_p\pm\tfrac12h_1,\qquad
H_\pm=[\tfrac12h_1\ h_2],\qquad
A_{0,\pm}=\tfrac12 A_{0,p}.
\]

Copy `G` and piecewise-constant constitutive/intensive state; set
`v±=vp±Cph₁/2` and retain `C`. The children exactly partition the parent
parallelogram. There is no empty-space search and no independent spawn distance.
Splitting the second coordinate is analogous. Splitting both produces four
children at `x_p ± h₁/2 ± h₂/2`, each with `H/2` and a quarter of the weight.

For a constant parent state, mass, grown rest area, spatial area, centroid, and
the entire represented material density are unchanged. Children carry the same
affine velocity field restricted to their domains. Thus continuum linear and
angular momentum and kinetic energy are unchanged too.

The second-moment identity makes the difference precise. A uniform domain has
covariance `Σp=Hp Hpᵀ/3`. With `δ=h₁/2`,

\[
\Sigma_{\rm child}+\delta\delta^T=\Sigma_p.
\]

The new separation is paid for by smaller internal domains. The current point
split introduces separation without reducing any corresponding internal domain.

Refine when a transported domain is too large to resolve, using a world-space
target length and grid support bound. A simple first rule bounds
`2 max(||h₁||,||h₂||)`, splitting the longer material edge; assess shear and
aspect ratio as well. Bounds on area alone miss thin stretched regions.
Severe shear may eventually require remapping, not endless bisection.

An anisotropic split correctly remembers the unsplit width. Isotropic expansion
can use four children or deterministic successive edge bisections. Passive
stretch triggers the same geometric rule with zero local growth command.
Neither operation needs growth direction or a morphology gradient.

Initialize a genuine partition, for example a clipped rectangular patch grid
for the first prototype. Arbitrarily assigning equal squares to the existing
hexagonal seed points does not produce a partition. Independent affine domains
can later develop gaps or overlaps under nonuniform deformation; subdivision
is exact within a parent but does not solve that inter-domain problem. Shared
corners or a more general domain representation are possible later extensions.

## 5. Transfers must agree with the representation

The geometric split alone is not a conservation fix for the current APIC code.
For a grid basis function `N_i`, the domain interpretation suggests

\[
m_i=\sum_p\int_{\Omega_p}\rho_p N_i(x)\,dx,
\quad
(mv)_i=\sum_p\int_{\Omega_p}\rho_pN_i(x)
[v_p+C_p(x-x_p)]\,dx.
\]

Stress uses corresponding domain integrals of basis gradients. Domain
subdivision preserves these integrals by additivity. A fixed world-space
sensing kernel integrated over the domains has the same property.

Exact domain integration is the mathematical target; corner formulas or finite
quadrature only approximate it and require error measurements. A CPDI-style
formulation is a relevant implementation family, not a plug-in replacement for
the current MLS-MPM formula. Its moment matrices, velocity reconstruction,
stress discretization, stencil size and stability must be derived together.
Do not retain `D⁻¹=4/Δx² I` after changing the transfer kernel without derivation.

Two reasonable implementation levels are:

| Level | Benefit | Limitation |
| --- | --- | --- |
| Domain-guided placement with current point transfers | Small prototype; removes arbitrary split geometry. | Nodal fields and APIC angular momentum can still jump. Measure this; do not call it fully conservative. |
| Domain-consistent transfers and subdivision | Defines refinement as a change in integration resolution with additive fields. | Larger physics change, potentially wider stencils, new transfer validation. |

I recommend a small CPU reference for the second level before choosing the GPU
implementation. Global Voronoi or optimal-transport redistribution is another
option, but it needs a reconstructed material boundary and conservative remap
of deformation, chemistry and history. Arbitrarily averaging `G` and `F` can
change stress and energy. Local subdivision is the clearer first design.

Before restoring fluidity/plasticity, explicitly separate total transported
deformation from elastic constitutive state. A full `F=Fe Fplastic G` model is
one option, with a stated ordering and plastic-volume law; the existing scalar
`jp` plus corrected `F` is insufficient to recover a general transported domain.

## 6. Budget, validation and discussion

Use separate parameters for maximum physical grown area `Σ A₀,p det(Gp)` and
maximum numerical sample count. At numerical capacity, report unmet resolution
demand and either stop the run as unresolved or use an explicit coarsening
strategy. Continuing growth can be acceptable only while error bounds remain
acceptable. A physical growth budget can inhibit growth independently of the
chosen sampling resolution.

Before learned-policy rollouts, test:

1. Repeated subdivision with growth and dynamics disabled: domain area, total
   mass, centroid, covariance, integrated density, nodal mass and stress force.
2. Constant velocity, rigid rotation, affine shear: linear/angular momentum
   and kinetic energy immediately before and after refinement.
3. Prescribed homogeneous isotropic and uniaxial deformation: exact expected
   domain tiling; separately test freely evolving growth and elastic relaxation.
4. Passive stretch with no growth: unchanged physical mass, increasing resolution.
5. Compression and prescribed rigid rotation: no artificial birth, no spurious
   axis preference; rotate both geometry and commands for an objectivity test.
6. A smoothly varying deformation and a thin neck: measure domain gaps/overlap
   and coupling before asserting absence of numerical tears.
7. Spatial and temporal convergence: vary sample scale, MPM grid scale and
   macro-step duration independently, at the same physical growth budget.
8. Restore chemistry, policy history and then plasticity/fluidity; measure
   field discontinuities and stress changes introduced by transfer/remapping.

The existing `trainer/continuous_growth_check.py` completed successfully on
Metal (Apple M2 Max): all 14 checks passed. This validates its current assertions,
not spatial-field invariance or a resolution-converged growth model. The
companion `trainer/growth_resampling_math_check.py` checks the split's moment
identities and the point-APIC angular-momentum counterexample on the CPU.

The most useful discussion decisions are whether to adopt explicit domains,
whether continuous volumetric production is the intended biology, and whether
tensile redirection is desired behavior. My starting choices are explicit
domains, the current continuous production model, and tensile redirection
disabled in the baseline until the sampling method stands on its own.

## References and scope of the proposal

The equations and diagnosis above are this review's derivation for this code.
These primary sources establish relevant foundations; none is a claim that
this proposed implementation is already validated:

- Rodriguez, Hoger & McCulloch (1994), [Stress-dependent finite growth in soft
  elastic tissues](https://pubmed.ncbi.nlm.nih.gov/8188726/): continuum growth
  decomposition and stress-dependent growth laws.
- Ruggirello & Schumacher (2013), [A dynamic adaptation technique for the
  material point method](https://www.osti.gov/servlets/purl/1080392): deforming
  particle domains and adaptive splitting in CPDI; discusses numerical fracture
  when particle separation breaks grid communication.
- Jiang et al. (2015), [The affine particle-in-cell method](https://doi.org/10.1145/2766996),
  and Jiang, Schroeder & Teran (2017), [An angular momentum conserving APIC
  method](https://arxiv.org/abs/1603.06188): affine state and transfer conservation.
- [A Momentum-Conserving Implicit Material Point Method for Surface Energies
  with Spatial Gradients](https://arxiv.org/abs/2101.12408): includes a resampling
  approach designed for linear and angular momentum conservation in APIC.

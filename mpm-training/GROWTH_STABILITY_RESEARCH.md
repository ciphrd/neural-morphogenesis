# Stable growth for training — initial research

Research date: 2026-09-09. Based on the current working tree, including its
uncommitted changes, and primary literature linked below. This is a code audit,
analytical diagnosis, and experiment proposal; no new GPU rollouts or training
comparisons were run. Production code and settings were not changed.

The strongest starting direction is **slower, smoothly regulated growth, reliable
transport of developmental state, and an objective that rewards persistence**.
These address different problems. Slowing growth alone does not produce an
attractor, and a persistent controller cannot reliably compensate for numerical
loss of material–field correspondence.

## What the current implementation implies

### Exponential growth is built into the material law

`trainer/mpm_core.py:growth_rate_for_duration` sets the maximum rate from
`log(2) / (growthDuration * substepsPerMacro * DT)`. In
`core/g2p.wgsl`, the growth tensor is updated by a matrix exponential. For grown
rest area `Ag = A0 det(G)`, this gives

```
d log(Ag) / dt = trace(Lg)
d Ag / dt = Ag * trace(Lg)
```

Consequently, a constant positive command produces exponential rest-area and
mass growth. This is independent of whether numerical samples are split well.
Subdivision divides existing material; it is not the material source.

With the current `growthDuration = 12`, the idealized uncompressed, aligned,
constant-command case gives:

| Effective positive trace, relative to maximum | Rest-area doubling time | Area multiplier after 120 macro steps |
| --- | ---: | ---: |
| 1.0 | 12 | 1024 |
| 0.2 | 60 | 4 |
| 0.05 | 240 | 1.414 |

The maximum per-macro increase is about 5.95%. These are analytical values
calculated from current configuration, not predictions of an actual rollout:
compression, cancellation, boundary contraction, deformation, and capacity alter
the trajectory. World-space occupied area can differ from grown rest area.

Growth projection already averages signed commands by represented material
weight. Increasing sample count does not simply multiply the local rate.
`trainer/experiments/growth_output/comparison.md` reports much less growth after
vector-first cancellation for one existing checkpoint, but not improved trained
stability. The older single-weight experiment used a non-growing policy and
cannot establish sensitivity of the present growing system.

### The compression brake has a discontinuity

Current configuration sets `growthCompressionStart = growthCompressionStop =
0.1`. The shader therefore switches expansion completely on or off at this
threshold instead of using its available smooth transition. Compression is
measured from elastic deformation after singular-value clamping. Small strain
changes near the threshold can change the commanded expansion sharply;
clamping also means this feedback does not expose all geometric distortion.

This is a plausible sensitivity amplifier, not a measured explanation of every
burst. Signed contraction remains active when positive growth is inhibited.
Stress inhibition by itself also provides no reliable size signal for a freely
expanding body that can relax its stress.

### Substrate transport exists, but differs from material transport

`trainer/training_sim.py:macro_step` advances the persistent field before the
first neural read, using the preceding mechanical block. The field shader
backtraces once across the whole macro interval using the available MPM velocity
grid. Material vertices instead take quadratic velocity gathers on every
mechanical substep (`core/g2p.wgsl`). The substrate uses bilinear velocity
interpolation (`core/environment.wgsl:sampleMpmVelocity`).

This is not simply a missing advection step. It is an approximation to the
material trajectory: a final velocity field over an entire interval cannot
generally reproduce a changing velocity history. Bilinear resampling also
smooths transported patterns. Furthermore, `core/gridUpdate.wgsl` sets velocity
to zero on nodes without mass. Field values near and beyond a moving boundary
can therefore sample inadequate transport support. The mechanical per-substep
velocity guard does not establish accurate macro-scale chemical transport.

There is a second representation mismatch. Growth commands are integrated over
triangles, but chemical deposits use centroid point splats weighted by grown
rest area (`core/agents.wgsl:depositMaterialSample`). Splitting conserves total
represented area but moves deposition sites; nodal deposits need not be
invariant. Existing chemical transfer checks cover coincident splitting, which
does not establish invariance for actual separated child centroids.

These mechanisms are candidates for the observed lag. A passive-marker test
should distinguish transport error from neural rewriting of the pattern.

### Increasing neural rounds currently changes chemical forcing

`EnvironmentGPU.set_communication_timestep` divides decay/diffusion time by the
number of rounds, but deposit gain remains per tick. The shader likewise applies
a full chemical residual on each tick, in both persistent and cell-owned modes.
Thus increasing `neuralUpdatesPerMacro` is not a pure refinement of the same
coupled dynamics: it also increases the number of full neural writes.

Current chemical outputs are linear, unbounded residuals; `maxEnvWrite` is a
gain rather than a bound. Bounded growth vectors do not imply bounded chemical
state or a stable chemical feedback loop. A local linearization of an isolated
field update has the schematic form `delta c_next = (d I + a J) delta c`,
where `d` is retention and `J` is the derivative of neural secretion with
respect to chemical state. Decay does not stabilize arbitrary positive feedback.
Spatial transport and mechanics make the actual Jacobian larger and coupled.

At communication speed one, current decay e-folding times are approximately
106 macro steps for global channels, 21.2 for regional channels, and 10.6 for
local channels. These are passive decay times, not measured closed-loop
response times. In particular, `fieldResponseTime` divides the source gain;
it is not by itself a relaxation time toward a prescribed target concentration.

### The objective rewards a good moment, not sustained stability

`trainer/evolve.py:_aggregate_scores` returns the minimum late-window loss.
Snapshots normally span 90–100% of the horizon. A good transient can therefore
win despite poor neighboring or subsequent states. Capacity terminates a rollout
and is scored immediately. At step 200, less than 10% growth in **sample count**
also terminates the run. Sample count mixes physical growth and refinement,
so this can work against a slower growth curriculum.

The configuration contains stable-stop and settling settings, but the inspected
rollout loop reports `stableMatch=False`, `settling=False`, and terminates for
capacity, low sample growth, or horizon. An explicit `growth_steps` cutoff can
disable further commands; that is different from learning to maintain shape
while the controller remains active.

## What carries over from NCAs

Growing NCA distinguishes policies trained only to reach a shape from policies
trained to persist and recover. It reuses evolved states in a training pool,
varies rollout length, and trains with damaged states. Its fixed lattice also
avoids transporting the lattice state through an independent mechanical solver.
These are useful design lessons, not guarantees that arbitrary NCA rules are
stable. [Mordvintsev et al., Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)

For this project, the important transfer is to make the target a region of
states the controller returns to after small disturbances. A frozen shape at a
hard growth cutoff does not demonstrate this. Neither does exactly replaying a
deterministic trajectory. Recovery should be tested with chemistry, growth,
and mechanics still active.

Growing-domain reaction–diffusion work derives the coupled effects of growth
and transport, and studies pattern formation when domain growth is slow.
This supports explicitly comparing developmental and growth timescales, rather
than assuming a chemical pattern can always keep up. Its one-dimensional,
slow-growth results do not establish stability of this MPM system.
[Crampin, Gaffney & Maini, 1999](https://people.maths.ox.ac.uk/maini/PKM%20publications/109.pdf)

Transporting a morphogen and scaling its pattern with a growing body are
different tasks. Work on dynamic gradient scaling examines advection,
diffusion, dilution, and degradation together. Increasing diffusion alone can
erase detail without maintaining relative coordinates across a larger shape.
[Fried & Iber, 2014](https://www.nature.com/articles/ncomms6077)

Mechanical homeostasis also needs a stability analysis: a study of growing
tubes finds that growth-response anisotropy affects stability. A separate
spheroid model adds energetic growth costs to obtain size control. These are
reasons to test homeostatic laws carefully, rather than assume that pressure
feedback ensures a finite stable shape.
[Erlich, Moulton & Goriely, 2018](https://arxiv.org/abs/1804.08107),
[Erlich & Recho, revised 2023](https://arxiv.org/abs/2011.00486)

## Changes worth testing, in order

### 1. Establish a fair persistence benchmark

Keep the best-snapshot score as a diagnostic. For selection, evaluate a mean
late-window shape loss plus an upper-tail or worst-window term, alongside a
post-target hold period. Begin with a modest window and increase it during
training. Retain active growth and chemistry during the hold. Also measure
longer rollouts, initially 2x and 4x the training horizon.

Score capacity termination as an unresolved numerical run rather than evidence
of stabilization. Replace the early sample-count cutoff for these experiments
with a physical-area/progress criterion or disable it. Use common seeds for
paired candidates, and several held-out seeds for finalists. The existing
rotating shared seed schedule is a useful foundation.

For CMA-ES, start by evaluating continuations of each candidate's own trajectory.
Later, a common pool of valid incumbent states can train recovery, but arbitrary
candidate-specific pools would confound comparisons. Pool snapshots must include
geometry, velocities, constitutive state, chemistry, and controller memory;
positions alone are insufficient.

### 2. Make growth gradual and the brake smooth

First sweep doubling durations of 12, 48, and 96 macro steps, with matched
developmental opportunity and enough capacity. Compare smooth compression
transitions against the current hard threshold; an initial experimental interval
could straddle 0.1, such as 0.05–0.15. These are ablation settings, not validated
defaults. Record gate switching and elastic distortion to determine whether
they help.

Keep exponential tensor integration: it preserves positive determinants in
exact arithmetic. Change the rate law around it. Optionally filter the proposed
tensor with an exponential moving average before application, using a defined
response time. Filtering reduces abrupt commands but adds feedback delay and
must be tested, especially for contraction.

To make growth amount predictable independently of current size, an explicit
area-supply cap is a clean diagnostic. If `a_p` is represented grown rest area
and `gamma_p+` is the positive spectral trace after local gating, calculate

```
P = sum_p a_p * gamma_p+
s = min(1, B / max(P, epsilon))
positive_growth_rate_p *= s
```

Here `B` is a total area-production rate, not a sample budget. This bounds the
instantaneous positive area source, replacing unconstrained exponential supply
with at-most-linear total production. For a finite timestep, enforce the budget
using predicted exponential area increments or sufficiently small steps;
the instantaneous formula alone is not an exact discrete budget guarantee.
Leave contraction separately controlled and bounded.

This global cap couples distant regions and is not a purely local NCA rule.
It is nevertheless useful for separating shape learning from uncontrolled
material supply. A fixed total supply alone does not determine where growth
stops. A finite reservoir or learned inhibitor would be a further design choice.

Boundary-only growth is another possibility: constant front speed makes a
roughly circular body's radius linear in time and its area quadratic, rather
than exponential. It can restrict interior remodeling and still does not select
a final size. It is a larger model change than the first experiments above.

### 3. Make chemical cadence and material transport consistent

Choose explicitly between a discrete NCA tick and a continuous-time neural
source. For the latter, scale chemical residuals by communication timestep;
subcycling should divide source increments as well as decay/diffusion time.
Then compare 1/2/4 communication rounds at matched integrated source strength.
For the discrete model, increasing ticks is a change of controller dynamics
and must be trained and compared as such.

For persistent fields, test advection during mechanical substeps or short
blocks, using velocity history as it becomes available and consistent spatial
interpolation. Transport support at material boundaries needs an explicit
extension policy. More frequent interpolation can also add diffusion, so judge
the result by marker accuracy and training throughput, not substep count alone.

The existing cell-owned chemistry option is a valuable independent ablation:
developmental memory follows material and is inherited by children. The grid
then mediates communication by projection. This removes the separate advection
path for that memory but retains projection error, and it changes the meaning of
the substrate. A hybrid of material-carried developmental state and persistent
external signals may ultimately fit the desired behavior best.

Define each channel's semantics before adding dilution. A material-carried
identity variable can satisfy `D_t c = reaction + diffusion` and be inherited
by new material. A conserved amount per spatial area instead obeys
`D_t c = reaction + diffusion - c div(v)`, with source terms chosen consistently
with newly produced material. Current signed neural latent channels are not
automatically physical concentrations. Adding dilution indiscriminately could
weaken the very developmental state intended to keep up with growth.

### 4. Learn recovery, then consider stronger local regulation

After the baseline is well behaved, train continuation from mild chemical noise,
small impulses, seed variations, and controlled geometry disturbances. Geometry
perturbations must preserve shared vertices and valid triangles. Introduce
damage only after ordinary persistence works. Penalize sustained deformation,
excess production, and temporal drift lightly enough that refusing to grow does
not beat constructing the target.

A local nutrient/inhibitor or maturation field could make growth decrease as
material matures. A diagnostic scale-separation condition is
`abs(gamma) * tau_signal << 1`: little relative growth during a signal response
time. For a transported front, also inspect `v_front * tau_signal / ell_signal`.
These are useful measured ratios, not universal thresholds or stability proofs.
For global diffusion, response time grows roughly as body length squared at
fixed diffusivity. Eventually a larger body may require hierarchical or
advected positional signals rather than merely stronger diffusion.

## Experiments that would resolve the main uncertainties

| Experiment | Controlled comparison | Main evidence |
| --- | --- | --- |
| Passive material marker | Disable neural writes, decay and intended diffusion; prescribe translation, rotation, expansion, then nonuniform flow | Field sampled along material trajectories; phase lag and amplitude loss; include boundary markers |
| Refinement-only chemistry | Fixed triangle and constant/linear marker, before and after actual conforming bisection | Change in nodal field and sensed values, with total area conserved |
| Prescribed growth | Constant or smooth spatial commands, with no learned feedback | Rest area vs occupied area, elastic strain, kinetic energy, distortion and refinement backlog |
| Chemical cadence | 1/2/4 rounds at matched source and elapsed time | Frozen-policy trajectory differences and cost; separate from deliberate extra NCA ticks |
| Controller continuation | Several actively growing checkpoints; identical seeds and small perturbations | Shape/chemistry drift and recovery while control stays enabled |
| Retrained ablations | Smooth gate, slower growth, transport changes, then persistence objective | Held-out late/long-horizon loss and failure rate at matched training compute |

Log physical grown area and occupied geometric area separately from sample count.
Track positive and negative production separately, capacity/refinement failures,
pruning losses, elastic strain quantiles, maximum domain extent, field norms,
and shape loss throughout each rollout. Compare at both matched elapsed time and
matched achieved area so that growing less is not mistaken for greater stability.

For perturbation sensitivity, use spatial fields or geometric distances when
refinement changes topology; same-slot particle comparisons become misleading.
Repeat small perturbations at multiple magnitudes and measure whether the
observable deviation contracts over a hold window. A stable morphology may
permit bounded internal dynamics or neutral translations/rotations, so define
stability in the same alignment frame used by fitness and also report raw drift.

The first implementation experiment should combine a passive transport
benchmark with a persistence evaluation mode. Those establish whether the
dominant problem is transport, an aggressive controller, or a training objective
that currently has little incentive to stay stable. Rate and gate ablations can
then be judged against that benchmark before redesigning the material model.

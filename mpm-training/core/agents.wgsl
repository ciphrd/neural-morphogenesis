// The evolved per-particle policy, GPU-resident — WGSL port of
// trainer/update_rule.py's stateless-128/stateful-64 dense policy variants,
// following envnca/frontend/src/gpu/agents.wgsl's own NN-forward-
// pass approach (weights as one flat buffer, plain loops rather than a
// matrix type, safeTanh on the OUTPUT layer's own squashing to avoid a
// real confirmed NaN failure mode — see safeTanh()) with the differences
// update_rule.py's own Python port already settled:
//
// Lives in core/, not viewer/src/gpu/, alongside p2g.wgsl/g2p.wgsl/etc
// — the single source of truth BOTH ../viewer/src/gpu/agents.ts (via
// Vite ?raw) AND ../trainer/agents_gpu.py (via
// ../trainer/shader_template.py's own load_core_shader(), same as
// mpm_core.py already uses for the other core/*.wgsl files) load their
// shader module from. This used to be viewer-only, with
// trainer/training_sim.py running an entirely separate torch/MPS
// forward pass — see trainer/training_sim.py's own module docstring for
// why that split was removed (it required a real, blocking GPU<->CPU
// host round-trip every macro step to bridge wgpu-native's own physics
// device and torch's own MPS/CUDA device, since the two share no
// buffers).
//
// - Local (heading-relative) frame, but heading is NOT derived from
//   velocity the way envnca's own agents.wgsl does it — that's an
//   undamped feedback loop (heading sets the local frame the network
//   senses/acts in, its action changes velocity, velocity *is* heading
//   next step, nothing smooths it) and was a real, confirmed source of
//   chaotic spin. Heading here is instead this shader's own persistent
//   per-particle state (the `heading`/`angularVelocity` buffers below),
//   driven toward the policy's desired heading by a proportional angular
//   controller and second-order angularVelocity integrator with damping —
//   accel -> velocity -> position already works, see agentStep()'s own
//   comments below for the exact integration.
// - No repulsion sampling. repulsion.wgsl already gives every particle a
//   real repulsion force as part of MpmCore's own physics — there's no
//   separate host-agent repulsion field to sense here the way envnca's
//   own agents.wgsl needs (its agents have no physics substrate of their
//   own at all).
// - CHIRALITY (simulation_settings.py's own CHIRALITY, on by default):
//   evalPolicy() runs twice per agent per macro step, once on the sensed
//   input as-is and once mirrored across the agent's own forward axis,
//   averaged (with the mirrored pass's own output un-mirrored first) —
//   see evalPolicy()/agentStep()'s own comments for the exact transform.
//   A real, deliberate compute cost (doubles the dominant per-agent
//   work) in exchange for ruling out an arbitrary, physically
//   unmotivated left/right turning bias by construction.
//
// Communication — each cell owns persistent chemicalState. A separate
// splatChemicalState pass projects every cell's OLD state as a live-tunable
// Gaussian before any brain runs. agentStep senses the resulting values and
// gradients, then interprets the chemical output head as a bounded local state
// delta. No post-brain value is written to persistent environmental state.
//
// Growth — each particle may spawn a copy of itself after completing its
// tensor-growth cycle. As a first boundary-tangent experiment, division uses
// the tangent perpendicular to the local morphology-density gradient and
// places the daughters symmetrically along that axis. A locally flat field has
// no defined tangent, so it falls back to the signed network growth direction.
// Growth admission integrates persistent
// hazard from the LAST channel's sensed VALUE (inputVec[CHANNELS-1u] — the same
// value already fed into this step's own NN input, clamped to [0,1];
// NOT a network output, so CHIRALITY's mirror-averaging never touches
// it) — an evolved policy can shape this "growth substrate" the exact
// same way it shapes any other channel, via its own existing env_write.
// See agentStep()'s own comment for the exact split logic.
//
// A new particle claims the next free slot via an atomic counter
// (`agentState.growthCount`, binding 7) — physics.maxActiveParticles caps
// how many claims actually get written. Training fixes it to evolve.py's
// --particles; the viewer may override it live for playback only. It is
// the growth CAP, not a fixed
// starting count — every rollout currently starts with two particles;
// see training_sim.py's own module docstring) —
// see trainer/training_sim.py's/gpu/simulation.ts's own module
// docstring for the CPU-side readback that turns this atomic's own
// value into the "official" activeCount every other pass (P2G/
// gridUpdate/G2P/repulsion, and this shader's own NEXT dispatch) reads.
// Deliberately DEFERRED like this rather than same-step: a newly
// claimed particle only starts getting its own agentStep()/physics
// once the CPU has propagated the grown count, one macro step after it
// split — a same-step alternative (always over-dispatching every pass
// to the particle-capacity ceiling, relying on a shared activeCount storage
// buffer instead of a uniform) was considered and rejected: P2G/
// gridUpdate/G2P/repulsion run once per PHYSICS SUBSTEP (many times
// per macro step, not once), so over-dispatching all of them to a
// generous cap rather than the actual live count would be a real,
// ALWAYS-PAID compute cost on this project's hottest inner loop, for
// every rollout, whether or not growth ever happens — a one-macro-step
// activation lag is a far cheaper, and imperceptible-at-training-scale,
// price.
//
// Division is conservative grow-then-split. The substrate probabilistically
// starts a cell cycle; g2p increases the parent's stress-free area and mass
// to the division target; agentStep then replaces it with two baseline
// daughters. Their F is rescaled so Fe and stress are continuous. APIC
// velocities remain centered around the pre-split velocity, preserving total
// linear momentum even when a directional signal deliberately shifts the
// daughters' positional center toward +n.
//
// agentState.particleMeta (binding 7, byte offset 256) packs FOUR
// per-particle scalars into one
// struct/buffer: rng+cooldown (growth's own state, only growth itself
// ever reads/writes them) and heading+angularVelocity (this shader's
// facing-direction integrator, see below). `rng` is retained as the ABI name
// but stores lineage generation in density model v3; lifecycle randomness is
// sampled from a rollout-seeded world-space field, never particle slot. cooldown is
// macro steps remaining before this slot can split again, counted down
// every step, reset to physics.divisionCooldown on BOTH the parent and
// the new child whenever a split succeeds (without this, a particle
// sitting on a strong, stable deposit could keep splitting every single
// step it's eligible, producing a burst of children from one spot
// rather than growth actually spreading out over time/distance).
// Packed into ONE struct/buffer, not four separate ones, specifically to
// keep this shader's own storage buffer count under the 10-per-stage
// hardware ceiling Chrome's own Dawn backend reports on real browser
// adapters (NOT the much higher number wgpu-native/Metal reports
// headlessly on the Python side — a real, confirmed
// CreateComputePipeline validation error the first time this shader
// tried to bind particleF/particleC/particleJp as 3 SEPARATE new
// bindings on top of heading/angularVelocity/growthState each having
// their own, not a hypothetical). rng/cooldown were already packed
// together for this exact reason once before (when `velocities` was
// added); heading/angularVelocity joined them for the same reason again
// when particleF/particleJp needed room (see this file's own module
// docstring for why those two matter enough to be worth the 2 slots
// this merge freed).
//
// Bound directly to MpmCore's own positions/velocities/activeCount
// buffers (see simulation.ts/agents_gpu.py).
//
// The policy proposes a local growth direction and anisotropy target;
// agentStep() relaxes persistent angle/anisotropy states toward them. A
// separate sigmoid controls signed division placement. Fg sees an axis, but
// division uses its sign to place the daughter toward +n. physics.maxStrafe
// remains an optional scale for also applying the direction as physical
// acceleration; it is zero by default and does not scale growth geometry.

// Chirality — see simulation_settings.py's own CHIRALITY for the full
// reasoning (why averaging a mirrored second pass is what actually
// enforces left-right symmetry, not just "adds noise"). Templated as a
// compile-time bool (agents_gpu.py/agents.ts's own CHIRALITY template
// var), not a live-tunable uniform: it changes how many times the net
// runs per agent, not a physics knob PhysicsPanel-style live tuning
// makes sense for.
const CHIRALITY: bool = __CHIRALITY__;
const STATEFUL: bool = __STATEFUL__;
const CELL_OWNED_CHEMISTRY: bool = __CELL_OWNED_CHEMISTRY__;
const ELASTIC_STRAIN_INPUTS_ENABLED: bool = __ELASTIC_STRAIN_INPUTS_ENABLED__;
const PRIVATE_STATE_DIM: u32 = 8u;

const CHANNELS: u32 = __CHANNELS__u;
const HIDDEN_DIM: u32 = __HIDDEN_DIM__u;
// Per chemical channel: value, heading-forward gradient, lateral gradient;
// followed by morphology occupancy/forward/lateral gradient and three
// heading-relative elastic Hencky-strain components.
const IN_DIM: u32 = __IN_DIM__u;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;
const CHEMICAL_VALUE_INPUT_SCALE: f32 = __CHEMICAL_VALUE_INPUT_SCALE__;
const MORPHOLOGY_GRADIENT_INPUT_SCALE: f32 = __MORPHOLOGY_GRADIENT_INPUT_SCALE__;
const GROWTH_DIRECTION_RESPONSE_RATE: f32 = __GROWTH_DIRECTION_RESPONSE_RATE__;
const GROWTH_ANISOTROPY_RESPONSE_RATE: f32 = __GROWTH_ANISOTROPY_RESPONSE_RATE__;
const DIRECTION_CONFIDENCE_SCALE: f32 = __DIRECTION_CONFIDENCE_SCALE__;
// One chemical-state delta per channel. The legacy name is retained in the
// checkpoint/output ABI.
const ENV_WRITE_DIM: u32 = CHANNELS;
const OUT_DIM: u32 = __OUT_DIM__u;

const PI: f32 = 3.14159265358979323846;

const FC1W_OFFSET: u32 = 0u;
const FC1B_OFFSET: u32 = FC1W_OFFSET + HIDDEN_DIM * IN_DIM;
const FC2W_OFFSET: u32 = FC1B_OFFSET + HIDDEN_DIM;
const FC2B_OFFSET: u32 = FC2W_OFFSET + OUT_DIM * HIDDEN_DIM;
// Total float count — agents.ts's flattenWeights() must produce exactly
// this many floats, in exactly this fc1w/fc1b/fc2w/fc2b order.

const FIELD_WIDTH: u32 = __FIELD_WIDTH__u;
const FIELD_HEIGHT: u32 = __FIELD_HEIGHT__u;
const FIELD_PLANE: u32 = FIELD_WIDTH * FIELD_HEIGHT;
const FIELD_TOTAL: u32 = FIELD_PLANE * CHANNELS;

// Must match environment.wgsl's own copy exactly.
const DEPOSIT_SCALE: f32 = 4096.0;
const SPATIAL_RANDOM_CELLS: u32 = __SPATIAL_RANDOM_CELLS__u;

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> activeCount: u32;
@group(0) @binding(3) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(4) var<storage, read> gradient: array<f32>; // gx:[0,FIELD_TOTAL), gy:[FIELD_TOTAL,2*FIELD_TOTAL)
@group(0) @binding(5) var<storage, read_write> depositScratch: array<atomic<i32>>;

struct AgentPhysics {
  maxAccel: f32,
  maxStrafe: f32,
  maxEnvWrite: f32,
  maxAngularAccel: f32,
  angularDamping: f32,
  maxAngularVelocity: f32,
  // Legacy ABI slot. Centered deposits no longer use a directional offset.
  depositDistance: f32,
  // World-domain (same [0,1]^2 units as `positions`, NOT field-pixels
  // like depositDistance) distance a newly split particle spawns behind
  // its own parent — see this file's own module docstring/agentStep()'s
  // own growth comment. trainer/simulation_settings.py's own
  // SPLIT_DISPLACEMENT is the starting value.
  splitDisplacement: f32,
  // Macro steps a particle refuses to split again for, right after
  // splitting (either as the parent OR as the new child — see
  // agentStep()'s own growth comment) — counted down once per macro
  // step in the `cooldown` buffer below, regardless of whether this
  // step's own split draw would otherwise have succeeded.
  // trainer/simulation_settings.py's own DIVISION_COOLDOWN is the
  // starting value; 0 disables cooldown entirely (a particle can split
  // again the very next step, this shader's own behavior before
  // cooldown existed).
  divisionCooldown: f32,
  // Per-macro-step velocity retention fraction (velocity *= friction,
  // applied after adding this step's own strafe-driven acceleration —
  // see agentStep()'s own comment) — strafe's own counterweight against
  // unbounded momentum buildup, since nothing else in this shader bleeds
  // velocity off between agentStep() invocations (MpmCore's own physics
  // substeps apply their own, separate damping to the SAME velocity
  // buffer, see mpm_core.py's own set_damping() — this is an additional,
  // independent knob, not a duplicate of that one). trainer/
  // simulation_settings.py's own FRICTION is the starting value.
  friction: f32,
  // Gaussian splat radius (sigma), field-pixel units (not domain [0,1]
  // units like core/repulsion.wgsl's own splat sigma), since it spreads
  // a deposit around the particle's field-pixel position. See
  // depositGaussian()'s own comment
  // for the exact kernel this drives. trainer/simulation_settings.py's
  // own DEPOSIT_SIGMA is the starting value — live-tunable via
  // PhysicsPanel, for testing this splat's own shape/spread.
  depositSigma: f32,
  // Host-controlled gate for STARTING new cell cycles. Already-active
  // cycles are allowed to finish, so switching this off creates a clean
  // settling phase rather than stranding partly-grown cells.
  growthEnabled: f32,
  // Legacy wire-layout slots. Spawn position remains part of rollout
  // initialization, but is deliberately no longer exposed to the policy.
  // Retaining these two floats avoids an unrelated AgentPhysics ABI shift.
  spawnX: f32,
  spawnY: f32,
  // Runtime growth cap. Training writes its fixed --particles value;
  // the frontend can lower/raise it without recompiling this shader.
  maxActiveParticles: u32,
  // Log-strain magnitude mapped to roughly tanh(1). This is input
  // normalization only; it does not change mechanics.
  elasticStrainScale: f32,
  // Density-resolved normalization for the field-pixel Sobel gradient.
  chemicalGradientInputScale: f32,
  // Represented material area carried by one chemical projection.  q times
  // as many particles each publish at 1/q so the fixed-resolution chemical
  // field observes the same continuum concentration across sampling density.
  chemicalProjectionWeight: f32,
  // Seed for a fixed world-space lifecycle random field. Nearby numerical
  // samples share thresholds regardless of particle slot or density.
  rolloutSeed: u32,
  // Below this morphology-gradient magnitude, the boundary tangent is treated
  // as undefined and division falls back to the network growth direction.
  boundaryTangentMinGradient: f32,
  // Lab-only lifecycle control. 0xffffffff disables it. Admission latches a
  // normal cycle once; the index/direction can remain active afterward to
  // keep the morphoelastic growth and eventual division axis deterministic.
  // A zero direction selects the local morphology-boundary tangent instead.
  forcedLifecycleIndex: u32,
  forcedCycleAdmission: u32,
  forcedDivisionDirection: vec2<f32>,
  // Inclusive end of a contiguous Lab-controlled particle range. Equal to
  // forcedLifecycleIndex for the existing single-particle scenarios.
  forcedLifecycleEndIndex: u32,
}
@group(0) @binding(6) var<uniform> physics: AgentPhysics;

// Persistent per-particle state, owned by this shader (not MpmCore, not
// Environment) — heading/angularVelocity used to each have their OWN
// binding, same as rng/cooldown once did before THEY got packed
// together; the scalar state and neural color are now ONE struct/buffer for the same reason:
// this shader hit the WebGPU CORE (not just spec-default) hard ceiling
// of 10 storage buffers per stage on real browser adapters (Dawn/Chrome
// on this project's own dev machine reports exactly 10, not the much
// higher number wgpu-native/Metal reports headlessly on the Python
// side — a real, confirmed CreateComputePipeline validation error the
// first time this shader tried to go to 13, not a hypothetical
// portability worry) — see below for what THAT room got spent on.
// heading is no longer derived from velocity (see this file's own
// module docstring). Every field zeroed/randomized by
// Agents.resetHeading() whenever a rollout restarts (simulation.ts's
// own restartRollout()). rng/cooldown are the lineage generation and
// growth-cooldown state respectively — unrelated to
// heading/angularVelocity otherwise, nothing besides growth's own
// agentStep() logic ever reads those two fields.
struct ParticleMeta {
  rng: u32,
  cooldown: f32,
  heading: f32,
  angularVelocity: f32,
  // Current neural cell color. vec4 keeps the packed record naturally
  // aligned while the unused alpha lane remains fixed at 1.
  color: vec4<f32>,
  // Integrated division clock. Positive growth signal adds hazard; the
  // particle starts a cycle when it crosses an exponentially-distributed
  // threshold. Unlike a fresh Bernoulli draw, sub-threshold drive is not
  // discarded when the signal later changes or temporarily vanishes.
  divisionHazard: f32,
  divisionThreshold: f32,
  // Eight private neural-memory channels. Stateless policies leave these at
  // zero. Stateful policies sense them and apply gated residual updates.
  privateState: array<f32, PRIVATE_STATE_DIM>,
  // Persistent chemical levels owned by this cell. Before every policy
  // invocation splatChemicalState() projects these values into the transient
  // substrate; agentStep() then applies the chemical head as a local delta.
  // The environment itself never preserves or receives a post-policy write.
  chemicalState: array<f32, CHANNELS>,
}
// Pack growth's counter and particle metadata into one allocation. The
// explicit 252-byte pad puts particleMeta at byte 256, allowing the render
// pipeline to bind that tail directly while satisfying WebGPU's storage-
// buffer offset alignment. This frees one storage binding for particleC.
struct AgentState {
  growthCount: atomic<u32>,
  _padding: array<u32, 63>,
  particleMeta: array<ParticleMeta>,
}
@group(0) @binding(7) var<storage, read_write> agentState: AgentState;
// MpmCore's APIC affine velocity field. Division copies C and samples its
// local velocity field at both daughter offsets, preserving linear and
// affine momentum rather than silently giving the child C=0 for a step.
@group(0) @binding(8) var<storage, read_write> particleC: array<vec4<f32>>;
// MpmCore's own velocity buffer. The growth-direction signal only affects
// this through the optional physics.maxStrafe scale (zero by default).
@group(0) @binding(9) var<storage, read_write> velocities: array<vec2<f32>>;
// MpmCore's own F/rest buffers (self.F/self.rest, same buffers
// core/p2g.wgsl and core/g2p.wgsl read/write every physics substep) —
// what the heading/angularVelocity merge above bought room for: a
// freshly-claimed particle can now inherit its parent's CURRENT
// deformation state at split time instead of starting from a fresh
// identity/zero rest state (see this file's own module docstring for why
// that reversal matters). The counter/metadata packing above also makes
// room for particleC, so daughters inherit the complete current MPM state
// rather than losing affine momentum for their first physics substep.
// Brings this shader's own
// storage-buffer count to exactly 10 — AT the real browser ceiling, not
// under it, so any FUTURE addition needs to free a slot first, the same
// way this one did.
@group(0) @binding(10) var<storage, read_write> particleF: array<vec4<f32>>;

// Per-particle REST-STATE bookkeeping — every quantity describing how a
// particle's own stress-free reference configuration differs from the
// pristine one it was seeded with. Tensor Fg and the two scalars remain
// PACKED into one struct/buffer, exactly like ParticleMeta above and for
// exactly the same reason: this shader is at the hard 10-storage-buffer
// Dawn ceiling, and growthF plus the cell-cycle latch (`cycleActive`)
// both need to be written HERE, at the split site, for a newly-claimed
// child. Three separate bindings were never an option; widening the
// buffer that was already bound (the old `particleJp: array<f32>`) costs
// zero new bindings in any of the four shaders that touch it
// (p2g/g2p/this/fieldDiagnostics).
//
// IMPORTANT for every writer: g2p.wgsl and this shader both used to
// overwrite the whole element unconditionally. Now that siblings share
// it, a partial update must preserve the fields it doesn't own.
struct ParticleRest {
  // Full row-major 2x2 growth tensor in F = Fe*Fg. g2p applies either an
  // isotropic increment (zero anisotropy) or a determinant-preserving
  // directional increment selected by growthAngle below.
  growthF: vec4<f32>,
  // Plastic Jacobian — g2p.wgsl's own yield clamp accumulates genuine
  // plastic damage here, and p2g.wgsl's own `e = exp(hardening*(1-jp))`
  // reads it for hardening. Growth NO LONGER touches this: an earlier
  // revision (growthJpRelief) piggybacked growth onto Jp precisely
  // because there was nowhere else to put it, which entangled two
  // unrelated meanings; `growthF` above is now that home.
  jp: f32,
  // Cell-cycle latch. The last substrate channel probabilistically sets
  // this to 1; g2p advances growth until division and both daughters reset it.
  cycleActive: f32,
  // Persistent growth polarity, stored as one angle relative to the
  // particle's heading rather than an instantly-overwritten world vector.
  growthAngle: f32,
  // Persistent anisotropy relaxed toward the policy's sigmoid target.
  growthAnisotropy: f32,
  // Instantaneous sigmoid strength used when division places a daughter.
  divisionBias: f32,
  // Cached heading for the physics passes, which do not bind ParticleMeta.
  // Together with growthAngle it reconstructs the world growth axis.
  growthFrameHeading: f32,
  // Rendering-only newborn area fraction. Seeded cells start at 1; mitosis
  // leaves the parent full-sized and starts the new daughter at 0. g2p grows
  // this with the same curve and compression response as rest area. It fills
  // the struct's former alignment lane, so the 48-byte ABI is unchanged.
  appearanceScale: f32,
  // Explicit tail padding preserves the 48-byte storage ABI.
  _padding: f32,
}
@group(0) @binding(11) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(12) var morphologyTexture: texture_2d<f32>;
__MORPHOLOGY_SAMPLER_DECLARATION__
struct StepMode {
  commitLifecycle: u32,
  // Chemical/orientation time represented by this neural evaluation.
  // Host sets this to communicationSpeed / neuralUpdatesPerMacro.
  communicationDt: f32,
  stateUpdateSpeed: f32,
  // Global cap on the policy's polarized daughter placement. 0 restores
  // center-preserving symmetric division; 1 grants full policy authority.
  divisionDirectionality: f32,
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
}
@group(0) @binding(13) var<uniform> stepMode: StepMode;

fn morphologyLoad(p: vec2<i32>) -> f32 {
  let n = i32(MORPHOLOGY_FIELD_N);
  let q = ((p % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n);
  return textureLoad(morphologyTexture, q, 0).x;
}

fn sampleMorphology(p: vec2<f32>) -> f32 {
  __MORPHOLOGY_SAMPLE_BODY__
}

fn fieldIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * FIELD_PLANE + y * FIELD_WIDTH + x;
}

// A real, confirmed failure mode on this backend: naive tanh computed as
// (e^2x-1)/(e^2x+1) overflows to Inf/Inf (NaN) for large |x| — clamping
// the input first is cheap insurance envnca/frontend/src/gpu/agents.wgsl
// already carries for the same reason, ported verbatim regardless of
// this file's other differences from that one.
fn safeTanh(x: f32) -> f32 {
  return tanh(clamp(x, -20.0, 20.0));
}

// Robust scales measured from a 2,000-step live rollout (5,005 tracked
// particle samples). These mappings keep zero exactly zero and bound rare
// outliers instead of subtracting rollout-specific means. Every chemical
// channel and both gradient directions share their respective scale,
// preserving channel-permutation and rotational symmetry.
fn normalizeChemicalValue(raw: f32) -> f32 {
  return safeTanh(raw / max(CHEMICAL_VALUE_INPUT_SCALE, 1e-6));
}

fn normalizeChemicalGradient(raw: f32) -> f32 {
  return safeTanh(raw / max(physics.chemicalGradientInputScale, 1e-6));
}

fn normalizeMorphologyGradient(raw: f32) -> f32 {
  return safeTanh(raw / max(MORPHOLOGY_GRADIENT_INPUT_SCALE, 1e-6));
}

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z,
    a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z,
    a.z * b.y + a.w * b.w
  );
}

fn matDet(m: vec4<f32>) -> f32 {
  return m.x * m.w - m.y * m.z;
}

fn matInverse(m: vec4<f32>) -> vec4<f32> {
  let det = matDet(m);
  if (abs(det) < 1e-8) {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
  }
  return vec4<f32>(m.w, -m.y, -m.z, m.x) / det;
}

// Objective elastic deformation input. The constitutive model uses
// F = Fe*Fg, so sensing raw F would incorrectly report stress-free growth as
// load. B=Fe*Fe^T removes elastic rigid rotation; rotating B into the
// particle's heading frame makes the result covariant with the chemical and
// morphology forward/lateral inputs. The closed-form eigensystem below
// computes H=0.5*log(B), i.e. spatial Hencky strain, without an SVD.
fn elasticStrainInput(F: vec4<f32>, Fg: vec4<f32>, forward: vec2<f32>, lateral: vec2<f32>, scale: f32) -> vec3<f32> {
  let Fe = matMul(F, matInverse(Fg));
  let bxx = Fe.x * Fe.x + Fe.y * Fe.y;
  let bxy = Fe.x * Fe.z + Fe.y * Fe.w;
  let byy = Fe.z * Fe.z + Fe.w * Fe.w;

  let bf = vec2<f32>(bxx * forward.x + bxy * forward.y, bxy * forward.x + byy * forward.y);
  let bl = vec2<f32>(bxx * lateral.x + bxy * lateral.y, bxy * lateral.x + byy * lateral.y);
  let a = dot(forward, bf);
  let b = dot(forward, bl);
  let d = dot(lateral, bl);

  let midpoint = 0.5 * (a + d);
  let radius = sqrt(max(0.25 * (a - d) * (a - d) + b * b, 0.0));
  let lambda1 = max(midpoint + radius, 1e-8);
  let lambda2 = max(midpoint - radius, 1e-8);
  let e1 = 0.5 * log(lambda1);
  let e2 = 0.5 * log(lambda2);
  let average = 0.5 * (e1 + e2);
  var h00 = average;
  var h11 = average;
  var h01 = 0.0;
  if (radius > 1e-7) {
    let factor = 0.25 * (e1 - e2) / radius;
    h00 = average + factor * (a - d);
    h11 = average - factor * (a - d);
    h01 = factor * 2.0 * b;
  }

  let invScale = 1.0 / max(scale, 1e-6);
  return vec3<f32>(
    safeTanh((h00 + h11) * invScale),
    safeTanh((h00 - h11) * invScale),
    safeTanh((2.0 * h01) * invScale),
  );
}

// Portable integer hash shared conceptually with trainer/agents_gpu.py and
// viewer/src/gpu/rng.ts. Position selects a fixed spatial cell; lineage
// generation supplies a fresh threshold after every conservative split.
fn hashU32(valueIn: u32) -> u32 {
  var value = valueIn;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

fn spatialUniform01(pos: vec2<f32>, generation: u32, domain: u32) -> f32 {
  let wrapped = fract(pos);
  let cellX = min(u32(floor(wrapped.x * f32(SPATIAL_RANDOM_CELLS))), SPATIAL_RANDOM_CELLS - 1u);
  let cellY = min(u32(floor(wrapped.y * f32(SPATIAL_RANDOM_CELLS))), SPATIAL_RANDOM_CELLS - 1u);
  let combined = physics.rolloutSeed
    ^ hashU32(cellX + 0x9e3779b9u)
    ^ hashU32(cellY + 0x85ebca6bu)
    ^ hashU32(generation + 0xc2b2ae35u)
    ^ domain;
  return f32(hashU32(combined) >> 8u) * (1.0 / 16777216.0);
}

struct Corners {
  x0: u32,
  x1: u32,
  y0: u32,
  y1: u32,
  wx0: f32,
  wx1: f32,
  wy0: f32,
  wy1: f32,
}

// WGSL's own `%` keeps the input's sign (like C fmod, not Python's %) —
// this folds a same-sign-as-divisor result back into [0,size) whichever
// side of 0 `v` started on. It is a no-op for values already in range.
fn wrapCoord(v: f32, size: f32) -> f32 {
  let m = v % size;
  return select(m, m + size, m < 0.0);
}

// Bilinear gather/scatter corners at a continuous field-pixel position,
// wrapped (toroidal) into [0,size) — matches trainer/environment.py's
// own _corners() exactly (see environment.wgsl's own module docstring).
fn corners(posIn: vec2<f32>) -> Corners {
  let pos = vec2<f32>(wrapCoord(posIn.x, f32(FIELD_WIDTH)), wrapCoord(posIn.y, f32(FIELD_HEIGHT)));
  let x0f = floor(pos.x);
  let y0f = floor(pos.y);
  var out: Corners;
  out.wx1 = pos.x - x0f;
  out.wx0 = 1.0 - out.wx1;
  out.wy1 = pos.y - y0f;
  out.wy0 = 1.0 - out.wy1;
  out.x0 = u32(x0f) % FIELD_WIDTH;
  out.x1 = (u32(x0f) + 1u) % FIELD_WIDTH;
  out.y0 = u32(y0f) % FIELD_HEIGHT;
  out.y1 = (u32(y0f) + 1u) % FIELD_HEIGHT;
  return out;
}

fn sampleValue(c: u32, k: Corners) -> f32 {
  let v00 = gridCurrent[fieldIndex(c, k.y0, k.x0)];
  let v10 = gridCurrent[fieldIndex(c, k.y0, k.x1)];
  let v01 = gridCurrent[fieldIndex(c, k.y1, k.x0)];
  let v11 = gridCurrent[fieldIndex(c, k.y1, k.x1)];
  return v00 * (k.wx0 * k.wy0) + v10 * (k.wx1 * k.wy0) + v01 * (k.wx0 * k.wy1) + v11 * (k.wx1 * k.wy1);
}

fn sampleGrad(planeOffset: u32, c: u32, k: Corners) -> f32 {
  let v00 = gradient[planeOffset + fieldIndex(c, k.y0, k.x0)];
  let v10 = gradient[planeOffset + fieldIndex(c, k.y0, k.x1)];
  let v01 = gradient[planeOffset + fieldIndex(c, k.y1, k.x0)];
  let v11 = gradient[planeOffset + fieldIndex(c, k.y1, k.x1)];
  return v00 * (k.wx0 * k.wy0) + v10 * (k.wx1 * k.wy0) + v01 * (k.wx0 * k.wy1) + v11 * (k.wx1 * k.wy1);
}

// Hard cap on the deposit splat's own texel footprint, regardless of how
// large physics.depositSigma is dragged via its own PhysicsPanel slider
// — same bounded-cost reasoning core/repulsion.wgsl's own
// MAX_KERNEL_RADIUS_TEXELS gives (see that const's own comment): this is
// what keeps depositGaussian()'s own per-call cost genuinely bounded
// rather than growing without limit alongside a live-tunable radius.
// Smaller than repulsion's own cap (5) to keep each particle's chemical
// write bounded.
const MAX_DEPOSIT_KERNEL_RADIUS: i32 = 3;

// Euclidean modulo, i32 in/out — same wraparound idea
// core/repulsion.wgsl's own wrapFieldIndex() already uses for its own
// (separate) splat/field, applied per-axis here since this file's own
// field can have independent FIELD_WIDTH/FIELD_HEIGHT.
fn wrapDepositIndex(i: i32, size: u32) -> i32 {
  let n = i32(size);
  return ((i % n) + n) % n;
}

// Scatter-adds one particle's per-channel chemical levels as a bounded
// Gaussian splat around `centerFieldPos` — replaces this shader's old
// 4-corner bilinear
// scatter (sensing still uses that: corners()/sampleValue()/
// sampleGrad() above are untouched, only the DEPOSIT side changed) per
// this project's own explicit request for finer, live-tunable control
// over deposit shape/spread than a hard 2x2 footprint allowed. Same
// kernel-loop shape as core/repulsion.wgsl's own splatDensity() —
// bounded radius, Gaussian falloff, toroidal-wrapped storage index with
// an UNWRAPPED distance calculation (see that function's own comment
// for why: the falloff needs a genuine continuous-space distance across
// the domain seam, only the buffer index itself wraps) — except this
// splat's own weight is computed ONCE per kernel tap and reused across
// all CHANNELS at that tap (same position, only the per-channel VALUE
// differs), rather than one call per channel each re-deriving its own
// weights the way the old per-corner depositAt() did.
//
// UNLIKE the old bilinear scatter, this is NOT mass-normalized — each
// tap's weight peaks at 1.0 (not 1/(2*pi*sigma^2)), matching
// splatDensity()'s own convention. That means the TOTAL deposited mass
// from one particle now grows with depositSigma (more full-weight taps
// stacking up), not just its spread — a real, visible behavior change
// worth knowing while testing this slider, not merely a smoother-
// looking version of the old, mass-conserving 4-corner deposit.
fn depositGaussian(
  envWrite: array<f32, ENV_WRITE_DIM>,
  centerFieldPos: vec2<f32>,
  contributionScale: f32,
) {
  let baseI = i32(floor(centerFieldPos.x));
  let baseJ = i32(floor(centerFieldPos.y));

  let sigmaTexels = max(physics.depositSigma, 1e-3);
  let sigma2 = sigmaTexels * sigmaTexels;
  // 3-sigma is where a Gaussian's own contribution is already <1.1% of
  // its peak — truncating there (subject to MAX_DEPOSIT_KERNEL_RADIUS's
  // own hard cap) loses nothing visible, same reasoning
  // core/repulsion.wgsl's own splatDensity() gives.
  let kernelRadius = min(i32(ceil(3.0 * sigmaTexels)), MAX_DEPOSIT_KERNEL_RADIUS);

  for (var di: i32 = -kernelRadius; di <= kernelRadius; di = di + 1) {
    for (var dj: i32 = -kernelRadius; dj <= kernelRadius; dj = dj + 1) {
      let ti = baseI + di;
      let tj = baseJ + dj;
      // Field samples use integer grid coordinates (corners() floors the
      // continuous field position), so the Gaussian is centered on that same
      // lattice. A cell exactly on a grid coordinate therefore senses its own
      // stored level at full strength instead of an unintended half-texel loss.
      let texelCenter = vec2<f32>(f32(ti), f32(tj));
      let delta = centerFieldPos - texelCenter;
      let d2 = dot(delta, delta);
      let weight = exp(-d2 / (2.0 * sigma2));

      let wx = u32(wrapDepositIndex(ti, FIELD_WIDTH));
      let wy = u32(wrapDepositIndex(tj, FIELD_HEIGHT));
      for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
        let scaled = envWrite[c] * weight * contributionScale
          * physics.chemicalProjectionWeight * DEPOSIT_SCALE;
        atomicAdd(&depositScratch[fieldIndex(c, wy, wx)], i32(round(scaled)));
      }
    }
  }
}

// Rebuild contribution pass, deliberately separate from agentStep so every
// cell publishes its OLD state before any cell's brain can update it.
@compute @workgroup_size(64)
fn splatChemicalState(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let fieldPos = fract(positions[pi]) * vec2<f32>(f32(FIELD_WIDTH), f32(FIELD_HEIGHT));
  var levels: array<f32, ENV_WRITE_DIM>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    levels[c] = agentState.particleMeta[pi].chemicalState[c];
  }
  // A daughter should not communicate as a full-area cell while its visible
  // area is still emerging. Scale only this transient projection: its owned
  // chemical state remains intact and reaches full strength with its disc.
  let contributionScale = clamp(particleRest[pi].appearanceScale, 0.0, 1.0);
  depositGaussian(levels, fieldPos, contributionScale);
}

// The bounded subset of the network's own raw output
// this shader actually consumes — one chemical-state delta per channel +
// desired heading, anisotropy/division-bias controls, and desired growth
// direction (optionally also physical acceleration), all in LOCAL frame.
struct PolicyOutput {
  envWrite: array<f32, ENV_WRITE_DIM>,
  headingTargetLocal: vec2<f32>,
  anisotropyTarget: f32,
  divisionBias: f32,
  growthTargetLocal: vec2<f32>,
  color: vec3<f32>,
  stateDelta: array<f32, PRIVATE_STATE_DIM>,
  stateGate: array<f32, PRIVATE_STATE_DIM>,
}

fn safeSigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

fn wrapAngle(angle: f32) -> f32 {
  return atan2(sin(angle), cos(angle));
}

// Continuous confidence: a zero vector has no directional authority, while
// increasingly decisive vectors approach one without a hard cutoff.
fn directionConfidence(v: vec2<f32>) -> f32 {
  let magnitude = length(v);
  return magnitude / (magnitude + max(DIRECTION_CONFIDENCE_SCALE, 1e-6));
}

fn directionAngle(v: vec2<f32>) -> f32 {
  if (dot(v, v) <= 1e-20) {
    return 0.0;
  }
  return atan2(v.y, v.x);
}

// One full Dense(HIDDEN_DIM) -> tanh -> Dense(OUT_DIM) forward pass,
// squashed/scaled into PolicyOutput. Pulled out of agentStep() into its
// own function specifically so CHIRALITY can call it twice, on two
// different (mirrored) `inputVec`s, without duplicating the actual
// matmul loops — see agentStep()'s own comment for how the two calls
// combine.
fn evalPolicy(inputVec: array<f32, IN_DIM>) -> PolicyOutput {
  var hidden: array<f32, HIDDEN_DIM>;
  for (var j: u32 = 0u; j < HIDDEN_DIM; j = j + 1u) {
    var acc = weights[FC1B_OFFSET + j];
    for (var i: u32 = 0u; i < IN_DIM; i = i + 1u) {
      acc = acc + inputVec[i] * weights[FC1W_OFFSET + j * IN_DIM + i];
    }
    // Bounded, monotonic, zero-centered response. The shared overflow-safe
    // implementation protects mutated policies with extreme preactivations.
    hidden[j] = safeTanh(acc);
  }

  var outVec: array<f32, OUT_DIM>;
  for (var j: u32 = 0u; j < OUT_DIM; j = j + 1u) {
    var acc = weights[FC2B_OFFSET + j];
    for (var i: u32 = 0u; i < HIDDEN_DIM; i = i + 1u) {
      acc = acc + hidden[i] * weights[FC2W_OFFSET + j * HIDDEN_DIM + i];
    }
    outVec[j] = acc;
  }

  var out: PolicyOutput;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    out.envWrite[c] = safeTanh(outVec[c]) * physics.maxEnvWrite;
  }
  out.headingTargetLocal = vec2<f32>(
    safeTanh(outVec[ENV_WRITE_DIM]), safeTanh(outVec[ENV_WRITE_DIM + 1u])
  );
  out.anisotropyTarget = safeSigmoid(outVec[ENV_WRITE_DIM + 2u]);
  out.divisionBias = safeSigmoid(outVec[ENV_WRITE_DIM + 3u]);
  out.growthTargetLocal = vec2<f32>(
    safeTanh(outVec[ENV_WRITE_DIM + 4u]), safeTanh(outVec[ENV_WRITE_DIM + 5u])
  );
  __POLICY_TAIL_DECODE__
  return out;
}

@compute @workgroup_size(64)
fn agentStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = positions[pi];

  // Local (heading-relative) frame — heading is this shader's own
  // persistent state (NOT derived from velocity, see this file's own
  // module docstring for why). No zero-guard needed the way an
  // atan2-of-velocity approach would (cos/sin of an arbitrary real angle
  // is always well-defined).
  let headingVal = agentState.particleMeta[pi].heading;
  let cosH = cos(headingVal);
  let sinH = sin(headingVal);

  // fract() first, not just a straight scale — this pass runs BEFORE
  // this macro step's own physics substeps (see simulation.ts's own
  // step() ordering), so `pos` is whatever the *previous* step's physics
  // left behind (always in [0,1), MpmCore's own domain is toroidal) —
  // except on the very first macro step, where it's straight off
  // rng.ts's own seedBlob(), which doesn't itself guarantee [0,1) for a
  // spawn center/half-width combination that reaches past an edge.
  // fract() makes this correct (wrapped) either way, not just in the
  // common case.
  let fieldPos = fract(pos) * vec2<f32>(f32(FIELD_WIDTH), f32(FIELD_HEIGHT));
  let k = corners(fieldPos);

  // Rotate each channel's world-frame gradient into this particle's own
  // local frame (forward = heading, lateral = 90° left) before it
  // reaches the network — matches trainer/training_sim.py's own
  // macro_step() exactly, same rotation envnca/simulation.py's own
  // step() applies.
  var inputVec: array<f32, IN_DIM>;
  var rawGrowthSignal = 0.0;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    let rawValue = sampleValue(c, k);
    if (c == CHANNELS - 1u) { rawGrowthSignal = rawValue; }
    inputVec[c] = normalizeChemicalValue(rawValue);
    let gx = sampleGrad(0u, c, k);
    let gy = sampleGrad(FIELD_TOTAL, c, k);
    inputVec[CHANNELS + c] = normalizeChemicalGradient(gx * cosH + gy * sinH);
    inputVec[2u * CHANNELS + c] = normalizeChemicalGradient(-gx * sinH + gy * cosH);
  }
  let morphologyPos = fract(pos) * f32(MORPHOLOGY_FIELD_N);
  let morphologyOccupancy = clamp(sampleMorphology(morphologyPos), 0.0, 1.0);
  let morphologyGx = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(1.0, 0.0)) - sampleMorphology(morphologyPos - vec2<f32>(1.0, 0.0)));
  let morphologyGy = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(0.0, 1.0)) - sampleMorphology(morphologyPos - vec2<f32>(0.0, 1.0)));
  inputVec[3u * CHANNELS] = 2.0 * morphologyOccupancy - 1.0;
  inputVec[3u * CHANNELS + 1u] = normalizeMorphologyGradient(morphologyGx * cosH + morphologyGy * sinH);
  inputVec[3u * CHANNELS + 2u] = normalizeMorphologyGradient(-morphologyGx * sinH + morphologyGy * cosH);
  let forward = vec2<f32>(cosH, sinH);
  let lateral = vec2<f32>(-sinH, cosH);
  var elasticInput = vec3<f32>(0.0);
  if (ELASTIC_STRAIN_INPUTS_ENABLED) {
    elasticInput = elasticStrainInput(
      particleF[pi], particleRest[pi].growthF, forward, lateral, physics.elasticStrainScale
    );
  }
  inputVec[3u * CHANNELS + 3u] = elasticInput.x;
  inputVec[3u * CHANNELS + 4u] = elasticInput.y;
  inputVec[3u * CHANNELS + 5u] = elasticInput.z;
  __PRIVATE_STATE_INPUTS__
  var result = evalPolicy(inputVec);

  // CHIRALITY: a second pass on the mirror-reflected input (lateral —
  // left/right, perpendicular to heading — gradient component negated;
  // VALUE and the forward component have no handedness, untouched),
  // then its OWN output un-mirrored and averaged with the first pass's.
  // That averaging is the actual symmetry-enforcing step: any left/right
  // handedness bias the raw network happens to have cancels out, only
  // the genuinely symmetric part of its response survives — see
  // simulation_settings.py's own CHIRALITY for the full reasoning.
  // Both desired-direction vectors have their lateral component un-mirrored.
  // A scalar chemical-state delta has no handedness, so it is averaged
  // channel-for-channel.
  if (CHIRALITY) {
    var mirroredInput = inputVec;
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      mirroredInput[2u * CHANNELS + c] = -mirroredInput[2u * CHANNELS + c];
    }
    mirroredInput[3u * CHANNELS + 2u] = -mirroredInput[3u * CHANNELS + 2u];
    // Reflection across the heading axis preserves volumetric and axial
    // strain while reversing the signed forward/lateral shear component.
    mirroredInput[3u * CHANNELS + 5u] = -mirroredInput[3u * CHANNELS + 5u];
    let mirrored = evalPolicy(mirroredInput);

    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      result.envWrite[c] = (result.envWrite[c] + mirrored.envWrite[c]) * 0.5;
    }
    result.headingTargetLocal = vec2<f32>(
      (result.headingTargetLocal.x + mirrored.headingTargetLocal.x) * 0.5,
      (result.headingTargetLocal.y - mirrored.headingTargetLocal.y) * 0.5
    );
    result.anisotropyTarget = (result.anisotropyTarget + mirrored.anisotropyTarget) * 0.5;
    result.divisionBias = (result.divisionBias + mirrored.divisionBias) * 0.5;
    result.growthTargetLocal = vec2<f32>(
      (result.growthTargetLocal.x + mirrored.growthTargetLocal.x) * 0.5,
      (result.growthTargetLocal.y - mirrored.growthTargetLocal.y) * 0.5
    );
    result.color = (result.color + mirrored.color) * 0.5;
    for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {
      result.stateDelta[s] = (result.stateDelta[s] + mirrored.stateDelta[s]) * 0.5;
      result.stateGate[s] = (result.stateGate[s] + mirrored.stateGate[s]) * 0.5;
    }
  }

  let communicationDt = max(stepMode.communicationDt, 0.0);

  if (CELL_OWNED_CHEMISTRY) {
    // Cell-owned-projection interprets the chemical head as a delta to
    // persistent per-cell chemistry. The separate splat pass publishes the
    // pre-update state on the following communication round.
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      agentState.particleMeta[pi].chemicalState[c] = clamp(
        agentState.particleMeta[pi].chemicalState[c] + result.envWrite[c] * communicationDt,
        -1.0,
        1.0,
      );
    }
  } else {
    // Persistent-environment retains the original interpretation: chemical
    // outputs are direct spatial writes. Environment decay/deposit-rate owns
    // timestep scaling, so result.envWrite is not integrated here.
    depositGaussian(
      result.envWrite,
      fieldPos,
      clamp(particleRest[pi].appearanceScale, 0.0, 1.0),
    );
  }

  if (STATEFUL) {
    for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {
      let residual = result.stateGate[s] * result.stateDelta[s]
        * communicationDt * max(stepMode.stateUpdateSpeed, 0.0);
      agentState.particleMeta[pi].privateState[s] = clamp(
        agentState.particleMeta[pi].privateState[s] + residual, -4.0, 4.0
      );
    }
    result.color = vec3<f32>(
      safeSigmoid(agentState.particleMeta[pi].privateState[0u]),
      safeSigmoid(agentState.particleMeta[pi].privateState[1u]),
      safeSigmoid(agentState.particleMeta[pi].privateState[2u]),
    );
  }

  // The NN proposes a LOCAL desired heading vector. Its shortest signed angle
  // from local-forward becomes a proportional angular acceleration; the
  // existing angularVelocity state below supplies inertia and damping.
  let headingConfidence = directionConfidence(result.headingTargetLocal);
  let desiredHeadingError = directionAngle(result.headingTargetLocal);
  let angularAccel = clamp(
    desiredHeadingError / PI * physics.maxAngularAccel * headingConfidence,
    -physics.maxAngularAccel,
    physics.maxAngularAccel,
  );

  // Growth polarity is a separate persistent angle RELATIVE to heading. The
  // policy supplies a desired local vector; the stored angle follows it by
  // the shortest arc with an exponential, timestep-invariant response.
  let growthConfidence = directionConfidence(result.growthTargetLocal);
  let desiredGrowthAngle = directionAngle(result.growthTargetLocal);
  // Confidence belongs inside the continuous-time rate so communication
  // supersampling does not change convergence over the same elapsed time.
  let growthAngleBlend = 1.0 - exp(
    -max(GROWTH_DIRECTION_RESPONSE_RATE, 0.0) * growthConfidence * communicationDt
  );
  let growthAngleError = wrapAngle(desiredGrowthAngle - particleRest[pi].growthAngle);
  particleRest[pi].growthAngle = wrapAngle(
    particleRest[pi].growthAngle + growthAngleError * growthAngleBlend
  );

  let anisotropyBlend = 1.0 - exp(
    -max(GROWTH_ANISOTROPY_RESPONSE_RATE, 0.0) * communicationDt
  );
  particleRest[pi].growthAnisotropy = clamp(
    particleRest[pi].growthAnisotropy
      + (result.anisotropyTarget - particleRest[pi].growthAnisotropy) * anisotropyBlend,
    0.0,
    1.0,
  );
  particleRest[pi].divisionBias = result.divisionBias;
  particleRest[pi].growthFrameHeading = headingVal;

  let forcedLifecycle = pi >= physics.forcedLifecycleIndex
    && pi <= physics.forcedLifecycleEndIndex;
  let forcedBoundaryTangent = forcedLifecycle
    && length(physics.forcedDivisionDirection) < 0.5;
  var growthWorldAngle = headingVal + particleRest[pi].growthAngle;
  var growthDirectionWorld = vec2<f32>(cos(growthWorldAngle), sin(growthWorldAngle));
  let lifecycleMorphologyGradient = vec2<f32>(morphologyGx, morphologyGy);
  let lifecycleMorphologyGradientMagnitude = length(lifecycleMorphologyGradient);
  if (forcedBoundaryTangent
      && lifecycleMorphologyGradientMagnitude > physics.boundaryTangentMinGradient) {
    // During a tangent-controlled Lab cycle, align morphoelastic growth with
    // the same local boundary tangent that will place the two daughters.
    growthDirectionWorld = vec2<f32>(
      -lifecycleMorphologyGradient.y,
      lifecycleMorphologyGradient.x,
    ) / lifecycleMorphologyGradientMagnitude;
    growthWorldAngle = atan2(growthDirectionWorld.y, growthDirectionWorld.x);
    particleRest[pi].growthAngle = wrapAngle(growthWorldAngle - headingVal);
    particleRest[pi].growthFrameHeading = headingVal;
  } else if (forcedLifecycle && !forcedBoundaryTangent) {
    // Lock the persistent world-frame growth axis, not merely the final
    // daughter offset. g2p therefore grows Fg along this same direction and
    // transfers its stress through the ordinary MPM grid before division.
    growthDirectionWorld = normalize(physics.forcedDivisionDirection);
    growthWorldAngle = atan2(growthDirectionWorld.y, growthDirectionWorld.x);
    particleRest[pi].growthAngle = wrapAngle(growthWorldAngle - headingVal);
    particleRest[pi].growthFrameHeading = headingVal;
  }
  agentState.particleMeta[pi].color = vec4<f32>(result.color, 1.0);

  if (stepMode.commitLifecycle != 0u) {
  // Strafe as acceleration: added onto this particle's own CURRENT
  // velocity, then damped by physics.friction (a per-macro-step
  // retention fraction — see AgentPhysics's own friction field comment
  // for why this is a separate knob from MpmCore's own damping on the
  // same buffer). positions[pi] is NOT written here at all anymore —
  // MpmCore's own G2P pass integrates this new velocity into position
  // during this macro step's own physics substeps (see
  // training_sim.py's/simulation.ts's own step ordering: agentStep()
  // always runs before those substeps).
  velocities[pi] = (velocities[pi] + growthDirectionWorld * physics.maxStrafe) * physics.friction;

  // Grow-then-divide cell cycle. The last substrate channel supplies a
  // bounded per-macro-step growth probability. Convert it to cumulative
  // hazard, h=-log(1-p), so a constant signal retains the former event
  // probability while partial drive persists instead of being discarded.
  // g2p grows Fg until g=2 after admission; only then is one grown parent
  // replaced by two baseline daughters.
  // Lifecycle semantics remain in raw substrate units. Input normalization is
  // exclusively a neural-sensing transform and must not accelerate division.
  let splitProb = clamp(rawGrowthSignal, 0.0, 1.0);
  // Division cooldown — counted down every step regardless of whether
  // the division clock would otherwise cross (so it's a clean
  // macro-step countdown, independent of the stochastic threshold),
  // clamped at 0 rather than going negative. Gates admission below;
  // see AgentPhysics's own
  // divisionCooldown field comment for the full reasoning.
  let cooldownNow = max(agentState.particleMeta[pi].cooldown - 1.0, 0.0);
  agentState.particleMeta[pi].cooldown = cooldownNow;
  // A cycle admitted while capacity still existed can be overtaken by
  // other divisions and find the population cap full on a later macro
  // step. Close that now-unfulfillable cycle before g2p runs. Preserve
  // its current growth exactly: rolling g back would delete already
  // accumulated rest area/mass and require a compensating F change,
  // while letting the latch remain active would keep injecting growth
  // forever even though this particle can never obtain a daughter slot.
  if (activeCount >= physics.maxActiveParticles && particleRest[pi].cycleActive > 0.5) {
    particleRest[pi].cycleActive = 0.0;
  }
  // Never start a cycle that cannot produce a daughter. This matters at
  // the population cap: without it, every terminal particle could grow
  // to g=2 despite having no free slot, doubling rest mass/area after
  // the visible particle count had already stopped changing.
  let lifecycleEligible =
    physics.growthEnabled > 0.5 &&
    activeCount < physics.maxActiveParticles &&
    particleRest[pi].cycleActive < 0.5 &&
    cooldownNow <= 0.0;
  if (
    forcedLifecycle &&
    physics.forcedCycleAdmission != 0u &&
    activeCount < physics.maxActiveParticles &&
    particleRest[pi].cycleActive < 0.5 &&
    cooldownNow <= 0.0
  ) {
    // This is the scenario's only synthetic action: admit the exact same
    // persistent cycle organic substrate hazard would admit. Everything from
    // Fg growth onward remains the production grow-then-divide path.
    particleRest[pi].cycleActive = 1.0;
    agentState.particleMeta[pi].divisionHazard = 0.0;
    agentState.particleMeta[pi].divisionThreshold = 0.0;
  }
  if (lifecycleEligible) {
    // A zero threshold is the reset sentinel. Draw once per prospective
    // cycle, not once per tick. -log(1-u) is Exp(1); top 24 RNG bits match
    // the precision convention used for the division-angle draw below.
    if (agentState.particleMeta[pi].divisionThreshold <= 0.0) {
      let u = spatialUniform01(pos, agentState.particleMeta[pi].rng, 0x54485245u);
      agentState.particleMeta[pi].divisionThreshold = max(-log(max(1.0 - u, 1e-7)), 1e-7);
    }
    var hazardIncrement = -log(max(1.0 - splitProb, 1e-7));
    // Preserve the old exact p=1 behavior: a saturated growth field admits
    // immediately even for the vanishingly rare Exp(1) threshold >16.1.
    if (splitProb >= 1.0) {
      hazardIncrement = agentState.particleMeta[pi].divisionThreshold + 1.0;
    }
    agentState.particleMeta[pi].divisionHazard =
      agentState.particleMeta[pi].divisionHazard + hazardIncrement;
    if (agentState.particleMeta[pi].divisionHazard >= agentState.particleMeta[pi].divisionThreshold) {
      particleRest[pi].cycleActive = 1.0;
      agentState.particleMeta[pi].divisionHazard = 0.0;
      agentState.particleMeta[pi].divisionThreshold = 0.0;
    }
  }

  let divisionTarget = 2.0;
  if (
    activeCount < physics.maxActiveParticles &&
    particleRest[pi].cycleActive > 0.5 &&
    matDet(particleRest[pi].growthF) >= divisionTarget * 0.9999
  ) {
    // atomicAdd returns the OLD value — the slot THIS particle just
    // claimed. Never gated before the add (that would need a compare-
    // exchange loop to stay race-free against every OTHER agent that
    // might also split this exact step) — instead a claim landing at or
    // past physics.maxActiveParticles is simply never written below (this
    // buffer is sized to the much larger MAX_PARTICLES — see
    // mpm_core.py's own module docstring — so an over-claimed index is
    // never actually out of bounds, just wasted atomic traffic on a step
    // where many agents split near the cap at once); trainer/
    // training_sim.py's/gpu/simulation.ts's own readback separately
    // clamps the *reported* activeCount to physics.maxActiveParticles
    // regardless of how high this atomic itself climbs.
    let newIndex = atomicAdd(&agentState.growthCount, 1u);
    if (newIndex < physics.maxActiveParticles) {
      let nextGeneration = agentState.particleMeta[pi].rng + 1u;
      let morphologyGradient = vec2<f32>(morphologyGx, morphologyGy);
      let morphologyGradientMagnitude = length(morphologyGradient);
      var spawnDir = growthDirectionWorld;
      let directionStrength = clamp(particleRest[pi].divisionBias, 0.0, 1.0)
        * clamp(stepMode.divisionDirectionality, 0.0, 1.0);
      var centerShift = spawnDir * (0.5 * physics.splitDisplacement) * directionStrength;
      // A scheduled Lab lifecycle is an axis-controlled diagnostic: keep the
      // real conservative division path, but replace the policy's polarized
      // placement with a pair centered on the original particle. Production
      // divisions retain their usual divisionBias-controlled center shift.
      if (forcedLifecycle) {
        centerShift = vec2<f32>(0.0);
      }
      if ((!forcedLifecycle || forcedBoundaryTangent)
          && morphologyGradientMagnitude > physics.boundaryTangentMinGradient) {
        // The gradient is the boundary normal. Rotating it by +90 degrees
        // gives either representative of the tangent axis; because daughter
        // placement is symmetric, the sign of that representative is
        // irrelevant.
        spawnDir = vec2<f32>(-morphologyGradient.y, morphologyGradient.x)
          / morphologyGradientMagnitude;
        centerShift = vec2<f32>(0.0);
      }
      let halfOffset = spawnDir * (0.5 * physics.splitDisplacement);
      // A measurable boundary (and every scheduled Lab split) keeps the pair
      // centered; only production's flat-field fallback retains the existing
      // network-controlled center shift.
      positions[pi] = fract(pos - halfOffset + centerShift);
      positions[newIndex] = fract(pos + halfOffset + centerShift);
      // "A copy of itself": heading/angularVelocity copied from this
      // particle's own CURRENT state (its pre-integration values — the
      // integrator below hasn't run yet at this point in the function).
      // Heading itself stays copied (not randomized) — only the spawn
      // POSITION is random now, not the child's own facing direction.
      agentState.particleMeta[newIndex].heading = headingVal;
      agentState.particleMeta[newIndex].angularVelocity = agentState.particleMeta[pi].angularVelocity;
      agentState.particleMeta[newIndex].color = agentState.particleMeta[pi].color;
      agentState.particleMeta[newIndex].divisionHazard = 0.0;
      agentState.particleMeta[newIndex].divisionThreshold = 0.0;
      for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {
        agentState.particleMeta[newIndex].privateState[s] = agentState.particleMeta[pi].privateState[s];
      }
      for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
        agentState.particleMeta[newIndex].chemicalState[c] = agentState.particleMeta[pi].chemicalState[c];
      }
      // Both daughters advance the same lineage generation. Their next
      // thresholds come from their world-space cells, not q-dependent slots.
      agentState.particleMeta[pi].rng = nextGeneration;
      agentState.particleMeta[newIndex].rng = nextGeneration;
      // BOTH the parent (this slot, `pi`) and the new child go on
      // cooldown — a freshly split child immediately splitting again
      // would defeat the whole point of throttling growth, same as a
      // parent that just split shouldn't either.
      agentState.particleMeta[pi].cooldown = physics.divisionCooldown;
      agentState.particleMeta[newIndex].cooldown = physics.divisionCooldown;
      // Preserve velocity and elastic/plastic state. Returning each
      // daughter to Fg=I requires Fdaughter=Fparent*inverse(Fgparent),
      // which preserves Fe exactly while one det(Fg)=2 weight becomes two
      // det(Fg)=1 weights. This remains correct for anisotropic Fg.
      let parentVelocity = velocities[pi];
      let parentC = particleC[pi];
      let affineOffsetVelocity = vec2<f32>(
        parentC.x * halfOffset.x + parentC.y * halfOffset.y,
        parentC.z * halfOffset.x + parentC.w * halfOffset.y,
      );
      velocities[pi] = parentVelocity - affineOffsetVelocity;
      velocities[newIndex] = parentVelocity + affineOffsetVelocity;
      particleC[newIndex] = parentC;
      let daughterF = matMul(particleF[pi], matInverse(particleRest[pi].growthF));
      particleF[pi] = daughterF;
      particleF[newIndex] = daughterF;
      let parentJp = particleRest[pi].jp;
      let parentGrowthAngle = particleRest[pi].growthAngle;
      let parentGrowthAnisotropy = particleRest[pi].growthAnisotropy;
      let parentDivisionBias = particleRest[pi].divisionBias;
      let parentGrowthFrameHeading = particleRest[pi].growthFrameHeading;
      let parentPadding = particleRest[pi]._padding;
      let identity = vec4<f32>(1.0, 0.0, 0.0, 1.0);
      particleRest[pi] = ParticleRest(
        identity, parentJp, 0.0, parentGrowthAngle, parentGrowthAnisotropy,
        parentDivisionBias, parentGrowthFrameHeading, 1.0, parentPadding
      );
      particleRest[newIndex] = ParticleRest(
        identity,
        parentJp,
        0.0,
        parentGrowthAngle,
        parentGrowthAnisotropy,
        parentDivisionBias,
        parentGrowthFrameHeading,
        0.0,
        parentPadding,
      );
    } else {
      particleRest[pi].cycleActive = 0.0;
      agentState.particleMeta[pi].cooldown = physics.divisionCooldown;
    }
  }
  }

  // Second-order heading integrator — angularAccel nudges angularVelocity,
  // angularVelocity (after its own damping, applied here since nothing
  // else bleeds it off between agentStep() invocations) nudges heading.
  // Hard-clamped to maxAngularVelocity afterward — damping alone still
  // lets angularVelocity settle at a nonzero steady state, which was
  // still visibly too fast a spin (see simulation_settings.py's own
  // comment on the exact bound); this clamp is the real turn-rate
  // ceiling. Matches training_sim.py's own macro_step() exactly.
  let roundDamping = pow(clamp(physics.angularDamping, 0.000001, 1.0), communicationDt);
  let newAngularVelocity = clamp(
    (agentState.particleMeta[pi].angularVelocity + angularAccel * communicationDt) * roundDamping,
    -physics.maxAngularVelocity,
    physics.maxAngularVelocity,
  );
  agentState.particleMeta[pi].angularVelocity = newAngularVelocity;
  let newHeading = headingVal + newAngularVelocity * communicationDt;
  agentState.particleMeta[pi].heading = newHeading;
  particleRest[pi].growthFrameHeading = newHeading;
  if (forcedLifecycle
      && (!forcedBoundaryTangent
          || lifecycleMorphologyGradientMagnitude > physics.boundaryTangentMinGradient)) {
    // Heading integration happens after lifecycle logic. Re-express the
    // selected fixed or boundary-tangent world direction in the new local
    // frame so the following g2p pass grows on precisely that same axis.
    particleRest[pi].growthAngle = wrapAngle(growthWorldAngle - newHeading);
  }

}

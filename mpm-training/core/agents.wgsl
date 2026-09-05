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
// - Gradient-based steering, following Mordvintsev et al.'s steerable NCA:
//   the local frame is the L2-clipped gradient of chemical channel 7.
//   There is no persistent cell heading, turn state, or neural heading
//   output. In a locally uniform field the frame is zero, so directional
//   perception loses authority. Growth direction and anisotropy still follow
//   the NN directly; optional strafe remains suppressed without a frame.
// - No repulsion sampling. repulsion.wgsl already gives every particle a
//   real repulsion force as part of MpmCore's own physics — there's no
//   separate host-agent repulsion field to sense here the way envnca's
//   own agents.wgsl needs (its agents have no physics substrate of their
//   own at all).
// - CHIRALITY (simulation_settings.py's own CHIRALITY, on by default):
//   evalPolicy() runs twice per agent per macro step, once on the sensed
//   input as-is and once mirrored across the agent's own forward axis,
//   averaged after parity-correcting local-frame outputs —
//   see evalPolicy()/agentStep()'s own comments for the exact transform.
//   A real, deliberate compute cost (doubles the dominant per-agent
//   work) in exchange for ruling out an arbitrary, physically
//   unmotivated left/right turning bias by construction.
//
// Communication — the chemical head emits one signed delta rate per channel.
// Cell-owned mode integrates it into persistent per-cell chemicalState and
// projects the old synchronous state before each brain round. Persistent-
// environment mode deposits only the final round's delta into its transported
// spatial field. Per-channel temporal scales divide both delta rates.
//
// Growth integration and adaptive resampling live in growthField.wgsl. This
// module publishes one bounded 2-D growth vector per material sample. Its
// magnitude is the requested local growth rate and its direction is rotated
// from the sample's sensed local frame into world space before publication.
//
// MPM particles are material samples, not cells. Growth therefore changes the
// continuum rest volume independently of sample count. The local neural vector
// is splatted and volume-normalized on the MPM grid; g2p exponentiates the
// resulting tensor into growthF every physics substep. There is no stochastic
// admission, angular bin, range, cooldown, or per-sample division event.
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
// Sample creation is adaptive quadrature refinement only. An oversized sample
// conservatively transfers one baseline rest volume to a new sample while
// preserving elastic, chemical, and neural state. If no slot is available,
// continuum growth still proceeds; only numerical resolution stops increasing.
//
// agentState.particleMeta (binding 7, byte offset 256) packs retained ABI
// bookkeeping, the channel-7-gradient alignment cache, and neural state into
// one struct/buffer. `rng` now acts as a refinement lineage generation;
// cooldown/hazard/threshold fields remain layout-compatible but are not part
// of the continuous growth decision.
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
// The policy proposes a signed local-space growth vector. Its magnitude is the
// rate; v and -v share an axial growth tensor but point refinement toward
// opposite sides. physics.maxStrafe
// remains an optional scale for applying the growth direction as physical
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
// Production policies use chemical channel index 7 as their orientation
// field. Reduced-channel shader checks fall back to their last channel so the
// shared shader remains valid when CHANNELS < 8.
const HEADING_CHANNEL: u32 = min(7u, CHANNELS - 1u);
const HIDDEN_DIM: u32 = __HIDDEN_DIM__u;
// Per chemical channel: value, heading-forward gradient, lateral gradient;
// followed by morphology occupancy/forward/lateral gradient and three
// heading-relative elastic Hencky-strain components.
const IN_DIM: u32 = __IN_DIM__u;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;
const MORPHOLOGY_GRADIENT_INPUT_SCALE: f32 = __MORPHOLOGY_GRADIENT_INPUT_SCALE__;
const GROWTH_ANISOTROPY_RESPONSE_RATE: f32 = __GROWTH_ANISOTROPY_RESPONSE_RATE__;
// One signed chemical delta rate per channel. The legacy name is retained in
// the checkpoint/output ABI.
const ENV_WRITE_DIM: u32 = CHANNELS;
const OUT_DIM: u32 = __OUT_DIM__u;

const PI: f32 = 3.14159265358979323846;

const FC1W_OFFSET: u32 = 0u;
const FC1B_OFFSET: u32 = FC1W_OFFSET + HIDDEN_DIM * IN_DIM;
const FC2W_OFFSET: u32 = FC1B_OFFSET + HIDDEN_DIM;
const FC2B_OFFSET: u32 = FC2W_OFFSET + OUT_DIM * HIDDEN_DIM;
// Total float count — agents.ts's flattenWeights() must produce exactly
// this many floats, in exactly this fc1w/fc1b/fc2w/fc2b order.

// Packed multi-resolution chemical layout. Every channel keeps its ordinary
// policy index while choosing its spatial/temporal scale through host data.
const FIELD_WIDTHS: array<u32, CHANNELS> = __FIELD_WIDTHS__;
const FIELD_HEIGHTS: array<u32, CHANNELS> = __FIELD_HEIGHTS__;
const FIELD_OFFSETS: array<u32, CHANNELS> = __FIELD_OFFSETS__;
const FIELD_RELAXATION_TIMES: array<f32, CHANNELS> = __FIELD_RELAXATION_TIMES__;
const FIELD_DEPOSIT_SIGMA_MULTIPLIERS: array<f32, CHANNELS> = __FIELD_DEPOSIT_SIGMA_MULTIPLIERS__;
const FIELD_TOTAL: u32 = __FIELD_TOTAL__u;
const FIELD_MAX_WIDTH: u32 = __FIELD_MAX_WIDTH__u;
const FIELD_MAX_HEIGHT: u32 = __FIELD_MAX_HEIGHT__u;

// Scratch stores one extra atomic after numerator+density. Every publisher
// writes the same dynamically derived fixed-point scale there before adding
// contributions. The scale uses the live capacity/projection bounds, giving
// sub-micro precision for ordinary runs while retaining explicit overflow
// headroom for much larger particle counts.
const DEPOSIT_SCALE_INDEX: u32 = FIELD_TOTAL * 2u;
const MAX_DEPOSIT_SCALE: f32 = 1048576.0;
const DEPOSIT_ACCUMULATOR_BUDGET: f32 = 1000000000.0;
const MAX_PROJECTION_GROWTH: f32 = 4.0;
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
  // Amplitude of the policy's signed chemical delta rate.
  maxEnvWrite: f32,
  // Legacy uniform slots retained so old run-setting records keep their wire
  // layout. Gradient-based steering does not read them.
  maxAngularAccel: f32,
  angularDamping: f32,
  maxAngularVelocity: f32,
  // Legacy ABI slot. Centered deposits no longer use a directional offset.
  depositDistance: f32,
  // World-domain (same [0,1]^2 units as `positions`, NOT field-pixels
  // like depositDistance) radial spacing of newly emitted material samples.
  // trainer/simulation_settings.py's own
  // SPLIT_DISPLACEMENT is the starting value.
  splitDisplacement: f32,
  // Macro steps a particle refuses to emit again for, right after
  // growth (either as the source OR as a new sample — see
  // agentStep()'s own growth comment) — counted down once per macro
  // step in the `cooldown` buffer below, regardless of whether this
  // step's own split draw would otherwise have succeeded.
  // trainer/simulation_settings.py's own DIVISION_COOLDOWN is the
  // starting value; 0 disables cooldown entirely (a particle can emit
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
  // Gaussian splat sigma in normalized world-domain units. Each channel
  // converts it to its native grid only to choose a bounded support; weights
  // themselves are evaluated in world space, so changing field resolution no
  // longer changes the physical deposit footprint.
  depositSigma: f32,
  // Host-controlled gate for admitting new material-emission events.
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
  // Below this morphology-gradient magnitude, the Lab's optional
  // boundary-tangent growth axis is treated as undefined.
  boundaryTangentMinGradient: f32,
  // Lab-only lifecycle control. 0xffffffff disables it. Admission latches a
  // normal cycle once; the index/direction can remain active afterward to
  // keep the morphoelastic growth axis deterministic.
  // A zero direction selects the local morphology-boundary tangent instead.
  forcedLifecycleIndex: u32,
  forcedCycleAdmission: u32,
  forcedDivisionDirection: vec2<f32>,
  // Inclusive end of a contiguous Lab-controlled particle range. Equal to
  // forcedLifecycleIndex for the existing single-particle scenarios.
  forcedLifecycleEndIndex: u32,
  // Mechanical contact-inhibition controls. Compression is the positive
  // elastic areal Hencky strain max(0,-log(det(Fe))).
  growthCompressionStart: f32,
  growthCompressionStop: f32,
  growthCompressionFeedback: f32,
  // Live gain on chemical concentration inputs. 1 uses their natural [-1,1]
  // scale; 0 removes values while leaving gradients intact.
  chemicalValueInputMultiplier: f32,
  // Blends growth drive from its signed meaning toward a full probability
  // remap: 0 keeps drive unchanged; 1 maps [-1,1] to [0,1].
  divisionDriveBoost: f32,
  // Lab-only growth-grid override; consumed by growthField.wgsl.
  forcedGrowthFieldMode: u32,
  materialAreaBudget: f32,
}
@group(0) @binding(6) var<uniform> physics: AgentPhysics;

// Persistent per-particle state, owned by this shader (not MpmCore, not
// Environment). The alignment vector is only a cache of the current channel-7
// chemical gradient for rendering; it is overwritten on every agent
// evaluation.
// binding, same as rng/cooldown once did before THEY got packed
// together; the scalar state and neural color are now ONE struct/buffer for the same reason:
// this shader hit the WebGPU CORE (not just spec-default) hard ceiling
// of 10 storage buffers per stage on real browser adapters (Dawn/Chrome
// on this project's own dev machine reports exactly 10, not the much
// higher number wgpu-native/Metal reports headlessly on the Python
// side — a real, confirmed CreateComputePipeline validation error the
// first time this shader tried to go to 13, not a hypothetical
// portability worry) — see below for what THAT room got spent on.
struct ParticleMeta {
  rng: u32,
  cooldown: f32,
  alignment: vec2<f32>,
  // Current neural cell color. vec4 keeps the packed record naturally
  // aligned while the alpha lane remains fixed at 1.
  color: vec4<f32>,
  // Integrated growth clock. Positive growth signal adds hazard; the
  // particle emits material when it crosses an exponentially-distributed
  // threshold. Unlike a fresh Bernoulli draw, sub-threshold drive is not
  // discarded when the signal later changes or temporarily vanishes.
  divisionHazard: f32,
  divisionThreshold: f32,
  // Neural growth drive after the global drive remapping. This is
  // signed at zero boost and shifts toward [0,1] as boost approaches one.
  // Kept explicitly so rendering reads the value used by the simulation.
  mitosisPropensity: f32,
  // Eight private neural-memory channels. Stateless policies leave these at
  // zero. Stateful policies sense them and apply gated residual updates.
  privateState: array<f32, PRIVATE_STATE_DIM>,
  // Persistent chemical levels owned by this cell. Before every policy
  // invocation splatChemicalState() projects these values into the transient
  // substrate; agentStep() then integrates the chemical head's signed delta.
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
// deformation state at split time instead of starting undeformed with zero
// rest-state history (see this file's own module docstring for why
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
  // One-step growth-event latch. The policy's dedicated growth drive
  // probabilistically sets this to 1; additive emission resets it immediately.
  cycleActive: f32,
  // Persistent growth polarity, stored as one angle relative to the
  // particle's heading rather than an instantly-overwritten world vector.
  growthAngle: f32,
  // Persistent anisotropy relaxed toward the policy's sigmoid target.
  growthAnisotropy: f32,
  // Normalized angular fan spread. The legacy field name preserves the ABI.
  divisionBias: f32, // Original world area (legacy ABI name).
  // Heading frame captured for physics passes, which do not bind ParticleMeta.
  // Together with growthAngle it reconstructs the world-space growth axis.
  growthFrameAngle: f32,
  // Rendering-only appearance fraction. Adaptive refinement copies it to both
  // weighted sites; quadratureWeight below conserves their combined disc area.
  appearanceScale: f32,
  // Numerical quadrature weight. Refinement divides this between samples;
  // material growth never changes it. q * det(growthF) is represented area.
  quadratureWeight: f32,
  // Transported world-space half edges, row major. Independent of plastic F.
  domain: vec4<f32>,
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
  // Global cap on the policy's angular fan: 0 is a ray, 1 permits 360 degrees.
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
  return FIELD_OFFSETS[c] + y * FIELD_WIDTHS[c] + x;
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
  return safeTanh(raw * max(physics.chemicalValueInputMultiplier, 0.0));
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

  let frameStrength = length(forward);
  let unitForward = select(vec2<f32>(1.0, 0.0), forward / max(frameStrength, 1e-10), frameStrength > 1e-10);
  let unitLateral = vec2<f32>(-unitForward.y, unitForward.x);
  let bf = vec2<f32>(bxx * unitForward.x + bxy * unitForward.y, bxy * unitForward.x + byy * unitForward.y);
  let bl = vec2<f32>(bxx * unitLateral.x + bxy * unitLateral.y, bxy * unitLateral.x + byy * unitLateral.y);
  let a = dot(unitForward, bf);
  let b = dot(unitForward, bl);
  let d = dot(unitLateral, bl);

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
    safeTanh((h00 - h11) * invScale) * frameStrength,
    safeTanh((2.0 * h01) * invScale) * frameStrength,
  );
}

fn mechanicalGrowthGate(F: vec4<f32>, Fg: vec4<f32>) -> f32 {
  let Fe = matMul(F, matInverse(Fg));
  let compression = max(0.0, -log(max(matDet(Fe), 1e-6)));
  var pressureGate = 0.0;
  if (physics.growthCompressionStop > physics.growthCompressionStart) {
    pressureGate = 1.0 - smoothstep(
      physics.growthCompressionStart,
      physics.growthCompressionStop,
      compression
    );
  } else if (compression < physics.growthCompressionStart) {
    pressureGate = 1.0;
  }
  return mix(1.0, pressureGate, clamp(physics.growthCompressionFeedback, 0.0, 1.0));
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
fn corners(c: u32, posIn: vec2<f32>) -> Corners {
  let width = FIELD_WIDTHS[c];
  let height = FIELD_HEIGHTS[c];
  let pos = vec2<f32>(wrapCoord(posIn.x, f32(width)), wrapCoord(posIn.y, f32(height)));
  let x0f = floor(pos.x);
  let y0f = floor(pos.y);
  var out: Corners;
  out.wx1 = pos.x - x0f;
  out.wx0 = 1.0 - out.wx1;
  out.wy1 = pos.y - y0f;
  out.wy0 = 1.0 - out.wy1;
  out.x0 = u32(x0f) % width;
  out.x1 = (u32(x0f) + 1u) % width;
  out.y0 = u32(y0f) % height;
  out.y1 = (u32(y0f) + 1u) % height;
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
const MAX_DEPOSIT_KERNEL_RADIUS: i32 = 6;

// Euclidean modulo, i32 in/out — same wraparound idea
// core/repulsion.wgsl's own wrapFieldIndex() already uses for its own
// (separate) splat/field, applied per-axis here since this file's own
// field can have independent FIELD_WIDTH/FIELD_HEIGHT.
fn wrapDepositIndex(i: i32, size: u32) -> i32 {
  let n = i32(size);
  return ((i % n) + n) % n;
}

fn currentDepositScale() -> f32 {
  let worstCaseMagnitude = max(
    f32(physics.maxActiveParticles)
      * max(physics.chemicalProjectionWeight, 1e-6)
      * MAX_PROJECTION_GROWTH
      * max(abs(physics.maxEnvWrite), 1.0),
    1.0,
  );
  return max(
    1.0,
    min(MAX_DEPOSIT_SCALE, floor(DEPOSIT_ACCUMULATOR_BUDGET / worstCaseMagnitude)),
  );
}

fn publishDepositScale() -> f32 {
  let scale = currentDepositScale();
  // Every invocation writes the same integer. Compute-pass ordering makes it
  // visible to environment.wgsl before materialization/merge.
  atomicStore(&depositScratch[DEPOSIT_SCALE_INDEX], i32(scale));
  return scale;
}

fn addChemicalDeposit(c: u32, y: u32, x: u32, value: f32, projectionWeight: f32) {
  let depositScale = f32(max(atomicLoad(&depositScratch[DEPOSIT_SCALE_INDEX]), 1));
  let scaled = value * projectionWeight * depositScale;
  atomicAdd(&depositScratch[fieldIndex(c, y, x)], i32(round(scaled)));
  atomicAdd(
    &depositScratch[FIELD_TOTAL + fieldIndex(c, y, x)],
    i32(round(projectionWeight * depositScale)),
  );
}

// Four-point quadrature over the texel centered at the unwrapped native-grid
// coordinate (ti,tj). Evaluating the deformed Gaussian in world coordinates
// makes its physical width independent of FIELD_WIDTHS/FIELD_HEIGHTS. This is
// an inexpensive approximation to the texel integral and handles rotated,
// anisotropic growthF, for which separable Gaussian-CDF differences do not.
fn integratedGaussianTexelWeight(
  centerFieldPos: vec2<f32>,
  fieldDimensions: vec2<f32>,
  ti: i32,
  tj: i32,
  inverseGrowth: vec4<f32>,
  sigmaWorld2: f32,
) -> f32 {
  var weight = 0.0;
  for (var sy: u32 = 0u; sy < 2u; sy = sy + 1u) {
    for (var sx: u32 = 0u; sx < 2u; sx = sx + 1u) {
      let sampleOffset = vec2<f32>(f32(sx) * 0.5 - 0.25, f32(sy) * 0.5 - 0.25);
      let sampleFieldPos = vec2<f32>(f32(ti), f32(tj)) + sampleOffset;
      let worldDelta = (centerFieldPos - sampleFieldPos) / fieldDimensions;
      let referenceWorldDelta = vec2<f32>(
        inverseGrowth.x * worldDelta.x + inverseGrowth.y * worldDelta.y,
        inverseGrowth.z * worldDelta.x + inverseGrowth.w * worldDelta.y,
      );
      weight = weight + exp(-dot(referenceWorldDelta, referenceWorldDelta) / (2.0 * sigmaWorld2));
    }
  }
  return weight * 0.25;
}

// Scatter-adds one particle's per-channel chemical expression using a
// normalized-world-space kernel. Sub-texel Gaussians use the grid's bilinear
// basis directly; resolved Gaussians use texel-integrated quadrature and a
// second normalization pass. Both paths conserve projection mass, remain
// smooth under sub-texel particle motion, and scale total projection by
// det(growthF), so conservative division preserves represented material area.
fn depositGaussian(
  envWrite: array<f32, ENV_WRITE_DIM>,
  centerWorldPos: vec2<f32>,
  growthF: vec4<f32>,
  quadratureWeight: f32,
) {
  _ = publishDepositScale();
  let growthDet = max(abs(matDet(growthF)), 1e-6);
  let inverseGrowth = matInverse(growthF);
  // Largest singular value of growthF. It bounds the deformed Gaussian in
  // every direction, while the inverse transform below supplies the exact
  // anisotropic weight within that conservative square footprint.
  let frobenius2 = dot(growthF, growthF);
  let largestStretch = sqrt(max(
    0.5 * (frobenius2 + sqrt(max(frobenius2 * frobenius2
      - 4.0 * growthDet * growthDet, 0.0))),
    1e-6,
  ));
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    let width = FIELD_WIDTHS[c];
    let height = FIELD_HEIGHTS[c];
    let fieldDimensions = vec2<f32>(f32(width), f32(height));
    // Storage texel i is centered at world coordinate (i + 0.5) / size.
    // Shift world coordinates into the integer-centered lattice used by
    // corners() and integratedGaussianTexelWeight().
    let centerFieldPos = fract(centerWorldPos) * fieldDimensions - vec2<f32>(0.5);
    let baseI = i32(floor(centerFieldPos.x));
    let baseJ = i32(floor(centerFieldPos.y));
    let sigmaWorld = max(
      physics.depositSigma * FIELD_DEPOSIT_SIGMA_MULTIPLIERS[c], 1e-8
    );
    let sigmaNative = sigmaWorld * max(fieldDimensions.x, fieldDimensions.y);
    let projectionScale = max(quadratureWeight, 1e-6) * growthDet
      * physics.chemicalProjectionWeight;

    // A Gaussian narrower than half a native texel is not meaningfully
    // resolved. Cloud-in-cell is its mass-conserving, motion-continuous limit
    // on the same bilinear grid used by sampleValue()/sampleGrad().
    if (sigmaNative < 0.5) {
      let k = corners(c, centerFieldPos);
      addChemicalDeposit(c, k.y0, k.x0, envWrite[c], projectionScale * k.wx0 * k.wy0);
      addChemicalDeposit(c, k.y0, k.x1, envWrite[c], projectionScale * k.wx1 * k.wy0);
      addChemicalDeposit(c, k.y1, k.x0, envWrite[c], projectionScale * k.wx0 * k.wy1);
      addChemicalDeposit(c, k.y1, k.x1, envWrite[c], projectionScale * k.wx1 * k.wy1);
      continue;
    }

    let sigmaWorld2 = sigmaWorld * sigmaWorld;
    let kernelRadius = min(
      i32(ceil(3.0 * sigmaNative * largestStretch + 0.5)),
      MAX_DEPOSIT_KERNEL_RADIUS,
    );

    var totalWeight = 0.0;
    for (var di: i32 = -kernelRadius; di <= kernelRadius; di = di + 1) {
      for (var dj: i32 = -kernelRadius; dj <= kernelRadius; dj = dj + 1) {
        totalWeight = totalWeight + integratedGaussianTexelWeight(
          centerFieldPos, fieldDimensions, baseI + di, baseJ + dj,
          inverseGrowth, sigmaWorld2,
        );
      }
    }
    let inverseTotalWeight = 1.0 / max(totalWeight, 1e-20);
    for (var di: i32 = -kernelRadius; di <= kernelRadius; di = di + 1) {
      for (var dj: i32 = -kernelRadius; dj <= kernelRadius; dj = dj + 1) {
        let ti = baseI + di;
        let tj = baseJ + dj;
        let weight = integratedGaussianTexelWeight(
          centerFieldPos, fieldDimensions, ti, tj, inverseGrowth, sigmaWorld2,
        ) * inverseTotalWeight;
        let wx = u32(wrapDepositIndex(ti, width));
        let wy = u32(wrapDepositIndex(tj, height));
        addChemicalDeposit(c, wy, wx, envWrite[c], projectionScale * weight);
      }
    }
  }
}

__DOMAIN_FUNCTIONS__

// Integrate a fixed-world chemical kernel over the transported domain. Growth
// scales material amount; it no longer enlarges the kernel a second time.
fn depositDomain(envWrite: array<f32, ENV_WRITE_DIM>, pos: vec2<f32>, rest: ParticleRest) {
  let representedArea = max(rest.quadratureWeight, 1e-6) * max(matDet(rest.growthF), 1e-6);
  for (var k = 0u; k < domainQuadratureCount(rest.domain); k++) {
    let q = domainQuadrature(rest.domain, k);
    depositGaussian(envWrite, pos+q.xy, vec4<f32>(1.0, 0.0, 0.0, 1.0), representedArea*q.z);
  }
}

// Rebuild contribution pass, deliberately separate from agentStep so every
// cell publishes its OLD state before any cell's brain can update it.
@compute @workgroup_size(64)
fn splatChemicalState(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  var levels: array<f32, ENV_WRITE_DIM>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    levels[c] = agentState.particleMeta[pi].chemicalState[c];
  }
  // appearanceScale is deliberately rendering-only. Physical mass and
  // chemistry are fully present immediately after division; growthF controls
  // the projection footprint before division so the substrate follows the
  // continuously growing material.
  depositDomain(levels, positions[pi], particleRest[pi]);
}

// The bounded subset of the network's raw output: one signed chemical delta
// per channel plus one continuous 2-D growth vector in LOCAL frame.
struct PolicyOutput {
  envWrite: array<f32, ENV_WRITE_DIM>,
  growthVectorLocal: vec2<f32>,
  color: vec3<f32>,
  stateDelta: array<f32, PRIVATE_STATE_DIM>,
  stateGate: array<f32, PRIVATE_STATE_DIM>,
}

fn safeSigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
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
  out.growthVectorLocal = vec2<f32>(
    safeTanh(outVec[ENV_WRITE_DIM]), safeTanh(outVec[ENV_WRITE_DIM + 1u])
  );
  __POLICY_TAIL_DECODE__
  return out;
}

@compute @workgroup_size(64)
fn agentStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = positions[pi];

  // Sample the orientation field before assembling heading-relative policy
  // inputs. Sobel gradients are expressed per native texel, so convert channel
  // 7 to the same reference-grid convention used by the chemical perception
  // loop below before applying the paper's L2 clipping. Strong gradients have
  // unit direction; weak/undefined gradients proportionally suppress every
  // directional lane. The angle is only a cache for growth physics/rendering.
  let headingFieldPos = fract(pos)
    * vec2<f32>(f32(FIELD_WIDTHS[HEADING_CHANNEL]), f32(FIELD_HEIGHTS[HEADING_CHANNEL]))
    - vec2<f32>(0.5);
  let headingCorners = corners(HEADING_CHANNEL, headingFieldPos);
  let headingGradient = vec2<f32>(
    sampleGrad(0u, HEADING_CHANNEL, headingCorners)
      * f32(FIELD_WIDTHS[HEADING_CHANNEL]) / f32(FIELD_MAX_WIDTH),
    sampleGrad(FIELD_TOTAL, HEADING_CHANNEL, headingCorners)
      * f32(FIELD_HEIGHTS[HEADING_CHANNEL]) / f32(FIELD_MAX_HEIGHT),
  );
  let alignmentStrength = min(length(headingGradient), 1.0);
  let forward = headingGradient / max(length(headingGradient), 1.0);
  let lateral = vec2<f32>(-forward.y, forward.x);
  let alignmentAngle = select(0.0, atan2(forward.y, forward.x), alignmentStrength > 1e-10);
  agentState.particleMeta[pi].alignment = forward;

  // Morphology remains a separate policy observation and supports the Lab's
  // optional boundary-tangent lifecycle control; it no longer defines heading.
  let morphologyPos = fract(pos) * f32(MORPHOLOGY_FIELD_N);
  let morphologyOccupancy = clamp(sampleMorphology(morphologyPos), 0.0, 1.0);
  let morphologyGx = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(1.0, 0.0)) - sampleMorphology(morphologyPos - vec2<f32>(1.0, 0.0)));
  let morphologyGy = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(0.0, 1.0)) - sampleMorphology(morphologyPos - vec2<f32>(0.0, 1.0)));
  let morphologyGradient = vec2<f32>(morphologyGx, morphologyGy);

  // fract() first, not just a straight scale — this pass runs BEFORE
  // this macro step's own physics substeps (see simulation.ts's own
  // step() ordering), so `pos` is whatever the *previous* step's physics
  // left behind (always in [0,1), MpmCore's own domain is toroidal) —
  // except on the very first macro step, where it's straight off
  // rng.ts's own seedBlob(), which doesn't itself guarantee [0,1) for a
  // spawn center/half-width combination that reaches past an edge.
  // fract() makes this correct (wrapped) either way, not just in the
  // common case.
  // Rotate each channel's world-frame gradient into this particle's own
  // local frame (forward = heading, lateral = 90° left) before it
  // reaches the network — matches trainer/training_sim.py's own
  // macro_step() exactly, same rotation envnca/simulation.py's own
  // step() applies.
  var inputVec: array<f32, IN_DIM>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    let fieldPos = fract(pos)
      * vec2<f32>(f32(FIELD_WIDTHS[c]), f32(FIELD_HEIGHTS[c]))
      - vec2<f32>(0.5);
    let k = corners(c, fieldPos);
    let rawValue = sampleValue(c, k);
    inputVec[c] = normalizeChemicalValue(rawValue);
    // Sobel is a derivative per native texel. Convert it to the reference
    // finest-grid convention so a world-space slope has comparable neural
    // magnitude regardless of the channel's chosen resolution.
    let gx = sampleGrad(0u, c, k) * f32(FIELD_WIDTHS[c]) / f32(FIELD_MAX_WIDTH);
    let gy = sampleGrad(FIELD_TOTAL, c, k) * f32(FIELD_HEIGHTS[c]) / f32(FIELD_MAX_HEIGHT);
    inputVec[CHANNELS + c] = normalizeChemicalGradient(dot(vec2<f32>(gx, gy), forward));
    inputVec[2u * CHANNELS + c] = normalizeChemicalGradient(dot(vec2<f32>(gx, gy), lateral));
  }
  inputVec[3u * CHANNELS] = 2.0 * morphologyOccupancy - 1.0;
  inputVec[3u * CHANNELS + 1u] = normalizeMorphologyGradient(dot(morphologyGradient, forward));
  inputVec[3u * CHANNELS + 2u] = normalizeMorphologyGradient(dot(morphologyGradient, lateral));
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
  // then its parity-sensitive outputs are un-mirrored and averaged with the
  // first pass's.
  // That averaging is the actual symmetry-enforcing step: any left/right
  // handedness bias the raw network happens to have cancels out, only
  // the genuinely symmetric part of its response survives — see
  // simulation_settings.py's own CHIRALITY for the full reasoning.
  // The local growth vector's lateral component changes sign under reflection.
  // A scalar chemical delta has no handedness, so it is averaged
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
    result.growthVectorLocal = vec2<f32>(
      (result.growthVectorLocal.x + mirrored.growthVectorLocal.x) * 0.5,
      (result.growthVectorLocal.y - mirrored.growthVectorLocal.y) * 0.5
    );
    result.color = (result.color + mirrored.color) * 0.5;
    for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {
      result.stateDelta[s] = (result.stateDelta[s] + mirrored.stateDelta[s]) * 0.5;
      result.stateGate[s] = (result.stateGate[s] + mirrored.stateGate[s]) * 0.5;
    }
  }

  let communicationDt = max(stepMode.communicationDt, 0.0);

  if (CELL_OWNED_CHEMISTRY) {
    // The head is a rate, so subdividing one communication interval into more
    // rounds preserves its elapsed-time scale. Slow channels still accumulate;
    // they simply integrate a smaller delta per unit communication time.
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      let chemicalDelta = result.envWrite[c] * communicationDt
        / max(FIELD_RELAXATION_TIMES[c], 1e-6);
      agentState.particleMeta[pi].chemicalState[c] = clamp(
        agentState.particleMeta[pi].chemicalState[c] + chemicalDelta,
        -1.0,
        1.0,
      );
    }
  } else if (stepMode.commitLifecycle != 0u) {
    // Persistent mode deliberates against a frozen field and deposits only the
    // final neural round. Environment depositRate owns macro-time scaling.
    depositDomain(result.envWrite, pos, particleRest[pi]);
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

  let forcedLifecycle = pi >= physics.forcedLifecycleIndex
    && pi <= physics.forcedLifecycleEndIndex;
  let forcedBoundaryTangent = forcedLifecycle
    && length(physics.forcedDivisionDirection) < 0.5;
  // The raw proposal can also drive optional motility/strafe. Material growth
  // itself is selected from the integrated field in growthField.wgsl.
  // A vanishing heading gradient has the same deterministic world-X fallback
  // as the former angle representation; it must not erase growth magnitude.
  let growthForward = vec2<f32>(cos(alignmentAngle), sin(alignmentAngle));
  let growthLateral = vec2<f32>(-growthForward.y, growthForward.x);
  var growthVectorWorld = growthForward * result.growthVectorLocal.x
    + growthLateral * result.growthVectorLocal.y;
  let lifecycleMorphologyGradient = vec2<f32>(morphologyGx, morphologyGy);
  let lifecycleMorphologyGradientMagnitude = length(lifecycleMorphologyGradient);
  if (forcedBoundaryTangent
      && lifecycleMorphologyGradientMagnitude > physics.boundaryTangentMinGradient) {
    // During a tangent-controlled Lab cycle, align morphoelastic growth and
    // daughter placement with the local boundary tangent.
    let forcedMagnitude = max(length(growthVectorWorld), select(0.0, 1.0, physics.forcedCycleAdmission != 0u));
    growthVectorWorld = vec2<f32>(
      -lifecycleMorphologyGradient.y,
      lifecycleMorphologyGradient.x,
    ) / lifecycleMorphologyGradientMagnitude * forcedMagnitude;
  } else if (forcedLifecycle && !forcedBoundaryTangent) {
    // Lock the persistent world-frame growth axis. g2p therefore grows Fg
    // along this direction and transfers its stress through the ordinary MPM
    // grid before division along the same axis.
    let forcedMagnitude = max(length(growthVectorWorld), select(0.0, 1.0, physics.forcedCycleAdmission != 0u));
    growthVectorWorld = normalize(physics.forcedDivisionDirection) * forcedMagnitude;
  }
  growthVectorWorld *= select(0.0, 1.0, physics.growthEnabled > 0.5);
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
  velocities[pi] = (velocities[pi] + growthVectorWorld * physics.maxStrafe) * physics.friction;
  // These two legacy ABI slots now store the world-space vector components.
  // growthField.wgsl splats them with the same quadratic kernel as MPM.
  particleRest[pi].cycleActive = growthVectorWorld.x;
  particleRest[pi].growthAngle = growthVectorWorld.y;
  particleRest[pi].growthAnisotropy = 0.0;
  // divisionBias stores original world area; preserve it across policy updates.
  particleRest[pi].growthFrameAngle = 0.0;
  agentState.particleMeta[pi].mitosisPropensity = length(growthVectorWorld);
  }

}

// The evolved per-particle policy, GPU-resident — WGSL port of
// trainer/update_rule.py's Dense(128) -> sin -> Dense(4*CHANNELS+5) (hidden
// activation swapped from tanh to sin, a periodic activation, per this
// project's own explicit request — see evalPolicy()'s own comment for
// why), following envnca/frontend/src/gpu/agents.wgsl's own NN-forward-
// pass approach (weights as one flat buffer, plain loops rather than a
// matrix type, safeTanh on the OUTPUT layer's own squashing to avoid a
// real confirmed NaN failure mode — see that layer's own comment, sin's
// own hidden-layer use doesn't share this) with the differences
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
//   driven by a second-order angularAccel -> angularVelocity -> heading
//   integrator with its own damping — symmetric with how translational
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
// Deposit — four independently-controlled writes per channel at
// heading-relative front/left/back/right spots. Their distance from the
// particle and Gaussian radius are live-tunable. See depositGaussian()
// and agentStep() for the bounded scatter and spot geometry.
// sensing (value/grad_forward/grad_lateral) is unchanged, still sampled
// once, at the agent's own position, still via the old bilinear
// corners()/sampleValue()/sampleGrad() (only the deposit SIDE changed).
//
// Growth — each particle may spawn a copy of itself after completing its
// tensor-growth cycle. The signed network growth vector polarizes division:
// the child is placed along +n and the pair's center shifts toward +n in
// proportion to signal magnitude. A zero signal retains the old symmetric,
// uniformly-random split exactly. Growth admission has a per-step probability read straight off
// the LAST channel's own sensed VALUE (inputVec[CHANNELS-1u] — the same
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
// facing-direction integrator, see below) — rng is xorshift32 state,
// seeded once per rollout by Agents.resetHeading() (see xorshift32()'s
// own comment for why growth's random draw needs dedicated state rather
// than hashing something that already changes each step); cooldown is
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
// Four independent outputs control growth geometry: the former strafe pair
// encodes a normalized LOCAL direction, while the two former acceleration
// channels pass through sigmoid to control tensor anisotropy and signed
// division placement. agentStep() rotates the direction to world space and
// stores all three values in ParticleRest. Fg sees an unoriented axis, but
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

const CHANNELS: u32 = __CHANNELS__u;
const HIDDEN_DIM: u32 = __HIDDEN_DIM__u;
// Per chemical channel: value, heading-forward gradient, lateral gradient;
// followed by morphology occupancy and its forward/lateral gradients.
const IN_DIM: u32 = CHANNELS * 3u + 3u;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;

// Four heading-relative deposit spots in spot-major order:
// front, left, back, right. Each spot has an independent write per channel.
const SPOTS: u32 = 4u;
const ENV_WRITE_DIM: u32 = CHANNELS * SPOTS;
const OUT_DIM: u32 = ENV_WRITE_DIM + 5u; // env_write(SPOTS*CHANNELS) + angular_accel(1) + accel(2) + strafe(2)

const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = PI * 0.5;

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
  // Field-pixel distance from the particle to each heading-relative
  // deposit spot. Zero collapses all four writes onto the particle.
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
}
@group(0) @binding(6) var<uniform> physics: AgentPhysics;

// Persistent per-particle state, owned by this shader (not MpmCore, not
// Environment) — heading/angularVelocity used to each have their OWN
// binding, same as rng/cooldown once did before THEY got packed
// together; all four are now ONE struct/buffer for the same reason:
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
// own restartRollout()). rng/cooldown: see xorshift32()'s own comment
// (rng) and the growth-cooldown logic below (cooldown) — unrelated to
// heading/angularVelocity otherwise, nothing besides growth's own
// agentStep() logic ever reads those two fields.
struct ParticleMeta {
  rng: u32,
  cooldown: f32,
  heading: f32,
  angularVelocity: f32,
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
  // isotropic increment (zero direction) or a determinant-preserving
  // directional increment selected by growthDirection below.
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
  // Normalized world-frame growth axis from the network's former strafe
  // channels. Zero means that no axis was selected.
  growthDirection: vec2<f32>,
  // x: sigmoid anisotropy, y: sigmoid signed-division center bias.
  // Stored independently so blob-like isotropic/symmetric growth does
  // not require destroying the direction signal itself.
  growthControls: vec2<f32>,
}
@group(0) @binding(11) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(12) var morphologyTexture: texture_2d<f32>;

fn morphologyLoad(p: vec2<i32>) -> f32 {
  let n = i32(MORPHOLOGY_FIELD_N);
  let q = ((p % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n);
  return textureLoad(morphologyTexture, q, 0).x;
}

fn sampleMorphology(p: vec2<f32>) -> f32 {
  let base = vec2<i32>(floor(p));
  let f = fract(p);
  let a = mix(morphologyLoad(base), morphologyLoad(base + vec2<i32>(1, 0)), f.x);
  let b = mix(morphologyLoad(base + vec2<i32>(0, 1)), morphologyLoad(base + vec2<i32>(1, 1)), f.x);
  return mix(a, b, f.y);
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

// Simple, deterministic PRNG (xorshift32) — growth's own random draw
// (agentStep()'s own comment) needs a DEDICATED, persistent per-particle
// state that evolves every call regardless of what else is happening,
// not a hash of something that might not change (position, heading,
// ...): a particle sitting still (e.g. pinned by a local repulsion/
// deposit balance) would keep re-deriving the exact SAME "random" draw
// every step instead of a properly evolving sequence — the same "don't
// derive persistent state from something that might not change" pitfall
// this file's own heading state already avoids re: velocity (see this
// file's own module docstring). Requires a nonzero state (xorshift's
// own fixed point at 0) — every seed this file ever writes into
// particleMeta[].rng is OR'd with 1u to guarantee that.
fn xorshift32(stateIn: u32) -> u32 {
  var state = stateIn;
  state = state ^ (state << 13u);
  state = state ^ (state >> 17u);
  state = state ^ (state << 5u);
  return state;
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

// Scatter-adds one particle's per-channel env_write values as a bounded
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
fn depositGaussian(envWrite: array<f32, ENV_WRITE_DIM>, spot: u32, centerFieldPos: vec2<f32>) {
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
      let texelCenter = vec2<f32>(f32(ti) + 0.5, f32(tj) + 0.5);
      let delta = centerFieldPos - texelCenter;
      let d2 = dot(delta, delta);
      let weight = exp(-d2 / (2.0 * sigma2));

      let wx = u32(wrapDepositIndex(ti, FIELD_WIDTH));
      let wy = u32(wrapDepositIndex(tj, FIELD_HEIGHT));
      for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
        let scaled = envWrite[spot * CHANNELS + c] * weight * DEPOSIT_SCALE;
        atomicAdd(&depositScratch[fieldIndex(c, wy, wx)], i32(round(scaled)));
      }
    }
  }
}

// The bounded subset of the network's own raw output
// this shader actually consumes — one env_write per spot per channel +
// angularAccel, independent anisotropy/division-bias controls, and the
// growth-direction signal (optionally also physical acceleration), all
// still in LOCAL frame.
struct PolicyOutput {
  envWrite: array<f32, ENV_WRITE_DIM>,
  angularAccel: f32,
  anisotropy: f32,
  divisionBias: f32,
  strafeLocal: vec2<f32>,
}

fn safeSigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

// One full Dense(HIDDEN_DIM) -> sin -> Dense(OUT_DIM) forward pass,
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
    // sin, not tanh — a periodic hidden activation, swapped in per this
    // project's own explicit request (a controlled, single-layer trial,
    // not a full SIREN rewrite — see this project's own design
    // discussion for why SIREN's actual value proposition, stable deep
    // gradient-based training of high-frequency signals, doesn't
    // transfer to a network that's mutated/selected, never backprop-
    // trained). No safeTanh-style input clamp needed the way the output
    // layer's own tanh below still has: sin's own native WGSL
    // implementation doesn't share naive tanh's specific (e^2x-1)/(e^2x+1)
    // overflow failure mode for large |x| (see safeTanh()'s own comment).
    hidden[j] = sin(acc);
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
  for (var spot: u32 = 0u; spot < SPOTS; spot = spot + 1u) {
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      let writeIndex = spot * CHANNELS + c;
      out.envWrite[writeIndex] = safeTanh(outVec[writeIndex]) * physics.maxEnvWrite;
    }
  }
  // No rotation for angularAccel — it nudges angularVelocity directly,
  // there's no "world frame" for a scalar turn rate to be rotated into.
  out.angularAccel = safeTanh(outVec[ENV_WRITE_DIM]) * physics.maxAngularAccel;
  // Reuse the two retained, formerly-unused acceleration neurons as
  // independent morphology controls.
  out.anisotropy = safeSigmoid(outVec[ENV_WRITE_DIM + 1u]);
  out.divisionBias = safeSigmoid(outVec[ENV_WRITE_DIM + 2u]);
  // Keep the raw bounded signal independent of maxStrafe. maxStrafe only
  // controls physical acceleration below; with MAX_STRAFE=0 these two NN
  // channels can still direct growth without moving particles directly.
  out.strafeLocal = vec2<f32>(safeTanh(outVec[ENV_WRITE_DIM + 3u]), safeTanh(outVec[ENV_WRITE_DIM + 4u]));
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
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    inputVec[c] = sampleValue(c, k);
    let gx = sampleGrad(0u, c, k);
    let gy = sampleGrad(FIELD_TOTAL, c, k);
    inputVec[CHANNELS + c] = gx * cosH + gy * sinH;
    inputVec[2u * CHANNELS + c] = -gx * sinH + gy * cosH;
  }
  let morphologyPos = fract(pos) * f32(MORPHOLOGY_FIELD_N);
  let morphologyGx = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(1.0, 0.0)) - sampleMorphology(morphologyPos - vec2<f32>(1.0, 0.0)));
  let morphologyGy = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(0.0, 1.0)) - sampleMorphology(morphologyPos - vec2<f32>(0.0, 1.0)));
  inputVec[3u * CHANNELS] = sampleMorphology(morphologyPos);
  inputVec[3u * CHANNELS + 1u] = morphologyGx * cosH + morphologyGy * sinH;
  inputVec[3u * CHANNELS + 2u] = -morphologyGx * sinH + morphologyGy * cosH;
  var result = evalPolicy(inputVec);

  // CHIRALITY: a second pass on the mirror-reflected input (lateral —
  // left/right, perpendicular to heading — gradient component negated;
  // VALUE and the forward component have no handedness, untouched),
  // then its OWN output un-mirrored and averaged with the first pass's.
  // That averaging is the actual symmetry-enforcing step: any left/right
  // handedness bias the raw network happens to have cancels out, only
  // the genuinely symmetric part of its response survives — see
  // simulation_settings.py's own CHIRALITY for the full reasoning.
  // angularAccel/strafeLocal's own lateral component flip sign right
  // back (a mirrored "turn right" is a "turn left"). Front/back lie on
  // the mirror axis; left/right swap when the mirrored result is mapped
  // back into the particle's real local frame.
  if (CHIRALITY) {
    var mirroredInput = inputVec;
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      mirroredInput[2u * CHANNELS + c] = -mirroredInput[2u * CHANNELS + c];
    }
    mirroredInput[3u * CHANNELS + 2u] = -mirroredInput[3u * CHANNELS + 2u];
    let mirrored = evalPolicy(mirroredInput);

    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      let front = result.envWrite[0u * CHANNELS + c];
      let left = result.envWrite[1u * CHANNELS + c];
      let back = result.envWrite[2u * CHANNELS + c];
      let right = result.envWrite[3u * CHANNELS + c];
      result.envWrite[0u * CHANNELS + c] = (front + mirrored.envWrite[0u * CHANNELS + c]) * 0.5;
      result.envWrite[2u * CHANNELS + c] = (back + mirrored.envWrite[2u * CHANNELS + c]) * 0.5;
      result.envWrite[1u * CHANNELS + c] = (left + mirrored.envWrite[3u * CHANNELS + c]) * 0.5;
      result.envWrite[3u * CHANNELS + c] = (right + mirrored.envWrite[1u * CHANNELS + c]) * 0.5;
    }
    result.angularAccel = (result.angularAccel - mirrored.angularAccel) * 0.5;
    result.anisotropy = (result.anisotropy + mirrored.anisotropy) * 0.5;
    result.divisionBias = (result.divisionBias + mirrored.divisionBias) * 0.5;
    result.strafeLocal = vec2<f32>(
      (result.strafeLocal.x + mirrored.strafeLocal.x) * 0.5,
      (result.strafeLocal.y - mirrored.strafeLocal.y) * 0.5
    );
  }

  // Four independent bounded Gaussian writes at the heading-relative
  // front/left/back/right spots. Positions are left unwrapped here so
  // depositGaussian can measure continuous distance across a seam; only
  // its destination indices wrap.
  for (var spot: u32 = 0u; spot < SPOTS; spot = spot + 1u) {
    let angle = headingVal + f32(spot) * HALF_PI;
    let spotDirection = vec2<f32>(cos(angle), sin(angle));
    let spotPos = fieldPos + physics.depositDistance * spotDirection;
    depositGaussian(result.envWrite, spot, spotPos);
  }

  let angularAccel = result.angularAccel;
  var directionLocal = vec2<f32>(0.0);
  let rawDirectionMagnitude = length(result.strafeLocal);
  if (rawDirectionMagnitude > 1e-6) {
    directionLocal = result.strafeLocal / rawDirectionMagnitude;
  }

  // Local -> world, exact inverse of the sensing rotation above.
  let strafeWorld = vec2<f32>(directionLocal.x * cosH - directionLocal.y * sinH, directionLocal.x * sinH + directionLocal.y * cosH);
  // Reuse the otherwise-disabled strafe signal as a normalized tensor
  // growth axis. Anisotropy and division bias are independent scalars.
  // Stored every macro step, including the step that starts a cell cycle,
  // so all following G2P substeps see the current policy decision.
  particleRest[pi].growthDirection = strafeWorld;
  particleRest[pi].growthControls = vec2<f32>(result.anisotropy, result.divisionBias);

  // Strafe as acceleration: added onto this particle's own CURRENT
  // velocity, then damped by physics.friction (a per-macro-step
  // retention fraction — see AgentPhysics's own friction field comment
  // for why this is a separate knob from MpmCore's own damping on the
  // same buffer). positions[pi] is NOT written here at all anymore —
  // MpmCore's own G2P pass integrates this new velocity into position
  // during this macro step's own physics substeps (see
  // training_sim.py's/simulation.ts's own step ordering: agentStep()
  // always runs before those substeps).
  velocities[pi] = (velocities[pi] + strafeWorld * physics.maxStrafe) * physics.friction;

  // Grow-then-divide cell cycle. The last substrate channel remains a
  // per-macro-step probability, but a successful draw STARTS growth
  // instead of inserting a full child immediately. g2p grows Fg until
  // g=2; only then is one grown parent replaced by two
  // baseline daughters.
  var rngNext = xorshift32(agentState.particleMeta[pi].rng);
  agentState.particleMeta[pi].rng = rngNext;
  // Top 24 bits -> a uniform float in [0,1) — an f32 only has 24 bits of
  // mantissa to begin with, so this uses every bit of precision it can.
  let draw = f32(rngNext >> 8u) * (1.0 / 16777216.0);
  // The LAST channel's sensed value is the probability of entering a
  // growth cycle this step, clamped into a valid range (the
  // chemical field itself isn't bounded to [0,1] — see
  // simulation_settings.py's own MAX_ENV_WRITE).
  let splitProb = clamp(inputVec[CHANNELS - 1u], 0.0, 1.0);
  // Division cooldown — counted down every step regardless of whether
  // this step's own draw would otherwise succeed (so it's a clean
  // macro-step countdown, not paused by bad luck on the random draw),
  // clamped at 0 rather than going negative. Gates the split below
  // alongside the probability draw; see AgentPhysics's own
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
  if (
    physics.growthEnabled > 0.5 &&
    activeCount < physics.maxActiveParticles &&
    particleRest[pi].cycleActive < 0.5 &&
    cooldownNow <= 0.0 &&
    draw < splitProb
  ) {
    particleRest[pi].cycleActive = 1.0;
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
      // Always consume the historical random-angle draw so zero-direction
      // policies remain bit-for-bit deterministic. A nonzero signed growth
      // vector replaces that random axis: +n is the NEW daughter's side.
      let angleState = xorshift32(rngNext);
      agentState.particleMeta[pi].rng = angleState; // parent's own rng now reflects BOTH draws this step
      let angleDraw = f32(angleState >> 8u) * (1.0 / 16777216.0);
      let spawnAngle = angleDraw * 2.0 * PI;
      var spawnDir = vec2<f32>(cos(spawnAngle), sin(spawnAngle));
      let directionMagnitude = length(strafeWorld);
      var directionStrength = 0.0;
      if (directionMagnitude > 1e-6) {
        spawnDir = strafeWorld / directionMagnitude;
        directionStrength = clamp(particleRest[pi].growthControls.y, 0.0, 1.0);
      }
      let halfOffset = spawnDir * (0.5 * physics.splitDisplacement);
      // Smoothly interpolate from the old center-preserving split (s=0)
      // to a fully polarized split (s=1): the parent remains at its old
      // position and the child appears one full splitDisplacement along
      // +n. Thus sign now matters even though n*n^T tensor stretch itself
      // remains axial. The deliberate pair-center shift is s*halfOffset.
      let centerShift = halfOffset * directionStrength;
      positions[pi] = fract(pos - halfOffset + centerShift);
      positions[newIndex] = fract(pos + halfOffset + centerShift);
      // "A copy of itself": heading/angularVelocity copied from this
      // particle's own CURRENT state (its pre-integration values — the
      // integrator below hasn't run yet at this point in the function).
      // Heading itself stays copied (not randomized) — only the spawn
      // POSITION is random now, not the child's own facing direction.
      agentState.particleMeta[newIndex].heading = headingVal;
      agentState.particleMeta[newIndex].angularVelocity = agentState.particleMeta[pi].angularVelocity;
      // Reseeded from the parent's own latest post-advance state (both
      // draws, `angleState`) mixed with the new slot index, not copied
      // outright — two particles sharing an identical RNG state would
      // draw the exact same "random" sequence forever after, a real,
      // easy-to-hit correlation since a child starts every subsequent
      // step from close to the same position/heading its parent had at
      // birth too.
      agentState.particleMeta[newIndex].rng = xorshift32(angleState ^ newIndex) | 1u;
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
      let identity = vec4<f32>(1.0, 0.0, 0.0, 1.0);
      particleRest[pi] = ParticleRest(identity, parentJp, 0.0, vec2<f32>(0.0), vec2<f32>(0.0));
      particleRest[newIndex] = ParticleRest(
        identity,
        parentJp,
        0.0,
        vec2<f32>(0.0),
        vec2<f32>(0.0),
      );
    } else {
      particleRest[pi].cycleActive = 0.0;
      agentState.particleMeta[pi].cooldown = physics.divisionCooldown;
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
  let newAngularVelocity = clamp((agentState.particleMeta[pi].angularVelocity + angularAccel) * physics.angularDamping, -physics.maxAngularVelocity, physics.maxAngularVelocity);
  agentState.particleMeta[pi].angularVelocity = newAngularVelocity;
  agentState.particleMeta[pi].heading = headingVal + newAngularVelocity;

}

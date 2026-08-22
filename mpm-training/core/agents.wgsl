// The evolved per-particle policy, GPU-resident — WGSL port of
// trainer/update_rule.py's Dense(128) -> sin -> Dense(16) (hidden
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
// Deposit — SPOTS=4 independently-controlled spots around the agent
// (front/left/back/right, evenly spaced 90° apart starting at the
// agent's own heading direction, each `physics.depositDistance` field-
// pixels away), not just a single deposit at the agent's own position
// the way this shader used to work. Each spot gets its own env_write per
// channel — see PolicyOutput's own comment for the resulting output
// layout, and depositGaussian()'s own comment (called from agentStep()'s
// own deposit loop) for the per-spot scatter math: a bounded Gaussian
// splat (sigma = physics.depositSigma, live-tunable), not the flat
// 4-corner bilinear scatter this shader used before. CHANNELS was
// reduced (see simulation_settings.py's own CHEM_CHANNELS comment)
// specifically to keep OUT_DIM (and so the evolved network's own output
// layer) from growing 4x along with the spot count — this shader's own
// sensing (value/grad_forward/grad_lateral) is unchanged, still sampled
// once, at the agent's own position, still via the old bilinear
// corners()/sampleValue()/sampleGrad() (only the deposit SIDE changed).
//
// Growth — each particle may spawn a copy of itself, a short
// `physics.splitDisplacement` away from its own current position, in a
// UNIFORMLY RANDOM direction (not fixed behind its own heading — see
// agentStep()'s own growth comment for why, and for the extra xorshift32
// draw this spends), with a per-step probability read straight off
// the LAST channel's own sensed VALUE (inputVec[CHANNELS-1u] — the same
// value already fed into this step's own NN input, clamped to [0,1];
// NOT a network output, so CHIRALITY's mirror-averaging never touches
// it) — an evolved policy can shape this "growth substrate" the exact
// same way it shapes any other channel, via its own existing env_write.
// See agentStep()'s own comment for the exact split logic.
//
// A new particle claims the next free slot via an atomic counter
// (`growthCount`, binding 9) — MAX_ACTIVE_PARTICLES (a compile-time
// const, like CHANNELS/HIDDEN_DIM: this shapes buffer-index bounds
// checking, not something live-tunable mid-rollout makes sense for) caps
// how many claims actually get written. Unlike CHANNELS/HIDDEN_DIM this
// isn't a fixed simulation_settings.py constant — it's evolve.py's own
// --particles (a per-run CLI choice, now the growth CAP, not a fixed
// starting count — every rollout always starts with exactly ONE
// particle; see evolve.py's own module docstring) —
// see trainer/training_sim.py's/gpu/simulation.ts's own module
// docstring for the CPU-side readback that turns this atomic's own
// value into the "official" activeCount every other pass (P2G/
// gridUpdate/G2P/repulsion, and this shader's own NEXT dispatch) reads.
// Deliberately DEFERRED like this rather than same-step: a newly
// claimed particle only starts getting its own agentStep()/physics
// once the CPU has propagated the grown count, one macro step after it
// split — a same-step alternative (always over-dispatching every pass
// to MAX_ACTIVE_PARTICLES, relying on a shared activeCount storage
// buffer instead of a uniform) was considered and rejected: P2G/
// gridUpdate/G2P/repulsion run once per PHYSICS SUBSTEP (many times
// per macro step, not once), so over-dispatching all of them to a
// generous cap rather than the actual live count would be a real,
// ALWAYS-PAID compute cost on this project's hottest inner loop, for
// every rollout, whether or not growth ever happens — a one-macro-step
// activation lag is a far cheaper, and imperceptible-at-training-scale,
// price.
//
// A newly claimed particle is a literal position+heading copy of its
// parent (see agentStep()'s own comment for exactly what's copied vs.
// reset) — F/C/Jp are NOT bound here at all, and this shader never
// writes them for a new particle: instead, every slot up to
// MAX_ACTIVE_PARTICLES is pre-reset to the exact same fresh MPM state
// (velocity=0, F=identity, C=0, Jp=1) seed_blob() already gives every
// genuinely-seeded particle, once per rollout (mpm_core.py's own
// reset_growth_buffers()) — a claimed slot already holds correct fresh
// F/C/Jp state the moment it's claimed, with zero extra bindings/writes
// needed here for those three specifically (`velocities` IS bound and
// written directly, for strafe's own sake — see below).
//
// growthState (binding 10) packs growth's own PER-PARTICLE state that
// only growth itself ever reads/writes — rng (xorshift32, seeded once
// per rollout by Agents.resetHeading() alongside heading/angularVelocity
// — see xorshift32()'s own comment for why growth's random draw needs
// dedicated state rather than hashing something that already changes
// each step) and cooldown (macro steps remaining before this slot can
// split again, counted down every step, reset to physics.divisionCooldown
// on BOTH the parent and the new child whenever a split succeeds —
// without this, a particle sitting on a strong, stable deposit could
// keep splitting every single step it's eligible, producing a burst of
// children from one spot rather than growth actually spreading out over
// time/distance). Packed into one struct/buffer, not two, specifically
// to keep this shader's own storage buffer count under the 10-per-stage
// hardware ceiling wgpu-native's Metal backend/Chrome's own Dawn backend
// both reported on this project's own dev hardware (`velocities`, added
// back below, would otherwise have pushed this shader to 11 — a real,
// confirmed uncaptured WebGPU validation error, not a hypothetical).
//
// Bound directly to MpmCore's own positions/velocities/activeCount
// buffers (see simulation.ts/agents_gpu.py).
//
// Strafe drives VELOCITY again (an acceleration, added onto the
// particle's own velocity then damped by physics.friction, integrated
// forward into position by MpmCore's own physics substeps) — see
// agentStep()'s own comment for the exact integration. This is the
// SECOND time this shader has flipped between the two interpretations:
// a direct, un-accumulating position nudge was tried in between (after
// measuring that a velocity-driving channel gets dominated by forces
// well outside this shader's own control — MpmCore's own repulsion
// produced ~20x more displacement than even a 1500x-larger strafe scale
// managed via velocity, and the corotated elastic material's own
// restoring stress measured losing 30-40% of an injected velocity within
// a single macro step's own substeps) — reverted back to velocity by
// explicit request despite those measurements, which still hold; the
// friction knob (trainer/simulation_settings.py's own FRICTION) is the
// tool available to fight that dominance, same as before. `accel` (a
// separate output channel, outVec[ENV_WRITE_DIM+1u]/[ENV_WRITE_DIM+2u])
// stays a real channel evalPolicy() computes but intentionally unread,
// unrelated to strafe either way.

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
// value + grad_x + grad_y, +2 for the agent's own (x,y) domain position
// — appended after the per-channel triples, not interleaved, so
// CHANNELS*3 stays every existing offset's own meaning unchanged
// (inputVec[CHANNELS-1u]'s own growth-probability read below, sampleValue/
// sampleGrad's own indexing, ...). See agentStep()'s own comment for
// where this gets populated and filled, and for why it's NOT mirrored
// under CHIRALITY the way grad_lateral is.
const IN_DIM: u32 = CHANNELS * 3u + 2u;

// 4 deposit spots (front/left/back/right — see agentStep()'s own
// comment for the exact angles) around the agent, each with its own
// independent env_write per channel. Must match
// trainer/simulation_settings.py's own DEPOSIT_SPOTS exactly (that
// constant is what agents_gpu.py's/agents.ts's own weight_layout()
// hardcode 4 against too) — not templated in like CHANNELS/HIDDEN_DIM
// since it's a fixed architecture choice, not something any run varies.
const SPOTS: u32 = 4u;
const ENV_WRITE_DIM: u32 = CHANNELS * SPOTS;
const OUT_DIM: u32 = ENV_WRITE_DIM + 5u; // env_write(SPOTS*CHANNELS) + angular_accel(1) + accel(2) + strafe(2)

const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = PI * 0.5;

// Growth cap — see this file's own module docstring for why this is
// templated in (like CHANNELS/HIDDEN_DIM) rather than a live uniform.
// Must match trainer/simulation_settings.py's own MAX_ACTIVE_PARTICLES
// exactly.
const MAX_ACTIVE_PARTICLES: u32 = __MAX_ACTIVE_PARTICLES__u;

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
  // Field-pixel distance from the agent's own position out to each of
  // its SPOTS deposit targets — see agentStep()'s own comment for the
  // exact per-spot direction. 0 collapses all SPOTS deposits back onto
  // the agent's own position (degenerates to this shader's old
  // single-spot behavior). trainer/simulation_settings.py's own
  // DEPOSIT_DISTANCE (3 field-pixels) is the starting value.
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
  // Gaussian splat radius (sigma), field-pixel units — SAME convention
  // depositDistance above already uses (not domain [0,1] units like
  // core/repulsion.wgsl's own splat sigma), since it's spreading a
  // deposit around a position already computed in field-pixel space
  // (agentStep()'s own `spotPos`). See depositGaussian()'s own comment
  // for the exact kernel this drives. trainer/simulation_settings.py's
  // own DEPOSIT_SIGMA is the starting value — live-tunable via
  // PhysicsPanel, for testing this splat's own shape/spread.
  depositSigma: f32,
  // This rollout's own spawn center (MpmCore's own [0,1]^2 domain units,
  // same as `positions` — trainer/evolve.py's own --spawn-x/--spawn-y /
  // types.ts's own SimulationConfig.spawnX/spawnY) — what the NN's own
  // position input (agentStep()'s own inputVec population) is measured
  // relative to, NOT the domain's own fixed (0.5,0.5) center. Unlike
  // every other field above, NOT written by setPhysics() (Agents.ts)/
  // set_physics() (agents_gpu.py) on every PhysicsPanel tick — see
  // Agents.setSpawnCenter()'s own docstring for why this gets its own,
  // separate, rollout-scoped setter into the same uniform buffer
  // instead.
  spawnX: f32,
  spawnY: f32,
}
@group(0) @binding(6) var<uniform> physics: AgentPhysics;

// Persistent per-particle state, owned by this shader (not MpmCore, not
// Environment) — see this file's own module docstring for why heading
// is no longer derived from velocity. Zeroed by Agents.resetHeading()
// whenever a rollout restarts (simulation.ts's own restartRollout()).
@group(0) @binding(7) var<storage, read_write> heading: array<f32>;
@group(0) @binding(8) var<storage, read_write> angularVelocity: array<f32>;
// Growth's own atomic "next free slot" counter — see this file's own
// module docstring for the full design. Kept in sync with the
// "official" activeCount by the CPU (Agents.setActiveCount() writes
// both together) every time it changes, not just at rollout start.
@group(0) @binding(9) var<storage, read_write> growthCount: atomic<u32>;

// rng: growth's own dedicated per-particle PRNG state (see this file's
// own module docstring and xorshift32()'s own comment for why this can't
// just be derived from other, already-changing per-particle state).
// cooldown: growth's own per-particle division cooldown, macro steps
// remaining before this slot can split again (see this file's own
// module docstring). Packed together into one struct/buffer purely to
// stay under this shader's own 10-storage-buffer ceiling — see this
// file's own module docstring for why; unrelated otherwise, nothing
// besides growth's own agentStep() logic ever reads either field.
struct GrowthState {
  rng: u32,
  cooldown: f32,
}
@group(0) @binding(10) var<storage, read_write> growthState: array<GrowthState>;
// MpmCore's own velocity buffer — see this file's own module docstring
// for why strafe drives this directly again.
@group(0) @binding(11) var<storage, read_write> velocities: array<vec2<f32>>;

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
// growthState[].rng is OR'd with 1u to guarantee that.
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
// side of 0 `v` started on. Needed now that corners() can be called with
// a deposit spot position that's stepped outside [0,size) in either
// direction (agentStep()'s own fieldPos + depositDistance*dir, which
// unlike the agent's own always-in-range fieldPos isn't pre-wrapped by
// the caller) — a no-op for anything already in [0,size), so this
// doesn't change corners()'s own behavior for that (previously the only)
// case.
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
// Smaller than repulsion's own cap (5) since this runs SPOTS(4) *
// CHANNELS(4) = 16 times per agent per macro step, not once — a wide-
// open cap here would multiply out much further than repulsion's single
// per-particle splat does.
const MAX_DEPOSIT_KERNEL_RADIUS: i32 = 3;

// Euclidean modulo, i32 in/out — same wraparound idea
// core/repulsion.wgsl's own wrapFieldIndex() already uses for its own
// (separate) splat/field, applied per-axis here since this file's own
// field can have independent FIELD_WIDTH/FIELD_HEIGHT.
fn wrapDepositIndex(i: i32, size: u32) -> i32 {
  let n = i32(size);
  return ((i % n) + n) % n;
}

// Scatter-adds one spot's own per-channel env_write values as a bounded
// Gaussian splat around `centerFieldPos` (continuous field-pixel
// coordinates, possibly outside [0,size) — see agentStep()'s own
// `spotPos` comment) — replaces this shader's old 4-corner bilinear
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
// from one spot now grows with depositSigma (more full-weight taps
// stacking up), not just its spread — a real, visible behavior change
// worth knowing while testing this slider, not merely a smoother-
// looking version of the old, mass-conserving 4-corner deposit.
fn depositGaussian(envWrite: array<f32, ENV_WRITE_DIM>, s: u32, centerFieldPos: vec2<f32>) {
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
        let scaled = envWrite[s * CHANNELS + c] * weight * DEPOSIT_SCALE;
        atomicAdd(&depositScratch[fieldIndex(c, wy, wx)], i32(round(scaled)));
      }
    }
  }
}

// The squashed, physics-scaled subset of the network's own raw output
// this shader actually consumes — env_write (per spot, per channel:
// envWrite[s * CHANNELS + c], spot-major — see this file's own module
// docstring for what each of the SPOTS=4 spots physically is) +
// angularAccel + strafeLocal (a velocity-driving acceleration, see this
// file's own module docstring), all still in LOCAL frame. accel
// (outVec[ENV_WRITE_DIM+1u]/[ENV_WRITE_DIM+2u]) is a real, separate
// output channel but intentionally not carried through here at all —
// unused.
struct PolicyOutput {
  envWrite: array<f32, ENV_WRITE_DIM>,
  angularAccel: f32,
  strafeLocal: vec2<f32>,
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
  for (var s: u32 = 0u; s < SPOTS; s = s + 1u) {
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      out.envWrite[s * CHANNELS + c] = safeTanh(outVec[s * CHANNELS + c]) * physics.maxEnvWrite;
    }
  }
  // No rotation for angularAccel — it nudges angularVelocity directly,
  // there's no "world frame" for a scalar turn rate to be rotated into.
  out.angularAccel = safeTanh(outVec[ENV_WRITE_DIM]) * physics.maxAngularAccel;
  out.strafeLocal = vec2<f32>(safeTanh(outVec[ENV_WRITE_DIM + 3u]), safeTanh(outVec[ENV_WRITE_DIM + 4u])) * physics.maxStrafe;
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
  let headingVal = heading[pi];
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
  // Agent's own domain position (`pos`, MpmCore's own [0,1]^2 — NOT
  // fieldPos's field-pixel space) RELATIVE TO THIS ROLLOUT'S OWN SPAWN
  // CENTER (physics.spawnX/spawnY), not the domain's own fixed (0.5,0.5)
  // center — "how far have I drifted from where growth started," not
  // "where am I in the unit square" (those two only coincide if
  // --spawn-x/--spawn-y happen to be exactly 0.5,0.5). Wrapped into
  // [-0.5,0.5) per axis (minimum-image convention: subtract the nearest
  // integer, correct since the domain is size 1 and toroidal — same
  // "shortest distance across the seam" reasoning core/repulsion.wgsl's
  // own splatDensity() gives for its own falloff calculation) rather
  // than a naive difference, which would be wrong by up to 1.0 for an
  // agent that's actually close to spawn center but wrapped across the
  // domain edge. Already naturally small-magnitude/roughly zero-centered
  // (growth starts AT spawn center and rarely needs to explore the
  // entire wrapped domain), so no extra scaling like a raw domain
  // position would need. World-frame, NOT rotated into heading-local
  // frame the way grad_forward/grad_lateral above are — there's no
  // meaningful "local-frame position." Appended after the per-channel
  // inputs (see IN_DIM's own comment for the offset).
  var dx = pos.x - physics.spawnX;
  dx = dx - round(dx);
  var dy = pos.y - physics.spawnY;
  dy = dy - round(dy);
  inputVec[3u * CHANNELS] = dx;
  inputVec[3u * CHANNELS + 1u] = dy;

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
  // back (a mirrored "turn right" is a "turn left"). envWrite's spots
  // are handled the same way, just per-spot instead of a single sign
  // flip: spot 0 (front) and spot 2 (back) sit ON the mirror axis, so
  // they un-mirror straight across, same as before (average with the
  // SAME spot index); spots 1 and 3 (left/right) sit OFF axis, so a
  // reflection swaps them — the mirrored pass's own "left spot" output
  // is what its "right spot" becomes once un-mirrored back into the real
  // frame, and vice versa (see agentStep()'s own comment below for why
  // spot 1 = left, spot 3 = right).
  if (CHIRALITY) {
    var mirroredInput = inputVec;
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      mirroredInput[2u * CHANNELS + c] = -mirroredInput[2u * CHANNELS + c];
    }
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
    result.strafeLocal = vec2<f32>(
      (result.strafeLocal.x + mirrored.strafeLocal.x) * 0.5,
      (result.strafeLocal.y - mirrored.strafeLocal.y) * 0.5
    );
  }

  // Deposit into SPOTS=4 independent spots around the agent — front
  // (heading direction, s=0), left (+90° from heading, s=1), back
  // (+180°, s=2), right (+270°/-90°, s=3), each `physics.depositDistance`
  // field-pixels out. +90° per step in a standard (cos,sin) frame is a
  // CCW rotation, which is the same direction the sensed gradient's own
  // lateral component already treats as "left" (inputVec's own
  // `-gx*sinH+gy*cosH` above is exactly a +90° rotation of the forward
  // vector) — so this ordering is what makes the CHIRALITY swap above
  // (spots 1 and 3) correct. depositDistance=0 collapses every spot back
  // onto the agent's own position. Each spot gets its own bounded
  // Gaussian splat (depositGaussian(), sigma = physics.depositSigma) —
  // see that function's own comment for why this replaced the old
  // 4-corner bilinear scatter, and for the mass-vs-spread tradeoff that
  // comes with it — since a spot can land in a different field cell than
  // the agent itself, or than another spot.
  for (var s: u32 = 0u; s < SPOTS; s = s + 1u) {
    let angle = headingVal + f32(s) * HALF_PI;
    let dir = vec2<f32>(cos(angle), sin(angle));
    let spotPos = fieldPos + dir * physics.depositDistance;
    depositGaussian(result.envWrite, s, spotPos);
  }

  let angularAccel = result.angularAccel;
  let strafeLocal = result.strafeLocal;

  // Local -> world, exact inverse of the sensing rotation above.
  let strafeWorld = vec2<f32>(strafeLocal.x * cosH - strafeLocal.y * sinH, strafeLocal.x * sinH + strafeLocal.y * cosH);

  // Strafe as acceleration: added onto this particle's own CURRENT
  // velocity, then damped by physics.friction (a per-macro-step
  // retention fraction — see AgentPhysics's own friction field comment
  // for why this is a separate knob from MpmCore's own damping on the
  // same buffer). positions[pi] is NOT written here at all anymore —
  // MpmCore's own G2P pass integrates this new velocity into position
  // during this macro step's own physics substeps (see
  // training_sim.py's/simulation.ts's own step ordering: agentStep()
  // always runs before those substeps).
  velocities[pi] = (velocities[pi] + strafeWorld) * physics.friction;

  // Growth: this particle may split, spawning a copy of itself a short
  // `physics.splitDisplacement` away from its own current position, in a
  // UNIFORMLY RANDOM direction (see the extra xorshift32 draw below) —
  // was fixed at exactly 180° behind its own heading, changed per this
  // project's own explicit request: children spawning around a parent,
  // not only ever trailing it. See this file's own module docstring for
  // the full design (probability source, deferred-by-one-macro-step
  // activation). Uses `pos` (this step's own STARTING position, read at
  // the top of this function) rather than anything strafe-adjusted,
  // since strafe no longer moves position directly at all.
  var rngNext = xorshift32(growthState[pi].rng);
  growthState[pi].rng = rngNext;
  // Top 24 bits -> a uniform float in [0,1) — an f32 only has 24 bits of
  // mantissa to begin with, so this uses every bit of precision it can.
  let draw = f32(rngNext >> 8u) * (1.0 / 16777216.0);
  // The LAST channel's own sensed VALUE — already computed above, into
  // this step's own NN input, NOT a network output (CHIRALITY's own
  // mirror-averaging never applies here) — doubles as this particle's
  // own split probability this step, clamped into a valid range (the
  // chemical field itself isn't bounded to [0,1] — see
  // simulation_settings.py's own MAX_ENV_WRITE).
  let splitProb = clamp(inputVec[CHANNELS - 1u], 0.0, 1.0);
  // Division cooldown — counted down every step regardless of whether
  // this step's own draw would otherwise succeed (so it's a clean
  // macro-step countdown, not paused by bad luck on the random draw),
  // clamped at 0 rather than going negative. Gates the split below
  // alongside the probability draw; see AgentPhysics's own
  // divisionCooldown field comment for the full reasoning.
  let cooldownNow = max(growthState[pi].cooldown - 1.0, 0.0);
  growthState[pi].cooldown = cooldownNow;
  if (cooldownNow <= 0.0 && draw < splitProb) {
    // atomicAdd returns the OLD value — the slot THIS particle just
    // claimed. Never gated before the add (that would need a compare-
    // exchange loop to stay race-free against every OTHER agent that
    // might also split this exact step) — instead a claim landing at or
    // past MAX_ACTIVE_PARTICLES is simply never written below (this
    // buffer is sized to the much larger MAX_PARTICLES — see
    // mpm_core.py's own module docstring — so an over-claimed index is
    // never actually out of bounds, just wasted atomic traffic on a step
    // where many agents split near the cap at once); trainer/
    // training_sim.py's/gpu/simulation.ts's own readback separately
    // clamps the *reported* activeCount to MAX_ACTIVE_PARTICLES
    // regardless of how high this atomic itself climbs.
    let newIndex = atomicAdd(&growthCount, 1u);
    if (newIndex < MAX_ACTIVE_PARTICLES) {
      // Uniformly random spawn direction — a second xorshift32 draw off
      // the SAME persistent per-particle stream `rngNext` above already
      // advanced (not a hash of position/heading/anything else that
      // might not change — see this file's own xorshift32() comment for
      // why persistent, always-advancing state matters here), only spent
      // when a split actually happens, not every step. Was a fixed
      // -1*(cosH,sinH) (directly behind heading) — now a fresh angle
      // each split instead, per this project's own explicit request.
      let angleState = xorshift32(rngNext);
      growthState[pi].rng = angleState; // parent's own rng now reflects BOTH draws this step
      let angleDraw = f32(angleState >> 8u) * (1.0 / 16777216.0);
      let spawnAngle = angleDraw * 2.0 * PI;
      let spawnDir = vec2<f32>(cos(spawnAngle), sin(spawnAngle));
      positions[newIndex] = fract(pos + spawnDir * physics.splitDisplacement);
      // "A copy of itself": heading/angularVelocity copied from this
      // particle's own CURRENT state (its pre-integration values — the
      // integrator below hasn't run yet at this point in the function).
      // Heading itself stays copied (not randomized) — only the spawn
      // POSITION is random now, not the child's own facing direction.
      heading[newIndex] = headingVal;
      angularVelocity[newIndex] = angularVelocity[pi];
      // Reseeded from the parent's own latest post-advance state (both
      // draws, `angleState`) mixed with the new slot index, not copied
      // outright — two particles sharing an identical RNG state would
      // draw the exact same "random" sequence forever after, a real,
      // easy-to-hit correlation since a child starts every subsequent
      // step from close to the same position/heading its parent had at
      // birth too.
      growthState[newIndex].rng = xorshift32(angleState ^ newIndex) | 1u;
      // BOTH the parent (this slot, `pi`) and the new child go on
      // cooldown — a freshly split child immediately splitting again
      // would defeat the whole point of throttling growth, same as a
      // parent that just split shouldn't either.
      growthState[pi].cooldown = physics.divisionCooldown;
      growthState[newIndex].cooldown = physics.divisionCooldown;
      // Fresh velocity, not copied from the parent — same "start from
      // the exact same rest state seed_blob() gives every genuinely-
      // seeded particle" convention F/C/Jp already follow via
      // mpm_core.py's own reset_growth_buffers() (see this file's own
      // module docstring). Explicit here (unlike F/C/Jp) because
      // `velocities` is bound/written directly in this shader now,
      // where those three still aren't.
      velocities[newIndex] = vec2<f32>(0.0, 0.0);
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
  let newAngularVelocity = clamp((angularVelocity[pi] + angularAccel) * physics.angularDamping, -physics.maxAngularVelocity, physics.maxAngularVelocity);
  angularVelocity[pi] = newAngularVelocity;
  heading[pi] = headingVal + newAngularVelocity;
}

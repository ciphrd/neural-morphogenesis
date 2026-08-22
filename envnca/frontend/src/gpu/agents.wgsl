// Per-agent sense -> MLP -> integrate motion -> deposit-scatter, one
// invocation per agent. Mirrors simulation.py::Simulation.step()'s exact
// ordering and math — see that file for the authoritative Python source
// this was ported from. All __NAME__ tokens are substituted by
// shaderTemplate.ts before compilation (WGSL has no preprocessor, and
// the scratch arrays below need compile-time-known sizes).

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;
const HIDDEN: u32 = __HIDDEN__u;
const AGENT_COUNT: u32 = __AGENT_COUNT__u;
const IN: u32 = 3u * CHANNELS;
const OUT: u32 = CHANNELS + 4u;
const PLANE_SIZE: u32 = CHANNELS * HEIGHT * WIDTH;

// repulsion.wgsl's own field resolution — independent of (and coarser
// than) WIDTH/HEIGHT above, see that file's header comment. Only needed
// here for sampleRepulsionGradient()'s bilinear lookup below.
const REPULSION_RESOLUTION: u32 = __REPULSION_RESOLUTION__u;
const REPULSION_FIELD_SIZE: u32 = REPULSION_RESOLUTION * REPULSION_RESOLUTION;


// Combined weights buffer layout — fc1w, fc1b, fc2w, fc2b back to back,
// row-major (out_features, in_features) per nn.Linear's own convention
// (see update_rule.py::export_weights()'s docstring: y = x @ W.T + b).
const FC1W_OFFSET: u32 = __FC1W_OFFSET__u;
const FC1B_OFFSET: u32 = __FC1B_OFFSET__u;
const FC2W_OFFSET: u32 = __FC2W_OFFSET__u;
const FC2B_OFFSET: u32 = __FC2B_OFFSET__u;

// Must match environment.wgsl's DEPOSIT_SCALE. env_write is tanh-squashed
// to physics.maxEnvWrite before it ever reaches depositScatter() below
// (see AgentPhysics), but this headroom is kept generous regardless — see
// gpu/agents.ts's own comment for the overflow-budget reasoning behind
// this constant and the pre-add clamp below.
const DEPOSIT_SCALE: f32 = 4096.0;
const DEPOSIT_CLAMP: f32 = 1073741824.0; // 2^30

@group(0) @binding(0) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> depositScratch: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> velocity: array<vec2<f32>>;
// Owned by gpu/repulsion.ts's GpuRepulsion, computed in the passes
// immediately before this one each step (see gpu/simulation.ts's own
// step()) — gx plane [0,REPULSION_FIELD_SIZE), gy plane
// [REPULSION_FIELD_SIZE, 2*REPULSION_FIELD_SIZE), same two-plane layout
// as `gradient` above.
@group(0) @binding(7) var<storage, read> repulsionGradient: array<f32>;

// Live-adjustable, unlike CHANNELS/WIDTH/HEIGHT/HIDDEN/AGENT_COUNT above
// (those are compile-time consts sizing arrays/dispatches/buffers — see
// this file's header). A real uniform buffer so the frontend's "Physics"
// panel can push new values on every slider tick via a cheap
// queue.writeBuffer, without recompiling this pipeline or touching
// positions/velocity — see gpu/agents.ts's setPhysics(). Initialized
// from the training run's own MAX_SPEED/MAX_ACCEL/MAX_STRAFE
// (constants.py).
struct AgentPhysics {
  maxSpeed: f32,
  maxAccel: f32,
  maxStrafe: f32,
  // Ceiling on env_write, deposited into the grid below — see
  // constants.MAX_ENV_WRITE's own docstring (simulation.py) for why this
  // needed to be bounded at all: unbounded, it accumulates over a
  // rollout (the grid decays slowly) until the network's own first
  // Linear -> Tanh layer saturates, killing gradient-descent training
  // dead (a failure mode evolutionary training is immune to, which is
  // why this wasn't caught until GD training was attempted). Squashed
  // here the same way accel/strafe already are, not left raw.
  maxEnvWrite: f32,
  // repulsion.wgsl's splatRepulsion reads this same field from this same
  // buffer (see gpu/repulsion.ts's bindAgents()) — struct layout must
  // match that file's own (separately declared) AgentPhysics exactly.
  repulsionSigma: f32,
  // Scales sampleRepulsionGradient()'s result before it's folded into
  // velocity below — 0 (the default) means exactly zero repulsion force,
  // not just a small one.
  repulsionStrength: f32,
}
@group(0) @binding(6) var<uniform> physics: AgentPhysics;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * HEIGHT * WIDTH + y * WIDTH + x;
}

struct BilinearWeights {
  x0: u32,
  x1: u32,
  y0: u32,
  y1: u32,
  w00: f32,
  w10: f32,
  w01: f32,
  w11: f32,
};

// Toroidal: corner indices wrap around each axis rather than clamping to
// an edge texel — mirrors environment.py's Environment._corners() (torch
// doesn't offer a circular padding_mode for grid_sample, which is why
// both that method and this one hand-roll the gather instead). `%` on a
// non-negative i32 is exact modulo, and x0i/y0i here are always
// non-negative since px/py arrive already wrapped into [0, WIDTH)/
// [0, HEIGHT) by this same shader's own position wrap below (and by
// rng.ts's spawn seeding on the very first step).
fn bilinearWeights(px: f32, py: f32) -> BilinearWeights {
  let x0f = floor(px);
  let y0f = floor(py);
  let x0i = i32(x0f);
  let y0i = i32(y0f);
  let wx1 = px - x0f;
  let wx0 = 1.0 - wx1;
  let wy1 = py - y0f;
  let wy0 = 1.0 - wy1;
  var out: BilinearWeights;
  out.x0 = u32(x0i % i32(WIDTH));
  out.x1 = u32((x0i + 1) % i32(WIDTH));
  out.y0 = u32(y0i % i32(HEIGHT));
  out.y1 = u32((y0i + 1) % i32(HEIGHT));
  out.w00 = wx0 * wy0;
  out.w10 = wx1 * wy0;
  out.w01 = wx0 * wy1;
  out.w11 = wx1 * wy1;
  return out;
}

fn sampleGrid(c: u32, w: BilinearWeights) -> f32 {
  return w.w00 * gridCurrent[gridIndex(c, w.y0, w.x0)]
       + w.w10 * gridCurrent[gridIndex(c, w.y0, w.x1)]
       + w.w01 * gridCurrent[gridIndex(c, w.y1, w.x0)]
       + w.w11 * gridCurrent[gridIndex(c, w.y1, w.x1)];
}

fn sampleGradX(c: u32, w: BilinearWeights) -> f32 {
  return w.w00 * gradient[gridIndex(c, w.y0, w.x0)]
       + w.w10 * gradient[gridIndex(c, w.y0, w.x1)]
       + w.w01 * gradient[gridIndex(c, w.y1, w.x0)]
       + w.w11 * gradient[gridIndex(c, w.y1, w.x1)];
}

fn sampleGradY(c: u32, w: BilinearWeights) -> f32 {
  return w.w00 * gradient[PLANE_SIZE + gridIndex(c, w.y0, w.x0)]
       + w.w10 * gradient[PLANE_SIZE + gridIndex(c, w.y0, w.x1)]
       + w.w01 * gradient[PLANE_SIZE + gridIndex(c, w.y1, w.x0)]
       + w.w11 * gradient[PLANE_SIZE + gridIndex(c, w.y1, w.x1)];
}

// Always-non-negative modulo — same reasoning as repulsion.wgsl's own
// wrapIndex (WGSL's `%` on a negative i32 keeps its sign, unlike
// Python's), needed here because a corner index one cell past
// REPULSION_RESOLUTION-1 must wrap back to 0, not go negative.
fn wrapRepulsionIndex(v: i32, m: i32) -> u32 {
  return u32(((v % m) + m) % m);
}

// Same bilinear-corner pattern as bilinearWeights()/sampleGrid() above,
// against repulsionGradient's own (independent, coarser) resolution
// instead of WIDTH/HEIGHT — see repulsion.wgsl's header comment for why
// that field has its own resolution at all.
fn sampleRepulsionGradient(px: f32, py: f32) -> vec2<f32> {
  let scale = f32(REPULSION_RESOLUTION) / f32(WIDTH); // assumes a square grid, WIDTH == HEIGHT
  let fx = px * scale;
  let fy = py * scale;
  let x0f = floor(fx);
  let y0f = floor(fy);
  let x0i = i32(x0f);
  let y0i = i32(y0f);
  let wx1 = fx - x0f;
  let wx0 = 1.0 - wx1;
  let wy1 = fy - y0f;
  let wy0 = 1.0 - wy1;
  let x0 = wrapRepulsionIndex(x0i, i32(REPULSION_RESOLUTION));
  let x1 = wrapRepulsionIndex(x0i + 1, i32(REPULSION_RESOLUTION));
  let y0 = wrapRepulsionIndex(y0i, i32(REPULSION_RESOLUTION));
  let y1 = wrapRepulsionIndex(y0i + 1, i32(REPULSION_RESOLUTION));

  let gx = wx0 * wy0 * repulsionGradient[y0 * REPULSION_RESOLUTION + x0]
         + wx1 * wy0 * repulsionGradient[y0 * REPULSION_RESOLUTION + x1]
         + wx0 * wy1 * repulsionGradient[y1 * REPULSION_RESOLUTION + x0]
         + wx1 * wy1 * repulsionGradient[y1 * REPULSION_RESOLUTION + x1];
  let gy = wx0 * wy0 * repulsionGradient[REPULSION_FIELD_SIZE + y0 * REPULSION_RESOLUTION + x0]
         + wx1 * wy0 * repulsionGradient[REPULSION_FIELD_SIZE + y0 * REPULSION_RESOLUTION + x1]
         + wx0 * wy1 * repulsionGradient[REPULSION_FIELD_SIZE + y1 * REPULSION_RESOLUTION + x0]
         + wx1 * wy1 * repulsionGradient[REPULSION_FIELD_SIZE + y1 * REPULSION_RESOLUTION + x1];
  return vec2<f32>(gx, gy);
}

// Scatter-add is the mathematical transpose of the gather above: same
// 4 corners, same weights, opposite direction — collisions (multiple
// agents landing on/near the same texel) are a real, expected, per-step
// occurrence (no collision pass in this project), so contributions must
// sum, not overwrite. No native float atomics exist in core WebGPU on
// buffers or textures, so this scatter goes through a fixed-point i32
// buffer (environment.wgsl's mergeDeposit decodes it back into the float
// grid next). SCALE=4096 gives ~524,288 headroom per texel before i32
// overflow even under a pathological worst case; the pre-add clamp
// guards against WGSL's silent (non-trapping) i32 wraparound poisoning
// a whole channel from one divergent agent.
fn depositAt(c: u32, y: u32, x: u32, v: f32) {
  let scaled = clamp(v * DEPOSIT_SCALE, -DEPOSIT_CLAMP, DEPOSIT_CLAMP);
  atomicAdd(&depositScratch[gridIndex(c, y, x)], i32(round(scaled)));
}

fn depositScatter(w: BilinearWeights, c: u32, value: f32) {
  depositAt(c, w.y0, w.x0, value * w.w00);
  depositAt(c, w.y0, w.x1, value * w.w10);
  depositAt(c, w.y1, w.x0, value * w.w01);
  depositAt(c, w.y1, w.x1, value * w.w11);
}

// tanh(x) mathematically saturates to ±1 well before |x|=20 (tanh(20) is
// 1.0 to float32 precision), so clamping the argument first changes
// nothing for any input that would ever legitimately occur — but it
// closes off a real failure mode confirmed on this GPU backend: a
// naive tanh built on (e^2x-1)/(e^2x+1) produces NaN, not ±1, once
// e^2x itself overflows to Infinity for large |x| (Infinity/Infinity).
// Even with env_write now squashed by physics.maxEnvWrite (see
// AgentPhysics), the raw pre-tanh dot product feeding *this* tanh call
// can still be large for an adversarial/diverged weight set — so this
// still only has to handle attacker-scale inputs, not typical ones.
fn safeTanh(x: f32) -> f32 {
  return tanh(clamp(x, -20.0, 20.0));
}

@compute @workgroup_size(64)
fn agentStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= AGENT_COUNT) { return; }

  let pos = positions[i];
  // Sanitize velocity BEFORE it's used for anything, rather than trying
  // to detect a NaN component via comparison (`vel.x == vel.x` is false
  // for NaN under strict IEEE 754, but this backend's shader compiler
  // evidently applies fast-math-style optimizations that constant-fold
  // self-comparison to `true` regardless — confirmed empirically: an
  // earlier version of this guard, built on exactly that comparison,
  // did not stop NaN headings from occurring). clamp() itself, on this
  // same backend, reliably resolves a NaN operand to one of its finite
  // bounds instead of propagating NaN (this is *why* a NaN position
  // consistently lands at exactly grid corner (0,0) rather than staying
  // NaN — see simulation()'s own position clamp) — so reuse that same,
  // empirically-proven-safe primitive here rather than a comparison.
  // Bounding to MAX_SPEED is also just correct on its own terms: valid
  // velocity is never supposed to exceed it anyway.
  let vel = clamp(velocity[i], vec2<f32>(-physics.maxSpeed), vec2<f32>(physics.maxSpeed));

  // heading never stored — derived fresh from velocity every step, same
  // convention as simulation.py (atan2(0,0) = 0, a resting agent has
  // heading 0 until it actually starts moving). Explicitly guarded,
  // NOT just `atan2(vel.y, vel.x)` — unlike torch.atan2, WGSL leaves
  // atan2(0,0) implementation-defined, and on at least one WebGPU
  // backend it produces NaN. Every agent starts at velocity (0,0) (see
  // agent_state.py's seed()), so an unguarded call here would poison
  // every agent's position with NaN on the very first step. `vel` is
  // now guaranteed finite by the clamp above, so this only has to
  // handle the literal (0,0) case.
  var heading: f32 = 0.0;
  if (vel.x != 0.0 || vel.y != 0.0) {
    heading = atan2(vel.y, vel.x);
  }
  let cosH = cos(heading);
  let sinH = sin(heading);

  // Sense at the CURRENT (pre-move) position, on the grid state left
  // over from the previous step's diffuseDecay — matches
  // simulation.py's step() ordering exactly.
  let senseWeights = bilinearWeights(pos.x, pos.y);

  var inputVec: array<f32, IN>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    inputVec[c] = sampleGrid(c, senseWeights);
  }
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    let gx = sampleGradX(c, senseWeights);
    let gy = sampleGradY(c, senseWeights);
    // World -> local-frame rotation (forward = heading, lateral = 90°
    // left), identical to simulation.py's own rotation.
    inputVec[CHANNELS + c] = gx * cosH + gy * sinH;
    inputVec[2u * CHANNELS + c] = -gx * sinH + gy * cosH;
  }
  // Heading (cosH/sinH) is NOT fed into the network — see
  // update_rule.py's own "Heading" docstring section. Still used above
  // (sensing rotation) and below (accel/strafe rotation back to world).

  // fc1 -> tanh
  var hidden: array<f32, HIDDEN>;
  for (var j: u32 = 0u; j < HIDDEN; j = j + 1u) {
    var acc: f32 = weights[FC1B_OFFSET + j];
    for (var k: u32 = 0u; k < IN; k = k + 1u) {
      acc = acc + inputVec[k] * weights[FC1W_OFFSET + j * IN + k];
    }
    hidden[j] = safeTanh(acc);
  }

  // fc2 (raw) -> env_write(C) | local_accel(2) | local_strafe(2) — env_write
  // gets squashed below (physics.maxEnvWrite), accel/strafe further down.
  var outVec: array<f32, OUT>;
  for (var j: u32 = 0u; j < OUT; j = j + 1u) {
    var acc: f32 = weights[FC2B_OFFSET + j];
    for (var k: u32 = 0u; k < HIDDEN; k = k + 1u) {
      acc = acc + hidden[k] * weights[FC2W_OFFSET + j * HIDDEN + k];
    }
    outVec[j] = acc;
  }

  // local_accel: per-component tanh squash, then scale — distinct from
  // strafe's magnitude-based squash below, see update_rule.py/
  // simulation.py for why these two conventions differ.
  let accelLocal = vec2<f32>(safeTanh(outVec[CHANNELS]) * physics.maxAccel, safeTanh(outVec[CHANNELS + 1u]) * physics.maxAccel);
  let accelWorld = vec2<f32>(
    accelLocal.x * cosH - accelLocal.y * sinH,
    accelLocal.x * sinH + accelLocal.y * cosH
  );

  // Repulsion force, from this same pre-move `pos` snapshot (see
  // gpu/simulation.ts's step() for why the splat/gradient passes feeding
  // repulsionGradient run before this one) — world-frame already (no
  // heading-relative notion for this force, unlike accel/strafe), so it
  // folds straight into velocity below with no rotation step of its own.
  let repulsionWorld = sampleRepulsionGradient(pos.x, pos.y) * -physics.repulsionStrength;

  // Integrate velocity, then clamp its MAGNITUDE (never scales up, only
  // down) — not each component independently.
  var newVel = vel + accelWorld + repulsionWorld;
  let speedSafe = max(length(newVel), 1e-9);
  newVel = newVel * min(physics.maxSpeed / speedSafe, 1.0);

  // local_strafe: magnitude-based tanh squash, direction preserved —
  // applied straight to position this step, never to velocity, never
  // persisted (recomputed fresh every step).
  let strafeRaw = vec2<f32>(outVec[CHANNELS + 2u], outVec[CHANNELS + 3u]);
  let strafeMagSafe = max(length(strafeRaw), 1e-9);
  let strafeLocal = strafeRaw * (safeTanh(strafeMagSafe) * physics.maxStrafe / strafeMagSafe);
  let strafeWorld = vec2<f32>(
    strafeLocal.x * cosH - strafeLocal.y * sinH,
    strafeLocal.x * sinH + strafeLocal.y * cosH
  );

  // Toroidal wrap, not a clamp — see simulation.py's _wrap_to_grid() and
  // environment.py's module docstring for why the grid has no edge.
  // `x - floor(x / w) * w` is true (always-non-negative) mathematical
  // mod, unlike WGSL's `%` on floats (which, like C's fmod, can return a
  // negative result for a negative operand) — matches torch.remainder's
  // semantics exactly.
  var newPos = pos + newVel + strafeWorld;
  newPos.x = newPos.x - floor(newPos.x / f32(WIDTH)) * f32(WIDTH);
  newPos.y = newPos.y - floor(newPos.y / f32(HEIGHT)) * f32(HEIGHT);

  positions[i] = newPos;
  velocity[i] = newVel;

  // Deposit env_write, squashed by physics.maxEnvWrite (matches
  // simulation.py's step(): torch.tanh(env_write) * MAX_ENV_WRITE), at
  // the NEW (post-move) position — matches simulation.py's ordering
  // (deposit after motion, diffuse+decay runs after all agents have
  // deposited).
  let depositWeights = bilinearWeights(newPos.x, newPos.y);
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    let envWrite = safeTanh(outVec[c]) * physics.maxEnvWrite;
    depositScatter(depositWeights, c, envWrite);
  }
}

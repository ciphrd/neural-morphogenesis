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

const MAX_SPEED: f32 = __MAX_SPEED__;
const MAX_ACCEL: f32 = __MAX_ACCEL__;
const MAX_STRAFE: f32 = __MAX_STRAFE__;
const EDGE_MARGIN: f32 = __EDGE_MARGIN__;

// Combined weights buffer layout — fc1w, fc1b, fc2w, fc2b back to back,
// row-major (out_features, in_features) per nn.Linear's own convention
// (see update_rule.py::export_weights()'s docstring: y = x @ W.T + b).
const FC1W_OFFSET: u32 = __FC1W_OFFSET__u;
const FC1B_OFFSET: u32 = __FC1B_OFFSET__u;
const FC2W_OFFSET: u32 = __FC2W_OFFSET__u;
const FC2B_OFFSET: u32 = __FC2B_OFFSET__u;

// Must match environment.wgsl's DEPOSIT_SCALE. env_write is the network's
// raw, unsquashed output (no tanh bounds it), so headroom matters more
// than for accel/strafe — see gpu/agents.ts's own comment for the
// overflow-budget reasoning behind this constant and the pre-add clamp
// below.
const DEPOSIT_SCALE: f32 = 4096.0;
const DEPOSIT_CLAMP: f32 = 1073741824.0; // 2^30

@group(0) @binding(0) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> depositScratch: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> velocity: array<vec2<f32>>;

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

// Matches PyTorch grid_sample(align_corners=True, padding_mode="border")
// exactly: corner indices clamp to the nearest edge texel rather than
// wrapping or zero-padding (a different convention from the
// zero-padded convolutions in environment.wgsl — see this project's own
// design notes on why the two must not be conflated).
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
  out.x0 = u32(clamp(x0i, 0, i32(WIDTH) - 1));
  out.x1 = u32(clamp(x0i + 1, 0, i32(WIDTH) - 1));
  out.y0 = u32(clamp(y0i, 0, i32(HEIGHT) - 1));
  out.y1 = u32(clamp(y0i + 1, 0, i32(HEIGHT) - 1));
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
// env_write is raw/unbounded (see this module's own docstring), and
// once deposited it can reappear as a large-but-finite *sensed* value
// on a later step (grid_current itself never overflows — only the raw,
// pre-tanh dot product computed from it can) — so this only has to
// handle attacker-scale inputs, not typical ones.
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
  let vel = clamp(velocity[i], vec2<f32>(-MAX_SPEED), vec2<f32>(MAX_SPEED));

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

  // fc1 -> tanh
  var hidden: array<f32, HIDDEN>;
  for (var j: u32 = 0u; j < HIDDEN; j = j + 1u) {
    var acc: f32 = weights[FC1B_OFFSET + j];
    for (var k: u32 = 0u; k < IN; k = k + 1u) {
      acc = acc + inputVec[k] * weights[FC1W_OFFSET + j * IN + k];
    }
    hidden[j] = safeTanh(acc);
  }

  // fc2 (raw, unsquashed) -> env_write(C) | local_accel(2) | local_strafe(2)
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
  let accelLocal = vec2<f32>(safeTanh(outVec[CHANNELS]) * MAX_ACCEL, safeTanh(outVec[CHANNELS + 1u]) * MAX_ACCEL);
  let accelWorld = vec2<f32>(
    accelLocal.x * cosH - accelLocal.y * sinH,
    accelLocal.x * sinH + accelLocal.y * cosH
  );

  // Integrate velocity, then clamp its MAGNITUDE (never scales up, only
  // down) — not each component independently.
  var newVel = vel + accelWorld;
  let speedSafe = max(length(newVel), 1e-9);
  newVel = newVel * min(MAX_SPEED / speedSafe, 1.0);

  // local_strafe: magnitude-based tanh squash, direction preserved —
  // applied straight to position this step, never to velocity, never
  // persisted (recomputed fresh every step).
  let strafeRaw = vec2<f32>(outVec[CHANNELS + 2u], outVec[CHANNELS + 3u]);
  let strafeMagSafe = max(length(strafeRaw), 1e-9);
  let strafeLocal = strafeRaw * (safeTanh(strafeMagSafe) * MAX_STRAFE / strafeMagSafe);
  let strafeWorld = vec2<f32>(
    strafeLocal.x * cosH - strafeLocal.y * sinH,
    strafeLocal.x * sinH + strafeLocal.y * cosH
  );

  var newPos = pos + newVel + strafeWorld;
  newPos.x = clamp(newPos.x, 0.0, f32(WIDTH) - EDGE_MARGIN);
  newPos.y = clamp(newPos.y, 0.0, f32(HEIGHT) - EDGE_MARGIN);

  positions[i] = newPos;
  velocity[i] = newVel;

  // Deposit env_write (raw, unclamped) at the NEW (post-move) position —
  // matches simulation.py's ordering (deposit after motion, diffuse+decay
  // runs after all agents have deposited).
  let depositWeights = bilinearWeights(newPos.x, newPos.y);
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    depositScatter(depositWeights, c, outVec[c]);
  }
}

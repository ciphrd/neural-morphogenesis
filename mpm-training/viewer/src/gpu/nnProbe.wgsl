// Viewer-only diagnostic: recomputes ONE particle's (always index 0 — the
// rollout's original seed particle, guaranteed to exist regardless of
// activeCount, since every rollout always starts with exactly one) full
// agentStep() sensing + evalPolicy() forward pass, for the "Network"
// panel's own brain visualization (see ui/NetworkPanel.tsx / gpu/nnProbe.ts).
//
// A close, deliberate DUPLICATE of ../../../core/agents.wgsl's own
// sensing/evalPolicy code — not a shared import (WGSL has no such
// mechanism), and this file lives in viewer/src/gpu/, NOT core/,
// specifically so this diagnostic never touches the training-critical
// shared shader — see gpu/interact.wgsl's own module docstring for the
// same, already-established reasoning. Skips agentStep()'s own deposit/
// growth/velocity side effects entirely (a probe, not a step) — nothing
// here computes them, keeps them in sync with core/agents.wgsl by hand,
// or risks silently drifting from it mattering, since they're not part
// of what the visualization reads.
//
// Dispatched at workgroup_size(1)/1 workgroup from gpu/nnProbe.ts's own
// probe(), on that class's own timer (NOT once per macro step) — a
// once-in-a-while diagnostic readback, so the obvious inefficiency here
// (re-reading the exact same weights/gridCurrent/gradient buffers
// agentStep() itself already reads this step, redundantly, on the CPU's
// own schedule rather than piggybacking on agentStep()'s own dispatch) is
// fine.

const CHANNELS: u32 = __CHANNELS__u;
const HIDDEN_DIM: u32 = __HIDDEN_DIM__u;
const IN_DIM: u32 = CHANNELS * 3u; // value + grad_forward + grad_lateral
const SPOTS: u32 = 4u;
const ENV_WRITE_DIM: u32 = CHANNELS * SPOTS;
const OUT_DIM: u32 = ENV_WRITE_DIM + 5u;
const CHIRALITY: bool = __CHIRALITY__;

const FIELD_WIDTH: u32 = __FIELD_WIDTH__u;
const FIELD_HEIGHT: u32 = __FIELD_HEIGHT__u;
const FIELD_PLANE: u32 = FIELD_WIDTH * FIELD_HEIGHT;
const FIELD_TOTAL: u32 = FIELD_PLANE * CHANNELS;

const FC1W_OFFSET: u32 = 0u;
const FC1B_OFFSET: u32 = FC1W_OFFSET + HIDDEN_DIM * IN_DIM;
const FC2W_OFFSET: u32 = FC1B_OFFSET + HIDDEN_DIM;
const FC2B_OFFSET: u32 = FC2W_OFFSET + OUT_DIM * HIDDEN_DIM;

// Output buffer layout — must match gpu/nnProbe.ts's own probeLayout()
// exactly (that function is this shader's ONLY reader).
const INPUT_OFFSET: u32 = 0u;
const HIDDEN_OFFSET: u32 = INPUT_OFFSET + IN_DIM;
const ENV_WRITE_OFFSET: u32 = HIDDEN_OFFSET + HIDDEN_DIM;
const ANGULAR_OFFSET: u32 = ENV_WRITE_OFFSET + ENV_WRITE_DIM;
const STRAFE_OFFSET: u32 = ANGULAR_OFFSET + 1u;
const SPLIT_PROB_OFFSET: u32 = STRAFE_OFFSET + 2u;

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> heading: array<f32>;
@group(0) @binding(3) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(4) var<storage, read> gradient: array<f32>;

// Identical layout to core/agents.wgsl's own AgentPhysics — declared
// again here (not shared) purely so this uniform's byte offsets line up;
// only maxEnvWrite/maxAngularAccel/maxStrafe are actually read below.
struct AgentPhysics {
  maxAccel: f32,
  maxStrafe: f32,
  maxEnvWrite: f32,
  maxAngularAccel: f32,
  angularDamping: f32,
  maxAngularVelocity: f32,
  depositDistance: f32,
  splitDisplacement: f32,
  divisionCooldown: f32,
  friction: f32,
}
@group(0) @binding(5) var<uniform> physics: AgentPhysics;

@group(0) @binding(6) var<storage, read_write> probeOut: array<f32>;

fn fieldIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * FIELD_PLANE + y * FIELD_WIDTH + x;
}

fn safeTanh(x: f32) -> f32 {
  return tanh(clamp(x, -20.0, 20.0));
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

fn wrapCoord(v: f32, size: f32) -> f32 {
  let m = v % size;
  return select(m, m + size, m < 0.0);
}

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

struct PolicyOutput {
  envWrite: array<f32, ENV_WRITE_DIM>,
  angularAccel: f32,
  strafeLocal: vec2<f32>,
  hidden: array<f32, HIDDEN_DIM>,
}

fn evalPolicy(inputVec: array<f32, IN_DIM>) -> PolicyOutput {
  var hidden: array<f32, HIDDEN_DIM>;
  for (var j: u32 = 0u; j < HIDDEN_DIM; j = j + 1u) {
    var acc = weights[FC1B_OFFSET + j];
    for (var i: u32 = 0u; i < IN_DIM; i = i + 1u) {
      acc = acc + inputVec[i] * weights[FC1W_OFFSET + j * IN_DIM + i];
    }
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
  out.hidden = hidden;
  for (var s: u32 = 0u; s < SPOTS; s = s + 1u) {
    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      out.envWrite[s * CHANNELS + c] = safeTanh(outVec[s * CHANNELS + c]) * physics.maxEnvWrite;
    }
  }
  out.angularAccel = safeTanh(outVec[ENV_WRITE_DIM]) * physics.maxAngularAccel;
  out.strafeLocal = vec2<f32>(safeTanh(outVec[ENV_WRITE_DIM + 3u]), safeTanh(outVec[ENV_WRITE_DIM + 4u])) * physics.maxStrafe;
  return out;
}

@compute @workgroup_size(1)
fn probe() {
  let pi = 0u;
  let pos = positions[pi];
  let headingVal = heading[pi];
  let cosH = cos(headingVal);
  let sinH = sin(headingVal);

  let fieldPos = fract(pos) * vec2<f32>(f32(FIELD_WIDTH), f32(FIELD_HEIGHT));
  let k = corners(fieldPos);

  var inputVec: array<f32, IN_DIM>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    inputVec[c] = sampleValue(c, k);
    let gx = sampleGrad(0u, c, k);
    let gy = sampleGrad(FIELD_TOTAL, c, k);
    inputVec[CHANNELS + c] = gx * cosH + gy * sinH;
    inputVec[2u * CHANNELS + c] = -gx * sinH + gy * cosH;
  }

  var result = evalPolicy(inputVec);

  // CHIRALITY averaging — mirrors core/agents.wgsl's own agentStep()
  // exactly (see that file's own comment for the full reasoning); the
  // hidden layer shown in the visualization stays the PRIMARY
  // (non-mirrored) pass's own activations regardless — there's no single
  // "the" hidden layer once two passes run, and the primary pass is the
  // representative one, but the displayed OUTPUT values below match what
  // agentStep() actually applies (the chirality-averaged result).
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

  for (var i: u32 = 0u; i < IN_DIM; i = i + 1u) {
    probeOut[INPUT_OFFSET + i] = inputVec[i];
  }
  for (var i: u32 = 0u; i < HIDDEN_DIM; i = i + 1u) {
    probeOut[HIDDEN_OFFSET + i] = result.hidden[i];
  }
  for (var i: u32 = 0u; i < ENV_WRITE_DIM; i = i + 1u) {
    probeOut[ENV_WRITE_OFFSET + i] = result.envWrite[i];
  }
  probeOut[ANGULAR_OFFSET] = result.angularAccel;
  probeOut[STRAFE_OFFSET] = result.strafeLocal.x;
  probeOut[STRAFE_OFFSET + 1u] = result.strafeLocal.y;
  // The LAST channel's own sensed VALUE, clamped — same growth-probability
  // source core/agents.wgsl's own agentStep() reads (inputVec[CHANNELS-1u]),
  // not a network output.
  probeOut[SPLIT_PROB_OFFSET] = clamp(inputVec[CHANNELS - 1u], 0.0, 1.0);
}

// GPU-resident chemical field the evolved policy senses and writes to —
// originally a WGSL port of trainer/environment.py's own Environment
// (bilinear sample/deposit, Sobel gradient, blur+decay dynamics),
// following envnca/frontend/src/gpu/environment.wgsl's own GPU-porting
// approach (flat (C,H,W) storage buffer, fixed-point atomic deposit
// scratch, ping-pong buffers for the blur pass). TOROIDAL, matching
// envnca's own version exactly: MpmCore's own MLS-MPM domain has no
// walls either now (gridUpdate.wgsl's own module docstring — a
// particle's 3x3 P2G/G2P stencil wraps at the domain edge, not clamps),
// so every neighbor lookup here wraps too, consistent with where
// particles actually are. An earlier revision of this file deliberately
// clamped instead, back when MpmCore's own domain still had walls.
//
// Lives in core/, not viewer/src/gpu/, alongside p2g.wgsl/g2p.wgsl/etc
// — the single source of truth BOTH ../viewer/src/gpu/environment.ts
// (via Vite ?raw) AND ../trainer/environment_gpu.py (via
// ../trainer/shader_template.py's own load_core_shader()) load their
// shader module from. trainer/environment.py (the torch/MPS version this
// originally ported) is gone — see trainer/training_sim.py's own module
// docstring for why running the chemical field on a separate GPU
// framework from MpmCore's own wgpu/Metal physics was removed.
//
// Layout: flat array<f32>, (C,H,W) row-major — gridIndex(c,y,x) =
// c*HEIGHT*WIDTH + y*WIDTH + x. Sensing (agents.wgsl) does its own
// bilinear gather straight out of gridCurrent/gradient at continuous
// particle positions — this file only maintains the grid itself (clear
// the deposit scratch, compute the whole grid's gradient once per macro
// step, merge deposits, diffuse+decay), same "one gradient pass shared by
// every sensor" reasoning environment.py's own module docstring gives for
// its conv2d-based gradient (cost independent of particle count).

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;
const PLANE_SIZE: u32 = WIDTH * HEIGHT;
const TOTAL: u32 = PLANE_SIZE * CHANNELS;

// Must match agents.wgsl's own copy of this constant — the fixed-point
// scale env_write deposits are encoded at before landing in
// depositScratch (see that file's own deposit scatter).
const DEPOSIT_SCALE: f32 = 4096.0;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * PLANE_SIZE + y * WIDTH + x;
}

// read_write (not read) on every binding below, even where a given entry
// point only ever reads through it — same convention ../core/repulsion.wgsl
// already uses for a buffer only some of its own entry points write
// through (see that file's own comment): WGSL bindings are declared once
// per module and shared by every entry point in it, so the access mode
// has to cover the most permissive use any of them needs.
@group(0) @binding(0) var<storage, read_write> gridCurrent: array<f32>;
// gx: [0, TOTAL), gy: [TOTAL, 2*TOTAL) — two planes back to back, same
// convention envnca/frontend/src/gpu/environment.wgsl uses.
@group(0) @binding(1) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> depositScratch: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> gridNext: array<f32>;

struct EnvPhysics {
  decay: f32,
  // Multiplier applied to this macro step's own accumulated
  // depositScratch just before it's folded into gridCurrent — see
  // mergeDeposit's own comment. 1.0 = unchanged (deposits land at
  // exactly whatever agents.wgsl's own agentStep() scattered); scaling
  // it down throttles how much of a step's own writes actually reach
  // the field without having to touch MAX_ENV_WRITE (that caps each
  // agent's own PER-DEPOSIT-SPOT magnitude before scatter — this scales
  // the whole macro step's already-accumulated total instead, evenly,
  // after every agent's contribution has already landed in the same
  // scratch).
  depositRate: f32,
  // Fraction of one legacy 3x3 blur applied by this communication
  // substep. 1 preserves the old one-round behavior; 1/N turns N neural
  // evaluations into numerical supersampling instead of N full-strength
  // diffusion steps.
  diffusionStep: f32,
  _padding: f32,
}
@group(0) @binding(4) var<uniform> physics: EnvPhysics;

@compute @workgroup_size(256)
fn clearScratch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= TOTAL) { return; }
  atomicStore(&depositScratch[i], 0);
}

// Matches trainer/environment.py's own _SOBEL_X (and its transpose for Y)
// exactly: [[-1,0,1],[-2,0,2],[-1,0,1]]/8 — magnitude 0.25 on the
// straight-adjacent taps, 0.125 on the diagonals, 0 on the center column
// (for X) / center row (for Y transposed).
fn sobelX(dy: i32, dx: i32) -> f32 {
  if (dx == 0) { return 0.0; }
  let mag = select(0.125, 0.25, dy == 0);
  return select(-mag, mag, dx > 0);
}
fn sobelY(dy: i32, dx: i32) -> f32 {
  return sobelX(dx, dy); // Y kernel is X's transpose
}

@compute @workgroup_size(16, 16, 1)
fn computeGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let c = gid.z;
  if (x >= WIDTH || y >= HEIGHT || c >= CHANNELS) { return; }

  var gx: f32 = 0.0;
  var gy: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
      let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));
      let v = gridCurrent[gridIndex(c, ny, nx)];
      gx = gx + v * sobelX(dy, dx);
      gy = gy + v * sobelY(dy, dx);
    }
  }
  let idx = gridIndex(c, y, x);
  gradient[idx] = gx;
  gradient[TOTAL + idx] = gy;
}

// Decodes the fixed-point scatter agents.wgsl's own deposit accumulated
// this macro step back into a float, scales it by physics.depositRate
// (see EnvPhysics's own comment for why this is the one place that
// multiplier applies), and adds it in place onto whichever buffer this
// pass's own binding 0 is bound to. That's a CALLER decision, not fixed
// by this shader — environment_gpu.py's own encode_merge_and_decay()/
// environment.ts's own encodeMergeAndDecay() dispatch this AFTER
// diffuseDecay below, pointed at the buffer diffuseDecay just wrote,
// deliberately DECAY-then-DEPOSIT rather than deposit-then-decay: a
// fresh deposit this way reaches its own full depositRate-scaled
// magnitude the very first time anything senses it, and only starts
// decaying/blurring the step after (see that method's own docstring for
// the full "why" — decaying a deposit in the same step it's merged,
// before it's ever sensed, silently caps its own effective magnitude at
// depositRate*decay instead of depositRate). This pass itself has no
// opinion on ordering — it just adds scratch onto binding 0, whatever
// that buffer represents this call.
@compute @workgroup_size(256)
fn mergeDeposit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= TOTAL) { return; }
  gridCurrent[i] = gridCurrent[i] + (f32(atomicLoad(&depositScratch[i])) / DEPOSIT_SCALE) * physics.depositRate;
}

fn blurWeight(dy: i32, dx: i32) -> f32 {
  // 3x3 binomial [[1,2,1],[2,4,2],[1,2,1]]/16 — mass-preserving (sums to
  // 1), matches environment.py's own _BLUR exactly.
  if (dy == 0 && dx == 0) { return 0.25; }
  if (abs(dy) == 1 && abs(dx) == 1) { return 0.0625; }
  return 0.125;
}

// Mass-preserving blur then a flat decay, reading gridCurrent and
// writing gridNext — can't be done in place, this is a spatial
// convolution. Dispatched BEFORE mergeDeposit above now (see that pass's
// own comment for why: decay-then-deposit, not deposit-then-decay), so
// gridCurrent here is the field as the PREVIOUS macro step left it —
// this step's own fresh deposit hasn't touched it yet, still sitting in
// depositScratch untouched. Caller (environment_gpu.py's own
// encode_merge_and_decay()/environment.ts's own
// encodeMergeAndDecay()) flips which of its two buffers is "current" vs
// "next" after mergeDeposit above (not this pass) finishes.
@compute @workgroup_size(16, 16, 1)
fn diffuseDecay(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let c = gid.z;
  if (x >= WIDTH || y >= HEIGHT || c >= CHANNELS) { return; }

  var acc: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
      let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));
      acc = acc + gridCurrent[gridIndex(c, ny, nx)] * blurWeight(dy, dx);
    }
  }
  let idx = gridIndex(c, y, x);
  let diffused = mix(gridCurrent[idx], acc, clamp(physics.diffusionStep, 0.0, 1.0));
  gridNext[idx] = diffused * physics.decay;
}

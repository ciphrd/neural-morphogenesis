// GPU-resident chemical communication field. The host selects one of two
// lifecycles without changing its storage interface: cell-owned-projection
// replaces gridCurrent from persistent per-cell chemistry every brain round;
// persistent-environment keeps a ping-pong spatial field, diffuses/decays its
// old contents, then merges the policy's fresh deposits. TOROIDAL, matching
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
// particle positions — this file only maintains the transient grid itself
// (clear splats, materialize, compute the whole grid's gradient once per brain
// invocation), same "one gradient pass shared by
// every sensor" reasoning environment.py's own module docstring gives for
// its conv2d-based gradient (cost independent of particle count).

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;
const PLANE_SIZE: u32 = WIDTH * HEIGHT;
const TOTAL: u32 = PLANE_SIZE * CHANNELS;

// Must match agents.wgsl's own copy of this constant — the fixed-point
// scale cell-state splats are encoded at before landing in depositScratch.
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
  depositRate: f32,
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

// Builds the sensed chemical field from this communication round's cell
// splats. Assignment (rather than addition) is the key lifecycle rule: the
// field has no memory of a previous round; persistence belongs exclusively to
// each cell's chemicalState in agents.wgsl.
@compute @workgroup_size(256)
fn materializeSplat(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= TOTAL) { return; }
  gridCurrent[i] = f32(atomicLoad(&depositScratch[i])) / DEPOSIT_SCALE;
}

// Persistent-environment only: add this round's policy writes after the old
// field has diffused and decayed, so a new write is sensed once at full
// depositRate before it begins aging on the following round.
@compute @workgroup_size(256)
fn mergeDeposit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= TOTAL) { return; }
  gridCurrent[i] = gridCurrent[i]
    + (f32(atomicLoad(&depositScratch[i])) / DEPOSIT_SCALE) * physics.depositRate;
}

fn blurWeight(dy: i32, dx: i32) -> f32 {
  if (dy == 0 && dx == 0) { return 0.25; }
  if (abs(dy) == 1 && abs(dx) == 1) { return 0.0625; }
  return 0.125;
}

// Persistent-environment only: a mass-preserving 3x3 binomial blur blended
// by the communication timestep, followed by exponential timestep-scaled
// decay. Ping-pong binding selection is owned by the host wrappers.
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

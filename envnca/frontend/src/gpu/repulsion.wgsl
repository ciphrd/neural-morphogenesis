// Cheap O(N) stand-in for pairwise agent-agent repulsion — mirrors
// repulsion.py exactly (see that module's own docstring for the full
// design rationale: splat every agent as a Gaussian blob onto a
// dedicated, low-resolution density field, compute that field's
// gradient once via Sobel, let agents.wgsl's agentStep sample the
// result back at each agent's own position — no agent ever looks at
// another agent directly).
//
// Independent resolution from the main (C,H,W) grid, on purpose — see
// repulsion.py's own docstring for why staying coarse/cheap is the
// entire point. __RESOLUTION__/__GRID_SIZE__/__AGENT_COUNT__ are
// substituted by shaderTemplate.ts before compilation.

const RESOLUTION: u32 = __RESOLUTION__u;
const GRID_SIZE: f32 = __GRID_SIZE__;
const AGENT_COUNT: u32 = __AGENT_COUNT__u;
const FIELD_SIZE: u32 = RESOLUTION * RESOLUTION;

// Fixed-point scatter-add convention — no native float atomics on
// buffers/textures in core WebGPU, same reasoning (and same headroom
// budget) as environment.wgsl/agents.wgsl's own DEPOSIT_SCALE.
const SPLAT_SCALE: f32 = 4096.0;
const SPLAT_CLAMP: f32 = 1073741824.0; // 2^30

@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> scratch: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> density: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradientOut: array<f32>; // gx [0,FIELD_SIZE), gy [FIELD_SIZE,2*FIELD_SIZE)

// Same struct as agents.wgsl's own AgentPhysics — two separate WGSL
// modules sharing one underlying uniform buffer at the TS orchestration
// layer (gpu/repulsion.ts's bindAgents()), same "must match exactly"
// convention DEPOSIT_SCALE already has across environment.wgsl/
// agents.wgsl. Only repulsionSigma is actually read here; the rest of
// the fields still need to be declared so this struct's byte layout
// matches the real one field-for-field.
struct AgentPhysics {
  maxSpeed: f32,
  maxAccel: f32,
  maxStrafe: f32,
  maxEnvWrite: f32,
  repulsionSigma: f32,
  repulsionStrength: f32,
}
@group(0) @binding(4) var<uniform> physics: AgentPhysics;

// Always-non-negative modulo — WGSL's `%` on a negative i32 keeps its
// sign (like C's fmod), unlike Python's; this brings any v (arbitrarily
// negative, not just "one wrap around") back into [0, m).
fn wrapIndex(v: i32, m: i32) -> u32 {
  return u32(((v % m) + m) % m);
}

@compute @workgroup_size(256)
fn clearRepulsionScratch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= FIELD_SIZE) { return; }
  atomicStore(&scratch[i], 0);
}

// One invocation per agent — splats a Gaussian blob (sum-combined via
// atomicAdd, see repulsion.py's rasterize_points_sum-style reasoning for
// why sum, not max) onto `scratch`, toroidally wrapped to match the main
// grid's own no-edge convention. Window radius depends on
// physics.repulsionSigma, a *live* uniform — this loop's bound is
// therefore a runtime value, not a compile-time one (still fine in WGSL,
// just variable per-invocation work depending on the current slider
// value).
@compute @workgroup_size(64)
fn splatRepulsion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= AGENT_COUNT) { return; }

  let scale = f32(RESOLUTION) / GRID_SIZE;
  let sigma = max(0.05, physics.repulsionSigma);
  let pos = positions[i];
  let cx = pos.x * scale;
  let cy = pos.y * scale;
  let ix = i32(round(cx));
  let iy = i32(round(cy));
  let radius = max(1, i32(ceil(3.0 * sigma)));
  let inv2sig2 = 1.0 / (2.0 * sigma * sigma);

  for (var dy: i32 = -radius; dy <= radius; dy = dy + 1) {
    let py = wrapIndex(iy + dy, i32(RESOLUTION));
    let fdy = f32(iy + dy) - cy;
    let fdy2 = fdy * fdy;
    for (var dx: i32 = -radius; dx <= radius; dx = dx + 1) {
      let px = wrapIndex(ix + dx, i32(RESOLUTION));
      let fdx = f32(ix + dx) - cx;
      let k = exp(-(fdy2 + fdx * fdx) * inv2sig2);
      let idx = py * RESOLUTION + px;
      let scaled = clamp(k * SPLAT_SCALE, -SPLAT_CLAMP, SPLAT_CLAMP);
      atomicAdd(&scratch[idx], i32(round(scaled)));
    }
  }
}

@compute @workgroup_size(256)
fn mergeRepulsionDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= FIELD_SIZE) { return; }
  let raw = atomicLoad(&scratch[i]);
  density[i] = f32(raw) / SPLAT_SCALE;
}

// Same Sobel-X/Y matrices as environment.wgsl's own (single-channel
// here, no per-channel loop needed).
const SOBEL_X: array<f32, 9> = array<f32, 9>(
  -0.125, 0.0, 0.125,
  -0.25,  0.0, 0.25,
  -0.125, 0.0, 0.125
);
const SOBEL_Y: array<f32, 9> = array<f32, 9>(
  -0.125, -0.25, -0.125,
   0.0,    0.0,    0.0,
   0.125,  0.25,  0.125
);

// Toroidal 3x3 neighborhood — same wrap reasoning as environment.wgsl's
// own computeGradient.
@compute @workgroup_size(16, 16, 1)
fn computeRepulsionGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= RESOLUTION || y >= RESOLUTION) { return; }

  var gx: f32 = 0.0;
  var gy: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    let ny = wrapIndex(i32(y) + dy, i32(RESOLUTION));
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let nx = wrapIndex(i32(x) + dx, i32(RESOLUTION));
      let v = density[ny * RESOLUTION + nx];
      let k = u32(dy + 1) * 3u + u32(dx + 1);
      gx = gx + v * SOBEL_X[k];
      gy = gy + v * SOBEL_Y[k];
    }
  }
  gradientOut[y * RESOLUTION + x] = gx;
  gradientOut[FIELD_SIZE + y * RESOLUTION + x] = gy;
}

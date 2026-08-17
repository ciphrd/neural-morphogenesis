// GPU chemical grid: gradient (Sobel) + deposit-merge + diffuse/decay.
// Mirrors environment.py exactly — see that module's docstring for the
// overall design (dense (C,H,W) grid, bilinear read/write, whole-grid
// conv2d for gradients). __CHANNELS__/__WIDTH__/__HEIGHT__/__DECAY__ are
// substituted by shaderTemplate.ts before compilation.

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;
const PLANE_SIZE: u32 = CHANNELS * HEIGHT * WIDTH;

// Must match agents.wgsl's DEPOSIT_SCALE exactly — see that file's
// comment for the headroom/precision reasoning behind this constant.
const DEPOSIT_SCALE: f32 = 4096.0;

@group(0) @binding(0) var<storage, read_write> gridCurrent: array<f32>;
@group(0) @binding(1) var<storage, read_write> gradient: array<f32>; // gx plane [0,PLANE_SIZE), gy plane [PLANE_SIZE,2*PLANE_SIZE)
@group(0) @binding(2) var<storage, read_write> depositScratch: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> gridNext: array<f32>;

// Live-adjustable, unlike CHANNELS/WIDTH/HEIGHT above (those are
// compile-time consts because they size arrays/dispatches — a struct
// shape change needs a real shader recompile). A real uniform buffer
// instead, so the frontend's "Physics" panel can push a new value on
// every slider tick via a cheap queue.writeBuffer — no pipeline
// recreation, no touching grid/agent state. Only diffuseDecay below
// actually reads this, so (per layout:"auto"'s reachability-based bind
// group derivation) it's the only pipeline whose layout includes binding
// 4 — see gpu/environment.ts's diffuseDecayBindGroups. Initialized from
// the training run's own DECAY (constants.py); see setDecay().
struct EnvPhysics {
  decay: f32,
}
@group(0) @binding(4) var<uniform> physics: EnvPhysics;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * HEIGHT * WIDTH + y * WIDTH + x;
}

// Sobel-X/Y and the mass-preserving blur, laid out row-major by
// (dy+1)*3+(dx+1) — see environment.py's _SOBEL_X/_BLUR for the source
// matrices this was verified against (kernel_y = kernel_x transposed).
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
const BLUR: array<f32, 9> = array<f32, 9>(
  0.0625, 0.125, 0.0625,
  0.125,  0.25,  0.125,
  0.0625, 0.125, 0.0625
);

@compute @workgroup_size(256)
fn clearScratch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= PLANE_SIZE) { return; }
  atomicStore(&depositScratch[i], 0);
}

// Toroidal (wrapped, not zero-padded) 3x3 neighborhood — mirrors
// environment.py's F.pad(mode="circular") + conv2d(padding=0). Fused
// Sobel-X + Sobel-Y in one pass since both read the same neighborhood.
// dy/dx are always in {-1,0,1} against y/x already in [0,HEIGHT)/
// [0,WIDTH), so i32(y)+dy is always in [-1, HEIGHT] — adding HEIGHT/WIDTH
// once before the i32 `%` is enough to land back in range, no loop
// needed (same reasoning as agents.wgsl's own wrap).
@compute @workgroup_size(16, 16, 1)
fn computeGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= WIDTH || y >= HEIGHT) { return; }

  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    var gx: f32 = 0.0;
    var gy: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
        let v = gridCurrent[gridIndex(c, ny, nx)];
        let k = u32(dy + 1) * 3u + u32(dx + 1);
        gx = gx + v * SOBEL_X[k];
        gy = gy + v * SOBEL_Y[k];
      }
    }
    gradient[gridIndex(c, y, x)] = gx;
    gradient[PLANE_SIZE + gridIndex(c, y, x)] = gy;
  }
}

// Merges this step's atomic-scattered deposits (agents.wgsl's
// depositScatter) back into the float grid — pure elementwise add, no
// ping-pong needed here (see gpu/environment.ts for why diffuseDecay
// below is the only pass that actually needs the second buffer).
@compute @workgroup_size(256)
fn mergeDeposit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= PLANE_SIZE) { return; }
  let raw = atomicLoad(&depositScratch[i]);
  gridCurrent[i] = gridCurrent[i] + f32(raw) / DEPOSIT_SCALE;
}

// Mass-preserving blur (BLUR sums to 1) * DECAY, toroidally wrapped (see
// computeGradient's own comment on the wrap formula) — reads gridCurrent
// (post-deposit), writes gridNext, since this is a spatial convolution
// reading neighbors of the same array it would otherwise write in place.
@compute @workgroup_size(16, 16, 1)
fn diffuseDecay(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= WIDTH || y >= HEIGHT) { return; }

  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    var acc: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
        let v = gridCurrent[gridIndex(c, ny, nx)];
        let k = u32(dy + 1) * 3u + u32(dx + 1);
        acc = acc + v * BLUR[k];
      }
    }
    gridNext[gridIndex(c, y, x)] = acc * physics.decay;
  }
}



const CHANNELS: u32 = __CHANNELS__u;

const FIELD_WIDTHS: array<u32, CHANNELS> = __FIELD_WIDTHS__;
const FIELD_HEIGHTS: array<u32, CHANNELS> = __FIELD_HEIGHTS__;
const FIELD_OFFSETS: array<u32, CHANNELS> = __FIELD_OFFSETS__;
const FIELD_RESPONSE_TIMES: array<f32, CHANNELS> = __FIELD_RESPONSE_TIMES__;
const FIELD_DECAY_EXPONENTS: array<f32, CHANNELS> = __FIELD_DECAY_EXPONENTS__;
const FIELD_DIFFUSION_MULTIPLIERS: array<f32, CHANNELS> = __FIELD_DIFFUSION_MULTIPLIERS__;
const FIELD_TOTAL: u32 = __FIELD_TOTAL__u;
const FIELD_MAX_WIDTH: u32 = __FIELD_MAX_WIDTH__u;
const FIELD_MAX_HEIGHT: u32 = __FIELD_MAX_HEIGHT__u;
const GRID_N: u32 = __GRID_N__u;

const SCRATCH_TOTAL: u32 = FIELD_TOTAL * 2u;
const CLEAR_WORKGROUP_SIZE: u32 = 256u;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return FIELD_OFFSETS[c] + y * FIELD_WIDTHS[c] + x;
}

fn channelForIndex(i: u32) -> u32 {
  var c = 0u;
  while (c + 1u < CHANNELS && i >= FIELD_OFFSETS[c + 1u]) {
    c = c + 1u;
  }
  return c;
}

@group(0) @binding(0) var<storage, read_write> gridCurrent: array<f32>;

@group(0) @binding(1) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> depositScratch: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> gridNext: array<f32>;

struct EnvPhysics {
  decay: f32,
  depositRate: f32,
  diffusionStep: f32,
  normalizeDeposits: f32,
  advectionDt: f32,
  _padding1: f32,
  _padding2: f32,
}
@group(0) @binding(4) var<uniform> physics: EnvPhysics;
@group(0) @binding(5) var<storage, read> mpmGridVelocity: array<vec2<f32>>;

fn flatDispatchIndex(gid: vec3<u32>, workgroups: vec3<u32>) -> u32 {
  return gid.x + gid.y * workgroups.x * CLEAR_WORKGROUP_SIZE;
}

@compute @workgroup_size(256)
fn clearScratch(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  let i = flatDispatchIndex(gid, workgroups);
  if (i >= SCRATCH_TOTAL) { return; }
  atomicStore(&depositScratch[i], 0);
}

fn resolvedDeposit(i: u32) -> f32 {
  let numerator = bitcast<f32>(atomicLoad(&depositScratch[i]));
  let area = bitcast<f32>(atomicLoad(&depositScratch[FIELD_TOTAL + i]));
  if (area <= 0.0) { return 0.0; }
  let c = channelForIndex(i);
  let inverseTexelArea = f32(FIELD_WIDTHS[c]) * f32(FIELD_HEIGHTS[c]);

  if (physics.normalizeDeposits < 0.5) { return numerator * inverseTexelArea; }

  let expression = numerator / area;
  let coverage = min(area * inverseTexelArea, 1.0);
  return expression * coverage;
}

fn channelRetention(c: u32) -> f32 {
  return pow(clamp(physics.decay, 0.0, 1.0), FIELD_DECAY_EXPONENTS[c]);
}

@compute @workgroup_size(256)
fn materializeSplat(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  let i = flatDispatchIndex(gid, workgroups);
  if (i >= FIELD_TOTAL) { return; }
  gridCurrent[i] = resolvedDeposit(i);
}

@compute @workgroup_size(256)
fn mergeDeposit(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  let i = flatDispatchIndex(gid, workgroups);
  if (i >= FIELD_TOTAL) { return; }
  let c = channelForIndex(i);
  gridCurrent[i] = gridCurrent[i]
    + resolvedDeposit(i) * max(physics.depositRate, 0.0)
      / max(FIELD_RESPONSE_TIMES[c], 1e-6);
}

fn blurWeight(dy: i32, dx: i32) -> f32 {
  if (dy == 0 && dx == 0) { return 0.25; }
  if (abs(dy) == 1 && abs(dx) == 1) { return 0.0625; }
  return 0.125;
}

fn wrapFloat(v: f32, size: f32) -> f32 {
  return v - floor(v / size) * size;
}

fn sampleMpmVelocity(worldPos: vec2<f32>) -> vec2<f32> {
  let p = fract(worldPos) * f32(GRID_N);
  let base = vec2<u32>(floor(p)) % vec2<u32>(GRID_N);
  let next = (base + vec2<u32>(1u)) % vec2<u32>(GRID_N);
  let f = fract(p);
  let stride = GRID_N + 1u;
  let v00 = mpmGridVelocity[base.x * stride + base.y];
  let v10 = mpmGridVelocity[next.x * stride + base.y];
  let v01 = mpmGridVelocity[base.x * stride + next.y];
  let v11 = mpmGridVelocity[next.x * stride + next.y];
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

fn sampleChemical(c: u32, fieldPos: vec2<f32>) -> f32 {
  let width = FIELD_WIDTHS[c];
  let height = FIELD_HEIGHTS[c];
  let x = wrapFloat(fieldPos.x, f32(width));
  let y = wrapFloat(fieldPos.y, f32(height));
  let base = vec2<u32>(floor(vec2<f32>(x, y)));
  let next = vec2<u32>((base.x + 1u) % width, (base.y + 1u) % height);
  let f = fract(vec2<f32>(x, y));
  let v00 = gridCurrent[gridIndex(c, base.y, base.x)];
  let v10 = gridCurrent[gridIndex(c, base.y, next.x)];
  let v01 = gridCurrent[gridIndex(c, next.y, base.x)];
  let v11 = gridCurrent[gridIndex(c, next.y, next.x)];
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

@compute @workgroup_size(16, 16, 1)
fn diffuseDecay(@builtin(global_invocation_id) gid: vec3<u32>) {
  evolveField(gid, true);
}

@compute @workgroup_size(16, 16, 1)
fn diffuseDecayStationary(@builtin(global_invocation_id) gid: vec3<u32>) {
  evolveField(gid, false);
}

// Passive diagnostic coordinates: exactly the substrate's backtrace and
// interpolation, without reaction, diffusion, or decay.
@compute @workgroup_size(16, 16, 1)
fn advectOnly(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.z;
  if (c >= CHANNELS) { return; }
  let dimensions = vec2<u32>(FIELD_WIDTHS[c], FIELD_HEIGHTS[c]);
  if (any(gid.xy >= dimensions)) { return; }
  let size = vec2<f32>(dimensions);
  let worldPos = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / size;
  let backtraced = fract(worldPos - sampleMpmVelocity(worldPos) * max(physics.advectionDt, 0.0));
  gridNext[gridIndex(c, gid.y, gid.x)] = sampleChemical(c, backtraced * size - vec2<f32>(0.5));
}

fn evolveField(gid: vec3<u32>, transport: bool) {
  let x = gid.x;
  let y = gid.y;
  let c = gid.z;
  if (c >= CHANNELS) { return; }
  let width = FIELD_WIDTHS[c];
  let height = FIELD_HEIGHTS[c];
  if (x >= width || y >= height) { return; }

  let fieldDimensions = vec2<f32>(f32(width), f32(height));
  let worldPos = (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5)) / fieldDimensions;
  let velocity = sampleMpmVelocity(worldPos);
  let backtracedWorld = fract(worldPos - velocity * select(0.0, max(physics.advectionDt, 0.0), transport));
  let backtracedField = backtracedWorld * fieldDimensions - vec2<f32>(0.5);
  let advected = sampleChemical(c, backtracedField);
  var acc: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      acc = acc + sampleChemical(c, backtracedField + vec2<f32>(f32(dx), f32(dy)))
        * blurWeight(dy, dx);
    }
  }
  let idx = gridIndex(c, y, x);
  let diffusion = clamp(
    physics.diffusionStep * FIELD_DIFFUSION_MULTIPLIERS[c], 0.0, 1.0
  );
  let diffused = mix(advected, acc, diffusion);
  let channelDecay = channelRetention(c);
  gridNext[idx] = diffused * channelDecay;
}

fn sobelX(dy: i32, dx: i32) -> f32 {
  if (dx == 0) { return 0.0; }
  let mag = select(0.125, 0.25, dy == 0);
  return select(-mag, mag, dx > 0);
}
fn sobelY(dy: i32, dx: i32) -> f32 {
  return sobelX(dx, dy);
}

@compute @workgroup_size(16, 16, 1)
fn computeGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let c = gid.z;
  if (c >= CHANNELS) { return; }
  let width = FIELD_WIDTHS[c];
  let height = FIELD_HEIGHTS[c];
  if (x >= width || y >= height) { return; }

  var gx: f32 = 0.0;
  var gy: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let nx = u32((i32(x) + dx + i32(width)) % i32(width));
      let ny = u32((i32(y) + dy + i32(height)) % i32(height));
      let v = gridCurrent[gridIndex(c, ny, nx)];
      gx = gx + v * sobelX(dy, dx);
      gy = gy + v * sobelY(dy, dx);
    }
  }
  let idx = gridIndex(c, y, x);
  gradient[idx] = gx;
  gradient[FIELD_TOTAL + idx] = gy;
}

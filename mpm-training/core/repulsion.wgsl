

const FIELD_N: u32 = __FIELD_N__u;
const TEXELS: u32 = FIELD_N * FIELD_N;
const DT: f32 = __DT__;
const CLEAR_WORKGROUP_SIZE: u32 = 64u;

const SCALE: f32 = 65536.0;

const MAX_KERNEL_RADIUS_TEXELS: i32 = 5;

@group(0) @binding(0) var<storage, read_write> densityAccum: array<atomic<i32>>;

@compute @workgroup_size(64)
fn clearDensity(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  let idx = gid.x + gid.y * workgroups.x * CLEAR_WORKGROUP_SIZE;
  if (idx >= TEXELS) { return; }
  atomicStore(&densityAccum[idx], 0);
}

struct SplatParams {
  sigma: f32,
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
}
@group(0) @binding(1) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> activeCount: u32;
@group(0) @binding(3) var<uniform> splatParams: SplatParams;
struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  growthVectorX: f32,
  growthVectorY: f32,
  verticesAB: vec4<f32>,
  vertexC: vec2<f32>,
  originalArea: f32,
  quadratureWeight: f32,
}
@group(0) @binding(8) var<storage, read> particleRest: array<ParticleRest>;

fn matDet(m: vec4<f32>) -> f32 { return m.x * m.w - m.y * m.z; }

fn wrapFieldIndex(i: i32) -> i32 {
  let n = i32(FIELD_N);
  return ((i % n) + n) % n;
}

@compute @workgroup_size(64)
fn splatDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let texPos = pos * f32(FIELD_N);
  let baseI = i32(floor(texPos.x));
  let baseJ = i32(floor(texPos.y));

  let sigmaTexels = max(splatParams.sigma * f32(FIELD_N), 1e-3);
  let sigma2 = sigmaTexels * sigmaTexels;

  let kernelRadius = min(i32(ceil(3.0 * sigmaTexels)), MAX_KERNEL_RADIUS_TEXELS);

  for (var di = -kernelRadius; di <= kernelRadius; di = di + 1) {
    for (var dj = -kernelRadius; dj <= kernelRadius; dj = dj + 1) {
      let ti = baseI + di;
      let tj = baseJ + dj;

      let texelCenter = vec2<f32>(f32(ti) + 0.5, f32(tj) + 0.5);
      let delta = texPos - texelCenter;
      let d2 = dot(delta, delta);
      let idx = u32(wrapFieldIndex(ti)) * FIELD_N + u32(wrapFieldIndex(tj));

      let representedArea = max(particleRest[pi].quadratureWeight, 1e-6)
        * max(abs(matDet(particleRest[pi].growthF)), 1e-6);
      let weight = representedArea * exp(-d2 / (2.0 * sigma2));
      atomicAdd(&densityAccum[idx], i32(round(weight * SCALE)));
    }
  }
}

@group(0) @binding(1) var densityTex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16, 1)
fn densityToTexture(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= FIELD_N || j >= FIELD_N) { return; }
  let idx = i * FIELD_N + j;
  let value = f32(atomicLoad(&densityAccum[idx])) / SCALE;
  textureStore(densityTex, vec2<i32>(i32(i), i32(j)), vec4<f32>(value, 0.0, 0.0, 0.0));
}

struct RepulsionParams {
  strength: f32,

  maxDelta: f32,
  _padding: vec2<f32>,
}
@group(0) @binding(4) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(5) var densityTexSampled: texture_2d<f32>;
@group(0) @binding(7) var<uniform> repulsionParams: RepulsionParams;

fn loadDensity(texel: vec2<i32>) -> f32 {
  let wrapped = vec2<i32>(wrapFieldIndex(texel.x), wrapFieldIndex(texel.y));
  return textureLoad(densityTexSampled, wrapped, 0).r;
}

fn sampleDensityBilinear(domainPos: vec2<f32>) -> f32 {
  let texPos = domainPos * f32(FIELD_N) - vec2<f32>(0.5, 0.5);
  let base = vec2<i32>(floor(texPos));
  let f = texPos - vec2<f32>(base);
  let d00 = loadDensity(base);
  let d10 = loadDensity(base + vec2<i32>(1, 0));
  let d01 = loadDensity(base + vec2<i32>(0, 1));
  let d11 = loadDensity(base + vec2<i32>(1, 1));
  return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

@compute @workgroup_size(64)
fn applyRepulsion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];

  let eps = 1.0 / f32(FIELD_N);
  let dx = sampleDensityBilinear(pos + vec2<f32>(eps, 0.0)) - sampleDensityBilinear(pos - vec2<f32>(eps, 0.0));
  let dy = sampleDensityBilinear(pos + vec2<f32>(0.0, eps)) - sampleDensityBilinear(pos - vec2<f32>(0.0, eps));
  let grad = vec2<f32>(dx, dy) / (2.0 * eps);

  var delta = grad * (repulsionParams.strength * DT);

  let deltaLen = length(delta);
  if (deltaLen > repulsionParams.maxDelta) {
    delta = delta * (repulsionParams.maxDelta / deltaLen);
  }
  particleVel[pi] = particleVel[pi] - delta;
}

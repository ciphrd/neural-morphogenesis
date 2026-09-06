

const GRID_N: u32 = __GRID_N__u;
const DX: f32 = __DX__;
const INV_DX: f32 = __INV_DX__;

const SCALE: f32 = 4096.0;
const PRESSURE_SCALE: f32 = 2.0;
const PRESSURE_CLAMP: f32 = 1.0e6;

const CH_J: u32 = 0u;
const CH_SHEAR: u32 = 1u;
const CH_PRESSURE: u32 = 2u;
const CH_MASS: u32 = 3u;

const CHANNELS: u32 = 4u;

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> particleF: array<vec4<f32>>;

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
@group(0) @binding(2) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(3) var<storage, read_write> diagnostics: array<atomic<i32>>;

struct Material {
  mu0: f32,
  lambda0: f32,
  hardening: f32,
  yieldLow: f32,
  yieldHigh: f32,
  growthRate: f32,
  growthAnisotropy: f32,
  particleMass: f32,
  particleVolume: f32,
  growthCompressionStart: f32,
  growthCompressionStop: f32,
  growthCompressionFeedback: f32,
}
@group(0) @binding(4) var<uniform> material: Material;
@group(0) @binding(5) var<uniform> activeCount: u32;

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z,
    a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z,
    a.z * b.y + a.w * b.w
  );
}

fn matTranspose(m: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(m.x, m.z, m.y, m.w);
}

fn matDet(m: vec4<f32>) -> f32 {
  return m.x * m.w - m.y * m.z;
}

fn matInverse(m: vec4<f32>) -> vec4<f32> {
  let det = matDet(m);
  if (abs(det) < 1e-8) {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
  }
  return vec4<f32>(m.w, -m.y, -m.z, m.x) / det;
}

struct Polar {
  r: vec4<f32>,
  s: vec4<f32>,
};

fn polarDecompose(m: vec4<f32>) -> Polar {
  let x = m.x + m.w;
  let y = m.z - m.y;
  let d = sqrt(x * x + y * y);
  var r: vec4<f32>;
  if (d < 1e-6) {
    r = vec4<f32>(1.0, 0.0, 0.0, 1.0);
  } else {
    let c = x / d;
    let s = y / d;
    r = vec4<f32>(c, -s, s, c);
  }
  var out: Polar;
  out.r = r;
  out.s = matMul(matTranspose(r), m);
  return out;
}

fn wrapIndex(i: i32) -> u32 {
  let n = i32(GRID_N);
  return u32(((i % n) + n) % n);
}

fn quadraticWeights(fx: vec2<f32>) -> array<vec2<f32>, 3> {
  var w: array<vec2<f32>, 3>;
  let a = vec2<f32>(1.5) - fx;
  let b = fx - vec2<f32>(1.0);
  let c = fx - vec2<f32>(0.5);
  w[0] = 0.5 * a * a;
  w[1] = vec2<f32>(0.75) - b * b;
  w[2] = 0.5 * c * c;
  return w;
}

@compute @workgroup_size(16, 16)
fn clearDiagnostics(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i > GRID_N || j > GRID_N) { return; }
  let base = (i * (GRID_N + 1u) + j) * CHANNELS;
  atomicStore(&diagnostics[base + CH_J], 0);
  atomicStore(&diagnostics[base + CH_SHEAR], 0);
  atomicStore(&diagnostics[base + CH_PRESSURE], 0);
  atomicStore(&diagnostics[base + CH_MASS], 0);
}

@compute @workgroup_size(64)
fn scatterDiagnostics(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let F = particleF[pi];
  let rest = particleRest[pi];
  let Jp = rest.jp;

  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  let e = exp(material.hardening * (1.0 - Jp));
  let lambda = material.lambda0 * e;

  let g = max(matDet(rest.growthF), 1e-6);
  let q = max(rest.quadratureWeight, 0.0);
  let Fe = matMul(F, matInverse(rest.growthF));
  let J = matDet(Fe);
  let polar = polarDecompose(Fe);
  let r = polar.r;

  let FminusR = Fe - r;
  let shearMag = sqrt(FminusR.x * FminusR.x + FminusR.y * FminusR.y + FminusR.z * FminusR.z + FminusR.w * FminusR.w);
  let pressure = clamp(-lambda * (J - 1.0), -PRESSURE_CLAMP, PRESSURE_CLAMP);

  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let ni = wrapIndex(base.x + i32(i));
      let nj = wrapIndex(base.y + i32(j));

      let wgt = w[i].x * w[j].y;

      let massContribution = wgt * material.particleMass * q * g;

      let nodeIndex = (ni * (GRID_N + 1u) + nj) * CHANNELS;
      atomicAdd(&diagnostics[nodeIndex + CH_J], i32(round(massContribution * J * SCALE)));
      atomicAdd(&diagnostics[nodeIndex + CH_SHEAR], i32(round(massContribution * shearMag * SCALE)));
      atomicAdd(&diagnostics[nodeIndex + CH_PRESSURE], i32(round(massContribution * pressure * PRESSURE_SCALE)));
      atomicAdd(&diagnostics[nodeIndex + CH_MASS], i32(round(massContribution * SCALE)));
    }
  }
}



const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);

const DX: f32 = __DX__;
const INV_DX: f32 = __INV_DX__;
const DT: f32 = __DT__;

const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CHANNELS: u32 = 3u;

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> particleC: array<vec4<f32>>;

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
@group(0) @binding(4) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(5) var<storage, read_write> gridAccum: array<atomic<i32>>;

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
@group(0) @binding(6) var<uniform> material: Material;

@group(0) @binding(7) var<uniform> activeCount: u32;

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

fn matAddScaledIdentity(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(m.x + s, m.y, m.z, m.w + s);
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

fn addGridFloat(index: u32, value: f32) {
  if (value == 0.0) { return; }
  var previous = atomicLoad(&gridAccum[index]);
  loop {
    let next = bitcast<i32>(bitcast<f32>(previous) + value);
    let result = atomicCompareExchangeWeak(&gridAccum[index], previous, next);
    if (result.exchanged) { return; }
    previous = result.old_value;
  }
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let vel = particleVel[pi];
  let F = particleF[pi];
  let C = particleC[pi];
  let rest = particleRest[pi];
  let Jp = rest.jp;

  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  let e = exp(material.hardening * (1.0 - Jp));
  let mu = material.mu0 * e;
  let lambda = material.lambda0 * e;

  let Fg = rest.growthF;
  let g = max(matDet(Fg), 1e-6);
  let q = max(rest.quadratureWeight, 0.0);
  let Fe = matMul(F, matInverse(Fg));
  let Je = matDet(Fe);
  let polar = polarDecompose(Fe);
  let r = polar.r;

  let volEff = material.particleVolume * q * g;
  let massEff = material.particleMass * q * g;

  let PF = matAddScaledIdentity(2.0 * mu * matMul(Fe - r, matTranspose(Fe)), lambda * (Je - 1.0) * Je);
  let Dinv = 4.0 * INV_DX * INV_DX;
  let stress = -(DT * volEff * Dinv) * PF;
  let affine = stress + massEff * C;
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      let ni = wrapIndex(base.x + i32(i));
      let nj = wrapIndex(base.y + i32(j));
      let dpos = (vec2<f32>(f32(i), f32(j)) - fx) * DX;
      let wgt = w[i].x * w[j].y;
      let affineDpos = vec2<f32>(
        affine.x*dpos.x + affine.y*dpos.y,
        affine.z*dpos.x + affine.w*dpos.y,
      );
      let momentum = wgt * (massEff * vel + affineDpos);
      let nodeIndex = (ni * (GRID_N+1u) + nj) * CHANNELS;
      addGridFloat(nodeIndex + CH_MOM_X, momentum.x);
      addGridFloat(nodeIndex + CH_MOM_Y, momentum.y);
      addGridFloat(nodeIndex + CH_MASS, massEff * wgt);
    }
  }
}

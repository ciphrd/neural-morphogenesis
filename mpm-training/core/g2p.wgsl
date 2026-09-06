

const GRID_N: u32 = __GRID_N__u;
const INV_DX: f32 = __INV_DX__;
const DT: f32 = __DT__;
const CHEMICAL_CHANNELS: u32 = __CHEMICAL_CHANNELS__u;
const GROWTH_FIELD_CHANNELS: u32 = __GROWTH_FIELD_CHANNELS__u;
const GROWTH_CH_TENSOR_XX: u32 = 2u;
const GROWTH_CH_TENSOR_XY: u32 = 3u;
const GROWTH_CH_TENSOR_YY: u32 = 4u;
const GROWTH_CH_WEIGHT: u32 = 5u;


@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particleC: array<vec4<f32>>;

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
@group(0) @binding(4) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(5) var<storage, read> gridVel: array<vec2<f32>>;

@group(0) @binding(6) var<uniform> activeCount: u32;

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
@group(0) @binding(7) var<uniform> material: Material;



@group(0) @binding(9) var<storage, read_write> growthField: array<atomic<i32>>;

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

fn identityPlusScaled(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(1.0 + s * m.x, s * m.y, s * m.z, 1.0 + s * m.w);
}

fn symmetricExp(m: vec4<f32>) -> vec4<f32> {
  let traceHalf = 0.5 * (m.x + m.w);
  let diagonal = 0.5 * (m.x - m.w);
  let offDiagonal = 0.5 * (m.y + m.z);
  let radius = sqrt(diagonal * diagonal + offDiagonal * offDiagonal);
  let radialScale = select(1.0, sinh(radius) / radius, radius > 1e-7);
  let traceScale = exp(traceHalf);
  let c = cosh(radius);
  return traceScale * vec4<f32>(
    c + radialScale * diagonal,
    radialScale * offDiagonal,
    radialScale * offDiagonal,
    c - radialScale * diagonal,
  );
}

// Spectral positive part. Negative eigenvalues are active contraction and
// must survive compression inhibition.
fn positivePart(m: vec4<f32>) -> vec4<f32> {
  let halfTrace = 0.5 * (m.x + m.w);
  let diagonal = 0.5 * (m.x - m.w);
  let radius = length(vec2<f32>(diagonal, m.y));
  if (radius < 1e-12) {
    return vec4<f32>(max(halfTrace, 0.0), 0.0, 0.0, max(halfTrace, 0.0));
  }
  let high = max(halfTrace + radius, 0.0);
  let low = max(halfTrace - radius, 0.0);
  let mean = 0.5 * (high + low);
  let scale = 0.5 * (high - low) / radius;
  return vec4<f32>(mean + scale*diagonal, scale*m.y, scale*m.y, mean - scale*diagonal);
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

struct Svd {
  u: vec4<f32>,
  sigma: vec2<f32>,
  v: vec4<f32>,
};

fn svd2(m: vec4<f32>) -> Svd {
  let polar = polarDecompose(m);
  let r = polar.r;
  let s = polar.s;
  let a = s.x;

  let b = 0.5 * (s.y + s.z);
  let d = s.w;

  let tr = a + d;
  let diff = a - d;
  let radius = sqrt(diff * diff * 0.25 + b * b);
  let lambda1 = tr * 0.5 + radius;
  let lambda2 = tr * 0.5 - radius;

  var v1 = vec2<f32>(lambda1 - d, b);
  if (length(v1) < 1e-6) {
    v1 = vec2<f32>(b, lambda1 - a);
  }
  if (length(v1) < 1e-6) {
    v1 = vec2<f32>(1.0, 0.0);
  }
  v1 = normalize(v1);
  let v2 = vec2<f32>(-v1.y, v1.x);

  var out: Svd;
  out.v = vec4<f32>(v1.x, v2.x, v1.y, v2.y);
  out.sigma = vec2<f32>(lambda1, lambda2);
  out.u = matMul(r, out.v);
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

fn velocityAtVertex(position: vec2<f32>) -> vec2<f32> {

  let y = fract(position) * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let w = quadraticWeights(y - vec2<f32>(base));
  var velocity = vec2<f32>(0.0);
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      let node = wrapIndex(base.x+i32(i)) * (GRID_N+1u) + wrapIndex(base.y+i32(j));
      velocity += w[i].x * w[j].y * gridVel[node];
    }
  }
  return velocity;
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let F0 = particleF[pi];
  let rest0 = particleRest[pi];
  let Jp0 = rest0.jp;
  let Fg0 = rest0.growthF;

  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  var v = vec2<f32>(0.0);
  var C = vec4<f32>(0.0);
  var growthTensor = vec3<f32>(0.0);
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      let ni = wrapIndex(base.x + i32(i));
      let nj = wrapIndex(base.y + i32(j));
      let dpos = vec2<f32>(f32(i), f32(j)) - fx;
      let nodeIndex = ni * (GRID_N+1u) + nj;
      let gv = gridVel[nodeIndex];
      let wgt = w[i].x * w[j].y;
      let wgv = wgt * gv;
      v += wgv;
      C += (4.0 * INV_DX) * vec4<f32>(
        wgv.x*dpos.x, wgv.x*dpos.y, wgv.y*dpos.x, wgv.y*dpos.y,
      );
      // One growth sample at the triangle centroid, sharing the velocity
      // stencil. Signed boundary tensors still integrate every physics step.
      if (material.growthRate > 0.0) {
        let offset = nodeIndex * GROWTH_FIELD_CHANNELS;
        let weight = bitcast<f32>(atomicLoad(&growthField[offset + GROWTH_CH_WEIGHT]));
        if (weight > 0.0) {
          growthTensor += wgt * vec3<f32>(
            bitcast<f32>(atomicLoad(&growthField[offset + GROWTH_CH_TENSOR_XX])),
            bitcast<f32>(atomicLoad(&growthField[offset + GROWTH_CH_TENSOR_XY])),
            bitcast<f32>(atomicLoad(&growthField[offset + GROWTH_CH_TENSOR_YY]))) / weight;
        }
      }
    }
  }
  var domainNew = rest0.verticesAB;
  var vertexCNew = rest0.vertexC;
  var newPos = fract(pos + DT*v);

  if (rest0.originalArea > 0.0) {
    let a = fract(rest0.verticesAB.xy + DT*velocityAtVertex(rest0.verticesAB.xy));
    let b = fract(rest0.verticesAB.zw + DT*velocityAtVertex(rest0.verticesAB.zw));
    let c = fract(rest0.vertexC + DT*velocityAtVertex(rest0.vertexC));
    domainNew = vec4<f32>(a,b);
    vertexCNew = c;

    let ab = b-a;
    let ac = c-a;
    let u = ab-floor(ab+vec2<f32>(0.5));
    let w = ac-floor(ac+vec2<f32>(0.5));
    newPos = fract(a+(u+w)/3.0);
  }

  let newVel = v;

  var F = matMul(identityPlusScaled(C, DT), F0);

  let FeTrial = matMul(F, matInverse(Fg0));
  let svd = svd2(FeTrial);

  let sigma = clamp(
    svd.sigma,
    vec2<f32>(material.yieldLow),
    vec2<f32>(material.yieldHigh)
  );

  let oldJe = matDet(FeTrial);
  let sigmaMat = vec4<f32>(sigma.x, 0.0, 0.0, sigma.y);
  let Fe = matMul(matMul(svd.u, sigmaMat), matTranspose(svd.v));
  F = matMul(Fe, Fg0);
  let newJe = matDet(Fe);

  let JpNew = clamp(Jp0 * oldJe / newJe, 0.6, 20.0);

  var FgNew = Fg0;

  let fieldRate = growthTensor.x + growthTensor.z;
  if (any(abs(growthTensor) > vec3<f32>(1e-12)) && material.growthRate > 0.0) {
    let compression = max(0.0, -log(max(newJe, 1e-6)));
    var pressureGate = 0.0;
    if (material.growthCompressionStop > material.growthCompressionStart) {
      pressureGate = 1.0 - smoothstep(
        material.growthCompressionStart,
        material.growthCompressionStop,
        compression
      );
    } else if (compression < material.growthCompressionStart) {

      pressureGate = 1.0;
    }
    let feedback = clamp(material.growthCompressionFeedback, 0.0, 1.0);
    let expansionGate = mix(1.0, pressureGate, feedback);

    let anisotropy = clamp(material.growthAnisotropy, 0.0, 1.0);
    let isotropic = 0.5 * fieldRate;
    var worldRate = vec4<f32>(
      mix(isotropic, growthTensor.x, anisotropy),
      growthTensor.y * anisotropy,
      growthTensor.y * anisotropy,
      mix(isotropic, growthTensor.z, anisotropy),
    );

    let positive = positivePart(worldRate);
    let negative = worldRate - positive;
    let growthDt = material.growthRate * DT;
    var positiveScale = expansionGate;
    let positiveTrace = max(positive.x + positive.w, 0.0);
    // Stop contraction at the solver's existing determinant floor, rather
    // than allowing vanishing rest area to make the constitutive inverse fail.
    let negativeTrace = max(-negative.x - negative.w, 0.0);
    var negativeScale = 1.0;
    if (negativeTrace > 0.0) {
      let availableLogArea = max(log(max(matDet(Fg0), 1e-6) / 1e-6), 0.0)
        + positiveScale * positiveTrace * growthDt;
      negativeScale = min(1.0, availableLogArea / (negativeTrace * growthDt));
    }
    worldRate = positiveScale * positive + negativeScale * negative;
    let rotation = polarDecompose(FeTrial).r;
    let materialRate = matMul(matMul(matTranspose(rotation), worldRate), rotation);
    let increment = symmetricExp(materialRate * growthDt);
    FgNew = matMul(increment, Fg0);
  }

  particlePos[pi] = newPos;
  particleVel[pi] = newVel;
  particleF[pi] = F;
  particleC[pi] = C;

  particleRest[pi] = ParticleRest(
    FgNew, JpNew, rest0.growthVectorX, rest0.growthVectorY,
    domainNew, vertexCNew,
    rest0.originalArea, rest0.quadratureWeight
  );
}

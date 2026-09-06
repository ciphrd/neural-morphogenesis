// Diagnostic-only sampler for both raw sensors and the exact normalized vector
// consumed by agents.wgsl. It is dispatched for a small stable list of
// particle-slot indices and used by trainer/capture_policy_inputs.py; it never
// participates in training.

const CHANNELS: u32 = __CHANNELS__u;
const FIELD_WIDTHS: array<u32, CHANNELS> = __FIELD_WIDTHS__;
const FIELD_HEIGHTS: array<u32, CHANNELS> = __FIELD_HEIGHTS__;
const FIELD_OFFSETS: array<u32, CHANNELS> = __FIELD_OFFSETS__;
const FIELD_TOTAL: u32 = __FIELD_TOTAL__u;
const FIELD_MAX_WIDTH: u32 = __FIELD_MAX_WIDTH__u;
const FIELD_MAX_HEIGHT: u32 = __FIELD_MAX_HEIGHT__u;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;
const TRACKED: u32 = __TRACKED__u;
const ELASTIC_SCALE: f32 = __ELASTIC_SCALE__;
const ELASTIC_ENABLED: bool = __ELASTIC_ENABLED__;
const IN_DIM: u32 = __IN_DIM__u;
const META_DIM: u32 = 12u;
const OUT_STRIDE: u32 = META_DIM + 2u * IN_DIM;
const CHEMICAL_VALUE_INPUT_MULTIPLIER: f32 = __CHEMICAL_VALUE_INPUT_MULTIPLIER__;
const CHEMICAL_GRADIENT_INPUT_SCALE: f32 = __CHEMICAL_GRADIENT_INPUT_SCALE__;
const MORPHOLOGY_GRADIENT_INPUT_SCALE: f32 = __MORPHOLOGY_GRADIENT_INPUT_SCALE__;

struct ParticleRest {
  growthF: vec4<f32>, jp: f32, cycleActive: f32,
  growthAngle: f32, growthAnisotropy: f32,
  divisionBias: f32, growthFrameAngle: f32, appearanceScale: f32, quadratureWeight: f32, domain: vec4<f32>, vertexC: vec2<f32>, domainPadding: vec2<f32>,
}
struct ParticleMeta {
  rng: u32, cooldown: f32, alignment: vec2<f32>,
  color: vec4<f32>, divisionHazard: f32, divisionThreshold: f32,
  mitosisPropensity: f32,
  privateState: array<f32, 8>, chemicalState: array<f32, CHANNELS>,
}
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(3) var<storage, read> gradient: array<f32>;
@group(0) @binding(4) var morphologyTexture: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> particleF: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(7) var<storage, read> particleMeta: array<ParticleMeta>;
@group(0) @binding(8) var<storage, read> trackedIndices: array<u32>;
@group(0) @binding(9) var<storage, read_write> output: array<f32>;

fn fieldIndex(c: u32, y: u32, x: u32) -> u32 {
  return FIELD_OFFSETS[c] + y * FIELD_WIDTHS[c] + x;
}
// Quadratic B-spline basis shared by chemical scatter and perception.
// Positions here use integer-centered texel coordinates (world * size - 0.5).
struct Corners {
  xs: array<u32, 3>,
  ys: array<u32, 3>,
  weights: array<vec2<f32>, 3>,
}

fn wrapDepositIndex(i: i32, size: u32) -> u32 {
  let n = i32(size);
  return u32(((i % n) + n) % n);
}

fn corners(c: u32, posIn: vec2<f32>) -> Corners {
  let base = vec2<i32>(floor(posIn - vec2<f32>(0.5)));
  let f = posIn - vec2<f32>(base);
  var out: Corners;
  out.weights[0] = 0.5 * (vec2<f32>(1.5) - f) * (vec2<f32>(1.5) - f);
  out.weights[1] = vec2<f32>(0.75) - (f - vec2<f32>(1.0)) * (f - vec2<f32>(1.0));
  out.weights[2] = 0.5 * (f - vec2<f32>(0.5)) * (f - vec2<f32>(0.5));
  for (var j = 0u; j < 3u; j = j + 1u) {
    out.xs[j] = wrapDepositIndex(base.x + i32(j), FIELD_WIDTHS[c]);
    out.ys[j] = wrapDepositIndex(base.y + i32(j), FIELD_HEIGHTS[c]);
  }
  return out;
}

fn sampleValue(c: u32, k: Corners) -> f32 {
  var value = 0.0;
  for (var x = 0u; x < 3u; x = x + 1u) {
    for (var y = 0u; y < 3u; y = y + 1u) {
      value = value + gridCurrent[fieldIndex(c, k.ys[y], k.xs[x])]
        * k.weights[x].x * k.weights[y].y;
    }
  }
  return value;
}

fn sampleGrad(planeOffset: u32, c: u32, k: Corners) -> f32 {
  var value = 0.0;
  for (var x = 0u; x < 3u; x = x + 1u) {
    for (var y = 0u; y < 3u; y = y + 1u) {
      value = value + gradient[planeOffset + fieldIndex(c, k.ys[y], k.xs[x])]
        * k.weights[x].x * k.weights[y].y;
    }
  }
  return value;
}

fn morphologyLoad(p: vec2<i32>) -> f32 {
  let n = i32(MORPHOLOGY_FIELD_N);
  let q = ((p % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n);
  return textureLoad(morphologyTexture, q, 0).x;
}
fn sampleMorphology(p: vec2<f32>) -> f32 {
  let base = vec2<i32>(floor(p));
  let f = fract(p);
  let a = mix(morphologyLoad(base), morphologyLoad(base + vec2<i32>(1, 0)), f.x);
  let b = mix(morphologyLoad(base + vec2<i32>(0, 1)), morphologyLoad(base + vec2<i32>(1, 1)), f.x);
  return mix(a, b, f.y);
}
fn safeTanh(x: f32) -> f32 { return tanh(clamp(x, -20.0, 20.0)); }
fn normalizeChemicalValue(raw: f32) -> f32 {
  return safeTanh(raw * max(CHEMICAL_VALUE_INPUT_MULTIPLIER, 0.0));
}
fn normalizeChemicalGradient(raw: f32) -> f32 { return safeTanh(raw/max(CHEMICAL_GRADIENT_INPUT_SCALE,1e-6)); }
fn normalizeMorphologyGradient(raw: f32) -> f32 { return safeTanh(raw/max(MORPHOLOGY_GRADIENT_INPUT_SCALE,1e-6)); }
fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(a.x*b.x+a.y*b.z, a.x*b.y+a.y*b.w, a.z*b.x+a.w*b.z, a.z*b.y+a.w*b.w);
}
fn matInverse(m: vec4<f32>) -> vec4<f32> {
  let det = m.x*m.w - m.y*m.z;
  if (abs(det) < 1e-8) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
  return vec4<f32>(m.w, -m.y, -m.z, m.x) / det;
}
fn elasticStrainInput(F: vec4<f32>, Fg: vec4<f32>, forward: vec2<f32>, lateral: vec2<f32>) -> vec3<f32> {
  let Fe = matMul(F, matInverse(Fg));
  let bxx = Fe.x*Fe.x + Fe.y*Fe.y;
  let bxy = Fe.x*Fe.z + Fe.y*Fe.w;
  let byy = Fe.z*Fe.z + Fe.w*Fe.w;
  let frameStrength = length(forward);
  let unitForward = select(vec2<f32>(1.0,0.0), forward/max(frameStrength,1e-10), frameStrength>1e-10);
  let unitLateral = vec2<f32>(-unitForward.y,unitForward.x);
  let bf = vec2<f32>(bxx*unitForward.x+bxy*unitForward.y, bxy*unitForward.x+byy*unitForward.y);
  let bl = vec2<f32>(bxx*unitLateral.x+bxy*unitLateral.y, bxy*unitLateral.x+byy*unitLateral.y);
  let a = dot(unitForward, bf);
  let b = dot(unitForward, bl);
  let d = dot(unitLateral, bl);
  let midpoint = 0.5*(a+d);
  let radius = sqrt(max(0.25*(a-d)*(a-d)+b*b, 0.0));
  let e1 = 0.5*log(max(midpoint+radius, 1e-8));
  let e2 = 0.5*log(max(midpoint-radius, 1e-8));
  let average = 0.5*(e1+e2);
  var h00 = average; var h11 = average; var h01 = 0.0;
  if (radius > 1e-7) {
    let factor = 0.25*(e1-e2)/radius;
    h00 = average + factor*(a-d); h11 = average - factor*(a-d); h01 = factor*2.0*b;
  }
  let invScale = 1.0/max(ELASTIC_SCALE, 1e-6);
  return vec3<f32>(safeTanh((h00+h11)*invScale), safeTanh((h00-h11)*invScale)*frameStrength, safeTanh(2.0*h01*invScale)*frameStrength);
}

@compute @workgroup_size(8)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= TRACKED) { return; }
  let baseOut = slot * OUT_STRIDE;
  let pi = trackedIndices[slot];
  if (pi >= activeCount) {
    for (var i = 0u; i < OUT_STRIDE; i = i + 1u) { output[baseOut+i] = 0.0; }
    return;
  }
  let pos = positions[pi];
  let agentState = particleMeta[pi];
  let rest = particleRest[pi];
  let growthArea = rest.growthF.x*rest.growthF.w - rest.growthF.y*rest.growthF.z;
  output[baseOut+0u]=1.0; output[baseOut+1u]=pos.x; output[baseOut+2u]=pos.y;
  let alignmentStrength = length(agentState.alignment);
  let heading = select(0.0, atan2(agentState.alignment.y, agentState.alignment.x), alignmentStrength > 1e-10);
  output[baseOut+3u]=heading; output[baseOut+4u]=agentState.cooldown;
  output[baseOut+5u]=agentState.divisionHazard; output[baseOut+6u]=agentState.divisionThreshold;
  output[baseOut+7u]=rest.cycleActive; output[baseOut+8u]=growthArea;
  output[baseOut+9u]=rest.growthAngle;
  output[baseOut+10u]=rest.growthAnisotropy;
  output[baseOut+11u]=rest.divisionBias;

  let forward = agentState.alignment;
  let lateral = vec2<f32>(-forward.y, forward.x);
  let rawBase = baseOut + META_DIM;
  let inputBase = rawBase + IN_DIM;
  for (var c=0u; c<CHANNELS; c = c + 1u) {
    let k = corners(
      c,
      fract(pos) * vec2<f32>(f32(FIELD_WIDTHS[c]), f32(FIELD_HEIGHTS[c]))
        - vec2<f32>(0.5),
    );
    let rawValue = sampleValue(c, k);
    output[rawBase+c] = rawValue;
    output[inputBase+c] = normalizeChemicalValue(rawValue);
    let gx = sampleGrad(0u, c, k) * f32(FIELD_WIDTHS[c]) / f32(FIELD_MAX_WIDTH);
    let gy = sampleGrad(FIELD_TOTAL, c, k) * f32(FIELD_HEIGHTS[c]) / f32(FIELD_MAX_HEIGHT);
    let rawForward = dot(vec2<f32>(gx, gy), forward);
    let rawLateral = dot(vec2<f32>(gx, gy), lateral);
    output[rawBase+CHANNELS+c] = rawForward;
    output[rawBase+2u*CHANNELS+c] = rawLateral;
    output[inputBase+CHANNELS+c] = normalizeChemicalGradient(rawForward);
    output[inputBase+2u*CHANNELS+c] = normalizeChemicalGradient(rawLateral);
  }
  let mp = fract(pos) * f32(MORPHOLOGY_FIELD_N);
  let mgx = 0.5*(sampleMorphology(mp+vec2<f32>(1.0,0.0))-sampleMorphology(mp-vec2<f32>(1.0,0.0)));
  let mgy = 0.5*(sampleMorphology(mp+vec2<f32>(0.0,1.0))-sampleMorphology(mp-vec2<f32>(0.0,1.0)));
  let rawOccupancy = sampleMorphology(mp);
  let rawMorphForward = dot(vec2<f32>(mgx, mgy), forward);
  let rawMorphLateral = dot(vec2<f32>(mgx, mgy), lateral);
  output[rawBase+3u*CHANNELS] = rawOccupancy;
  output[rawBase+3u*CHANNELS+1u] = rawMorphForward;
  output[rawBase+3u*CHANNELS+2u] = rawMorphLateral;
  output[inputBase+3u*CHANNELS] = clamp(2.0*rawOccupancy-1.0,-1.0,1.0);
  output[inputBase+3u*CHANNELS+1u] = normalizeMorphologyGradient(rawMorphForward);
  output[inputBase+3u*CHANNELS+2u] = normalizeMorphologyGradient(rawMorphLateral);
  var elastic = vec3<f32>(0.0);
  if (ELASTIC_ENABLED) { elastic = elasticStrainInput(particleF[pi], rest.growthF, forward, lateral); }
  output[rawBase+3u*CHANNELS+3u] = elastic.x;
  output[rawBase+3u*CHANNELS+4u] = elastic.y;
  output[rawBase+3u*CHANNELS+5u] = elastic.z;
  output[inputBase+3u*CHANNELS+3u] = elastic.x;
  output[inputBase+3u*CHANNELS+4u] = elastic.y;
  output[inputBase+3u*CHANNELS+5u] = elastic.z;
  __PRIVATE_STATE_PROBE__
}

// Diagnostic-only sampler for both raw sensors and the exact normalized vector
// consumed by agents.wgsl. It is dispatched for a small stable list of
// particle-slot indices and used by trainer/capture_policy_inputs.py; it never
// participates in training.

const CHANNELS: u32 = __CHANNELS__u;
const FIELD_WIDTH: u32 = __FIELD_WIDTH__u;
const FIELD_HEIGHT: u32 = __FIELD_HEIGHT__u;
const FIELD_PLANE: u32 = FIELD_WIDTH * FIELD_HEIGHT;
const FIELD_TOTAL: u32 = FIELD_PLANE * CHANNELS;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;
const TRACKED: u32 = __TRACKED__u;
const ELASTIC_SCALE: f32 = __ELASTIC_SCALE__;
const ELASTIC_ENABLED: bool = __ELASTIC_ENABLED__;
const IN_DIM: u32 = __IN_DIM__u;
const META_DIM: u32 = 12u;
const OUT_STRIDE: u32 = META_DIM + 2u * IN_DIM;
const CHEMICAL_VALUE_INPUT_SCALE: f32 = __CHEMICAL_VALUE_INPUT_SCALE__;
const CHEMICAL_GRADIENT_INPUT_SCALE: f32 = __CHEMICAL_GRADIENT_INPUT_SCALE__;
const MORPHOLOGY_GRADIENT_INPUT_SCALE: f32 = __MORPHOLOGY_GRADIENT_INPUT_SCALE__;

struct ParticleRest {
  growthF: vec4<f32>, jp: f32, cycleActive: f32,
  growthAngle: f32, growthAnisotropy: f32,
  divisionBias: f32, growthFrameHeading: f32,
}
struct ParticleMeta {
  rng: u32, cooldown: f32, heading: f32, angularVelocity: f32,
  color: vec4<f32>, divisionHazard: f32, divisionThreshold: f32,
  privateState: array<f32, 8>, chemicalState: array<f32, CHANNELS>,
}
struct Corners {
  x0: u32, x1: u32, y0: u32, y1: u32,
  wx0: f32, wx1: f32, wy0: f32, wy1: f32,
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
  return c * FIELD_PLANE + y * FIELD_WIDTH + x;
}
fn wrapCoord(v: f32, size: f32) -> f32 {
  let m = v % size;
  return select(m, m + size, m < 0.0);
}
fn corners(posIn: vec2<f32>) -> Corners {
  let p = vec2<f32>(wrapCoord(posIn.x, f32(FIELD_WIDTH)), wrapCoord(posIn.y, f32(FIELD_HEIGHT)));
  let x0f = floor(p.x);
  let y0f = floor(p.y);
  var out: Corners;
  out.wx1 = p.x - x0f; out.wx0 = 1.0 - out.wx1;
  out.wy1 = p.y - y0f; out.wy0 = 1.0 - out.wy1;
  out.x0 = u32(x0f) % FIELD_WIDTH; out.x1 = (out.x0 + 1u) % FIELD_WIDTH;
  out.y0 = u32(y0f) % FIELD_HEIGHT; out.y1 = (out.y0 + 1u) % FIELD_HEIGHT;
  return out;
}
fn sampleValue(c: u32, k: Corners) -> f32 {
  let v00 = gridCurrent[fieldIndex(c, k.y0, k.x0)];
  let v10 = gridCurrent[fieldIndex(c, k.y0, k.x1)];
  let v01 = gridCurrent[fieldIndex(c, k.y1, k.x0)];
  let v11 = gridCurrent[fieldIndex(c, k.y1, k.x1)];
  return v00*k.wx0*k.wy0 + v10*k.wx1*k.wy0 + v01*k.wx0*k.wy1 + v11*k.wx1*k.wy1;
}
fn sampleGrad(offset: u32, c: u32, k: Corners) -> f32 {
  let v00 = gradient[offset + fieldIndex(c, k.y0, k.x0)];
  let v10 = gradient[offset + fieldIndex(c, k.y0, k.x1)];
  let v01 = gradient[offset + fieldIndex(c, k.y1, k.x0)];
  let v11 = gradient[offset + fieldIndex(c, k.y1, k.x1)];
  return v00*k.wx0*k.wy0 + v10*k.wx1*k.wy0 + v01*k.wx0*k.wy1 + v11*k.wx1*k.wy1;
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
fn normalizeChemicalValue(raw: f32) -> f32 { return safeTanh(raw/max(CHEMICAL_VALUE_INPUT_SCALE,1e-6)); }
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
  let bf = vec2<f32>(bxx*forward.x+bxy*forward.y, bxy*forward.x+byy*forward.y);
  let bl = vec2<f32>(bxx*lateral.x+bxy*lateral.y, bxy*lateral.x+byy*lateral.y);
  let a = dot(forward, bf);
  let b = dot(forward, bl);
  let d = dot(lateral, bl);
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
  return vec3<f32>(safeTanh((h00+h11)*invScale), safeTanh((h00-h11)*invScale), safeTanh(2.0*h01*invScale));
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
  output[baseOut+3u]=agentState.heading; output[baseOut+4u]=agentState.cooldown;
  output[baseOut+5u]=agentState.divisionHazard; output[baseOut+6u]=agentState.divisionThreshold;
  output[baseOut+7u]=rest.cycleActive; output[baseOut+8u]=growthArea;
  output[baseOut+9u]=rest.growthAngle;
  output[baseOut+10u]=rest.growthAnisotropy;
  output[baseOut+11u]=rest.divisionBias;

  let cosH = cos(agentState.heading); let sinH = sin(agentState.heading);
  let forward = vec2<f32>(cosH, sinH); let lateral = vec2<f32>(-sinH, cosH);
  let k = corners(fract(pos) * vec2<f32>(f32(FIELD_WIDTH), f32(FIELD_HEIGHT)));
  let rawBase = baseOut + META_DIM;
  let inputBase = rawBase + IN_DIM;
  for (var c=0u; c<CHANNELS; c = c + 1u) {
    let rawValue = sampleValue(c, k);
    output[rawBase+c] = rawValue;
    output[inputBase+c] = normalizeChemicalValue(rawValue);
    let gx = sampleGrad(0u, c, k); let gy = sampleGrad(FIELD_TOTAL, c, k);
    let rawForward = gx*cosH + gy*sinH;
    let rawLateral = -gx*sinH + gy*cosH;
    output[rawBase+CHANNELS+c] = rawForward;
    output[rawBase+2u*CHANNELS+c] = rawLateral;
    output[inputBase+CHANNELS+c] = normalizeChemicalGradient(rawForward);
    output[inputBase+2u*CHANNELS+c] = normalizeChemicalGradient(rawLateral);
  }
  let mp = fract(pos) * f32(MORPHOLOGY_FIELD_N);
  let mgx = 0.5*(sampleMorphology(mp+vec2<f32>(1.0,0.0))-sampleMorphology(mp-vec2<f32>(1.0,0.0)));
  let mgy = 0.5*(sampleMorphology(mp+vec2<f32>(0.0,1.0))-sampleMorphology(mp-vec2<f32>(0.0,1.0)));
  let rawOccupancy = sampleMorphology(mp);
  let rawMorphForward = mgx*cosH + mgy*sinH;
  let rawMorphLateral = -mgx*sinH + mgy*cosH;
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

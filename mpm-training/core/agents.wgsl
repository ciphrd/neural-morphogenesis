


const STATEFUL: bool = __STATEFUL__;
const CELL_OWNED_CHEMISTRY: bool = __CELL_OWNED_CHEMISTRY__;
const ELASTIC_STRAIN_INPUTS_ENABLED: bool = __ELASTIC_STRAIN_INPUTS_ENABLED__;
const PRIVATE_STATE_DIM: u32 = 8u;

const CHANNELS: u32 = __CHANNELS__u;

const HEADING_CHANNEL: u32 = min(3u, CHANNELS - 1u);
const HIDDEN_DIM: u32 = __HIDDEN_DIM__u;

const IN_DIM: u32 = __IN_DIM__u;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;
const MORPHOLOGY_GRADIENT_INPUT_SCALE: f32 = __MORPHOLOGY_GRADIENT_INPUT_SCALE__;


const ENV_WRITE_DIM: u32 = CHANNELS;
const OUT_DIM: u32 = __OUT_DIM__u;

const PI: f32 = 3.14159265358979323846;

const FC1W_OFFSET: u32 = 0u;
const FC1B_OFFSET: u32 = FC1W_OFFSET + HIDDEN_DIM * IN_DIM;
const FC2W_OFFSET: u32 = FC1B_OFFSET + HIDDEN_DIM;
const FC2B_OFFSET: u32 = FC2W_OFFSET + OUT_DIM * HIDDEN_DIM;

const FIELD_WIDTHS: array<u32, CHANNELS> = __FIELD_WIDTHS__;
const FIELD_HEIGHTS: array<u32, CHANNELS> = __FIELD_HEIGHTS__;
const FIELD_OFFSETS: array<u32, CHANNELS> = __FIELD_OFFSETS__;
const FIELD_RELAXATION_TIMES: array<f32, CHANNELS> = __FIELD_RELAXATION_TIMES__;

const FIELD_TOTAL: u32 = __FIELD_TOTAL__u;
const FIELD_MAX_WIDTH: u32 = __FIELD_MAX_WIDTH__u;
const FIELD_MAX_HEIGHT: u32 = __FIELD_MAX_HEIGHT__u;



@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> activeCount: u32;
@group(0) @binding(3) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(4) var<storage, read> gradient: array<f32>;
@group(0) @binding(5) var<storage, read_write> depositScratch: array<atomic<i32>>;

struct AgentPhysics {
  maxEnvWrite: f32,
  sampleSpacing: f32,
  friction: f32,
  growthEnabled: f32,
  spawnX: f32,
  spawnY: f32,
  maxActiveParticles: u32,
  elasticStrainScale: f32,
  chemicalGradientInputScale: f32,
  forcedGrowthStart: u32,
  forceGrowthMagnitude: u32,
  forcedGrowthDirection: vec2<f32>,
  forcedGrowthEnd: u32,
  chemicalValueInputMultiplier: f32,
  forcedGrowthFieldMode: u32,
}
@group(0) @binding(6) var<uniform> physics: AgentPhysics;

struct ParticleMeta {
  color: vec4<f32>,
  alignment: vec2<f32>,
  growthMagnitude: f32,
  privateState: array<f32, 8>,
  chemicalState: array<f32, __CHANNELS__>,
}

struct AgentState {
  sampleCount: atomic<u32>,
  _padding: array<u32, 63>,
  particleMeta: array<ParticleMeta>,
}
@group(0) @binding(7) var<storage, read_write> agentState: AgentState;

@group(0) @binding(8) var<storage, read_write> particleC: array<vec4<f32>>;

@group(0) @binding(9) var<storage, read_write> velocities: array<vec2<f32>>;

@group(0) @binding(10) var<storage, read_write> particleF: array<vec4<f32>>;

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
@group(0) @binding(11) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(12) var morphologyTexture: texture_2d<f32>;
__MORPHOLOGY_SAMPLER_DECLARATION__
struct StepMode {
  commitGrowth: u32,
  communicationDt: f32,
  stateUpdateSpeed: f32,
}
@group(0) @binding(13) var<uniform> stepMode: StepMode;

fn morphologyLoad(p: vec2<i32>) -> f32 {
  let n = i32(MORPHOLOGY_FIELD_N);
  let q = ((p % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n);
  return textureLoad(morphologyTexture, q, 0).x;
}

fn sampleMorphology(p: vec2<f32>) -> f32 {
  __MORPHOLOGY_SAMPLE_BODY__
}

fn fieldIndex(c: u32, y: u32, x: u32) -> u32 {
  return FIELD_OFFSETS[c] + y * FIELD_WIDTHS[c] + x;
}

fn safeTanh(x: f32) -> f32 {
  return tanh(clamp(x, -20.0, 20.0));
}

fn normalizeChemicalValue(raw: f32) -> f32 {
  return safeTanh(raw * max(physics.chemicalValueInputMultiplier, 0.0));
}

fn normalizeChemicalGradient(raw: f32) -> f32 {
  return safeTanh(raw / max(physics.chemicalGradientInputScale, 1e-6));
}

fn normalizeMorphologyGradient(raw: f32) -> f32 {
  return safeTanh(raw / max(MORPHOLOGY_GRADIENT_INPUT_SCALE, 1e-6));
}

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z,
    a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z,
    a.z * b.y + a.w * b.w
  );
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

fn elasticStrainInput(F: vec4<f32>, Fg: vec4<f32>, forward: vec2<f32>, lateral: vec2<f32>, scale: f32) -> vec3<f32> {
  let Fe = matMul(F, matInverse(Fg));
  let bxx = Fe.x * Fe.x + Fe.y * Fe.y;
  let bxy = Fe.x * Fe.z + Fe.y * Fe.w;
  let byy = Fe.z * Fe.z + Fe.w * Fe.w;

  let frameStrength = length(forward);
  let unitForward = select(vec2<f32>(1.0, 0.0), forward / max(frameStrength, 1e-10), frameStrength > 1e-10);
  let unitLateral = vec2<f32>(-unitForward.y, unitForward.x);
  let bf = vec2<f32>(bxx * unitForward.x + bxy * unitForward.y, bxy * unitForward.x + byy * unitForward.y);
  let bl = vec2<f32>(bxx * unitLateral.x + bxy * unitLateral.y, bxy * unitLateral.x + byy * unitLateral.y);
  let a = dot(unitForward, bf);
  let b = dot(unitForward, bl);
  let d = dot(unitLateral, bl);

  let midpoint = 0.5 * (a + d);
  let radius = sqrt(max(0.25 * (a - d) * (a - d) + b * b, 0.0));
  let lambda1 = max(midpoint + radius, 1e-8);
  let lambda2 = max(midpoint - radius, 1e-8);
  let e1 = 0.5 * log(lambda1);
  let e2 = 0.5 * log(lambda2);
  let average = 0.5 * (e1 + e2);
  var h00 = average;
  var h11 = average;
  var h01 = 0.0;
  if (radius > 1e-7) {
    let factor = 0.25 * (e1 - e2) / radius;
    h00 = average + factor * (a - d);
    h11 = average - factor * (a - d);
    h01 = factor * 2.0 * b;
  }

  let invScale = 1.0 / max(scale, 1e-6);
  return vec3<f32>(
    safeTanh((h00 + h11) * invScale),
    safeTanh((h00 - h11) * invScale) * frameStrength,
    safeTanh((2.0 * h01) * invScale) * frameStrength,
  );
}







struct Corners {
  xs: array<u32, 3>,
  ys: array<u32, 3>,
  weights: array<vec2<f32>, 3>,
}



fn wrapDepositIndex(i: i32, size: u32) -> u32 {
  return u32(((i % i32(size)) + i32(size)) % i32(size));
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

fn addDepositFloat(index: u32, value: f32) {
  if (value == 0.0) { return; }
  var previous = atomicLoad(&depositScratch[index]);
  loop {
    let next = bitcast<i32>(bitcast<f32>(previous) + value);
    let result = atomicCompareExchangeWeak(&depositScratch[index], previous, next);
    if (result.exchanged) { return; }
    previous = result.old_value;
  }
}

fn depositMaterialSample(envWrite: array<f32, ENV_WRITE_DIM>, pos: vec2<f32>, rest: ParticleRest) {

  let originalArea = select(
    physics.sampleSpacing * physics.sampleSpacing * max(rest.quadratureWeight, 0.0),
    rest.originalArea, rest.originalArea > 0.0,
  );
  let area = originalArea * max(matDet(rest.growthF), 0.0);
  for (var c = 0u; c < CHANNELS; c = c + 1u) {
    let k = corners(c, fract(pos)
      * vec2<f32>(f32(FIELD_WIDTHS[c]), f32(FIELD_HEIGHTS[c])) - vec2<f32>(0.5));
    for (var x = 0u; x < 3u; x = x + 1u) {
      for (var y = 0u; y < 3u; y = y + 1u) {
        let index = fieldIndex(c, k.ys[y], k.xs[x]);
        let weightedArea = area * k.weights[x].x * k.weights[y].y;
        addDepositFloat(index, envWrite[c] * weightedArea);
        addDepositFloat(FIELD_TOTAL + index, weightedArea);
      }
    }
  }
}

@compute @workgroup_size(64)
fn splatChemicalState(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  var levels: array<f32, ENV_WRITE_DIM>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    levels[c] = agentState.particleMeta[pi].chemicalState[c];
  }

  depositMaterialSample(levels, positions[pi], particleRest[pi]);
}

struct PolicyOutput {
  envWrite: array<f32, ENV_WRITE_DIM>,
  growthVectorLocal: vec2<f32>,
  color: vec3<f32>,
  stateDelta: array<f32, PRIVATE_STATE_DIM>,
  stateGate: array<f32, PRIVATE_STATE_DIM>,
}

fn safeSigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

fn evalPolicy(inputVec: array<f32, IN_DIM>) -> PolicyOutput {
  var hidden: array<f32, HIDDEN_DIM>;
  for (var j: u32 = 0u; j < HIDDEN_DIM; j = j + 1u) {
    var acc = weights[FC1B_OFFSET + j];
    for (var i: u32 = 0u; i < IN_DIM; i = i + 1u) {
      acc = acc + inputVec[i] * weights[FC1W_OFFSET + j * IN_DIM + i];
    }

    hidden[j] = safeTanh(acc);
  }

  var outVec: array<f32, OUT_DIM>;
  for (var j: u32 = 0u; j < OUT_DIM; j = j + 1u) {
    var acc = weights[FC2B_OFFSET + j];
    for (var i: u32 = 0u; i < HIDDEN_DIM; i = i + 1u) {
      acc = acc + hidden[i] * weights[FC2W_OFFSET + j * HIDDEN_DIM + i];
    }
    outVec[j] = acc;
  }

  var out: PolicyOutput;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    out.envWrite[c] = safeTanh(outVec[c]) * physics.maxEnvWrite;
  }
  out.growthVectorLocal = vec2<f32>(
    safeTanh(outVec[ENV_WRITE_DIM]), safeTanh(outVec[ENV_WRITE_DIM + 1u])
  );
  __POLICY_TAIL_DECODE__
  return out;
}

@compute @workgroup_size(64)
fn agentStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = positions[pi];

  let headingFieldPos = fract(pos)
    * vec2<f32>(f32(FIELD_WIDTHS[HEADING_CHANNEL]), f32(FIELD_HEIGHTS[HEADING_CHANNEL]))
    - vec2<f32>(0.5);
  let headingCorners = corners(HEADING_CHANNEL, headingFieldPos);
  let headingGradient = vec2<f32>(
    sampleGrad(0u, HEADING_CHANNEL, headingCorners)
      * f32(FIELD_WIDTHS[HEADING_CHANNEL]) / f32(FIELD_MAX_WIDTH),
    sampleGrad(FIELD_TOTAL, HEADING_CHANNEL, headingCorners)
      * f32(FIELD_HEIGHTS[HEADING_CHANNEL]) / f32(FIELD_MAX_HEIGHT),
  );
  let alignmentStrength = min(length(headingGradient), 1.0);
  let forward = headingGradient / max(length(headingGradient), 1.0);
  let lateral = vec2<f32>(-forward.y, forward.x);
  let alignmentAngle = select(0.0, atan2(forward.y, forward.x), alignmentStrength > 1e-10);
  agentState.particleMeta[pi].alignment = forward;

  let morphologyPos = fract(pos) * f32(MORPHOLOGY_FIELD_N);
  let morphologyOccupancy = clamp(sampleMorphology(morphologyPos), 0.0, 1.0);
  let morphologyGx = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(1.0, 0.0)) - sampleMorphology(morphologyPos - vec2<f32>(1.0, 0.0)));
  let morphologyGy = 0.5 * (sampleMorphology(morphologyPos + vec2<f32>(0.0, 1.0)) - sampleMorphology(morphologyPos - vec2<f32>(0.0, 1.0)));
  let morphologyGradient = vec2<f32>(morphologyGx, morphologyGy);

  var inputVec: array<f32, IN_DIM>;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    let fieldPos = fract(pos)
      * vec2<f32>(f32(FIELD_WIDTHS[c]), f32(FIELD_HEIGHTS[c]))
      - vec2<f32>(0.5);
    let k = corners(c, fieldPos);
    let rawValue = sampleValue(c, k);
    inputVec[c] = normalizeChemicalValue(rawValue);

    let gx = sampleGrad(0u, c, k) * f32(FIELD_WIDTHS[c]) / f32(FIELD_MAX_WIDTH);
    let gy = sampleGrad(FIELD_TOTAL, c, k) * f32(FIELD_HEIGHTS[c]) / f32(FIELD_MAX_HEIGHT);
    inputVec[CHANNELS + c] = normalizeChemicalGradient(dot(vec2<f32>(gx, gy), forward));
    inputVec[2u * CHANNELS + c] = normalizeChemicalGradient(dot(vec2<f32>(gx, gy), lateral));
  }
  inputVec[3u * CHANNELS] = 2.0 * morphologyOccupancy - 1.0;
  inputVec[3u * CHANNELS + 1u] = normalizeMorphologyGradient(dot(morphologyGradient, forward));
  inputVec[3u * CHANNELS + 2u] = normalizeMorphologyGradient(dot(morphologyGradient, lateral));
  var elasticInput = vec3<f32>(0.0);
  if (ELASTIC_STRAIN_INPUTS_ENABLED) {
    elasticInput = elasticStrainInput(
      particleF[pi], particleRest[pi].growthF, forward, lateral, physics.elasticStrainScale
    );
  }
  inputVec[3u * CHANNELS + 3u] = elasticInput.x;
  inputVec[3u * CHANNELS + 4u] = elasticInput.y;
  inputVec[3u * CHANNELS + 5u] = elasticInput.z;
  __PRIVATE_STATE_INPUTS__
  let result = evalPolicy(inputVec);



  let communicationDt = max(stepMode.communicationDt, 0.0);

  if (CELL_OWNED_CHEMISTRY) {

    for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
      let chemicalDelta = result.envWrite[c] * communicationDt
        / max(FIELD_RELAXATION_TIMES[c], 1e-6);
      agentState.particleMeta[pi].chemicalState[c] = clamp(
        agentState.particleMeta[pi].chemicalState[c] + chemicalDelta,
        -1.0,
        1.0,
      );
    }
  } else {

    depositMaterialSample(result.envWrite, pos, particleRest[pi]);
  }

  if (STATEFUL) {
    for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {
      let residual = result.stateGate[s] * result.stateDelta[s]
        * communicationDt * max(stepMode.stateUpdateSpeed, 0.0);
      agentState.particleMeta[pi].privateState[s] = clamp(
        agentState.particleMeta[pi].privateState[s] + residual, -4.0, 4.0
      );
    }
  }

  let forcedGrowth = pi >= physics.forcedGrowthStart
    && pi <= physics.forcedGrowthEnd;

  let growthForward = vec2<f32>(cos(alignmentAngle), sin(alignmentAngle));
  let growthLateral = vec2<f32>(-growthForward.y, growthForward.x);
  var growthVectorWorld = growthForward * result.growthVectorLocal.x
    + growthLateral * result.growthVectorLocal.y;
  if (forcedGrowth) {
    let forcedMagnitude = max(length(growthVectorWorld), select(0.0, 1.0, physics.forceGrowthMagnitude != 0u));
    growthVectorWorld = normalize(physics.forcedGrowthDirection) * forcedMagnitude;
  }
  growthVectorWorld *= select(0.0, 1.0, physics.growthEnabled > 0.5);
  agentState.particleMeta[pi].color = vec4<f32>(result.color, 1.0);

  if (stepMode.commitGrowth != 0u) {

  velocities[pi] = velocities[pi] * physics.friction;

  particleRest[pi].growthVectorX = growthVectorWorld.x;
  particleRest[pi].growthVectorY = growthVectorWorld.y;

  agentState.particleMeta[pi].growthMagnitude = length(growthVectorWorld);
  }

}

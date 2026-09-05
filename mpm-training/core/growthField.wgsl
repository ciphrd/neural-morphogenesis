// Continuous MPM-grid growth integration and conservative adaptive sampling.
//
// Each numerical material sample carries only a world-space growth vector,
// written by agents.wgsl after rotating the NN's local vector through the
// sample's local frame. Its magnitude is the local areal growth rate and its
// direction defines an unoriented expansion axis. This pass splats both the
// signed vector and its positive-semidefinite outer-product tensor through the
// same quadratic B-spline stencil used by MPM. Division by splatted represented
// volume happens when g2p gathers the field, making the continuum decision
// independent of how densely it is sampled.
//
// Sample creation is a separate quadrature refinement operation. Whenever a
// sample's det(growthF) reaches two baseline volumes, one baseline volume is
// conservatively moved into a new sample. Growth therefore happens continuously
// even when no slot is available; sample count only controls integration
// precision.

const CHANNELS: u32 = __CHANNELS__u;
const PRIVATE_STATE_DIM: u32 = 8u;
const GRID_N: u32 = __GRID_N__u;
const INV_DX: f32 = __INV_DX__;
const MORPHOLOGY_FIELD_N: u32 = __MORPHOLOGY_FIELD_N__u;
const NODE_STRIDE: u32 = GRID_N + 1u;
const NODE_COUNT: u32 = NODE_STRIDE * NODE_STRIDE;
const FIELD_CHANNELS: u32 = 7u;
const CH_VECTOR_X: u32 = 0u;
const CH_VECTOR_Y: u32 = 1u;
const CH_TENSOR_XX: u32 = 2u;
const CH_TENSOR_XY: u32 = 3u;
const CH_TENSOR_YY: u32 = 4u;
const CH_WEIGHT: u32 = 5u;
const CH_CLAIM: u32 = 6u;
// 8192 retains sub-millipercent precision while leaving signed-i32 headroom
// even if all MAX_PARTICLES samples occupy the same node. A sample is normally
// refined before it exceeds two baseline volumes, so that is also the safe
// weighting cap if the refinement capacity has been exhausted.
const FIELD_SCALE: f32 = 8192.0;

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  // Packed world-space NN growth vector. Legacy scalar field names preserve
  // the 48-byte ParticleRest ABI shared by the physics and renderer shaders.
  cycleActive: f32,
  growthAngle: f32,
  growthAnisotropy: f32,
  divisionBias: f32,
  growthFrameAngle: f32,
  appearanceScale: f32,
  resampleAngle: f32,
}

struct ParticleMeta {
  rng: u32,
  cooldown: f32,
  alignment: vec2<f32>,
  color: vec4<f32>,
  divisionHazard: f32,
  divisionThreshold: f32,
  mitosisPropensity: f32,
  privateState: array<f32, PRIVATE_STATE_DIM>,
  chemicalState: array<f32, CHANNELS>,
}

struct AgentState {
  growthCount: atomic<u32>,
  _padding: array<u32, 63>,
  particleMeta: array<ParticleMeta>,
}

struct AgentPhysics {
  maxAccel: f32,
  maxStrafe: f32,
  maxEnvWrite: f32,
  maxAngularAccel: f32,
  angularDamping: f32,
  maxAngularVelocity: f32,
  depositDistance: f32,
  splitDisplacement: f32,
  divisionCooldown: f32,
  friction: f32,
  depositSigma: f32,
  growthEnabled: f32,
  spawnX: f32,
  spawnY: f32,
  maxActiveParticles: u32,
  elasticStrainScale: f32,
  chemicalGradientInputScale: f32,
  chemicalProjectionWeight: f32,
  rolloutSeed: u32,
  boundaryTangentMinGradient: f32,
  forcedLifecycleIndex: u32,
  forcedCycleAdmission: u32,
  forcedDivisionDirection: vec2<f32>,
  forcedLifecycleEndIndex: u32,
  growthCompressionStart: f32,
  growthCompressionStop: f32,
  growthCompressionFeedback: f32,
  chemicalValueInputMultiplier: f32,
  divisionDriveBoost: f32,
  _physicsPadding2: f32,
  _physicsPadding3: f32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(3) var<storage, read_write> agentState: AgentState;
@group(0) @binding(4) var<storage, read_write> particleC: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> particleF: array<vec4<f32>>;
@group(0) @binding(7) var morphologyTexture: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> growthField: array<atomic<i32>>;
@group(0) @binding(9) var<uniform> physics: AgentPhysics;

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z, a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z, a.z * b.y + a.w * b.w,
  );
}

fn matDet(m: vec4<f32>) -> f32 { return m.x * m.w - m.y * m.z; }

fn matInverse(m: vec4<f32>) -> vec4<f32> {
  let d = matDet(m);
  if (abs(d) < 1e-8) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
  return vec4<f32>(m.w, -m.y, -m.z, m.x) / d;
}

fn hashU32(valueIn: u32) -> u32 {
  var value = valueIn;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
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

fn fieldIndex(node: u32, channel: u32) -> u32 {
  return node * FIELD_CHANNELS + channel;
}

fn nearestNode(pos: vec2<f32>) -> u32 {
  let xy = vec2<u32>(floor(fract(pos) * f32(GRID_N))) % vec2<u32>(GRID_N);
  return xy.x * NODE_STRIDE + xy.y;
}

fn morphologyLoad(p: vec2<i32>) -> f32 {
  let n = i32(MORPHOLOGY_FIELD_N);
  let q = vec2<i32>(((p.x % n) + n) % n, ((p.y % n) + n) % n);
  return textureLoad(morphologyTexture, q, 0).x;
}

fn morphologySample(pos: vec2<f32>) -> f32 {
  let p = fract(pos) * f32(MORPHOLOGY_FIELD_N) - vec2<f32>(0.5);
  let b = vec2<i32>(floor(p));
  let f = fract(p);
  let a = mix(morphologyLoad(b), morphologyLoad(b + vec2<i32>(1, 0)), f.x);
  let c = mix(morphologyLoad(b + vec2<i32>(0, 1)), morphologyLoad(b + vec2<i32>(1, 1)), f.x);
  return mix(a, c, f.y);
}

fn refinementDirection(pi: u32) -> vec2<f32> {
  let h = 1.0 / f32(MORPHOLOGY_FIELD_N);
  let pos = positions[pi];
  let gradient = vec2<f32>(
    morphologySample(pos + vec2<f32>(h, 0.0)) - morphologySample(pos - vec2<f32>(h, 0.0)),
    morphologySample(pos + vec2<f32>(0.0, h)) - morphologySample(pos - vec2<f32>(0.0, h)),
  );
  if (length(gradient) > 1e-5) { return -normalize(gradient); }
  let proposal = vec2<f32>(particleRest[pi].cycleActive, particleRest[pi].growthAngle);
  if (length(proposal) > 1e-6) {
    let axis = normalize(proposal);
    let radius = max(physics.splitDisplacement, h);
    return select(axis, -axis, morphologySample(pos - axis * radius) < morphologySample(pos + axis * radius));
  }
  let angle = f32(hashU32(agentState.particleMeta[pi].rng ^ physics.rolloutSeed) >> 8u)
    * (6.28318530718 / 16777216.0);
  return vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(256)
fn clearGrowthField(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= FIELD_CHANNELS * NODE_COUNT) { return; }
  atomicStore(&growthField[i], select(0, -1, i % FIELD_CHANNELS == CH_CLAIM));
}

@compute @workgroup_size(64)
fn scatterGrowthIntent(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let pos = positions[pi];
  let representedVolume = clamp(matDet(particleRest[pi].growthF), 1e-6, 2.0);
  var vector = vec2<f32>(particleRest[pi].cycleActive, particleRest[pi].growthAngle);
  let magnitude = length(vector);
  if (magnitude > 1.0) { vector = vector / magnitude; }
  let rate = min(magnitude, 1.0);
  let direction = vector / max(rate, 1e-8);
  let tensor = rate * vec3<f32>(direction.x * direction.x, direction.x * direction.y, direction.y * direction.y);

  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);
  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let ni = wrapIndex(base.x + i32(i));
      let nj = wrapIndex(base.y + i32(j));
      let node = ni * NODE_STRIDE + nj;
      let contribution = representedVolume * w[i].x * w[j].y;
      atomicAdd(&growthField[fieldIndex(node, CH_VECTOR_X)], i32(round(contribution * vector.x * FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_VECTOR_Y)], i32(round(contribution * vector.y * FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_XX)], i32(round(contribution * tensor.x * FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_XY)], i32(round(contribution * tensor.y * FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_YY)], i32(round(contribution * tensor.z * FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_WEIGHT)], i32(round(contribution * FIELD_SCALE)));
    }
  }
}

@compute @workgroup_size(64)
fn proposeResample(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount || matDet(particleRest[pi].growthF) < 1.9998) { return; }
  let spawnPos = fract(positions[pi] + refinementDirection(pi) * physics.splitDisplacement);
  atomicMax(&growthField[fieldIndex(nearestNode(spawnPos), CH_CLAIM)], i32(pi));
}

fn claimGrowthSlot() -> u32 {
  var observed = atomicLoad(&agentState.growthCount);
  while (observed < physics.maxActiveParticles) {
    let exchanged = atomicCompareExchangeWeak(&agentState.growthCount, observed, observed + 1u);
    if (exchanged.exchanged) { return observed; }
    observed = exchanged.old_value;
  }
  return physics.maxActiveParticles;
}

@compute @workgroup_size(64)
fn commitResample(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let sourceRest = particleRest[pi];
  let sourceArea = matDet(sourceRest.growthF);
  if (sourceArea < 1.9998) { return; }
  let direction = refinementDirection(pi);
  let spawnPos = fract(positions[pi] + direction * physics.splitDisplacement);
  if (atomicLoad(&growthField[fieldIndex(nearestNode(spawnPos), CH_CLAIM)]) != i32(pi)) { return; }
  let newIndex = claimGrowthSlot();
  if (newIndex >= physics.maxActiveParticles) { return; }

  let identity = vec4<f32>(1.0, 0.0, 0.0, 1.0);
  let baselineF = matMul(particleF[pi], matInverse(sourceRest.growthF));
  let offset = direction * physics.splitDisplacement;
  positions[newIndex] = spawnPos;
  velocities[newIndex] = velocities[pi] + vec2<f32>(
    particleC[pi].x * offset.x + particleC[pi].y * offset.y,
    particleC[pi].z * offset.x + particleC[pi].w * offset.y,
  );
  particleC[newIndex] = particleC[pi];
  particleF[newIndex] = baselineF;
  particleRest[newIndex] = sourceRest;
  particleRest[newIndex].growthF = identity;
  particleRest[newIndex].appearanceScale = 0.0;
  particleRest[newIndex].resampleAngle = atan2(direction.y, direction.x);
  agentState.particleMeta[newIndex] = agentState.particleMeta[pi];
  agentState.particleMeta[newIndex].rng = agentState.particleMeta[pi].rng + 1u;
  agentState.particleMeta[newIndex].cooldown = 0.0;
  agentState.particleMeta[newIndex].divisionHazard = 0.0;
  agentState.particleMeta[newIndex].divisionThreshold = 0.0;

  let remainingArea = max(sourceArea - 1.0, 1.0);
  let remainingFg = sourceRest.growthF * sqrt(remainingArea / max(sourceArea, 1e-8));
  particleF[pi] = matMul(baselineF, remainingFg);
  particleRest[pi].growthF = remainingFg;
  particleRest[pi].resampleAngle = atan2(direction.y, direction.x);
  agentState.particleMeta[pi].rng = agentState.particleMeta[pi].rng + 1u;
  agentState.particleMeta[pi].cooldown = 0.0;
}

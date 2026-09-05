// Continuous field-driven material growth and geometric domain subdivision.
// Domains carry transported half edges. Bisection partitions a parent exactly;
// shared Gauss quadrature approximates its grid integrals. No morphology search
// or insertion arbitration is needed. See GROWTH_MODEL.md for conservation,
// physical area budgets, and the distinct numerical capacity safety stop.

const CHANNELS: u32 = __CHANNELS__u;
const PRIVATE_STATE_DIM: u32 = 8u;
const GRID_N: u32 = __GRID_N__u;
const INV_DX: f32 = __INV_DX__;
const NODE_STRIDE: u32 = GRID_N + 1u;
const NODE_COUNT: u32 = NODE_STRIDE * NODE_STRIDE;
const FIELD_CHANNELS: u32 = 10u;
const CH_VECTOR_X: u32 = 0u;
const CH_VECTOR_Y: u32 = 1u;
const CH_TENSOR_XX: u32 = 2u;
const CH_TENSOR_XY: u32 = 3u;
const CH_TENSOR_YY: u32 = 4u;
const CH_WEIGHT: u32 = 5u;
// Fixed-point growth projection; headroom depends on local represented mass.
// Channels 6/7 of node zero hold total world rest area / budget ratio.
const FIELD_SCALE: f32 = 8192.0;
// Refine when the longest transported full edge exceeds sqrt(1.75) times
// the target spacing. This criterion is independent of material-growth state.
const REFINEMENT_THRESHOLD: f32 = 1.75;

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  // Packed world-space NN growth vector. Legacy scalar field names preserve
  // the 64-byte ParticleRest ABI shared by the physics and renderer shaders.
  cycleActive: f32,
  growthAngle: f32,
  growthAnisotropy: f32,
  divisionBias: f32, // Original world area (legacy ABI name).
  growthFrameAngle: f32,
  appearanceScale: f32,
  quadratureWeight: f32,
  // Transported world-space half edges, row major. Independent of plastic F.
  domain: vec4<f32>,
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
  unresolvedSamples: atomic<u32>,
  _padding: array<u32, 62>,
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
  // Lab-only analytic growth-field override. 0 disables it; 1 writes a
  // unit radial-inward field centered on spawnX/spawnY at every MPM node.
  forcedGrowthFieldMode: u32,
  materialAreaBudget: f32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(3) var<storage, read_write> agentState: AgentState;
@group(0) @binding(4) var<storage, read_write> particleC: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> particleF: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> growthField: array<atomic<i32>>;
@group(0) @binding(9) var<uniform> physics: AgentPhysics;
fn matDet(m: vec4<f32>) -> f32 { return m.x * m.w - m.y * m.z; }

fn refinementDemand(pi: u32) -> f32 {
  let h = particleRest[pi].domain;
  let edge2 = max(h.x*h.x+h.z*h.z, h.y*h.y+h.w*h.w);
  return 4.0 * edge2 / max(physics.splitDisplacement*physics.splitDisplacement, 1e-12);
}

fn splitFirstAxis(h: vec4<f32>) -> bool {
  return h.x*h.x+h.z*h.z >= h.y*h.y+h.w*h.w;
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

__DOMAIN_FUNCTIONS__

@compute @workgroup_size(256)
fn clearGrowthField(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i == 0u) { atomicStore(&agentState.unresolvedSamples, 0u); }
  if (i < FIELD_CHANNELS * NODE_COUNT) {
    atomicStore(&growthField[i], 0);
  }

}

@compute @workgroup_size(64)
fn scatterGrowthIntent(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  // Legacy externally loaded scenes have no domain. Initialize once from
  // their deformation and target spacing; production seeders supply a tiling.
  if (dot(particleRest[pi].domain, particleRest[pi].domain) == 0.0) {
    particleRest[pi].domain = particleF[pi] * (0.5 * physics.splitDisplacement);
  }
  if (particleRest[pi].divisionBias <= 0.0) {
    particleRest[pi].divisionBias = 4.0 * abs(matDet(particleRest[pi].domain))
      / max(abs(matDet(particleF[pi])), 1e-8);
  }
  // Snapshot each lineage's current growth for proportional physical-budget
  // allocation over this macro interval. Subdivision copies this value.
  particleRest[pi].growthAnisotropy = max(matDet(particleRest[pi].growthF), 1e-6);
  let worldRestArea = particleRest[pi].divisionBias * particleRest[pi].growthAnisotropy;
  atomicAdd(&growthField[6], i32(round(worldRestArea * 100000000.0)));
  let pos = positions[pi];
  let representedVolume = max(particleRest[pi].quadratureWeight, 1e-6)
    * max(matDet(particleRest[pi].growthF), 1e-6);
  var vector = vec2<f32>(particleRest[pi].cycleActive, particleRest[pi].growthAngle);
  let magnitude = length(vector);
  if (magnitude > 1.0) { vector = vector / magnitude; }
  let rate = min(magnitude, 1.0);
  let direction = vector / max(rate, 1e-8);
  let tensor = rate * vec3<f32>(direction.x * direction.x, direction.x * direction.y, direction.y * direction.y);

  for (var k = 0u; k < domainQuadratureCount(particleRest[pi].domain); k++) {
    let quadrature = domainQuadrature(particleRest[pi].domain, k);
    let samplePos = pos + quadrature.xy;
    let base = vec2<i32>(floor(samplePos * INV_DX - vec2<f32>(0.5)));
    let fx = samplePos * INV_DX - vec2<f32>(base);
    let w = quadraticWeights(fx);
    for (var i = 0u; i < 3u; i++) {
      for (var j = 0u; j < 3u; j++) {
        let node = wrapIndex(base.x+i32(i))*NODE_STRIDE + wrapIndex(base.y+i32(j));
        let contribution = representedVolume * quadrature.z * w[i].x * w[j].y;
        atomicAdd(&growthField[fieldIndex(node, CH_VECTOR_X)], i32(round(contribution*vector.x*FIELD_SCALE)));
        atomicAdd(&growthField[fieldIndex(node, CH_VECTOR_Y)], i32(round(contribution*vector.y*FIELD_SCALE)));
        atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_XX)], i32(round(contribution*tensor.x*FIELD_SCALE)));
        atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_XY)], i32(round(contribution*tensor.y*FIELD_SCALE)));
        atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_YY)], i32(round(contribution*tensor.z*FIELD_SCALE)));
        atomicAdd(&growthField[fieldIndex(node, CH_WEIGHT)], i32(round(contribution*FIELD_SCALE)));
      }
    }
  }
}

@compute @workgroup_size(256)
fn enforceGrowthField(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = gid.x;
  if (node >= NODE_COUNT || physics.forcedGrowthFieldMode != 1u) { return; }
  let ix = node / NODE_STRIDE;
  let iy = node % NODE_STRIDE;
  let nodePos = vec2<f32>(f32(ix), f32(iy)) / f32(GRID_N);
  let center = vec2<f32>(physics.spawnX, physics.spawnY);
  let centerDelta = fract(center - nodePos + vec2<f32>(0.5)) - vec2<f32>(0.5);
  let direction = select(
    vec2<f32>(1.0, 0.0),
    normalize(centerDelta),
    length(centerDelta) > 1e-8,
  );
  // This Lab case is a planar material-expansion test, not a conical-defect
  // test. A pure radial d*d^T metric asks every radius to lengthen without
  // lengthening its circumference; in 2-D that incompatible rest metric must
  // open a seam. Keep the signed inward vector for visualization and sampling
  // intent, but inject its unit trace as compatible isotropic area growth.
  let tensor = vec3<f32>(0.5, 0.0, 0.5);
  // Preserve the sampling channels accumulated by scatterGrowthIntent.
  atomicStore(&growthField[fieldIndex(node, CH_VECTOR_X)], i32(round(direction.x * FIELD_SCALE)));
  atomicStore(&growthField[fieldIndex(node, CH_VECTOR_Y)], i32(round(direction.y * FIELD_SCALE)));
  atomicStore(&growthField[fieldIndex(node, CH_TENSOR_XX)], i32(round(tensor.x * FIELD_SCALE)));
  atomicStore(&growthField[fieldIndex(node, CH_TENSOR_XY)], i32(round(tensor.y * FIELD_SCALE)));
  atomicStore(&growthField[fieldIndex(node, CH_TENSOR_YY)], i32(round(tensor.z * FIELD_SCALE)));
  atomicStore(&growthField[fieldIndex(node, CH_WEIGHT)], i32(FIELD_SCALE));
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
  if (refinementDemand(pi) < REFINEMENT_THRESHOLD) { return; }
  let sourceRest = particleRest[pi];
  let h = sourceRest.domain;
  let first = splitFirstAxis(h);
  let offset = 0.5 * select(vec2<f32>(h.y, h.w), vec2<f32>(h.x, h.z), first);
  let childDomain = select(vec4<f32>(h.x, 0.5*h.y, h.z, 0.5*h.w),
                           vec4<f32>(0.5*h.x, h.y, 0.5*h.z, h.w), first);
  let spawnPos = fract(positions[pi] + offset);
  let newIndex = claimGrowthSlot();
  if (newIndex >= physics.maxActiveParticles) {
    atomicAdd(&agentState.unresolvedSamples, 1u);
    return;
  }

  // Center the replacement pair around the old quadrature point. Leaving the
  // parent fixed and placing every child on the weak-coverage side introduces
  // a first moment at every resample and compounds into radial spokes.
  positions[pi] = fract(positions[pi] - offset);
  positions[newIndex] = spawnPos;
  velocities[newIndex] = velocities[pi] + vec2<f32>(
    particleC[pi].x * offset.x + particleC[pi].y * offset.y,
    particleC[pi].z * offset.x + particleC[pi].w * offset.y,
  );
  velocities[pi] = velocities[pi] - vec2<f32>(
    particleC[pi].x * offset.x + particleC[pi].y * offset.y,
    particleC[pi].z * offset.x + particleC[pi].w * offset.y,
  );
  particleC[newIndex] = particleC[pi];
  particleF[newIndex] = particleF[pi];
  particleRest[newIndex] = sourceRest;
  let childWeight = 0.5 * max(sourceRest.quadratureWeight, 1e-6);
  particleRest[newIndex].domain = childDomain;
  particleRest[pi].domain = childDomain;
  particleRest[pi].divisionBias = 0.5 * sourceRest.divisionBias;
  particleRest[newIndex].divisionBias = 0.5 * sourceRest.divisionBias;
  particleRest[newIndex].quadratureWeight = childWeight;
  particleRest[pi].quadratureWeight = childWeight;
  agentState.particleMeta[newIndex] = agentState.particleMeta[pi];
  agentState.particleMeta[newIndex].rng = agentState.particleMeta[pi].rng + 1u;
  agentState.particleMeta[newIndex].cooldown = 0.0;
  agentState.particleMeta[newIndex].divisionHazard = 0.0;
  agentState.particleMeta[newIndex].divisionThreshold = 0.0;
  agentState.particleMeta[pi].rng = agentState.particleMeta[pi].rng + 1u;
  agentState.particleMeta[pi].cooldown = 0.0;
}

@compute @workgroup_size(256)
fn stopGrowthAtCapacity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= FIELD_CHANNELS * NODE_COUNT) { return; }
  if (atomicLoad(&agentState.growthCount) < physics.maxActiveParticles) {
    if (i == 7u) {
      let totalArea = f32(atomicLoad(&growthField[6])) / 100000000.0;
      let ratio = select(0.0, max(1.0, physics.materialAreaBudget / max(totalArea, 1e-12)),
                         physics.materialAreaBudget > 0.0);
      atomicStore(&growthField[7], bitcast<i32>(ratio));
    }
    return;
  }
  // commitResample can consume the final slot after this macro step's field
  // was scattered. Clear it before G2P sees it so growth stops in that same
  // step rather than overshooting for one full controller interval.
  atomicStore(&growthField[i], 0);
}

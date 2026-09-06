// Continuous field-driven material growth and geometric domain subdivision.
// Domains carry explicit triangle vertices used only for refinement geometry;
// physics and field coupling remain ordinary point-sample operations. Bisection
// partitions a parent exactly. Shared longest edges are committed atomically
// as a pair; longest-edge dependencies advance in conforming stages.
// See GROWTH_MODEL.md for conservation, physical area budgets, and capacity.

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
// Equilateral-equivalent area: sum(edge length squared)/(4 sqrt(3)).
// Unlike determinant area, this detects unresolved isochoric stretching too.
const REFINEMENT_THRESHOLD: f32 = 1.75;
const REFINE_CAPACITY: u32 = __REFINE_CAPACITY__u;
const REFINE_HASH_SIZE: u32 = __REFINE_HASH_SIZE__u;
// Scratch: half-edge hash, then edge choice, neighbor, root, request count,
// allocation, and one capacity-blocked flag. All rebuilt each macro interval.
const CHOICE: u32 = REFINE_HASH_SIZE;
const NEIGHBOR: u32 = CHOICE + REFINE_CAPACITY;
const ROOT: u32 = NEIGHBOR + REFINE_CAPACITY;
const REQUESTS: u32 = ROOT + REFINE_CAPACITY;
const ALLOCATION: u32 = REQUESTS + REFINE_CAPACITY;
const BLOCKED: u32 = ALLOCATION + REFINE_CAPACITY;
const INVALID: u32 = 0xffffffffu;

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  // Packed world-space NN growth vector. Legacy scalar field names preserve
  // original scalar offsets in the shared 80-byte ParticleRest ABI.
  cycleActive: f32,
  growthAngle: f32,
  growthAnisotropy: f32,
  divisionBias: f32, // Original world area (legacy ABI name).
  growthFrameAngle: f32,
  appearanceScale: f32,
  quadratureWeight: f32,
  // Explicit wrapped vertices: domain.xy=A, domain.zw=B, vertexC=C.
  domain: vec4<f32>, vertexC: vec2<f32>, domainPadding: vec2<f32>,
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
  capacityBlocked: u32, // Header word 2; written once after reservation.
  _padding: array<u32, 61>,
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
@group(0) @binding(7) var<storage, read_write> refinement: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> growthField: array<atomic<i32>>;
@group(0) @binding(9) var<uniform> physics: AgentPhysics;
fn matDet(m: vec4<f32>) -> f32 { return m.x * m.w - m.y * m.z; }
fn edgeBetween(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let d = b-a;
  return d-floor(d+vec2<f32>(0.5));
}
fn triangleArea(rest: ParticleRest) -> f32 {
  let u = edgeBetween(rest.domain.xy,rest.domain.zw);
  let v = edgeBetween(rest.domain.xy,rest.vertexC);
  return 0.5*abs(u.x*v.y-u.y*v.x);
}
fn triangleCenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  return fract(a+(edgeBetween(a,b)+edgeBetween(a,c))/3.0);
}

fn vertex(pi: u32, vi: u32) -> vec2<f32> {
  if (vi == 0u) { return particleRest[pi].domain.xy; }
  if (vi == 1u) { return particleRest[pi].domain.zw; }
  return particleRest[pi].vertexC;
}
fn vertexLess(a: vec2<f32>, b: vec2<f32>) -> bool {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}
fn edgeKey(pi: u32, ei: u32) -> vec4<f32> {
  let a = vertex(pi,ei); let b = vertex(pi,(ei+1u)%3u);
  if (vertexLess(b,a)) { return vec4<f32>(b,a); }
  return vec4<f32>(a,b);
}
fn keyLess(a: vec4<f32>, b: vec4<f32>) -> bool {
  if (any(a.xy != b.xy)) { return vertexLess(a.xy,b.xy); }
  return vertexLess(a.zw,b.zw);
}
fn edgeLengthSquared(key: vec4<f32>) -> f32 {
  let e = edgeBetween(key.xy,key.zw);
  return dot(e,e);
}
fn refinementDemand(pi: u32) -> f32 {
  var sum = 0.0;
  for (var ei=0u; ei<3u; ei++) { sum += edgeLengthSquared(edgeKey(pi,ei)); }
  return sum / (4.0*sqrt(3.0)*max(physics.splitDisplacement*physics.splitDisplacement,1e-12));
}
fn hashKey(key: vec4<f32>) -> u32 {
  // Float equality handles signed zero; hash must do so as well.
  let k = bitcast<vec4<u32>>(select(key,vec4<f32>(0.0),key == vec4<f32>(0.0)));
  var h = 2166136261u;
  for (var i=0u; i<4u; i++) { h = (h ^ k[i])*16777619u; }
  h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u;
  return h & (REFINE_HASH_SIZE-1u);
}

@compute @workgroup_size(256)
fn clearRefinement(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x <= BLOCKED) { atomicStore(&refinement[gid.x],0u); }
}

@compute @workgroup_size(64)
fn indexRefinementEdges(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  var best = 0u; var bestKey = edgeKey(pi,0u);
  var bestLength = edgeLengthSquared(bestKey);
  for (var ei=0u; ei<3u; ei++) {
    let key = edgeKey(pi,ei); let len = edgeLengthSquared(key);
    // A global edge order breaks equal-length cycles in the propagation graph.
    if (len > bestLength || (len == bestLength && keyLess(bestKey,key))) {
      best=ei; bestKey=key; bestLength=len;
    }
    var slot = hashKey(key);
    loop {
      let result = atomicCompareExchangeWeak(&refinement[slot],0u,3u*pi+ei+1u);
      if (result.exchanged) { break; }
      // Weak CAS may fail spuriously on an empty slot; retry it before probing.
      if (result.old_value != 0u) { slot=(slot+1u)&(REFINE_HASH_SIZE-1u); }
    }
  }
  atomicStore(&refinement[CHOICE+pi],best);
}

@compute @workgroup_size(64)
fn linkRefinementEdges(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi=gid.x;
  if (pi >= activeCount) { return; }
  let ei=atomicLoad(&refinement[CHOICE+pi]);
  let key=edgeKey(pi,ei);
  var slot=hashKey(key); var neighbor=INVALID; var neighborEdge=0u; var matches=0u;
  loop {
    let entry=atomicLoad(&refinement[slot]);
    if (entry == 0u) { break; }
    let other=(entry-1u)/3u; let otherEdge=(entry-1u)%3u;
    if (other != pi && all(edgeKey(other,otherEdge) == key)) {
      neighbor=other; neighborEdge=otherEdge; matches++;
    }
    slot=(slot+1u)&(REFINE_HASH_SIZE-1u);
  }
  var root=pi;
  if (matches > 1u) {
    // Non-manifold/overlapping input is ambiguous: never split only one copy.
    root=INVALID;
  } else if (matches == 1u) {
    root=neighbor;
    if (atomicLoad(&refinement[CHOICE+neighbor]) == neighborEdge) { root=min(pi,neighbor); }
  }
  atomicStore(&refinement[NEIGHBOR+pi],neighbor);
  atomicStore(&refinement[ROOT+pi],root);
}

@compute @workgroup_size(64)
fn propagateRefinement(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi=gid.x;
  if (pi >= activeCount) { return; }
  let parent=atomicLoad(&refinement[ROOT+pi]);
  if (parent != INVALID) {
    // Atomic pointer jumping: roots only move forward along increasing edges.
    // Host dispatch barriers bound a path of N triangles in ceil(log2(N)) steps.
    atomicStore(&refinement[ROOT+pi],atomicLoad(&refinement[ROOT+parent]));
  }
}

@compute @workgroup_size(64)
fn requestRefinement(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi=gid.x;
  if (pi >= activeCount || refinementDemand(pi) < REFINEMENT_THRESHOLD) { return; }
  let root=atomicLoad(&refinement[ROOT+pi]);
  if (root == INVALID) { atomicAdd(&agentState.unresolvedSamples,1u); return; }
  atomicAdd(&refinement[REQUESTS+root],1u);
}

@compute @workgroup_size(64)
fn reserveRefinement(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi=gid.x;
  if (pi >= activeCount) { return; }
  let requests=atomicLoad(&refinement[REQUESTS+pi]);
  if (requests == 0u) { return; }
  let neighbor=atomicLoad(&refinement[NEIGHBOR+pi]);
  let slots=select(2u,1u,neighbor == INVALID);
  var observed=atomicLoad(&agentState.growthCount);
  loop {
    if (observed+slots > physics.maxActiveParticles) {
      atomicAdd(&agentState.unresolvedSamples,requests);
      atomicStore(&refinement[BLOCKED],1u);
      return;
    }
    let result=atomicCompareExchangeWeak(&agentState.growthCount,observed,observed+slots);
    if (result.exchanged) { break; }
    observed=result.old_value;
  }
  atomicStore(&refinement[ALLOCATION+pi],observed+1u);
  if (neighbor != INVALID) { atomicStore(&refinement[ALLOCATION+neighbor],observed+2u); }
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
  if (dot(particleRest[pi].domain, particleRest[pi].domain) == 0.0 && dot(particleRest[pi].vertexC,particleRest[pi].vertexC) == 0.0) {
    // Right triangle with undeformed area spacing², transported by F.
    let e = particleF[pi] * (sqrt(2.0) * physics.splitDisplacement);
    let u = vec2<f32>(e.x,e.z);
    let v = vec2<f32>(e.y,e.w);
    let a = positions[pi]-(u+v)/3.0;
    particleRest[pi].domain = vec4<f32>(fract(a),fract(a+u));
    particleRest[pi].vertexC = fract(a+v);
  }
  if (particleRest[pi].divisionBias <= 0.0) {
    particleRest[pi].divisionBias = triangleArea(particleRest[pi])
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

  let base = vec2<i32>(floor(pos * INV_DX - vec2<f32>(0.5)));
  let fx = pos * INV_DX - vec2<f32>(base);
  let w = quadraticWeights(fx);
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      let node = wrapIndex(base.x+i32(i))*NODE_STRIDE + wrapIndex(base.y+i32(j));
      let contribution = representedVolume * w[i].x * w[j].y;
      atomicAdd(&growthField[fieldIndex(node, CH_VECTOR_X)], i32(round(contribution*vector.x*FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_VECTOR_Y)], i32(round(contribution*vector.y*FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_XX)], i32(round(contribution*tensor.x*FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_XY)], i32(round(contribution*tensor.y*FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_TENSOR_YY)], i32(round(contribution*tensor.z*FIELD_SCALE)));
      atomicAdd(&growthField[fieldIndex(node, CH_WEIGHT)], i32(round(contribution*FIELD_SCALE)));
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

@compute @workgroup_size(64)
fn commitResample(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let allocation=atomicLoad(&refinement[ALLOCATION+pi]);
  if (allocation == 0u) { return; }
  let newIndex=allocation-1u;
  let sourceRest = particleRest[pi];
  let ei=atomicLoad(&refinement[CHOICE+pi]);
  let a=vertex(pi,ei); let b=vertex(pi,(ei+1u)%3u); let c=vertex(pi,(ei+2u)%3u);
  // Canonical endpoint ordering makes independently split copies of the same
  // edge generate a bit-identical midpoint even with opposite edge winding.
  var first = a; var second = b;
  if (b.x < a.x || (b.x == a.x && b.y < a.y)) { first = b; second = a; }
  let midpoint = fract(first+0.5*edgeBetween(first,second));
  let minusDomain = vec4<f32>(a,midpoint);
  let plusDomain = vec4<f32>(midpoint,b);
  let spawnPos = triangleCenter(midpoint,b,c);

  // Center the replacement pair around the old material point. Leaving the
  // parent fixed and placing every child on the weak-coverage side introduces
  // a first moment at every resample and compounds into radial spokes.
  positions[pi] = triangleCenter(a,midpoint,c);
  positions[newIndex] = spawnPos;
  // Standard point MPM: copy the parent's particle velocity to both children,
  // as in Ruggirello and Schumacher's adaptation algorithm. Sampling the
  // affine field at the displaced centers would add orbital angular momentum
  // on top of each child's retained APIC affine state.
  velocities[newIndex] = velocities[pi];
  particleC[newIndex] = particleC[pi];
  particleF[newIndex] = particleF[pi];
  particleRest[newIndex] = sourceRest;
  let childWeight = 0.5 * sourceRest.quadratureWeight;
  particleRest[newIndex].domain = plusDomain;
  particleRest[pi].domain = minusDomain;
  particleRest[newIndex].vertexC = c;
  particleRest[pi].vertexC = c;
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
  if (i == 0u) { agentState.capacityBlocked = atomicLoad(&refinement[BLOCKED]); }
  if (atomicLoad(&agentState.growthCount) < physics.maxActiveParticles && atomicLoad(&refinement[BLOCKED]) == 0u) {
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

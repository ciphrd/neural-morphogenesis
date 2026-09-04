// Experimental organism-attached developmental coordinate fields. This is a
// separate three-channel bank (anterior, posterior, inhibitor), deliberately
// outside the policy's chemical ABI. Values are non-negative concentrations.

const FIELD_N: u32 = __FIELD_N__u;
const MPM_GRID_N: u32 = __MPM_GRID_N__u;
const ANTERIOR: u32 = 0u;
const POSTERIOR: u32 = 1u;
const INHIBITOR: u32 = 2u;
const PLANE: u32 = FIELD_N * FIELD_N;

struct Params {
  dt: f32,
  advectionDt: f32,
  seedSigma: f32,
  enabled: f32,

  activatorDiffusion: f32,
  inhibitorDiffusion: f32,
  sourceProduction: f32,
  activatorDecay: f32,

  inhibitorProduction: f32,
  inhibitorDecay: f32,
  inhibitorSuppression: f32,
  occupancyHalfSaturation: f32,

  occupancyHillExponent: f32,
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
}

struct Organizers {
  anterior: vec2<f32>,
  posterior: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> fieldCurrent: array<f32>;
@group(0) @binding(1) var<storage, read_write> fieldNext: array<f32>;
@group(0) @binding(2) var<storage, read_write> fieldGradient: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> mpmGridVelocity: array<vec2<f32>>;
@group(0) @binding(5) var morphology: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> organizers: Organizers;

fn index(c: u32, y: u32, x: u32) -> u32 {
  return c * PLANE + y * FIELD_N + x;
}

fn wrapFloat(v: f32) -> f32 {
  return v - floor(v / f32(FIELD_N)) * f32(FIELD_N);
}

fn toroidalDelta(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let d = a - b;
  return d - round(d);
}

fn sampleField(c: u32, fieldPos: vec2<f32>) -> f32 {
  let p = vec2<f32>(wrapFloat(fieldPos.x), wrapFloat(fieldPos.y));
  let base = vec2<u32>(floor(p));
  let next = (base + vec2<u32>(1u)) % vec2<u32>(FIELD_N);
  let f = fract(p);
  let v00 = fieldCurrent[index(c, base.y, base.x)];
  let v10 = fieldCurrent[index(c, base.y, next.x)];
  let v01 = fieldCurrent[index(c, next.y, base.x)];
  let v11 = fieldCurrent[index(c, next.y, next.x)];
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

fn sampleVelocity(worldPos: vec2<f32>) -> vec2<f32> {
  let p = fract(worldPos) * f32(MPM_GRID_N);
  let base = vec2<u32>(floor(p)) % vec2<u32>(MPM_GRID_N);
  let next = (base + vec2<u32>(1u)) % vec2<u32>(MPM_GRID_N);
  let f = fract(p);
  let stride = MPM_GRID_N + 1u;
  let v00 = mpmGridVelocity[base.x * stride + base.y];
  let v10 = mpmGridVelocity[next.x * stride + base.y];
  let v01 = mpmGridVelocity[base.x * stride + next.y];
  let v11 = mpmGridVelocity[next.x * stride + next.y];
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

fn occupancy(worldPos: vec2<f32>) -> f32 {
  let dimensions = vec2<i32>(textureDimensions(morphology));
  let texel = clamp(
    vec2<i32>(floor(fract(worldPos) * vec2<f32>(dimensions))),
    vec2<i32>(0), dimensions - vec2<i32>(1)
  );
  let rho = max(textureLoad(morphology, texel, 0).r, 0.0);
  let exponent = max(params.occupancyHillExponent, 1.0);
  let numerator = pow(rho, exponent);
  let threshold = pow(max(params.occupancyHalfSaturation, 1e-6), exponent);
  return numerator / max(numerator + threshold, 1e-12);
}

@compute @workgroup_size(8, 8)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N) { return; }
  let world = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / f32(FIELD_N);
  let sigma2 = max(params.seedSigma * params.seedSigma, 1e-10);
  let deltaA = toroidalDelta(world, organizers.anterior);
  let deltaP = toroidalDelta(world, organizers.posterior);
  let a = exp(-0.5 * dot(deltaA, deltaA) / sigma2);
  let p = exp(-0.5 * dot(deltaP, deltaP) / sigma2);
  fieldNext[index(ANTERIOR, gid.y, gid.x)] = a * params.enabled;
  fieldNext[index(POSTERIOR, gid.y, gid.x)] = p * params.enabled;
  fieldNext[index(INHIBITOR, gid.y, gid.x)] = 0.0;
}

fn laplacian(c: u32, p: vec2<f32>) -> f32 {
  let center = sampleField(c, p);
  let sum = sampleField(c, p + vec2<f32>(1.0, 0.0))
    + sampleField(c, p - vec2<f32>(1.0, 0.0))
    + sampleField(c, p + vec2<f32>(0.0, 1.0))
    + sampleField(c, p - vec2<f32>(0.0, 1.0));
  // Convert the texel-space stencil to a normalized-world Laplacian.
  return (sum - 4.0 * center) * f32(FIELD_N * FIELD_N);
}

// Exact solution of dc/dt = source - loss*c while source/loss are held
// constant over this operator-split substep. This keeps decay and inhibition
// positive and timestep-consistent instead of approximating them with 1-loss*dt.
fn integrateLinear(value: f32, source: f32, loss: f32) -> f32 {
  if (loss < 1e-7) { return value + params.dt * source; }
  let retention = exp(-loss * params.dt);
  return value * retention + source * (1.0 - retention) / loss;
}

@compute @workgroup_size(1)
fn advectOrganizers(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u || params.enabled < 0.5) { return; }
  organizers.anterior = fract(
    organizers.anterior + sampleVelocity(organizers.anterior) * params.advectionDt
      + vec2<f32>(1.0)
  );
  organizers.posterior = fract(
    organizers.posterior + sampleVelocity(organizers.posterior) * params.advectionDt
      + vec2<f32>(1.0)
  );
}

@compute @workgroup_size(8, 8)
fn evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N) { return; }
  let world = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / f32(FIELD_N);
  let backtracedWorld = fract(world - sampleVelocity(world) * params.advectionDt + vec2<f32>(1.0));
  let fieldPos = backtracedWorld * f32(FIELD_N) - vec2<f32>(0.5);
  let a = sampleField(ANTERIOR, fieldPos);
  let p = sampleField(POSTERIOR, fieldPos);
  let inhibitor = sampleField(INHIBITOR, fieldPos);
  let gate = occupancy(world);
  let sigma2 = max(params.seedSigma * params.seedSigma, 1e-10);
  let deltaA = toroidalDelta(world, organizers.anterior);
  let deltaP = toroidalDelta(world, organizers.posterior);
  let organizerA = exp(-0.5 * dot(deltaA, deltaA) / sigma2);
  let organizerP = exp(-0.5 * dot(deltaP, deltaP) / sigma2);

  // Explicit normalized-coordinate diffusion followed by an exact local
  // production/loss solve. The configured UI bounds satisfy the 2-D CFL limit
  // D*dt/dx^2 <= 1/4 at the canonical resolution and four substeps.
  let transportedA = max(a + params.dt * params.activatorDiffusion * laplacian(ANTERIOR, fieldPos), 0.0);
  let transportedP = max(p + params.dt * params.activatorDiffusion * laplacian(POSTERIOR, fieldPos), 0.0);
  let transportedI = max(inhibitor + params.dt * params.inhibitorDiffusion * laplacian(INHIBITOR, fieldPos), 0.0);
  let sourceAttenuation = 1.0 / (1.0 + params.inhibitorSuppression * transportedI);
  // Organizer emission is not clipped by a grainy edge texel. Occupancy
  // instead makes signals clear four times faster outside the organism,
  // retaining an organism-shaped coordinate field without biasing either pole.
  let exteriorLossMultiplier = 4.0 - 3.0 * gate;
  let nextA = integrateLinear(
    transportedA, params.sourceProduction * organizerA * sourceAttenuation,
    params.activatorDecay * exteriorLossMultiplier
  );
  let nextP = integrateLinear(
    transportedP, params.sourceProduction * organizerP * sourceAttenuation,
    params.activatorDecay * exteriorLossMultiplier
  );
  let inhibitorSource = params.inhibitorProduction * (organizerA + organizerP);
  let nextI = integrateLinear(
    transportedI, inhibitorSource,
    params.inhibitorDecay * exteriorLossMultiplier + inhibitorSource
  );
  let enabledScale = params.enabled;
  fieldNext[index(ANTERIOR, gid.y, gid.x)] = clamp(nextA, 0.0, 1.0) * enabledScale;
  fieldNext[index(POSTERIOR, gid.y, gid.x)] = clamp(nextP, 0.0, 1.0) * enabledScale;
  fieldNext[index(INHIBITOR, gid.y, gid.x)] = clamp(nextI, 0.0, 1.0) * enabledScale;
}

@compute @workgroup_size(8, 8)
fn computeGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N || gid.z >= 3u) { return; }
  let p = vec2<f32>(gid.xy);
  let gx = 0.5 * (sampleField(gid.z, p + vec2<f32>(1.0, 0.0)) - sampleField(gid.z, p - vec2<f32>(1.0, 0.0)));
  let gy = 0.5 * (sampleField(gid.z, p + vec2<f32>(0.0, 1.0)) - sampleField(gid.z, p - vec2<f32>(0.0, 1.0)));
  let i = index(gid.z, gid.y, gid.x);
  fieldGradient[i] = gx * f32(FIELD_N);
  fieldGradient[3u * PLANE + i] = gy * f32(FIELD_N);
}

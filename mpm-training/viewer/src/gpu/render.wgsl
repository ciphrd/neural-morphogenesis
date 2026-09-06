

struct VOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

const QUAD_OFFSETS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@group(0) @binding(0) var<storage, read> pointPositions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> pointRadius: f32;
@group(0) @binding(2) var<uniform> pointColor: vec4<f32>;

@group(1) @binding(0) var<uniform> viewStyle: vec4<f32>;

fn viewCenter(center: vec2<f32>) -> vec2<f32> {
  return center * max(viewStyle.x, 1e-4);
}

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  growthVectorX: f32,
  growthVectorY: f32,
  budgetGrowthRatio: f32,
  verticesAB: vec4<f32>,
  vertexC: vec2<f32>,
  originalArea: f32,
  quadratureWeight: f32,
}

struct DomainOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec2<f32>,
}

fn domainWorldPosition(cornerIndex: u32, tile: u32, particleIndex: u32) -> vec2<f32> {
  let rest = particleRest[particleIndex];
  let a = rest.verticesAB.xy;
  let b = rest.verticesAB.zw-floor(rest.verticesAB.zw-a+vec2<f32>(0.5));
  let c = rest.vertexC-floor(rest.vertexC-a+vec2<f32>(0.5));
  let corners = array<vec2<f32>, 3>(a,b,c);
  let low = min(a, min(b, c));
  let high = max(a, max(b, c));
  let wrapShift = select(
    select(vec2<f32>(0.0), vec2<f32>(-1.0), high >= vec2<f32>(1.0)),
    vec2<f32>(1.0),
    low < vec2<f32>(0.0),
  );
  let useWrapX = (tile & 1u) != 0u;
  let useWrapY = (tile & 2u) != 0u;
  let shift = vec2<f32>(
    select(0.0, wrapShift.x, useWrapX),
    select(0.0, wrapShift.y, useWrapY),
  );
  let valid = (!useWrapX || wrapShift.x != 0.0) && (!useWrapY || wrapShift.y != 0.0);
  return select(vec2<f32>(-2.0), corners[cornerIndex] + shift, valid);
}

@vertex
fn domainVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> DomainOut {
  let ends = array<u32, 6>(0u, 1u, 1u, 2u, 2u, 0u);
  let tile = vertexIndex / 6u;
  // A sub-half-period triangle can cross at most one seam on each axis.
  // These combinations cover the base image and only needed neighbors.
  let world = domainWorldPosition(ends[vertexIndex % 6u], tile, instanceIndex);
  var out: DomainOut;
  out.position = vec4<f32>(viewCenter(world*2.0-vec2<f32>(1.0)), 0.0, 1.0);
  out.world = world;
  return out;
}

@fragment
fn domainFragment(in: DomainOut) -> @location(0) vec4<f32> {
  if (any(in.world < vec2<f32>(0.0)) || any(in.world >= vec2<f32>(1.0))) { discard; }
  return vec4<f32>(0.15, 0.85, 1.0, 0.9);
}

struct ParticleMeta {
  color: vec4<f32>,
  alignment: vec2<f32>,
  growthMagnitude: f32,
  privateState: array<f32, 8>,
  chemicalState: array<f32, __CHANNELS__>,
}

@group(0) @binding(4) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(3) var<storage, read> particleMeta: array<ParticleMeta>;

const TRIANGLE_OFFSETS = array<vec2<f32>, 6>(
  vec2<f32>(1.4, 0.0), vec2<f32>(-0.9, 0.9), vec2<f32>(-0.9, -0.9),
  vec2<f32>(-0.9, -0.9), vec2<f32>(-0.9, -0.9), vec2<f32>(-0.9, -0.9),
);

fn particleOffset(vertexIndex: u32, instanceIndex: u32) -> vec2<f32> {
  if (viewStyle.y < 0.5) {
    return QUAD_OFFSETS[vertexIndex];
  }
  let local = TRIANGLE_OFFSETS[vertexIndex];
  let alignment = particleMeta[instanceIndex].alignment;
  let strength = length(alignment);
  if (strength <= 1e-10) { return local; }
  let forward = alignment / strength;
  let lateral = vec2<f32>(-forward.y, forward.x);
  return local.x * forward + local.y * lateral;
}

fn outsideParticleShape(uv: vec2<f32>) -> bool {
  return viewStyle.y < 0.5 && dot(uv, uv) > 1.0;
}

fn materialRadiusScale(instanceIndex: u32) -> f32 {
  return sqrt(clamp(1.0, 0.0, 1.0));
}

struct ParticleGeometry {
  position: vec4<f32>,
  uv: vec2<f32>,
  particleIndex: u32,
}

fn particleGeometry(vertexIndex: u32, instanceIndex: u32) -> ParticleGeometry {
  var geometry: ParticleGeometry;
  if (viewStyle.y > 1.5) {
    geometry.particleIndex = instanceIndex / 4u;
    let world = domainWorldPosition(vertexIndex, instanceIndex % 4u, geometry.particleIndex);
    geometry.position = vec4<f32>(viewCenter(world * 2.0 - vec2<f32>(1.0)), 0.0, 1.0);
    geometry.uv = vec2<f32>(0.0);
    return geometry;
  }
  geometry.particleIndex = instanceIndex;
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  geometry.position = vec4<f32>(
    center + offset * pointRadius * materialRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0,
  );
  geometry.uv = offset;
  return geometry;
}

@vertex
fn particleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  var out: VOut;
  out.position = geometry.position;
  out.uv = geometry.uv;
  return out;
}

@vertex
fn targetVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VOut {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = QUAD_OFFSETS[vertexIndex];
  var out: VOut;
  out.position = vec4<f32>(center + offset * pointRadius, 0.0, 1.0);
  out.uv = offset;
  return out;
}

@fragment
fn particleFragment(in: VOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) {
    discard;
  }
  return vec4<f32>(pointColor.rgb, viewStyle.z);
}

@fragment
fn targetFragment(in: VOut) -> @location(0) vec4<f32> {
  if (dot(in.uv, in.uv) > 1.0) { discard; }
  return pointColor;
}

struct ActivationDotOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) activation: vec2<f32>,
}

@vertex
fn activationParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> ActivationDotOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  var out: ActivationDotOut;
  out.position = geometry.position;
  out.uv = geometry.uv;
  let rest = particleRest[geometry.particleIndex];
  out.activation = vec2<f32>(rest.growthVectorX, rest.growthVectorY);
  return out;
}

fn hueRgb(hue: f32) -> vec3<f32> {
  let p = abs(fract(vec3<f32>(hue) + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3<f32>(3.0));
  return clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn neuronActivationColor(raw: vec2<f32>) -> vec3<f32> {
  let magnitude = length(raw);
  let strength = clamp(magnitude, 0.0, 1.0);
  var direction = vec2<f32>(1.0, 0.0);
  if (magnitude > 1e-6) {
    direction = raw / magnitude;
  }
  let hue = fract(atan2(direction.y, direction.x) / (2.0 * 3.14159265359));
  let vivid = mix(vec3<f32>(1.0), hueRgb(hue), 0.82);
  return mix(vec3<f32>(0.12), vivid, strength);
}

@fragment
fn activationParticleFragment(in: ActivationDotOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) {
    discard;
  }
  return vec4<f32>(neuronActivationColor(in.activation), viewStyle.z);
}

@group(0) @binding(7) var<uniform> neuralColorStyle: vec4<f32>;
struct InternalStateStyle {
  channels: vec4<u32>,
  alpha: f32,
  opponentSubtraction: f32,

  _padding1: f32,
  _padding2: f32,
}
@group(0) @binding(8) var<uniform> internalStateStyle: InternalStateStyle;

fn stateSigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

fn privateStateDisplayColor(rawState: vec3<f32>) -> vec3<f32> {
  var normalized = rawState;

  let maxComponent = max(normalized.x, max(normalized.y, normalized.z));
  if (maxComponent > 1.0) {
    normalized = normalized / vec3<f32>(maxComponent);
  }
  return vec3<f32>(
    stateSigmoid(normalized.x),
    stateSigmoid(normalized.y),
    stateSigmoid(normalized.z),
  );
}

struct NeuralColorDotOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
}

@vertex
fn neuralColorParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  var out: NeuralColorDotOut;
  out.position = geometry.position;
  out.uv = geometry.uv;
  out.color = particleMeta[geometry.particleIndex].color.rgb;
  return out;
}

@fragment
fn neuralColorParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  let radiusSquared = dot(in.uv, in.uv);
  if (viewStyle.y < 0.5 && radiusSquared > 1.0) {
    discard;
  }

  let contrasted = vec3<f32>(0.5) + (in.color - vec3<f32>(0.5)) * neuralColorStyle.z;
  let luminance = dot(contrasted, vec3<f32>(0.2126, 0.7152, 0.0722));
  let boosted = clamp(
    vec3<f32>(luminance) + (contrasted - vec3<f32>(luminance)) * neuralColorStyle.y,
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
  return vec4<f32>(boosted, viewStyle.z);
}

const BERLIN = array<vec3<f32>, 17>(
  vec3<f32>(0.62108, 0.69018, 0.99951),
  vec3<f32>(0.47324, 0.67153, 0.92975),
  vec3<f32>(0.31849, 0.62455, 0.82794),
  vec3<f32>(0.21017, 0.52319, 0.67838),
  vec3<f32>(0.15674, 0.40615, 0.52486),
  vec3<f32>(0.11373, 0.29378, 0.37955),
  vec3<f32>(0.077286, 0.18914, 0.24359),
  vec3<f32>(0.06510, 0.10085, 0.12357),
  vec3<f32>(0.098319, 0.047041, 0.034683),
  vec3<f32>(0.16781, 0.054240, 0.0019629),
  vec3<f32>(0.25339, 0.071986, 0.0029984),
  vec3<f32>(0.35795, 0.11256, 0.030456),
  vec3<f32>(0.49191, 0.20352, 0.11819),
  vec3<f32>(0.61998, 0.31787, 0.24762),
  vec3<f32>(0.74490, 0.43635, 0.38864),
  vec3<f32>(0.87457, 0.55988, 0.53622),
  vec3<f32>(0.99987, 0.68007, 0.67995),
);

fn berlin(value: f32) -> vec3<f32> {
  let scaled = clamp(value, 0.0, 1.0) * 16.0;
  let lower = min(u32(floor(scaled)), 15u);
  return mix(BERLIN[lower], BERLIN[lower + 1u], fract(scaled));
}

@vertex
fn growthMagnitudeParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  var out: NeuralColorDotOut;
  out.position = geometry.position;
  out.uv = geometry.uv;
  let boostedMagnitude = clamp(
    particleMeta[geometry.particleIndex].growthMagnitude * neuralColorStyle.w,
    0.0,
    1.0,
  );
  out.color = berlin(boostedMagnitude);
  return out;
}

@fragment
fn growthMagnitudeParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) { discard; }
  return vec4<f32>(in.color, viewStyle.z);
}

@vertex
fn internalStateParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  let state = particleMeta[geometry.particleIndex].privateState;
  var out: NeuralColorDotOut;
  out.position = geometry.position;
  out.uv = geometry.uv;
  let colorState = vec3<f32>(
    state[internalStateStyle.channels.x],
    state[internalStateStyle.channels.y],
    state[internalStateStyle.channels.z],
  );

  let opponentState = vec3<f32>(
    state[(internalStateStyle.channels.x + 3u) % 8u],
    state[(internalStateStyle.channels.y + 3u) % 8u],
    state[(internalStateStyle.channels.z + 3u) % 8u],
  );

  out.color = clamp(
    privateStateDisplayColor(colorState)
      - privateStateDisplayColor(opponentState) * internalStateStyle.opponentSubtraction,
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
  return out;
}

@vertex
fn chemicalLevelsParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  let levels = particleMeta[geometry.particleIndex].chemicalState;
  var out: NeuralColorDotOut;
  out.position = geometry.position;
  out.uv = geometry.uv;

  let raw = vec3<f32>(
    levels[internalStateStyle.channels.x],
    levels[internalStateStyle.channels.y],
    levels[internalStateStyle.channels.z],
  );
  var color: vec3<f32> = clamp(raw + 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
  let maxComponent = max(color.x, max(color.y, color.z));
  if (maxComponent > 1.0) {
    color = color / vec3<f32>(maxComponent);
  }
  out.color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  return out;
}

@fragment
fn internalStateParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) { discard; }
  return vec4<f32>(in.color, viewStyle.z);
}

@group(0) @binding(9) var boundaryMorphologyTexture: texture_2d<f32>;
@group(0) @binding(10) var<uniform> boundaryGradientScale: f32;

fn boundaryMorphologyLoad(p: vec2<i32>) -> f32 {
  let dims = vec2<i32>(textureDimensions(boundaryMorphologyTexture));
  let q = ((p % dims) + dims) % dims;
  return textureLoad(boundaryMorphologyTexture, q, 0).x;
}

fn sampleBoundaryMorphology(p: vec2<f32>) -> f32 {
  let base = vec2<i32>(floor(p));
  let f = fract(p);
  let a = mix(
    boundaryMorphologyLoad(base),
    boundaryMorphologyLoad(base + vec2<i32>(1, 0)),
    f.x,
  );
  let b = mix(
    boundaryMorphologyLoad(base + vec2<i32>(0, 1)),
    boundaryMorphologyLoad(base + vec2<i32>(1, 1)),
    f.x,
  );
  return mix(a, b, f.y);
}

fn boundaryValueColor(value: f32) -> vec3<f32> {
  let low = vec3<f32>(0.075, 0.12, 0.31);
  let middle = vec3<f32>(0.05, 0.63, 0.60);
  let high = vec3<f32>(0.99, 0.86, 0.25);
  if (value < 0.5) {
    return mix(low, middle, value * 2.0);
  }
  return mix(middle, high, (value - 0.5) * 2.0);
}

@vertex
fn boundaryValueParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let geometry = particleGeometry(vertexIndex, instanceIndex);
  let position = fract(pointPositions[geometry.particleIndex]);
  let dims = vec2<f32>(textureDimensions(boundaryMorphologyTexture));
  let fieldPos = position * dims;
  let gx = 0.5 * (
    sampleBoundaryMorphology(fieldPos + vec2<f32>(1.0, 0.0))
      - sampleBoundaryMorphology(fieldPos - vec2<f32>(1.0, 0.0))
  );
  let gy = 0.5 * (
    sampleBoundaryMorphology(fieldPos + vec2<f32>(0.0, 1.0))
      - sampleBoundaryMorphology(fieldPos - vec2<f32>(0.0, 1.0))
  );
  let gradientMagnitude = length(vec2<f32>(gx, gy));
  let g0 = max(boundaryGradientScale, 1e-8);
  let boundaryValue = gradientMagnitude / (gradientMagnitude + g0);

  var out: NeuralColorDotOut;
  out.position = geometry.position;
  out.uv = geometry.uv;
  out.color = boundaryValueColor(boundaryValue);
  return out;
}

@fragment
fn boundaryValueParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) { discard; }
  return vec4<f32>(in.color, viewStyle.z);
}

@group(0) @binding(5) var<uniform> directionalLineStyle: vec4<f32>;
@vertex
fn headingLineVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> @builtin(position) vec4<f32> {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let alignment = particleMeta[instanceIndex].alignment;
  let strength = length(alignment);
  let direction = select(vec2<f32>(0.0), alignment / max(strength, 1e-10), strength > 1e-10);
  let offset = select(vec2<f32>(0.0), direction * directionalLineStyle.y, vertexIndex == 1u);
  return vec4<f32>(center + offset * viewStyle.x, 0.0, 1.0);
}

@fragment
fn headingLineFragment() -> @location(0) vec4<f32> {
  return vec4<f32>(pointColor.rgb, viewStyle.z);
}

@vertex
fn growthLineVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> @builtin(position) vec4<f32> {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let rest = particleRest[instanceIndex];
  let vector = vec2<f32>(rest.growthVectorX, rest.growthVectorY);
  let magnitude = min(length(vector), 1.0);
  let direction = select(vec2<f32>(0.0), vector / max(length(vector), 1e-8), magnitude > 1e-8);
  let offset = select(vec2<f32>(0.0), direction * directionalLineStyle.y * 1.5 * magnitude, vertexIndex == 1u);
  return vec4<f32>(center + offset * viewStyle.x, 0.0, 1.0);
}

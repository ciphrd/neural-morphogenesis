// Particle rendering — instanced quads drawn straight from a positions
// storage buffer, same approach mls-mpm/src/gpu/render.wgsl uses for its
// own particles (no density/color texture, no full-screen present pass):
// each instance is a 6-vertex unit quad offset by pointRadius (an NDC-
// space, device-pixel-derived constant — see render.ts's own
// setCanvasSizePx()) around the particle's own position, a circle carved
// out of it in the fragment shader by discarding outside the unit disc.
// One pipeline, two bind groups (see render.ts) — the exact same shader
// draws both the grown particles and the target point cloud overlay,
// just with a different positions buffer/color/radius bound in.
//
// Domain is [0,1]^2, +Y-up (core/gridUpdate.wgsl's own convention,
// gravity pulls toward y=0) — same as WebGPU's own NDC, so `pos*2-1` is
// the entire mapping, no Y-flip needed (matches mls-mpm/src/gpu/render.wgsl's
// own comment on this exact point).

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
// View zoom is applied in the geometry vertex stage. Particle quads are still
// rasterized directly at the canvas's native resolution; no completed image
// is enlarged in a later presentation pass.
// x: zoom, y: particle shape (0 dot, 1 heading-oriented triangle),
// z: unified particle alpha.
@group(1) @binding(0) var<uniform> viewStyle: vec4<f32>;

fn viewCenter(center: vec2<f32>) -> vec2<f32> {
  return center * max(viewStyle.x, 1e-4);
}

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
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

// appearanceScale is visible AREA. Radius therefore scales by sqrt(area),
// making a newborn emerge from a point without making its early disc area
// grow quadratically faster than the morphoelastic rest area it mirrors.
fn appearanceRadiusScale(instanceIndex: u32) -> f32 {
  return sqrt(clamp(particleRest[instanceIndex].appearanceScale, 0.0, 1.0));
}

@vertex
fn particleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VOut {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  var out: VOut;
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0
  );
  out.uv = offset;
  return out;
}

// Target points have no particle rest state and remain constant-sized.
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

// --- growth-neuron activation dots ------------------------------------------
// Hue encodes the growth-vector direction and saturation/brightness
// increases with its magnitude. This reads ParticleRest
// directly, exactly like the directional-arrow pass below.

struct ActivationDotOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) activation: vec2<f32>,
}

@vertex
fn activationParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> ActivationDotOut {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  var out: ActivationDotOut;
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0
  );
  out.uv = offset;
  let rest = particleRest[instanceIndex];
  out.activation = vec2<f32>(rest.cycleActive, rest.growthAngle);
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

// --- heading triangles: same positions/radius/color bindings as the
// circle pipeline above, plus Agents' own per-particle alignment cache
// buffer (binding 3 — additive, not colliding with 0-2). Only ever bound
// against MpmCore's/Agents' own live particle buffers, never the static
// target-point overlay (which has no heading to point toward) — see
// render.ts's own Renderer for which pipeline draws which.
//
// Heading is NOT derived from velocity here (an earlier revision did
// atan2(vel.y,vel.x) — see agents.wgsl's own module docstring for why
// that coupling was removed project-wide): it is the channel-7-gradient
// alignment cache agents.wgsl refreshes every controller evaluation.
//
// ParticleMeta is a small, deliberate DUPLICATE of core/agents.wgsl's
// own struct of the same name (WGSL has no cross-module share
// mechanism) — heading used to be its own tightly-packed array<f32>
// buffer; it got folded into this packed struct alongside rng/cooldown/
// angularVelocity and neural color specifically to free storage-buffer slots
// core/agents.wgsl needed for growth's parent-state inheritance (see
// that file's own module docstring) — this pipeline only ever reads the
// few fields each render mode needs, but the FULL struct layout (every
// field, in this exact order) has to match agents.wgsl's own for the
// stride/offsets to line up, since both shaders bind the exact same
// GPUBuffer. ---

// x: alpha, y: saturation amplification, z: contrast around sigmoid neutral,
// w: visualization-only gain for growth-vector magnitude.
@group(0) @binding(7) var<uniform> neuralColorStyle: vec4<f32>;
struct InternalStateStyle {
  channels: vec4<u32>,
  alpha: f32,
  opponentSubtraction: f32,
  // Scalars deliberately keep this uniform at 32 bytes. A vec3 here would
  // align to the next 16-byte boundary and inflate the required binding to 48.
  _padding1: f32,
  _padding2: f32,
}
@group(0) @binding(8) var<uniform> internalStateStyle: InternalStateStyle;

fn stateSigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

fn privateStateDisplayColor(rawState: vec3<f32>) -> vec3<f32> {
  var normalized = rawState;
  // Preserve relative channel strengths instead of letting a large positive
  // state drive one or more sigmoid-mapped RGB components into saturation.
  // Values already within the display range retain the previous mapping.
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

// --- neural RGB dots --------------------------------------------------------

struct NeuralColorDotOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
}

@vertex
fn neuralColorParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  var out: NeuralColorDotOut;
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0
  );
  out.uv = offset;
  out.color = particleMeta[instanceIndex].color.rgb;
  return out;
}

@fragment
fn neuralColorParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  let radiusSquared = dot(in.uv, in.uv);
  if (viewStyle.y < 0.5 && radiusSquared > 1.0) {
    discard;
  }
  // Expand small differences between sigmoid RGB channels so early neural
  // colors don't all read as neutral gray. This affects visualization only;
  // the inspector and particle state retain the exact raw values.
  let contrasted = vec3<f32>(0.5) + (in.color - vec3<f32>(0.5)) * neuralColorStyle.z;
  let luminance = dot(contrasted, vec3<f32>(0.2126, 0.7152, 0.0722));
  let boosted = clamp(
    vec3<f32>(luminance) + (contrasted - vec3<f32>(luminance)) * neuralColorStyle.y,
    vec3<f32>(0.0),
    vec3<f32>(1.0),
  );
  return vec4<f32>(boosted, viewStyle.z);
}

// --- growth-magnitude dots -------------------------------------------------

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
fn mitosisPropensityParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  var out: NeuralColorDotOut;
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0
  );
  out.uv = offset;
  let boostedMagnitude = clamp(
    particleMeta[instanceIndex].mitosisPropensity * neuralColorStyle.w,
    0.0,
    1.0,
  );
  out.color = berlin(boostedMagnitude);
  return out;
}

@fragment
fn mitosisPropensityParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) { discard; }
  return vec4<f32>(in.color, viewStyle.z);
}

@vertex
fn internalStateParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> NeuralColorDotOut {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  let state = particleMeta[instanceIndex].privateState;
  var out: NeuralColorDotOut;
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0
  );
  out.uv = offset;
  let colorState = vec3<f32>(
    state[internalStateStyle.channels.x],
    state[internalStateStyle.channels.y],
    state[internalStateStyle.channels.z],
  );
  // The next triplet acts as opponent color channels. Wrap across the eight
  // private-state slots so every selectable RGB window has a valid opponent.
  let opponentState = vec3<f32>(
    state[(internalStateStyle.channels.x + 3u) % 8u],
    state[(internalStateStyle.channels.y + 3u) % 8u],
    state[(internalStateStyle.channels.z + 3u) % 8u],
  );
  // Transform both triplets into the particle display color space first;
  // opponent subtraction is intentionally a color operation, not a mutation
  // or comparison of the underlying chemical-memory values.
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
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
  let levels = particleMeta[instanceIndex].chemicalState;
  var out: NeuralColorDotOut;
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0, 1.0
  );
  out.uv = offset;
  // Match substrate background's signed graypoint convention at accent=0:
  // raw 0 is neutral 0.5 and SUBSTRATE_MAX=2 maps levels linearly around it.
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

// --- boundary-value dots ---------------------------------------------------
//
// Shows the exact scalar proposed for boundary weighting:
//   wb = |grad(rho)| / (|grad(rho)| + g0)
// rho is MpmCore's policy morphology texture, sampled with the same wrapped
// bilinear interpolation and +/- one-texel central difference as
// core/agents.wgsl. This is deliberately a particle mode rather than a
// background: it shows the value each policy-controlled cell observes at its
// own position. The sequential blue -> teal -> yellow ramp maps wb in [0,1].

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
  let position = fract(pointPositions[instanceIndex]);
  let center = viewCenter(position * 2.0 - vec2<f32>(1.0, 1.0));
  let offset = particleOffset(vertexIndex, instanceIndex);
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
  out.position = vec4<f32>(
    center + offset * pointRadius * appearanceRadiusScale(instanceIndex) * viewStyle.x,
    0.0,
    1.0,
  );
  out.uv = offset;
  out.color = boundaryValueColor(boundaryValue);
  return out;
}

@fragment
fn boundaryValueParticleFragment(in: NeuralColorDotOut) -> @location(0) vec4<f32> {
  if (outsideParticleShape(in.uv)) { discard; }
  return vec4<f32>(in.color, viewStyle.z);
}

// Optional one-pixel heading indicator, independently composited over either
// particle shape and every color mode. ParticleMeta.alignment stores the
// L2-clipped chemical-channel-7 gradient, so its magnitude is confidence, not
// a useful display length. Normalize it here to keep every defined heading
// visible; a flat field still produces a zero-length line.
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

// World-space continuous growth vector written by the neural policy.
@vertex
fn growthLineVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> @builtin(position) vec4<f32> {
  let center = viewCenter(pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0));
  let rest = particleRest[instanceIndex];
  let vector = vec2<f32>(rest.cycleActive, rest.growthAngle);
  let magnitude = min(length(vector), 1.0);
  let direction = select(vec2<f32>(0.0), vector / max(length(vector), 1e-8), magnitude > 1e-8);
  let offset = select(vec2<f32>(0.0), direction * directionalLineStyle.y * 1.5 * magnitude, vertexIndex == 1u);
  return vec4<f32>(center + offset * viewStyle.x, 0.0, 1.0);
}



const GRID_N: u32 = __GRID_N__u;
const NODES: u32 = GRID_N + 1u;

const CH_MASS: u32 = 2u;
const SCALE: f32 = 4096.0;

const MODE_NONE: u32 = 0u;
const MODE_DENSITY: u32 = 1u;
const MODE_SPEED: u32 = 2u;
const MODE_DEFORMATION: u32 = 3u;
const MODE_PRESSURE: u32 = 4u;
const MODE_SHEAR: u32 = 5u;

const BG: vec3<f32> = vec3<f32>(0.02, 0.02, 0.02);

const DENSITY_MAX: f32 = 10.0;
const SPEED_MAX: f32 = 4.0;
const DEFORMATION_MAX: f32 = 0.15;
const SHEAR_MAX: f32 = 1.0;
const PRESSURE_MAX: f32 = 4000.0;
const PRESSURE_SCALE: f32 = 2.0;

const MIN_MASS: f32 = 1e-6;

const BATLOW_N: u32 = 32u;
const BATLOW: array<vec3<f32>, 32> = array<vec3<f32>, 32>(
  vec3<f32>(0.0052, 0.0982, 0.3498),
  vec3<f32>(0.0321, 0.1468, 0.3582),
  vec3<f32>(0.0494, 0.1911, 0.3658),
  vec3<f32>(0.0592, 0.2298, 0.3723),
  vec3<f32>(0.0679, 0.2671, 0.3782),
  vec3<f32>(0.0771, 0.2970, 0.3824),
  vec3<f32>(0.0903, 0.3257, 0.3849),
  vec3<f32>(0.1097, 0.3531, 0.3842),
  vec3<f32>(0.1413, 0.3812, 0.3771),
  vec3<f32>(0.1778, 0.4030, 0.3638),
  vec3<f32>(0.2201, 0.4219, 0.3443),
  vec3<f32>(0.2662, 0.4386, 0.3201),
  vec3<f32>(0.3208, 0.4560, 0.2898),
  vec3<f32>(0.3709, 0.4711, 0.2621),
  vec3<f32>(0.4225, 0.4861, 0.2345),
  vec3<f32>(0.4762, 0.5013, 0.2081),
  vec3<f32>(0.5402, 0.5186, 0.1831),
  vec3<f32>(0.6005, 0.5336, 0.1706),
  vec3<f32>(0.6627, 0.5475, 0.1740),
  vec3<f32>(0.7243, 0.5596, 0.1954),
  vec3<f32>(0.7906, 0.5714, 0.2362),
  vec3<f32>(0.8457, 0.5817, 0.2826),
  vec3<f32>(0.8958, 0.5938, 0.3379),
  vec3<f32>(0.9378, 0.6096, 0.4023),
  vec3<f32>(0.9708, 0.6324, 0.4833),
  vec3<f32>(0.9864, 0.6555, 0.5571),
  vec3<f32>(0.9923, 0.6790, 0.6283),
  vec3<f32>(0.9930, 0.7020, 0.6958),
  vec3<f32>(0.9914, 0.7276, 0.7703),
  vec3<f32>(0.9891, 0.7510, 0.8380),
  vec3<f32>(0.9860, 0.7753, 0.9084),
  vec3<f32>(0.9814, 0.8004, 0.9813),
);

fn batlow(t: f32) -> vec3<f32> {
  let c = clamp(t, 0.0, 1.0) * f32(BATLOW_N - 1u);
  let i0 = u32(floor(c));
  let i1 = min(i0 + 1u, BATLOW_N - 1u);
  return mix(BATLOW[i0], BATLOW[i1], fract(c));
}

@group(0) @binding(13) var<uniform> accent: f32;

fn accentedMagnitude(t: f32) -> f32 {
  return pow(clamp(t, 0.0, 1.0), exp(-accent));
}

fn accentedSigned(t: f32) -> f32 {
  let c = clamp(t, -1.0, 1.0);
  return sign(c) * pow(abs(c), exp(-accent));
}

fn scalarColor(t: f32) -> vec3<f32> {
  let c = accentedMagnitude(t);
  return mix(BG, batlow(c), c);
}

fn graypoint(value: f32, scale: f32) -> f32 {
  let s = max(scale, 1e-6);
  return accentedSigned(value / s) * 0.5 + 0.5;
}

@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> gridVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> mode: u32;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(7) var<storage, read> diagnostics: array<i32>;

@compute @workgroup_size(16, 16)
fn colorizeField(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= NODES || j >= NODES) { return; }
  let idx = i * NODES + j;
  let diagBase = idx * 4u;

  var color = BG;
  if (mode == MODE_DENSITY) {
    let mass = bitcast<f32>(atomicLoad(&gridAccum[idx * 3u + CH_MASS]));
    color = scalarColor(mass / DENSITY_MAX);
  } else if (mode == MODE_SPEED) {
    let speed = length(gridVel[idx]);
    color = scalarColor(speed / SPEED_MAX);
  } else if (mode == MODE_DEFORMATION || mode == MODE_PRESSURE || mode == MODE_SHEAR) {
    let mass = f32(diagnostics[diagBase + 3u]) / SCALE;
    if (mass > MIN_MASS) {
      if (mode == MODE_DEFORMATION) {
        let avgJ = (f32(diagnostics[diagBase + 0u]) / SCALE) / mass;
        let g = graypoint(avgJ - 1.0, DEFORMATION_MAX);
        color = vec3<f32>(g, g, g);
      } else if (mode == MODE_PRESSURE) {
        let avgPressure = (f32(diagnostics[diagBase + 2u]) / PRESSURE_SCALE) / mass;
        let g = graypoint(avgPressure, PRESSURE_MAX);
        color = vec3<f32>(g, g, g);
      } else {
        let avgShear = (f32(diagnostics[diagBase + 1u]) / SCALE) / mass;
        color = scalarColor(avgShear / SHEAR_MAX);
      }
    }
  }

  textureStore(outputTex, vec2<i32>(i32(i), i32(j)), vec4<f32>(color, 1.0));
}

const QUAD_POSITIONS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

struct QuadOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@group(0) @binding(22) var<uniform> fieldViewZoom: f32;

@vertex
fn fieldVertex(@builtin(vertex_index) vertexIndex: u32) -> QuadOut {
  let p = QUAD_POSITIONS[vertexIndex];
  var out: QuadOut;
  out.position = vec4<f32>(p * max(fieldViewZoom, 1e-4), 0.0, 1.0);
  out.uv = (p + vec2<f32>(1.0)) * 0.5;
  return out;
}

@group(0) @binding(4) var fieldTex: texture_2d<f32>;
@group(0) @binding(5) var fieldSampler: sampler;

@fragment
fn fieldFragment(in: QuadOut) -> @location(0) vec4<f32> {
  return textureSample(fieldTex, fieldSampler, in.uv);
}

const REPULSION_FIELD_N: u32 = __REPULSION_FIELD_N__u;
const REPULSION_DISPLAY_MAX: f32 = 3.0;

@group(0) @binding(6) var repulsionTex: texture_2d<f32>;

@fragment
fn repulsionFragment(in: QuadOut) -> @location(0) vec4<f32> {
  let texel = clamp(vec2<i32>(in.uv * f32(REPULSION_FIELD_N)), vec2<i32>(0), vec2<i32>(i32(REPULSION_FIELD_N) - 1));
  let density = textureLoad(repulsionTex, texel, 0).r;
  let t = accentedMagnitude(density / REPULSION_DISPLAY_MAX);
  let color = mix(BG, vec3<f32>(1.0, 0.75, 0.35), t);
  return vec4<f32>(color, 1.0);
}

@group(0) @binding(19) var morphologyTex: texture_2d<f32>;
struct MorphologyDisplay {
  gradientEnabled: f32,
  densityEnabled: f32,
}
@group(0) @binding(20) var<uniform> morphologyDisplay: MorphologyDisplay;
const MORPHOLOGY_GRADIENT_DISPLAY_MAX: f32 = 0.05;

fn morphologyAt(p: vec2<i32>) -> f32 {
  let n = i32(REPULSION_FIELD_N);
  let wrapped = ((p % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n);
  return textureLoad(morphologyTex, wrapped, 0).r;
}

@fragment
fn morphologyFragment(in: QuadOut) -> @location(0) vec4<f32> {
  let texel = clamp(vec2<i32>(in.uv * f32(REPULSION_FIELD_N)), vec2<i32>(0), vec2<i32>(i32(REPULSION_FIELD_N) - 1));
  let density = morphologyAt(texel);
  let gx = 0.5 * (morphologyAt(texel + vec2<i32>(1, 0)) - morphologyAt(texel - vec2<i32>(1, 0)));
  let gy = 0.5 * (morphologyAt(texel + vec2<i32>(0, 1)) - morphologyAt(texel - vec2<i32>(0, 1)));
  let red = morphologyDisplay.gradientEnabled * (accentedSigned(gx / MORPHOLOGY_GRADIENT_DISPLAY_MAX) * 0.5 + 0.5);
  let green = morphologyDisplay.gradientEnabled * (accentedSigned(gy / MORPHOLOGY_GRADIENT_DISPLAY_MAX) * 0.5 + 0.5);
  let blue = morphologyDisplay.densityEnabled * accentedMagnitude(density);
  return vec4<f32>(red, green, blue, 1.0);
}

const SUBSTRATE_WIDTH: u32 = __FIELD_MAX_WIDTH__u;
const SUBSTRATE_HEIGHT: u32 = __FIELD_MAX_HEIGHT__u;
const SUBSTRATE_CHANNELS: u32 = __CHANNELS__u;
const SUBSTRATE_WIDTHS: array<u32, SUBSTRATE_CHANNELS> = __FIELD_WIDTHS__;
const SUBSTRATE_HEIGHTS: array<u32, SUBSTRATE_CHANNELS> = __FIELD_HEIGHTS__;
const SUBSTRATE_OFFSETS: array<u32, SUBSTRATE_CHANNELS> = __FIELD_OFFSETS__;
const SUBSTRATE_MAX: f32 = 2.0;

fn substrateIndex(c: u32, y: u32, x: u32) -> u32 {
  return SUBSTRATE_OFFSETS[c] + y * SUBSTRATE_WIDTHS[c] + x;
}

fn substrateValue(c: u32, outputX: u32, outputY: u32) -> f32 {
  let uv = (vec2<f32>(f32(outputX), f32(outputY)) + vec2<f32>(0.5))
    / vec2<f32>(f32(SUBSTRATE_WIDTH), f32(SUBSTRATE_HEIGHT));
  let width = SUBSTRATE_WIDTHS[c];
  let height = SUBSTRATE_HEIGHTS[c];
  let x = min(u32(floor(uv.x * f32(width))), width - 1u);
  let y = min(u32(floor(uv.y * f32(height))), height - 1u);
  return substrateGrid[substrateIndex(c, y, x)];
}

@group(0) @binding(8) var<storage, read> substrateGrid: array<f32>;
@group(0) @binding(9) var substrateOutputTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(21) var<uniform> substrateDisplay: vec4<u32>;

@group(0) @binding(23) var<uniform> backgroundZeroIsBlack: vec2<u32>;

fn substrateDisplayValue(value: f32) -> f32 {
  if (backgroundZeroIsBlack.x != 0u) {
    return max(accentedSigned(value / max(SUBSTRATE_MAX, 1e-6)), 0.0);
  }
  return graypoint(value, SUBSTRATE_MAX);
}

@compute @workgroup_size(16, 16)
fn colorizeSubstrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= SUBSTRATE_WIDTH || y >= SUBSTRATE_HEIGHT) { return; }

  let channelStart = substrateDisplay.x;
  let r = substrateValue(channelStart, x, y);
  let g = substrateValue(min(channelStart + 1u, SUBSTRATE_CHANNELS - 1u), x, y);
  let b = substrateValue(min(channelStart + 2u, SUBSTRATE_CHANNELS - 1u), x, y);
  var color = vec3<f32>(substrateDisplayValue(r), substrateDisplayValue(g), substrateDisplayValue(b));
  if (substrateDisplay.z < 3u) { color.b = 0.0; }
  if (substrateDisplay.z < 2u) { color.g = 0.0; }
  if (substrateDisplay.y != 0u) {

    let orientationChannel = min(__HEADING_CHANNEL_INDEX__u, SUBSTRATE_CHANNELS - 1u);
    let value = substrateValue(orientationChannel, x, y);
    // Preserve the diagnostic orientation display's signed scale.
    let orientationValue = select(graypoint(value, SUBSTRATE_MAX),
      max(accentedSigned(value / SUBSTRATE_MAX), 0.0), backgroundZeroIsBlack.x != 0u);
    color = vec3<f32>(orientationValue);
  }

  textureStore(substrateOutputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}

@group(0) @binding(10) var substrateTex: texture_2d<f32>;

@fragment
fn substrateFragment(in: QuadOut) -> @location(0) vec4<f32> {
  return textureSample(substrateTex, fieldSampler, in.uv);
}

const GRADIENT_MAX: f32 = 0.25;

fn densityAt(x: i32, y: i32) -> f32 {
  let n = i32(REPULSION_FIELD_N);
  let wrapped = vec2<i32>((x + n) % n, (y + n) % n);
  return textureLoad(repulsionTex, wrapped, 0).r;
}

const BLUR_MAX_RADIUS: i32 = 6;

@group(0) @binding(14) var<uniform> blurSigma: f32;
@group(0) @binding(17) var blurredDensityOutputTex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn blurDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= REPULSION_FIELD_N || gid.y >= REPULSION_FIELD_N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);

  if (blurSigma <= 1e-4) {
    textureStore(blurredDensityOutputTex, vec2<i32>(x, y), vec4<f32>(densityAt(x, y), 0.0, 0.0, 0.0));
    return;
  }

  let radius = min(BLUR_MAX_RADIUS, i32(ceil(blurSigma * 3.0)));
  var sum: f32 = 0.0;
  var weightSum: f32 = 0.0;
  for (var dy: i32 = -radius; dy <= radius; dy = dy + 1) {
    for (var dx: i32 = -radius; dx <= radius; dx = dx + 1) {
      let w = exp(-f32(dx * dx + dy * dy) / (2.0 * blurSigma * blurSigma));
      sum = sum + densityAt(x + dx, y + dy) * w;
      weightSum = weightSum + w;
    }
  }
  textureStore(blurredDensityOutputTex, vec2<i32>(x, y), vec4<f32>(sum / weightSum, 0.0, 0.0, 0.0));
}

@group(0) @binding(18) var blurredDensityTex: texture_2d<f32>;

fn blurredDensityAt(x: i32, y: i32) -> f32 {
  let n = i32(REPULSION_FIELD_N);
  let wrapped = vec2<i32>((x + n) % n, (y + n) % n);
  return textureLoad(blurredDensityTex, wrapped, 0).r;
}

fn sobelX(dy: i32, dx: i32) -> f32 {
  if (dx == 0) { return 0.0; }
  let mag = select(0.125, 0.25, dy == 0);
  return select(-mag, mag, dx > 0);
}
fn sobelY(dy: i32, dx: i32) -> f32 {
  return sobelX(dx, dy);
}

@group(0) @binding(15) var gradientOutputTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(19) var<uniform> gradientExponent: f32;

@compute @workgroup_size(16, 16)
fn colorizeGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= REPULSION_FIELD_N || gid.y >= REPULSION_FIELD_N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);

  var gx: f32 = 0.0;
  var gy: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let v = blurredDensityAt(x + dx, y + dy);
      gx = gx + v * sobelX(dy, dx);
      gy = gy + v * sobelY(dy, dx);
    }
  }

  let normalized = clamp(vec2<f32>(gx, gy) / GRADIENT_MAX, vec2<f32>(-1.0), vec2<f32>(1.0));
  let mag = length(normalized);
  let shapedMag = pow(min(mag, 1.0), gradientExponent);
  var dir = vec2<f32>(0.0, 0.0);
  if (mag > 1e-6) {
    dir = normalized / mag;
  }
  let shaped = dir * shapedMag;

  var color = vec3<f32>(shaped.x * 0.5 + 0.5, shaped.y * 0.5 + 0.5, 0.5);
  if (backgroundZeroIsBlack.y != 0u) {
    color = vec3<f32>(max(shaped.x, 0.0), max(shaped.y, 0.0), 0.0);
  }
  textureStore(gradientOutputTex, vec2<i32>(x, y), vec4<f32>(color, 1.0));
}

@group(0) @binding(16) var gradientTex: texture_2d<f32>;

@fragment
fn gradientFragment(in: QuadOut) -> @location(0) vec4<f32> {
  return textureSample(gradientTex, fieldSampler, in.uv);
}

const GROWTH_NODE_STRIDE: u32 = GRID_N + 1u;
const GROWTH_FIELD_CHANNELS: u32 = __GROWTH_FIELD_CHANNELS__u;

@group(0) @binding(24) var<storage, read> integratedGrowthField: array<f32>;
@group(0) @binding(25) var growthOutputTex: texture_storage_2d<rgba8unorm, write>;

fn cyclicDirectionColor(angle: f32) -> vec3<f32> {
  let phase = vec3<f32>(angle, angle - 2.0943951, angle + 2.0943951);
  return vec3<f32>(0.5) + 0.5 * cos(phase);
}

@compute @workgroup_size(16, 16)
fn colorizeGrowth(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > GRID_N || gid.y > GRID_N) { return; }
  let node = gid.x * GROWTH_NODE_STRIDE + gid.y;
  let base = node * GROWTH_FIELD_CHANNELS;
  let sampleWeight = max(integratedGrowthField[base + 5u], 0.0);
  let vector = vec2<f32>(
    integratedGrowthField[base],
    integratedGrowthField[base + 1u],
  ) / max(sampleWeight, 1e-30);
  let tensor = vec3<f32>(
    integratedGrowthField[base + 2u],
    integratedGrowthField[base + 3u],
    integratedGrowthField[base + 4u],
  ) / max(sampleWeight, 1e-30);
  // Sum of absolute eigenvalues keeps contraction and zero-trace remodeling visible.
  let averageDrive = clamp(max(abs(tensor.x + tensor.z),
    length(vec2<f32>(tensor.x - tensor.z, 2.0 * tensor.y))), 0.0, 1.0);
  let signedStrength = clamp(length(vector) / max(averageDrive, 1e-8), 0.0, 1.0);
  let axialStrength = clamp(
    sqrt((tensor.x - tensor.z) * (tensor.x - tensor.z) + 4.0 * tensor.y * tensor.y)
      / max(averageDrive, 1e-8), 0.0, 1.0);
  var angle = 0.5 * atan2(2.0 * tensor.y, tensor.x - tensor.z);
  if (signedStrength > 0.05) {
    angle = atan2(vector.y, vector.x);
  }
  let directionalStrength = max(signedStrength, axialStrength);
  let neutralGrowth = vec3<f32>(0.10, 0.72, 0.24);
  let directionalGrowth = cyclicDirectionColor(angle);
  let signalColor = mix(neutralGrowth, directionalGrowth, directionalStrength);

  let intensity = accentedMagnitude(averageDrive);
  let color = mix(BG, signalColor, intensity);
  textureStore(growthOutputTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}

struct GrowthVectorOut {
  @builtin(position) position: vec4<f32>,
  @location(0) strength: f32,
}

fn growthVectorAt(node: u32) -> vec2<f32> {
  let base = node * GROWTH_FIELD_CHANNELS;
  let weight = max(integratedGrowthField[base + 5u], 0.0);
  return vec2<f32>(
    integratedGrowthField[base],
    integratedGrowthField[base + 1u],
  ) / max(weight, 1e-30);
}

fn arrowSegmentVertex(
  vertex: u32,
  start: vec2<f32>,
  finish: vec2<f32>,
  halfWidth: f32,
) -> vec2<f32> {
  let along = finish - start;
  let normal = normalize(vec2<f32>(-along.y, along.x)) * halfWidth;
  let corners = array<vec2<f32>, 6>(
    start - normal, finish - normal, finish + normal,
    start - normal, finish + normal, start + normal,
  );
  return corners[vertex];
}

@vertex
fn growthVectorVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> GrowthVectorOut {
  let nodeX = instanceIndex / NODES;
  let nodeY = instanceIndex % NODES;
  let zoom = max(fieldViewZoom, 1.0);
  let vector = growthVectorAt(instanceIndex);
  let magnitude = length(vector);
  let strength = clamp(magnitude, 0.0, 1.0);
  let direction = select(vec2<f32>(1.0, 0.0), vector / max(magnitude, 1e-8), magnitude > 1e-8);

  let center = vec2<f32>(f32(nodeX), f32(nodeY)) / f32(GRID_N) * 2.0 - vec2<f32>(1.0);
  let glyphSpacing = 2.0 / f32(GRID_N);

  let displayedMagnitude = clamp(magnitude / 0.12, 0.0, 1.0);
  let arrowLength = min(glyphSpacing * 0.62 * displayedMagnitude, 0.085 / zoom);
  let shaftHalfWidth = min(glyphSpacing * 0.020, 0.003 / zoom);
  let tail = center;
  let tip = center + direction * arrowLength;
  let headBase = tip - direction * arrowLength * 0.32;
  let perpendicular = vec2<f32>(-direction.y, direction.x);
  let headWidth = arrowLength * 0.15;
  var p = vec2<f32>(2.0);
  if (vertexIndex < 6u) {
    p = arrowSegmentVertex(vertexIndex, tail, headBase, shaftHalfWidth);
  } else {

    let head = array<vec2<f32>, 3>(
      headBase + perpendicular * headWidth,
      headBase - perpendicular * headWidth,
      tip,
    );
    p = head[vertexIndex - 6u];
  }

  p = select(vec2<f32>(2.0), p, strength > 1e-5);
  var out: GrowthVectorOut;
  out.position = vec4<f32>(p * zoom, 0.0, 1.0);
  out.strength = strength;
  return out;
}

@fragment
fn growthVectorFragment(in: GrowthVectorOut) -> @location(0) vec4<f32> {
  return vec4<f32>(vec3<f32>(0.60, 1.0, 0.66), 0.95);
}

@group(0) @binding(26) var<storage, read> policyGradient: array<f32>;

// Match core/agents.wgsl: periodic quadratic sampling of the cached gradient,
// including the per-channel resolution correction used by agentStep.
fn policyOrientationAt(uv: vec2<f32>) -> vec2<f32> {
  let c = min(__HEADING_CHANNEL_INDEX__u, SUBSTRATE_CHANNELS - 1u);
  let size = vec2<u32>(SUBSTRATE_WIDTHS[c], SUBSTRATE_HEIGHTS[c]);
  let pos = fract(uv) * vec2<f32>(size) - vec2<f32>(0.5);
  let base = vec2<i32>(floor(pos - vec2<f32>(0.5)));
  let f = pos - vec2<f32>(base);
  let weights = array<vec2<f32>, 3>(
    0.5 * (vec2<f32>(1.5) - f) * (vec2<f32>(1.5) - f),
    vec2<f32>(0.75) - (f - vec2<f32>(1.0)) * (f - vec2<f32>(1.0)),
    0.5 * (f - vec2<f32>(0.5)) * (f - vec2<f32>(0.5)),
  );
  var gradient = vec2<f32>(0.0);
  for (var x = 0u; x < 3u; x++) {
    for (var y = 0u; y < 3u; y++) {
      let cell = ((base + vec2<i32>(i32(x), i32(y))) % vec2<i32>(size) + vec2<i32>(size)) % vec2<i32>(size);
      let index = substrateIndex(c, u32(cell.y), u32(cell.x));
      gradient += vec2<f32>(policyGradient[index], policyGradient[__FIELD_TOTAL__u + index]) * weights[x].x * weights[y].y;
    }
  }
  return gradient * vec2<f32>(size) / vec2<f32>(f32(SUBSTRATE_WIDTH), f32(SUBSTRATE_HEIGHT));
}

@vertex
fn policyOrientationVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> GrowthVectorOut {
  let uv = (vec2<f32>(f32(instanceIndex % 32u), f32(instanceIndex / 32u)) + vec2<f32>(0.5)) / 32.0;
  let gradient = policyOrientationAt(uv);
  let magnitude = length(gradient);
  let direction = gradient / max(magnitude, 1e-10);
  // Constant-length glyphs expose direction even where alignment is weak.
  // A flat field has no defined direction and therefore no arrow.
  let center = uv * 2.0 - vec2<f32>(1.0);
  let tail = center - direction * 0.018;
  let tip = center + direction * 0.018;
  let headBase = tip - direction * 0.011;
  let normal = vec2<f32>(-direction.y, direction.x);
  var p: vec2<f32>;
  if (vertexIndex < 6u) {
    p = arrowSegmentVertex(vertexIndex, tail, headBase, 0.0012);
  } else {
    let head = array<vec2<f32>, 3>(headBase + normal * 0.006, headBase - normal * 0.006, tip);
    p = head[vertexIndex - 6u];
  }
  var out: GrowthVectorOut;
  out.position = vec4<f32>(select(vec2<f32>(2.0), p, magnitude > 1e-10) * max(fieldViewZoom, 1.0), 0.0, 1.0);
  out.strength = min(magnitude, 1.0);
  return out;
}

@fragment
fn policyOrientationFragment(in: GrowthVectorOut) -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.12, 0.05, 0.95);
}

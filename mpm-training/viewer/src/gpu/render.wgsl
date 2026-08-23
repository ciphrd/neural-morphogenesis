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

@vertex
fn particleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VOut {
  let center = pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  let offset = QUAD_OFFSETS[vertexIndex];
  var out: VOut;
  out.position = vec4<f32>(center + offset * pointRadius, 0.0, 1.0);
  out.uv = offset;
  return out;
}

@fragment
fn particleFragment(in: VOut) -> @location(0) vec4<f32> {
  if (dot(in.uv, in.uv) > 1.0) {
    discard;
  }
  return pointColor;
}

// --- growth-neuron activation dots ------------------------------------------
// Hue encodes the normalized growth direction and saturation/brightness
// increases with the independent anisotropy output. This reads ParticleRest
// directly, exactly like the directional-arrow pass below.

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  cycleActive: f32,
  growthDirection: vec2<f32>,
  growthControls: vec2<f32>,
}

@group(0) @binding(4) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(6) var<uniform> activationAlpha: f32;

struct ActivationDotOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) activation: vec2<f32>,
}

@vertex
fn activationParticleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> ActivationDotOut {
  let center = pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  let offset = QUAD_OFFSETS[vertexIndex];
  var out: ActivationDotOut;
  out.position = vec4<f32>(center + offset * pointRadius, 0.0, 1.0);
  out.uv = offset;
  let rest = particleRest[instanceIndex];
  out.activation = rest.growthDirection * rest.growthControls.x;
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
  if (dot(in.uv, in.uv) > 1.0) {
    discard;
  }
  return vec4<f32>(neuronActivationColor(in.activation), activationAlpha);
}

// --- heading triangles: same positions/radius/color bindings as the
// circle pipeline above, plus Agents' own persistent per-particle state
// buffer (binding 3 — additive, not colliding with 0-2). Only ever bound
// against MpmCore's/Agents' own live particle buffers, never the static
// target-point overlay (which has no heading to point toward) — see
// render.ts's own Renderer for which pipeline draws which.
//
// Heading is NOT derived from velocity here (an earlier revision did
// atan2(vel.y,vel.x) — see agents.wgsl's own module docstring for why
// that coupling was removed project-wide): it's agents.wgsl's own
// persistent per-particle state, the same buffer that shader integrates
// every macro step.
//
// ParticleMeta is a small, deliberate DUPLICATE of core/agents.wgsl's
// own struct of the same name (WGSL has no cross-module share
// mechanism) — heading used to be its own tightly-packed array<f32>
// buffer; it got folded into this 4-field struct alongside rng/cooldown/
// angularVelocity specifically to free storage-buffer slots
// core/agents.wgsl needed for growth's parent-state inheritance (see
// that file's own module docstring) — this pipeline only ever reads the
// one field it needs (.heading), but the FULL struct layout (all 4
// fields, in this exact order) has to match agents.wgsl's own for the
// stride/offsets to line up, since both shaders bind the exact same
// GPUBuffer. ---

struct ParticleMeta {
  rng: u32,
  cooldown: f32,
  heading: f32,
  angularVelocity: f32,
}
@group(0) @binding(3) var<storage, read> particleMeta: array<ParticleMeta>;

// Local-space wedge pointing along +X, rotated by each particle's own
// heading before translating to its position — an isoceles triangle, not
// a quad-carved shape, so this is its own 3-vertex draw (render.ts's own
// draw(3, count) call), not a discard-based circle.
const TRI_LOCAL = array<vec2<f32>, 3>(
  vec2<f32>(1.4, 0.0), vec2<f32>(-0.9, 0.9), vec2<f32>(-0.9, -0.9),
);

@vertex
fn triangleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> @builtin(position) vec4<f32> {
  let center = pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  let heading = particleMeta[instanceIndex].heading;
  let c = cos(heading);
  let s = sin(heading);
  let local = TRI_LOCAL[vertexIndex];
  let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
  return vec4<f32>(center + rotated * pointRadius, 0.0, 1.0);
}

@fragment
fn triangleFragment() -> @location(0) vec4<f32> {
  return pointColor;
}

// --- directional-growth axes -------------------------------------------------
//
// ParticleRest.growthDirection is the world-frame signal consumed by
// core/g2p.wgsl and core/agents.wgsl's polarized division. Tensor stretch
// alone is axial, but division now uses the SIGN: the new daughter and pair
// center bias toward +n. Therefore this is a one-way arrow pointing toward
// the selected axis. The independent anisotropy output controls glyph length;
// division bias is reported separately in the network inspector.
// This pass reads the live GPU buffers directly; there is no diagnostic
// readback or duplicated frontend approximation.

// x=max half-length, y=shaft half-width, z=head length, w=head half-width,
// all in NDC and derived from device pixels by render.ts.
@group(0) @binding(5) var<uniform> growthAxisStyle: vec4<f32>;

struct GrowthAxisOut {
  @builtin(position) position: vec4<f32>,
  @location(0) strength: f32,
}

@vertex
fn growthAxisVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> GrowthAxisOut {
  let center = pointPositions[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  let rest = particleRest[instanceIndex];
  let raw = rest.growthDirection;
  let rawMagnitude = length(raw);
  let strength = select(0.0, clamp(rest.growthControls.x, 0.0, 1.0), rawMagnitude > 1e-6);
  var axis = vec2<f32>(1.0, 0.0);
  if (rawMagnitude > 1e-6) {
    axis = raw / rawMagnitude;
  }
  let normal = vec2<f32>(-axis.y, axis.x);

  let halfLength = growthAxisStyle.x * strength;
  let widthScale = sqrt(strength);
  let halfWidth = growthAxisStyle.y * widthScale;
  let headLength = min(growthAxisStyle.z * widthScale, halfLength * 0.55);
  let headHalfWidth = growthAxisStyle.w * widthScale;
  let inner = max(halfLength - headLength, 0.0);

  // 0..5: shaft quad; 6..8: arrow head pointing toward +n.
  var along = 0.0;
  var across = 0.0;
  switch vertexIndex {
    case 0u: { along = -inner; across = -halfWidth; }
    case 1u: { along =  inner; across = -halfWidth; }
    case 2u: { along =  inner; across =  halfWidth; }
    case 3u: { along = -inner; across = -halfWidth; }
    case 4u: { along =  inner; across =  halfWidth; }
    case 5u: { along = -inner; across =  halfWidth; }
    case 6u: { along =  halfLength; across = 0.0; }
    case 7u: { along =  inner; across = -headHalfWidth; }
    case 8u: { along =  inner; across =  headHalfWidth; }
    default: { along = inner; across = headHalfWidth; }
  }

  var out: GrowthAxisOut;
  out.position = vec4<f32>(center + axis * along + normal * across, 0.0, 1.0);
  out.strength = strength;
  return out;
}

@fragment
fn growthAxisFragment(in: GrowthAxisOut) -> @location(0) vec4<f32> {
  if (in.strength < 0.01) {
    discard;
  }
  return vec4<f32>(pointColor.rgb, pointColor.a * (0.35 + 0.65 * in.strength));
}

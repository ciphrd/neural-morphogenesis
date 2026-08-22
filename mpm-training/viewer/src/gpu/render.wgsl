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

// --- heading triangles: same positions/radius/color bindings as the
// circle pipeline above, plus Agents' own persistent heading buffer
// (binding 3 — additive, not colliding with 0-2). Only ever bound
// against MpmCore's/Agents' own live particle buffers, never the static
// target-point overlay (which has no heading to point toward) — see
// render.ts's own Renderer for which pipeline draws which.
//
// Heading is NOT derived from velocity here (an earlier revision did
// atan2(vel.y,vel.x) — see agents.wgsl's own module docstring for why
// that coupling was removed project-wide): it's agents.wgsl's own
// persistent per-particle state, the same buffer that shader integrates
// every macro step. ---

@group(0) @binding(3) var<storage, read> pointHeading: array<f32>;

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
  let heading = pointHeading[instanceIndex];
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

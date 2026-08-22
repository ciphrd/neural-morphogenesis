// Presentation: particles as small instanced circles (a per-pixel
// discard against `uv`, the quad's own [-1,1] offset — see
// particleFragment below), plus a static boundary-box outline — the
// WebGPU equivalent of mls-mpm88-explained's own taichi GUI
// (canvas.circle(...)/canvas.rect(...)). At the original ~1px point
// size a circle and a filled square were visually indistinguishable
// (not worth the extra per-pixel discard), but main.ts's Particle Size
// slider can now push individual particles up to 16px — worlds/growth.ts
// in particular renders sparse, individually-placed dots at that size,
// where a square reads as visibly blocky. The simulation's [0,1]^2
// domain is already +Y-up (gravity subtracts from v.y, same orientation
// as WebGPU's NDC), so `pos*2-1` maps straight to NDC with no Y-flip —
// unlike envnca's grid, which is +Y-down and needs one.

// --- Particles ---

var<private> quadOffsets: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct ParticleVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> particleColor: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> pointRadius: f32; // NDC-space radius, kept in sync with canvas size

@vertex
fn particleVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> ParticleVOut {
  let center = particlePos[instanceIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  let offset = quadOffsets[vertexIndex];
  var out: ParticleVOut;
  out.position = vec4<f32>(center + offset * pointRadius, 0.0, 1.0);
  out.uv = offset;
  out.color = particleColor[instanceIndex];
  return out;
}

@fragment
fn particleFragment(in: ParticleVOut) -> @location(0) vec4<f32> {
  // `uv` is the quad's own offset ([-1,1] per axis, unit circle
  // inscribed) — discard anything outside it so the quad reads as a
  // circle instead of a square.
  if (dot(in.uv, in.uv) > 1.0) { discard; }
  return in.color;
}

// --- Boundary box (mirrors canvas.rect(Vec(0.04), Vec(0.96))) ---

var<private> boundaryPoints: array<vec2<f32>, 5> = array<vec2<f32>, 5>(
  vec2<f32>(0.04, 0.04),
  vec2<f32>(0.96, 0.04),
  vec2<f32>(0.96, 0.96),
  vec2<f32>(0.04, 0.96),
  vec2<f32>(0.04, 0.04)
);

@vertex
fn boundaryVertex(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  let p = boundaryPoints[vertexIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  return vec4<f32>(p, 0.0, 1.0);
}

@fragment
fn boundaryFragment() -> @location(0) vec4<f32> {
  // Reference's boundary color, 0x4FB99F = rgb(79, 185, 159).
  return vec4<f32>(79.0 / 255.0, 185.0 / 255.0, 159.0 / 255.0, 1.0);
}

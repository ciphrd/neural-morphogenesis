// Presentation: a full-screen background quad sampling the colorized
// grid texture, plus a shared instanced-quad pipeline drawn twice —
// small white agent markers first, small red target-point markers last
// (see gpu/render.ts's own draw-order comment for why: the target
// outline should stay visible as a reference even where agents
// currently sit on top of it). No templated constants needed here: grid
// dimensions arrive via MarkerUniforms at runtime rather than being
// baked in, since this module doesn't need any compile-time-sized
// scratch arrays.
//
// Grid pixel space is +Y-down (row 0 = top, matching environment.py's
// (C,H,W) layout); WebGPU NDC is +Y-up — both the background UV mapping
// and the marker's pixel->NDC mapping apply the same flip, consistently,
// so one layer never ends up mirrored relative to the other.
//
// The grid is always square, but the <canvas> it's presented into
// usually isn't (it fills whatever rectangle the surrounding layout
// gives it). Letterboxing used to be faked here by scaling NDC positions
// by a CPU-computed (scaleX, scaleY) uniform — removed: scaling the
// background's overscanning "big triangle" (see backgroundVertex)
// anisotropically breaks the property that makes that trick work at
// all. The unscaled triangle's hypotenuse is constructed to lie exactly
// outside the canonical [-1,1]^2 clip volume (touching it at only one
// corner), so the GPU's own clip-space clipping trims it to precisely
// fill that square with no visible diagonal edge — but scale x and y by
// *different* factors (exactly what non-square letterboxing needs) and
// that guarantee no longer holds: part of the hypotenuse can swing back
// inside [-1,1]^2, surviving clipping and rasterizing as a visible
// diagonal seam through the letterbox bar. (Confirmed directly: with
// scaleX=0.5, scaleY=1, the hypotenuse point at t=0.4 lands at (0.7,
// 0.6) — inside the clip volume, so it's drawn.)
//
// Both vertex shaders below now emit plain, unscaled NDC — the *real*
// GPU viewport (gpu/render.ts's render() calls setViewport with the
// letterboxed pixel rect before these draws) does the letterboxing
// instead, remapping this shared [-1,1]^2 space onto that sub-rect for
// every draw in the pass, keeping the background and both marker layers
// in registration without any per-shader scale factor at all.

// --- Background: full-screen triangle sampling the colorized texture ---

var<private> quadPositions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(3.0, -1.0),
  vec2<f32>(-1.0, 3.0)
);

struct BackgroundVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var backgroundTex: texture_2d<f32>;
@group(0) @binding(1) var backgroundSampler: sampler;

@vertex
fn backgroundVertex(@builtin(vertex_index) vertexIndex: u32) -> BackgroundVOut {
  let base = quadPositions[vertexIndex];
  var out: BackgroundVOut;
  out.position = vec4<f32>(base.x, base.y, 0.0, 1.0);
  // uv.y = 0 (top of texture, grid row 0) must land at ndc.y = +1 (top
  // of NDC) — see module docstring's Y-convention note.
  out.uv = vec2<f32>((base.x + 1.0) * 0.5, (1.0 - base.y) * 0.5);
  return out;
}

@fragment
fn backgroundFragment(in: BackgroundVOut) -> @location(0) vec4<f32> {
  return textureSample(backgroundTex, backgroundSampler, in.uv);
}

// --- Markers: instanced quads for agents/target points, no vertex buffer ---

var<private> quadOffsets: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct MarkerUniforms {
  color: vec4<f32>,
  halfSizePixels: f32,
  gridWidth: f32,
  gridHeight: f32,
  _pad: f32,
};

@group(0) @binding(0) var<storage, read> markerPositions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> marker: MarkerUniforms;

struct MarkerVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn markerVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> MarkerVOut {
  let center = markerPositions[instanceIndex];
  let offset = quadOffsets[vertexIndex];
  let px = center.x + offset.x * marker.halfSizePixels;
  let py = center.y + offset.y * marker.halfSizePixels;
  let ndcX = (px / marker.gridWidth) * 2.0 - 1.0;
  let ndcY = 1.0 - (py / marker.gridHeight) * 2.0;
  var out: MarkerVOut;
  out.position = vec4<f32>(ndcX, ndcY, 0.0, 1.0);
  out.color = marker.color;
  return out;
}

@fragment
fn markerFragment(in: MarkerVOut) -> @location(0) vec4<f32> {
  return in.color;
}

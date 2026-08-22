// Grid's first 3 channels -> RGB, matching the deleted render.py's
// render_frame(): each channel independently clipped to [-scale,scale]
// and mapped to [0,1] symmetric around 0 (0 always maps to 0.5, i.e.
// mid-gray) rather than a min/max stretch — see gpu/render.ts for where
// `scale` (the EMA'd per-channel magnitude) comes from.
// __CHANNELS__/__WIDTH__/__HEIGHT__ substituted by shaderTemplate.ts.
//
// Four background render modes (gpu/render.ts's BackgroundMode), picked
// by `scale.w` — packed into the same uniform as the EMA scale rather
// than a second buffer, since it's an otherwise-unused float: 0 = flat
// gray, 1 = flat black, 2 = the chemical substrate itself (this module's
// original, only, behavior before modes existed), 3 = repulsion.ts's
// own density field (see gpu/repulsion.wgsl — a debug/exploration view
// of the same field agents.wgsl's agentStep pushes agents away from).

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;
const REPULSION_RESOLUTION: u32 = __REPULSION_RESOLUTION__u;

const MODE_GRAY: u32 = 0u;
const MODE_BLACK: u32 = 1u;
const MODE_SUBSTRATE: u32 = 2u;
const MODE_REPULSION: u32 = 3u;

// render.py's BACKGROUND_GRAY = 127, i.e. what a value of exactly 0 maps
// to under the substrate mapping below — reused here so flat "gray"
// mode matches that same shade exactly, not an arbitrary different gray.
const GRAY_LEVEL: f32 = 127.0 / 255.0;

// Fixed baseline for the repulsion-density heatmap — density is an
// unnormalized sum of peak-1 Gaussian splats (see repulsion.wgsl), so a
// couple of nearby agents already produces values around this range;
// not tied to the EMA-tracked chemical scale/intensity above (an
// unrelated field, on an unrelated numeric scale) — a fixed constant
// here, not yet a live slider (see gpu/repulsion.ts's own docstring on
// what's still frontend-only/unexposed for repulsion generally).
const REPULSION_DISPLAY_SCALE: f32 = 3.0;

@group(0) @binding(0) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(1) var<uniform> scale: vec4<f32>; // xyz = per-channel EMA scale, w = mode

@group(0) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> repulsionDensity: array<f32>;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * HEIGHT * WIDTH + y * WIDTH + x;
}

// Always-non-negative modulo — same reasoning as repulsion.wgsl's own
// wrapIndex (WGSL's `%` on a negative i32 keeps its sign, unlike
// Python's).
fn wrapRepulsionIndex(v: i32, m: i32) -> u32 {
  return u32(((v % m) + m) % m);
}

// Bilinear sample of repulsionDensity (its own, independent, usually
// coarser resolution — see repulsion.wgsl's header comment) at a WIDTH/
// HEIGHT-space pixel position — same corner/weight pattern as agents.wgsl's
// sampleRepulsionGradient, one scalar instead of a two-plane gradient.
fn sampleRepulsionDensity(px: f32, py: f32) -> f32 {
  let scale2 = f32(REPULSION_RESOLUTION) / f32(WIDTH); // assumes a square grid, WIDTH == HEIGHT
  let fx = px * scale2;
  let fy = py * scale2;
  let x0f = floor(fx);
  let y0f = floor(fy);
  let x0i = i32(x0f);
  let y0i = i32(y0f);
  let wx1 = fx - x0f;
  let wx0 = 1.0 - wx1;
  let wy1 = fy - y0f;
  let wy0 = 1.0 - wy1;
  let x0 = wrapRepulsionIndex(x0i, i32(REPULSION_RESOLUTION));
  let x1 = wrapRepulsionIndex(x0i + 1, i32(REPULSION_RESOLUTION));
  let y0 = wrapRepulsionIndex(y0i, i32(REPULSION_RESOLUTION));
  let y1 = wrapRepulsionIndex(y0i + 1, i32(REPULSION_RESOLUTION));

  return wx0 * wy0 * repulsionDensity[y0 * REPULSION_RESOLUTION + x0]
       + wx1 * wy0 * repulsionDensity[y0 * REPULSION_RESOLUTION + x1]
       + wx0 * wy1 * repulsionDensity[y1 * REPULSION_RESOLUTION + x0]
       + wx1 * wy1 * repulsionDensity[y1 * REPULSION_RESOLUTION + x1];
}

@compute @workgroup_size(16, 16, 1)
fn colorize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= WIDTH || y >= HEIGHT) { return; }

  let mode = u32(round(scale.w));
  var color: vec3<f32>;

  if (mode == MODE_BLACK) {
    color = vec3<f32>(0.0, 0.0, 0.0);
  } else if (mode == MODE_GRAY) {
    color = vec3<f32>(GRAY_LEVEL, GRAY_LEVEL, GRAY_LEVEL);
  } else if (mode == MODE_REPULSION) {
    let d = sampleRepulsionDensity(f32(x), f32(y));
    let v = clamp(d / REPULSION_DISPLAY_SCALE, 0.0, 1.0);
    // Warm heatmap tint (not grayscale) so this reads as visually
    // distinct from the substrate/gray/black modes at a glance.
    color = vec3<f32>(v, v * 0.85, v * 0.6);
  } else {
    let r = gridCurrent[gridIndex(0u, y, x)];
    let g = gridCurrent[gridIndex(1u, y, x)];
    let b = gridCurrent[gridIndex(2u, y, x)];

    let sr = max(scale.x, 1e-6);
    let sg = max(scale.y, 1e-6);
    let sb = max(scale.z, 1e-6);

    color = vec3<f32>(
      clamp(r, -sr, sr) / (2.0 * sr) + 0.5,
      clamp(g, -sg, sg) / (2.0 * sg) + 0.5,
      clamp(b, -sb, sb) / (2.0 * sb) + 0.5
    );
  }

  textureStore(outputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}

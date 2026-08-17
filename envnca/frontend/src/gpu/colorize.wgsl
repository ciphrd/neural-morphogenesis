// Grid's first 3 channels -> RGB, matching the deleted render.py's
// render_frame(): each channel independently clipped to [-scale,scale]
// and mapped to [0,1] symmetric around 0 (0 always maps to 0.5, i.e.
// mid-gray) rather than a min/max stretch — see gpu/render.ts for where
// `scale` (the EMA'd per-channel magnitude) comes from.
// __CHANNELS__/__WIDTH__/__HEIGHT__ substituted by shaderTemplate.ts.
//
// Three background render modes (gpu/render.ts's BackgroundMode),
// picked by `scale.w` — packed into the same uniform as the EMA scale
// rather than a second buffer, since it's an otherwise-unused float:
// 0 = flat gray, 1 = flat black, 2 = the chemical substrate itself
// (this module's original, only, behavior before modes existed).

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;

const MODE_GRAY: u32 = 0u;
const MODE_BLACK: u32 = 1u;
const MODE_SUBSTRATE: u32 = 2u;

// render.py's BACKGROUND_GRAY = 127, i.e. what a value of exactly 0 maps
// to under the substrate mapping below — reused here so flat "gray"
// mode matches that same shade exactly, not an arbitrary different gray.
const GRAY_LEVEL: f32 = 127.0 / 255.0;

@group(0) @binding(0) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(1) var<uniform> scale: vec4<f32>; // xyz = per-channel EMA scale, w = mode

@group(0) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * HEIGHT * WIDTH + y * WIDTH + x;
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

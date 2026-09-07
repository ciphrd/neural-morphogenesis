

const FIELD_N: u32 = __FIELD_N__u;
const MAX_RADIUS: i32 = __MORPHOLOGY_MAX_RADIUS__;

struct MorphologyParams {
  radius: f32,
  densityReference: f32,
  _padding: vec2<f32>,
  weights: array<vec4<f32>, __MORPHOLOGY_WEIGHT_VECTORS__>,
}

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: MorphologyParams;

fn wrapped(v: i32) -> i32 {
  let n = i32(FIELD_N);
  return ((v % n) + n) % n;
}

@compute @workgroup_size(16, 16)
fn blurHorizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N) { return; }
  let radius = i32(params.radius);
  var sum = 0.0;
  for (var offset = -radius; offset <= radius; offset += 1) {
    if (abs(offset) <= radius) {
      let wi = u32(offset + MAX_RADIUS);
      let w = params.weights[wi / 4u][wi % 4u];
      sum += textureLoad(sourceTexture, vec2<i32>(wrapped(i32(gid.x) + offset), i32(gid.y)), 0).x * w;
    }
  }
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(sum, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn blurVerticalAndNormalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N) { return; }
  let radius = i32(params.radius);
  var sum = 0.0;
  for (var offset = -radius; offset <= radius; offset += 1) {
    if (abs(offset) <= radius) {
      let wi = u32(offset + MAX_RADIUS);
      let w = params.weights[wi / 4u][wi % 4u];
      sum += textureLoad(sourceTexture, vec2<i32>(i32(gid.x), wrapped(i32(gid.y) + offset)), 0).x * w;
    }
  }
  let rho = sum;
  let reference = max(params.densityReference, 1e-6);
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(rho / (rho + reference), 0.0, 0.0, 0.0));
}

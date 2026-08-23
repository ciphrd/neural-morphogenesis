// Simulation-owned morphology pre-pass. Renderer blur settings cannot alter
// this field or the policy inputs derived from it. A separable Gaussian keeps
// the once-per-controller-tick cost linear in radius rather than radius².

const FIELD_N: u32 = __FIELD_N__u;
const MAX_RADIUS: i32 = 8;

struct MorphologyParams {
  sigmaDomain: f32,
  densityReference: f32,
}

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: MorphologyParams;

fn wrapped(v: i32) -> i32 {
  let n = i32(FIELD_N);
  return ((v % n) + n) % n;
}

fn kernelWeight(offset: i32, sigma: f32) -> f32 {
  return select(select(0.0, 1.0, offset == 0), exp(-0.5 * f32(offset * offset) / max(sigma * sigma, 1e-8)), sigma > 1e-5);
}

@compute @workgroup_size(16, 16)
fn blurHorizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N) { return; }
  let sigma = max(params.sigmaDomain * f32(FIELD_N), 0.0);
  let radius = min(i32(ceil(3.0 * sigma)), MAX_RADIUS);
  var sum = 0.0;
  var weightSum = 0.0;
  for (var offset = -MAX_RADIUS; offset <= MAX_RADIUS; offset += 1) {
    if (abs(offset) <= radius) {
      let w = kernelWeight(offset, sigma);
      sum += textureLoad(sourceTexture, vec2<i32>(wrapped(i32(gid.x) + offset), i32(gid.y)), 0).x * w;
      weightSum += w;
    }
  }
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(sum / max(weightSum, 1e-8), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn blurVerticalAndNormalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FIELD_N || gid.y >= FIELD_N) { return; }
  let sigma = max(params.sigmaDomain * f32(FIELD_N), 0.0);
  let radius = min(i32(ceil(3.0 * sigma)), MAX_RADIUS);
  var sum = 0.0;
  var weightSum = 0.0;
  for (var offset = -MAX_RADIUS; offset <= MAX_RADIUS; offset += 1) {
    if (abs(offset) <= radius) {
      let w = kernelWeight(offset, sigma);
      sum += textureLoad(sourceTexture, vec2<i32>(i32(gid.x), wrapped(i32(gid.y) + offset)), 0).x * w;
      weightSum += w;
    }
  }
  let rho = sum / max(weightSum, 1e-8);
  let reference = max(params.densityReference, 1e-6);
  textureStore(outputTexture, vec2<i32>(gid.xy), vec4<f32>(rho / (rho + reference), 0.0, 0.0, 0.0));
}

struct DownsampleSettings { texelSize: vec2<f32>, threshold: f32, prefilter: f32 }
struct UpsampleSettings { texelSize: vec2<f32>, radius: f32, scatter: f32 }
struct CompositeSettings { intensity: f32, bloomEnabled: f32 }

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> downsampleSettings: DownsampleSettings;
@group(1) @binding(0) var detailTexture: texture_2d<f32>;
@group(1) @binding(1) var coarseTexture: texture_2d<f32>;
@group(1) @binding(2) var upsampleSampler: sampler;
@group(1) @binding(3) var<uniform> upsampleSettings: UpsampleSettings;
@group(2) @binding(0) var sceneTexture: texture_2d<f32>;
@group(2) @binding(1) var bloomTexture: texture_2d<f32>;
@group(2) @binding(2) var compositeSampler: sampler;
@group(2) @binding(3) var<uniform> compositeSettings: CompositeSettings;

struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex
fn fullscreenVertex(@builtin(vertex_index) index: u32) -> VertexOut {
  let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var out: VertexOut;
  out.position = vec4<f32>(positions[index], 0.0, 1.0);
  out.uv = positions[index] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return out;
}

fn luminance(color: vec3<f32>) -> f32 { return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn prefilter(color: vec3<f32>) -> vec3<f32> {
  if (downsampleSettings.prefilter < 0.5) { return color; }
  let brightness = max(color.r, max(color.g, color.b));
  let knee = max(downsampleSettings.threshold * 0.5, 1e-4);
  var soft = clamp(brightness - downsampleSettings.threshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 1e-4);
  let contribution = max(soft, brightness - downsampleSettings.threshold) / max(brightness, 1e-4);
  return color * contribution;
}

// First-reduction Karis weighting stops a single hot pixel from dominating
// every coarser level of the pyramid.
fn karis(color: vec3<f32>) -> vec3<f32> {
  if (downsampleSettings.prefilter < 0.5) { return color; }
  return color / (1.0 + luminance(color));
}

@fragment
fn downsampleFragment(in: VertexOut) -> @location(0) vec4<f32> {
  let t = downsampleSettings.texelSize;
  let a = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>(-2.0,  2.0) * t).rgb));
  let b = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 0.0,  2.0) * t).rgb));
  let c = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 2.0,  2.0) * t).rgb));
  let d = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>(-2.0,  0.0) * t).rgb));
  let e = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv).rgb));
  let f = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 2.0,  0.0) * t).rgb));
  let g = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>(-2.0, -2.0) * t).rgb));
  let h = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 0.0, -2.0) * t).rgb));
  let i = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 2.0, -2.0) * t).rgb));
  let j = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>(-1.0,  1.0) * t).rgb));
  let k = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 1.0,  1.0) * t).rgb));
  let l = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>(-1.0, -1.0) * t).rgb));
  let m = karis(prefilter(textureSample(sourceTexture, linearSampler, in.uv + vec2<f32>( 1.0, -1.0) * t).rgb));
  let result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  return vec4<f32>(result, 1.0);
}

@fragment
fn upsampleFragment(in: VertexOut) -> @location(0) vec4<f32> {
  let t = upsampleSettings.texelSize * upsampleSettings.radius;
  let low = (
      textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>(-t.x,  t.y)).rgb
    + textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>( t.x,  t.y)).rgb
    + textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>(-t.x, -t.y)).rgb
    + textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>( t.x, -t.y)).rgb
    + (textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>(-t.x, 0.0)).rgb
      + textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>( t.x, 0.0)).rgb
      + textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>(0.0, -t.y)).rgb
      + textureSample(coarseTexture, upsampleSampler, in.uv + vec2<f32>(0.0,  t.y)).rgb) * 2.0
    + textureSample(coarseTexture, upsampleSampler, in.uv).rgb * 4.0
  ) / 16.0;
  let detail = textureSample(detailTexture, upsampleSampler, in.uv).rgb;
  return vec4<f32>(detail + low * upsampleSettings.scatter, 1.0);
}

@fragment
fn compositeFragment(in: VertexOut) -> @location(0) vec4<f32> {
  let scene = textureSample(sceneTexture, compositeSampler, in.uv).rgb;
  let bloom = textureSample(bloomTexture, compositeSampler, in.uv).rgb;
  return vec4<f32>(scene + bloom * compositeSettings.intensity * compositeSettings.bloomEnabled, 1.0);
}

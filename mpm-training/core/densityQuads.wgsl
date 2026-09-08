// Gaussian particle deposition into a floating-point additive render target.
// Nine periodic images preserve the compute splat's wrapped boundary condition.
const FIELD_N: u32 = __FIELD_N__u;
struct ParticleRest {
  growthF: vec4<f32>, jp: f32, growthVectorX: f32, growthVectorY: f32,
  verticesAB: vec4<f32>, vertexC: vec2<f32>, originalArea: f32, quadratureWeight: f32,
}
struct SplatParams { sigma: f32, }
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> rest: array<ParticleRest>;
@group(0) @binding(2) var<uniform> params: SplatParams;
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) center: vec2<f32>,
  @location(1) @interpolate(flat) area: f32,
}
@vertex fn vertexMain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
  let pi = instance / 9u;
  let image = instance % 9u;
  let n = f32(FIELD_N);
  let shift = vec2<f32>(f32(image % 3u) - 1.0, f32(image / 3u) - 1.0) * n;
  let center = fract(positions[pi]) * n + shift;
  let radius = min(ceil(3.0 * max(params.sigma * n, 1e-3)), 5.0);
  let low = floor(center) - vec2<f32>(radius);
  let high = floor(center) + vec2<f32>(radius + 1.0);
  let corners = array<vec2<f32>, 6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),
                                    vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
  let pixel = mix(low, high, corners[vertex]);
  var out: VertexOutput;
  out.position = vec4<f32>(pixel.x / n * 2.0 - 1.0, 1.0 - pixel.y / n * 2.0, 0.0, 1.0);
  out.center = center;
  let g = rest[pi].growthF;
  out.area = max(rest[pi].quadratureWeight, 1e-6) * max(abs(g.x*g.w-g.y*g.z), 1e-6);
  return out;
}
@fragment fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let delta = in.center - in.position.xy;
  let sigma = max(params.sigma * f32(FIELD_N), 1e-3);
  let weight = in.area * exp(-dot(delta, delta) / (2.0*sigma*sigma));
  // Match the compute path's per-contribution fixed-point rounding.
  return vec4<f32>(round(weight * 65536.0) / 65536.0, 0.0, 0.0, 0.0);
}

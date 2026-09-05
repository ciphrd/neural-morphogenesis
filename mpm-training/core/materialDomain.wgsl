// Shared domain quadrature, injected by both shader template loaders.
// Three-point Gauss-Legendre per material coordinate. Zero domains retain
// point transfers for externally supplied legacy test scenes.
fn domainQuadratureCount(h: vec4<f32>) -> u32 {
  return select(9u, 1u, dot(h, h) == 0.0);
}

fn domainQuadrature(h: vec4<f32>, k: u32) -> vec3<f32> {
  if (dot(h, h) == 0.0) { return vec3<f32>(0.0, 0.0, 1.0); }
  let nodes = array<f32, 3>(-0.7745966692414834, 0.0, 0.7745966692414834);
  let weights = array<f32, 3>(0.2777777777777778, 0.4444444444444444, 0.2777777777777778);
  let a = k / 3u;
  let b = k % 3u;
  let xi = vec2<f32>(nodes[a], nodes[b]);
  return vec3<f32>(h.x * xi.x + h.y * xi.y, h.z * xi.x + h.w * xi.y, weights[a] * weights[b]);
}

fn domainMoment(h: vec4<f32>, dx: f32) -> vec4<f32> {
  let kernel = 0.25 * dx * dx;
  let xy = (h.x * h.z + h.y * h.w) / 3.0;
  return vec4<f32>(kernel + (h.x*h.x + h.y*h.y)/3.0, xy,
                   xy, kernel + (h.z*h.z + h.w*h.w)/3.0);
}

fn domainBasisGradient(fx: vec2<f32>, i: u32, j: u32, invDx: f32) -> vec2<f32> {
  let weights = array<vec2<f32>, 3>(0.5 * (1.5-fx)*(1.5-fx),
    vec2<f32>(0.75)-(fx-1.0)*(fx-1.0), 0.5*(fx-0.5)*(fx-0.5));
  let derivatives = array<vec2<f32>, 3>(fx-1.5, -2.0*(fx-1.0), fx-0.5);
  return invDx * vec2<f32>(derivatives[i].x * weights[j].y, weights[i].x * derivatives[j].y);
}

// Seven-point, degree-five triangle quadrature. Weights sum to one, so
// splitting a domain preserves integrated constant commands. Periodic edges
// are unwrapped before interpolation, never by averaging canonical vertices.
const GROWTH_QUADRATURE_COUNT: u32 = 7u;
const GROWTH_QUADRATURE = array<vec3<f32>, 7>(
  vec3<f32>(0.333333333333, 0.333333333333, 0.225),
  vec3<f32>(0.470142064105, 0.470142064105, 0.132394152789),
  vec3<f32>(0.059715871790, 0.470142064105, 0.132394152789),
  vec3<f32>(0.470142064105, 0.059715871790, 0.132394152789),
  vec3<f32>(0.101286507323, 0.101286507323, 0.125939180544),
  vec3<f32>(0.797426985353, 0.101286507323, 0.125939180544),
  vec3<f32>(0.101286507323, 0.797426985353, 0.125939180544),
);

fn growthQuadraturePosition(rest: ParticleRest, qi: u32) -> vec2<f32> {
  let a = rest.verticesAB.xy;
  let ab = rest.verticesAB.zw - a;
  let ac = rest.vertexC - a;
  let uv = GROWTH_QUADRATURE[qi].xy;
  return fract(a + uv.x * (ab - floor(ab + vec2<f32>(0.5)))
                 + uv.y * (ac - floor(ac + vec2<f32>(0.5))));
}

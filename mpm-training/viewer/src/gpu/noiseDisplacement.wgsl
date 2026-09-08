// Viewer-only animated simplex-noise displacement. This pass runs after a
// completed simulation step and before rendering, so it changes the visible
// and subsequent physical particle positions without affecting training.

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  growthVectorX: f32,
  growthVectorY: f32,
  verticesAB: vec4<f32>,
  vertexC: vec2<f32>,
  originalArea: f32,
  quadratureWeight: f32,
}
@group(0) @binding(3) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(4) var<storage, read_write> particleF: array<vec4<f32>>;

struct NoiseParams {
  strength: f32,
  time: f32,
  spatialScale: f32,
  padding: f32,
}
@group(0) @binding(2) var<uniform> params: NoiseParams;

// Noise-space units per real second. Keep this deliberately slow: the field
// should read as a continuous, drifting flow during a performance rather than
// as rapidly changing jitter. At this rate a major feature takes roughly
// 25 seconds to travel by one noise-space unit.
const TEMPORAL_SPEED: f32 = 0.04;

fn mod289_2(value: vec2<f32>) -> vec2<f32> {
  return value - floor(value * (1.0 / 289.0)) * 289.0;
}

fn mod289_3(value: vec3<f32>) -> vec3<f32> {
  return value - floor(value * (1.0 / 289.0)) * 289.0;
}

fn permute(value: vec3<f32>) -> vec3<f32> {
  return mod289_3(((value * 34.0) + 1.0) * value);
}

// Ashima-style 2D simplex noise, expressed directly in WGSL.
fn simplexNoise(point: vec2<f32>) -> f32 {
  let constants = vec4<f32>(
    0.211324865405187,
    0.366025403784439,
    -0.577350269189626,
    0.024390243902439,
  );
  var cell = floor(point + dot(point, constants.yy));
  let origin = point - cell + dot(cell, constants.xx);
  var corner = vec2<f32>(0.0, 1.0);
  if (origin.x > origin.y) {
    corner = vec2<f32>(1.0, 0.0);
  }
  var offsets = origin.xyxy + constants.xxzz;
  offsets.x = offsets.x - corner.x;
  offsets.y = offsets.y - corner.y;
  cell = mod289_2(cell);
  let permutation = permute(
    permute(cell.y + vec3<f32>(0.0, corner.y, 1.0))
      + cell.x + vec3<f32>(0.0, corner.x, 1.0),
  );
  var attenuation = max(
    0.5 - vec3<f32>(
      dot(origin, origin),
      dot(offsets.xy, offsets.xy),
      dot(offsets.zw, offsets.zw),
    ),
    vec3<f32>(0.0),
  );
  attenuation = attenuation * attenuation;
  attenuation = attenuation * attenuation;
  let gradients = 2.0 * fract(permutation * constants.www) - 1.0;
  let heights = abs(gradients) - 0.5;
  let gradientCells = floor(gradients + 0.5);
  let gradientX = gradients - gradientCells;
  attenuation = attenuation * (
    1.79284291400159
      - 0.85373472095314 * (gradientX * gradientX + heights * heights)
  );
  let contribution = vec3<f32>(
    gradientX.x * origin.x + heights.x * origin.y,
    gradientX.y * offsets.x + heights.y * offsets.y,
    gradientX.z * offsets.z + heights.z * offsets.w,
  );
  return 130.0 * dot(attenuation, contribution);
}

fn displace(position: vec2<f32>) -> vec2<f32> {
  let domain = position * params.spatialScale;
  let animatedTime = params.time * TEMPORAL_SPEED;
  let first = simplexNoise(domain + vec2<f32>(animatedTime, -animatedTime * 0.65));
  let second = simplexNoise(domain + vec2<f32>(31.416, 17.903)
    + vec2<f32>(-animatedTime * 0.76, animatedTime * 1.12));
  return fract(position + vec2<f32>(first, second) * params.strength * 0.00075);
}

fn edge(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let delta = b - a;
  return delta - floor(delta + vec2<f32>(0.5));
}

@compute @workgroup_size(64)
fn displaceWithNoise(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= activeCount || params.strength <= 0.0) { return; }
  let rest = particleRest[i];
  if (rest.originalArea <= 0.0) {
    particlePos[i] = displace(particlePos[i]);
    return;
  }
  // Evaluate at shared vertex positions so neighboring domains stay joined.
  let a = displace(rest.verticesAB.xy);
  let b = displace(rest.verticesAB.zw);
  let c = displace(rest.vertexC);
  let oldU = edge(rest.verticesAB.xy, rest.verticesAB.zw);
  let oldV = edge(rest.verticesAB.xy, rest.vertexC);
  let newU = edge(a, b);
  let newV = edge(a, c);
  let determinant = oldU.x * oldV.y - oldV.x * oldU.y;
  if (abs(determinant) > 1e-12) {
    let inverseOld = mat2x2<f32>(vec2<f32>(oldV.y, -oldU.y), vec2<f32>(-oldV.x, oldU.x)) * (1.0 / determinant);
    let f = particleF[i];
    let nextF = mat2x2<f32>(newU, newV) * inverseOld * mat2x2<f32>(f.xz, f.yw);
    particleF[i] = vec4<f32>(nextF[0].x, nextF[1].x, nextF[0].y, nextF[1].y);
  }
  particleRest[i].verticesAB = vec4<f32>(a, b);
  particleRest[i].vertexC = c;
  particlePos[i] = fract(a + (newU + newV) / 3.0);
}

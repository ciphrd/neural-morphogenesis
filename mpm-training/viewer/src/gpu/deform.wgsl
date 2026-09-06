

const MODE_VELOCITY: u32 = 0u;
const MODE_DEFORMATION: u32 = 1u;

const VELOCITY_SCALE: f32 = 40.0;
const DEFORMATION_SCALE: f32 = 1.5;

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<uniform> clickPos: vec2<f32>;
@group(0) @binding(3) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> particleF: array<vec4<f32>>;

struct DeformParams {
  strength: f32,
  radius: f32,

  outward: f32,
  mode: f32,
}
@group(0) @binding(5) var<uniform> params: DeformParams;

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z,
    a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z,
    a.z * b.y + a.w * b.w
  );
}

fn identityPlusScaled(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(1.0 + s * m.x, s * m.y, s * m.z, 1.0 + s * m.w);
}

@compute @workgroup_size(64)
fn injectDeform(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  var delta = particlePos[pi] - clickPos;
  delta.x = delta.x - round(delta.x);
  delta.y = delta.y - round(delta.y);
  let dist = length(delta);
  if (dist > params.radius) { return; }

  let t = clamp(1.0 - dist / max(params.radius, 1e-6), 0.0, 1.0);
  let falloff = t * t * (3.0 - 2.0 * t);

  var dir = vec2<f32>(0.0, 0.0);
  if (dist > 1e-6) {
    dir = delta / dist;
  }
  if (params.outward < 0.5) {
    dir = -dir;
  }

  if (params.mode < 0.5) {
    particleVel[pi] = particleVel[pi] + dir * params.strength * VELOCITY_SCALE * falloff;
  } else {
    let stretch = identityPlusScaled(
      vec4<f32>(dir.x * dir.x, dir.x * dir.y, dir.y * dir.x, dir.y * dir.y),
      params.strength * DEFORMATION_SCALE * falloff
    );
    particleF[pi] = matMul(stretch, particleF[pi]);
  }
}

// Boundary-localized tissue cohesion derived from the simulation-owned
// morphology occupancy field. This pass nudges exposed particles toward
// increasing occupancy before the same substep's P2G transfer. Dense interior
// particles and isolated/self-symmetric particles receive little or no force.

const FIELD_N: u32 = __FIELD_N__u;
const DT: f32 = __DT__;

struct TissueTensionParams {
  strength: f32,
  // Hard cap on one physics substep's velocity delta. Kept independent from
  // strength so large experimental forces cannot violate the MPM stencil CFL.
  maxDelta: f32,
  _padding0: f32,
  _padding1: f32,
}

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> activeCount: u32;
@group(0) @binding(3) var morphologyTexture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: TissueTensionParams;

fn wrapFieldIndex(i: i32) -> i32 {
  let n = i32(FIELD_N);
  return ((i % n) + n) % n;
}

fn loadOccupancy(texel: vec2<i32>) -> f32 {
  let wrapped = vec2<i32>(wrapFieldIndex(texel.x), wrapFieldIndex(texel.y));
  return textureLoad(morphologyTexture, wrapped, 0).r;
}

// Manual bilinear filtering preserves smooth forces without requiring the
// optional float32-filterable WebGPU feature.
fn sampleOccupancy(domainPos: vec2<f32>) -> f32 {
  let texPos = domainPos * f32(FIELD_N) - vec2<f32>(0.5);
  let base = vec2<i32>(floor(texPos));
  let f = texPos - vec2<f32>(base);
  let d00 = loadOccupancy(base);
  let d10 = loadOccupancy(base + vec2<i32>(1, 0));
  let d01 = loadOccupancy(base + vec2<i32>(0, 1));
  let d11 = loadOccupancy(base + vec2<i32>(1, 1));
  return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

@compute @workgroup_size(64)
fn applyTissueTension(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount || params.strength <= 0.0 || params.maxDelta <= 0.0) {
    return;
  }

  let pos = particlePos[pi];
  let occupancy = clamp(sampleOccupancy(pos), 0.0, 1.0);
  // Zero in empty/dense regions and maximal at the occupancy transition.
  let boundaryGate = 4.0 * occupancy * (1.0 - occupancy);
  let eps = 1.0 / f32(FIELD_N);
  let dx = sampleOccupancy(pos + vec2<f32>(eps, 0.0))
    - sampleOccupancy(pos - vec2<f32>(eps, 0.0));
  let dy = sampleOccupancy(pos + vec2<f32>(0.0, eps))
    - sampleOccupancy(pos - vec2<f32>(0.0, eps));
  let gradient = vec2<f32>(dx, dy) / (2.0 * eps);
  let gradientLength = length(gradient);
  if (gradientLength <= 1e-7 || boundaryGate <= 0.0) {
    return;
  }

  // Positive morphology gradient points inward, toward denser tissue.
  // Soft normalization prevents tiny finite-difference/texture-rounding
  // gradients in a symmetric interior from becoming full-strength forces.
  let inward = gradient / max(gradientLength, 1.0);
  var delta = inward * (params.strength * boundaryGate * DT);
  let deltaLength = length(delta);
  if (deltaLength > params.maxDelta) {
    delta = delta * (params.maxDelta / deltaLength);
  }
  particleVel[pi] = particleVel[pi] + delta;
}

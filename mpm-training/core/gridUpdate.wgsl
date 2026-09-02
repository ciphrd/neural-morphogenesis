const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);
const DT: f32 = __DT__;
const SCALE: f32 = 4096.0;
const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CHANNELS: u32 = 3u;
@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> gridVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> gravity: f32;
@group(0) @binding(3) var<uniform> damping: f32;

@compute @workgroup_size(64)
fn gridUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NODE_COUNT) { return; }
  let base = idx * CHANNELS;
  let mass = f32(atomicLoad(&gridAccum[base + CH_MASS])) / SCALE;
  if (mass <= 0.0) {
    gridVel[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  let momX = f32(atomicLoad(&gridAccum[base + CH_MOM_X])) / SCALE;
  let momY = f32(atomicLoad(&gridAccum[base + CH_MOM_Y])) / SCALE;
  var v = (vec2<f32>(momX, momY) / mass) * damping;
  // `gravity` is a center-seeking acceleration magnitude. Grid nodes use the
  // same [0,1]^2 coordinates as particles, so applying the radial direction
  // here gives every particle a consistent pull toward the domain center via
  // the ordinary G2P gather. The epsilon avoids an undefined normalization at
  // the exact center node, where the desired acceleration is simply zero.
  let rowWidth = GRID_N + 1u;
  let node = vec2<f32>(f32(idx / rowWidth), f32(idx % rowWidth)) / f32(GRID_N);
  let towardCenter = vec2<f32>(0.5, 0.5) - node;
  let centerDistance = length(towardCenter);
  if (centerDistance > 1e-6) {
    v = v + DT * gravity * towardCenter / centerDistance;
  }
  gridVel[idx] = v;
}

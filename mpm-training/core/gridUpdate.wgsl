const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);
const DT: f32 = __DT__;
const SCALE: f32 = 4096.0;
const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CHANNELS: u32 = 4u;
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
  v.y = v.y - DT * gravity;
  gridVel[idx] = v;
}

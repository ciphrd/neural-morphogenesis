// Two-pass GPU max-abs reduction over the grid's first 3 channels, for
// render.py's deleted DisplayScale EMA auto-scale (per-channel
// max(abs(...)) tracked across frames so the background shade doesn't
// flicker every frame — see gpu/render.ts for the JS-side EMA that
// consumes this). __CHANNELS__/__WIDTH__/__HEIGHT__ substituted by
// shaderTemplate.ts.

const CHANNELS: u32 = __CHANNELS__u;
const WIDTH: u32 = __WIDTH__u;
const HEIGHT: u32 = __HEIGHT__u;
const PIXEL_COUNT: u32 = WIDTH * HEIGHT;
const WORKGROUP_SIZE: u32 = 256u;
const NUM_WORKGROUPS: u32 = (PIXEL_COUNT + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

fn gridIndex(c: u32, y: u32, x: u32) -> u32 {
  return c * HEIGHT * WIDTH + y * WIDTH + x;
}

// --- Pass A: per-workgroup partial max, one vec4 (xyz=channels, w unused) per workgroup ---

@group(0) @binding(0) var<storage, read> gridCurrent: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> shared_partial: array<vec3<f32>, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reducePartial(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let idx = gid.x;
  var v: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
  if (idx < PIXEL_COUNT) {
    let y = idx / WIDTH;
    let x = idx % WIDTH;
    v = vec3<f32>(
      abs(gridCurrent[gridIndex(0u, y, x)]),
      abs(gridCurrent[gridIndex(1u, y, x)]),
      abs(gridCurrent[gridIndex(2u, y, x)])
    );
  }
  shared_partial[lid.x] = v;
  workgroupBarrier();

  var stride: u32 = WORKGROUP_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      shared_partial[lid.x] = max(shared_partial[lid.x], shared_partial[lid.x + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    partials[wid.x] = vec4<f32>(shared_partial[0], 0.0);
  }
}

// --- Pass B: single workgroup reduces all partials down to one vec4 ---

@group(0) @binding(2) var<storage, read> partialsIn: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> finalOut: array<vec4<f32>>; // size 1

var<workgroup> shared_final: array<vec3<f32>, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn reduceFinal(@builtin(local_invocation_id) lid: vec3<u32>) {
  var v: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
  var i: u32 = lid.x;
  loop {
    if (i >= NUM_WORKGROUPS) { break; }
    v = max(v, partialsIn[i].xyz);
    i = i + WORKGROUP_SIZE;
  }
  shared_final[lid.x] = v;
  workgroupBarrier();

  var stride: u32 = WORKGROUP_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      shared_final[lid.x] = max(shared_final[lid.x], shared_final[lid.x + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    finalOut[0] = vec4<f32>(shared_final[0], 0.0);
  }
}

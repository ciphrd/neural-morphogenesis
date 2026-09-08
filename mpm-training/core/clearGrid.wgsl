

const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);
const CHANNELS: u32 = 3u;

@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NODE_COUNT) { return; }
  let base = idx * CHANNELS;
  for (var c: u32 = 0u; c < CHANNELS; c = c + 1u) {
    atomicStore(&gridAccum[base + c], 0);
  }
}

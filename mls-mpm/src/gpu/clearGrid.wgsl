// Zeroes p2g.wgsl's grid accumulator buffer — momentum-x, momentum-y,
// mass, plus the field-visualize diagnostics J-sum/shear-sum/pressure-sum
// (see that file's own header for what each holds and why they're all
// packed into one `array<atomic<i32>>` with a fixed per-node stride
// rather than one buffer per channel: WebGPU cores at 8 storage buffers
// per compute stage, and p2g.wgsl's 5 read-only particle buffers already
// leave room for only a few more — a combined buffer keeps this at 1
// binding regardless of how many per-node scalars it carries). Runs once
// per substep, before p2g.

const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);
const CHANNELS: u32 = 6u; // must match p2g.wgsl/gridUpdate.wgsl/field.wgsl's own CHANNELS

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

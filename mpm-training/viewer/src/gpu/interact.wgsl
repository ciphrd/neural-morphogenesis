

const GRAB_RADIUS: f32 = 0.06;

const UNGRABBED: vec2<f32> = vec2<f32>(1.0e6, 1.0e6);

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<uniform> pickPos: vec2<f32>;
@group(0) @binding(3) var<storage, read_write> grabOffset: array<vec2<f32>>;

@compute @workgroup_size(64)
fn beginGrab(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let offset = particlePos[pi] - pickPos;
  grabOffset[pi] = select(UNGRABBED, offset, length(offset) <= GRAB_RADIUS);
}

@group(0) @binding(4) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> dragTarget: vec2<f32>;

const DRAG_GAIN: f32 = 2000.0;

@compute @workgroup_size(64)
fn applyDrag(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let offset = grabOffset[pi];

  if (offset.x > 1.0) { return; }
  let targetPos = dragTarget + offset;
  let toTarget = targetPos - particlePos[pi];
  particleVel[pi] = toTarget * DRAG_GAIN;
}

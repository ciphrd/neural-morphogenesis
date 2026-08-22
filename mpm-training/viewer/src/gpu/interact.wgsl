// Viewer-only "Move Particles" tool — grabs *every* particle within
// GRAB_RADIUS of the click point (not just the nearest one) and drags
// the whole clump together as a rigid body: each grabbed particle keeps
// its own offset from the pointer for the rest of the gesture, so the
// clump translates without collapsing onto the cursor or onto each
// other. Lives outside ../core/ for the same reason
// fieldDiagnostics.wgsl does (see that file's own module docstring):
// this is UI-only interactivity the headless training path never needs
// to pay for.
//
// "Add Particle" (the other new tool) needs no WGSL at all — appending
// one particle is a handful of plain buffer writes, see mpmCore.ts's own
// addParticleAt().
//
// Two passes, each its own compute pass for the same pass-boundary-
// visibility reason every other multi-pass file in this project's own
// lineage already documents: beginGrab runs once, on pointerdown,
// recomputing EVERY particle's own grabOffset from scratch (so a
// particle grabbed by a previous gesture but out of range this time
// correctly reverts to "ungrabbed," no separate clear pass needed);
// applyDrag runs every animation frame afterward, for as long as the
// pointer stays down, reading the offsets beginGrab already resolved.
//
// Every binding below is declared read_write even where a given entry
// point only ever reads through it (particlePos, activeCount, grabOffset
// are all shared by both passes) — same "declared permissive, used
// narrowly" convention core/repulsion.wgsl's own particlePos binding
// already documents: WGSL bindings are declared once per module and
// shared by every entry point that references them, so the access mode
// has to cover the most permissive use any of them needs.

// Domain units — how far from the click point a particle can be and
// still get grabbed. A few particles' worth at this project's own
// typical density (hundreds, spread across a fraction of [0,1]^2), not
// tuned against any real data — same "starting guess" caveat this
// file's own MAX_DIST predecessor carried.
const GRAB_RADIUS: f32 = 0.06;
// Sentinel offset a real grabbed particle can never produce (bounded by
// GRAB_RADIUS, itself well under 1.0) — "this particle isn't part of
// the current grab," checked by applyDrag below via a cheap magnitude
// comparison rather than an exact float equality check.
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

// EXPERIMENT (see this tool's own git history / conversation): drives
// grabbed particles via VELOCITY instead of a direct position pin, to
// see whether routing the drag back through the normal P2G -> gridUpdate
// -> G2P momentum transfer produces any of mls-mpm's own "drag pulls
// neighbors along" cohesion — see gridUpdate.wgsl's own mass-weighted
// average for why that's expected to be weak-to-nonexistent (the
// grabbed particles' momentum gets diluted by however much stationary
// mass shares their grid nodes), NOT a kinematic override the way
// mls-mpm's own MODE_MOVE is. Kept as a plain proportional (spring-
// toward-target) velocity, not an exact-arrival calculation — robust
// regardless of substeps-per-macro, same reasoning attract.wgsl's own
// spring has, just instantaneous (`vel = toTarget * DRAG_GAIN`) rather
// than an additive per-substep nudge (this only runs once per rendered
// frame, not once per substep, so there's no accumulated-nudge history
// to add onto). SET, not added to, every call — still a "let go and it
// just falls" tool, not a throw.
// Empirically: this project's own DT (core/constants.json) is 0.000125,
// and dragTo() gets one shot at velocity per RENDERED FRAME, consumed by
// however many substeps run per macro step (--substeps-per-macro, as
// little as 1 on some configs) before the next dragTo() overwrites it
// again — so closing even a modest gap in one substep needs a gain on
// the order of 1/DT (~8000), not a "reasonable-looking" small number the
// way attract.wgsl's own additive per-substep nudge does.
const DRAG_GAIN: f32 = 2000.0;

@compute @workgroup_size(64)
fn applyDrag(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let offset = grabOffset[pi];
  // Cheap stand-in for "offset == UNGRABBED": any real grabbed offset's
  // components are bounded by +/-GRAB_RADIUS (<< 1.0), so a component
  // this large can only be the sentinel.
  if (offset.x > 1.0) { return; }
  let targetPos = dragTarget + offset;
  let toTarget = targetPos - particlePos[pi];
  particleVel[pi] = toTarget * DRAG_GAIN;
}

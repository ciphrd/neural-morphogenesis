// "Attract to Point" tool (tools/types.ts) — a 2-click gesture: click a
// particle to pick it (pickParticle + highlightPicked below), then click
// a position to set that as its target (commitAttractor); from then on
// applyAttraction pulls it there every substep, gently. Like repulsion.
// wgsl's own applyRepulsion, this writes directly to particlePos, not
// just particleVel — a PLAIN grid-routed velocity nudge was the first
// thing tried here and confirmed LIVE not to work: this pulls only ONE
// particle, so there's no opposing-sign cancellation the way repulsion
// has, but a different pathology hits just as hard in a dense/cohesive
// world (worlds/blocks.ts's own solid blocks) — P2G splats that one
// particle's extra momentum onto a grid node shared with dozens of
// stationary neighbors, and gridUpdate.wgsl's mass-weighted average
// dilutes it down to near-nothing before G2P ever hands it back. Read
// back live via a direct velocity-buffer probe: the nudge WAS being
// added every substep (a real, nonzero, stable value), it just never
// survived the grid round-trip enough to move the particle any visible
// distance in 10+ real seconds. The direct particlePos write below
// bypasses that round-trip the same way repulsion's does, guaranteeing
// the picked particle actually closes distance toward its target every
// substep regardless of how many neighbors it shares a grid cell with —
// see applyAttraction's own comment for the exact form (an exponential
// approach, not a fixed step, so it decelerates and settles smoothly
// near the target with no separate damping term needed). The velocity
// nudge is kept alongside it purely for downstream momentum/APIC-C
// consistency (same reasoning as repulsion.wgsl's own comment), not as
// the thing that actually moves the particle.
//
// Four passes, each its own compute pass (not chained within one) for
// the same pass-boundary-visibility reason clearGrid/p2g/gridUpdate/g2p
// are separate passes — see gpu/mpm.ts's own class comment. pickParticle
// and highlightPicked run together on a "pick a particle" click;
// commitAttractor runs alone on the following "pick a position" click
// (a separate user gesture, arbitrarily far apart in time — no same-
// command-buffer ordering concern between the two clicks, WebGPU's own
// queue already serializes separately-submitted command buffers);
// applyAttraction runs every substep, same cadence as gravity/mouse-
// force/repulsion.

const DT: f32 = __DT__;

// GPU-side nearest-particle search: rather than reading particle
// positions back to the CPU (this project has no readback path anywhere
// else, and mapAsync's latency would make picking feel laggy), each
// particle's own distance to the click point is packed together with its
// own index into ONE u32 key (distance in the high bits, index in the
// low bits) and atomicMin-reduced across all particles — the smallest
// key is necessarily the smallest distance (ties broken by smallest
// index), decoded back into "which particle" by the passes below. This
// only works because distance is non-negative: a non-negative f32's own
// IEEE-754 bit pattern, reinterpreted as an integer, already sorts the
// same way the float value does, so a coarser fixed-point requantization
// (not the float bits themselves) into DIST_BITS is enough precision for
// "which particle is nearest," which is all a UI pick needs.
const INDEX_BITS: u32 = 18u; // 2^18 = 262144 > MAX_PARTICLES headroom
const INDEX_MASK: u32 = (1u << INDEX_BITS) - 1u;
const DIST_BITS: u32 = 32u - INDEX_BITS;
const DIST_SCALE: f32 = f32((1u << DIST_BITS) - 1u);
// Domain diagonal is sqrt(2)~1.414 — the largest distance two points in
// [0,1]^2 can ever be apart; clamping to this before quantizing means
// every reachable distance gets a distinct, correctly-ordered bucket.
const MAX_DIST: f32 = 1.5;
// Sentinel a real key can never produce (quantized distance maxes out at
// DIST_SCALE, well under this) — "no particle has been picked yet /
// activeCount was 0," checked by highlightPicked/commitAttractor below.
const NO_PICK: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<uniform> pickPos: vec2<f32>;
@group(0) @binding(3) var<storage, read_write> pickResult: array<atomic<u32>>; // 1 element

@compute @workgroup_size(64)
fn pickParticle(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }
  let d = clamp(distance(particlePos[pi], pickPos), 0.0, MAX_DIST);
  let quantized = u32((d / MAX_DIST) * DIST_SCALE);
  let key = (quantized << INDEX_BITS) | (pi & INDEX_MASK);
  atomicMin(&pickResult[0], key);
}

// Highlights whichever particle pickParticle just resolved as nearest —
// visual confirmation of "this one" before the second (target) click.
// Single-thread dispatch: pickResult already holds the fully-reduced
// answer by the time this pass runs (separate compute pass, guaranteed
// visibility after pickParticle's own pass ends).
@group(0) @binding(4) var<storage, read_write> colors: array<vec4<f32>>;
// Bright white-yellow — distinct from every world's own particle palette
// (see worlds/*.ts's own hexToRgb-based colors), reads as "selected" at
// a glance regardless of what world/material color scheme is active.
const HIGHLIGHT_COLOR: vec4<f32> = vec4<f32>(1.0, 0.95, 0.3, 1.0);

@compute @workgroup_size(1)
fn highlightPicked() {
  let key = atomicLoad(&pickResult[0]);
  if (key == NO_PICK) { return; }
  let index = key & INDEX_MASK;
  colors[index] = HIGHLIGHT_COLOR;
}

// One committed particle-index/target pair — see gpu/mpm.ts's own
// MAX_ATTRACTORS for why this is a small fixed-capacity array (not a
// dynamically-growing one) and how attractorSlot/attractorCount below
// are managed host-side.
struct Attractor {
  particleIndex: u32,
  targetX: f32,
  targetY: f32,
}

@group(0) @binding(5) var<uniform> commitTarget: vec2<f32>;
@group(0) @binding(6) var<uniform> attractorSlot: u32;
@group(0) @binding(7) var<storage, read_write> attractors: array<Attractor>;

// Reads pickResult (still holding the LAST pickParticle click's own
// answer — nothing overwrites it between a "pick particle" click and the
// "pick position" click that follows, since only pickParticle itself
// ever writes it) and appends a new Attractor entry at attractorSlot
// (gpu/mpm.ts's own host-side attractorCount before this call — the one
// place this project tracks a GPU-resident list's own length on the CPU
// side rather than via an atomic counter, safe here because commits only
// ever happen one at a time, driven by discrete mouse clicks, never
// concurrently). No-ops (leaves that slot's old contents alone) if
// nothing was ever successfully picked.
@compute @workgroup_size(1)
fn commitAttractor() {
  let key = atomicLoad(&pickResult[0]);
  if (key == NO_PICK) { return; }
  let index = key & INDEX_MASK;
  attractors[attractorSlot].particleIndex = index;
  attractors[attractorSlot].targetX = commitTarget.x;
  attractors[attractorSlot].targetY = commitTarget.y;
}

// --- Per-substep pull toward each committed target ---

@group(0) @binding(0) var<storage, read> attractorsRead: array<Attractor>;
@group(0) @binding(1) var<uniform> attractorCount: u32;
@group(0) @binding(2) var<storage, read_write> particlePosForPull: array<vec2<f32>>;
// Same margin repulsion.wgsl's own applyRepulsion clamps particlePos to
// after its own direct write — keeps a dragged particle from ever being
// placed exactly on (or past) the domain boundary.
const POS_MARGIN: f32 = 0.02;
@group(0) @binding(3) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> activeCountForPull: u32;
@group(0) @binding(5) var<uniform> attractStrength: f32;

// Dispatched over attractorCount threads (typically a handful), NOT
// MAX_PARTICLES/activeCount — this list is small by construction (one
// entry per completed 2-click gesture), so a dedicated pass keyed by
// attractor index is far cheaper than scanning every particle for a
// per-particle flag.
@compute @workgroup_size(64)
fn applyAttraction(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= attractorCount) { return; }
  let a = attractorsRead[i];
  // Defensive — see g2p.wgsl's own MIN_POS/MAX_POS comment for this
  // project's general stance on real (if rare) safety nets: a world
  // reset between the pick and now would leave a stale particleIndex
  // pointing past the new scene's own (possibly smaller) activeCount.
  // gpu/mpm.ts's own loadScene() already clears attractorCount to 0 on
  // every reset specifically to prevent this, so this should never
  // actually engage — kept anyway, cheap insurance against an
  // out-of-bounds array access.
  if (a.particleIndex >= activeCountForPull) { return; }

  let pos = particlePosForPull[a.particleIndex];
  let targetPos = vec2<f32>(a.targetX, a.targetY);
  // Exponential approach, not a constant-speed thruster or an undamped
  // spring: each substep closes a small FRACTION of the remaining
  // distance (delta scales with toTarget itself), so the step size
  // shrinks continuously as the particle nears its target — decelerates
  // and settles smoothly on its own, no separate damping term needed the
  // way a velocity-only spring would (see this file's own header for why
  // a velocity-only version was tried first and confirmed live not to
  // move the particle at all in a dense/cohesive world). "Slowly
  // attracted," per this tool's own name, is controlled by how small
  // attractStrength is (main.ts's "Attraction strength" slider).
  let toTarget = targetPos - pos;
  let delta = toTarget * (attractStrength * DT);
  particleVel[a.particleIndex] = particleVel[a.particleIndex] + delta;
  particlePosForPull[a.particleIndex] = clamp(pos + delta, vec2<f32>(POS_MARGIN), vec2<f32>(1.0 - POS_MARGIN));
}

// --- Marker overlay — drawn by render.ts AFTER the main particle pass
// (see that file's own render() ordering) so a targeted particle stays
// visibly marked regardless of instance-draw order. Confirmed live this
// is a real, not hypothetical, need: highlightPicked's own color write
// above is genuinely correct (verified via a direct buffer readback) but
// was still frequently invisible on screen in a dense world — this
// project's particle draw call has no depth buffer, so it's a plain
// painter's-algorithm overdraw: any later-INDEXED neighbor sharing the
// same screen pixels simply draws over an earlier one, regardless of
// which one is "highlighted." Reads live positions every frame (not a
// fixed spot from when it was picked), so the ring actually tracks a
// drifting particle rather than marking where it used to be. ---

@group(0) @binding(0) var<storage, read> attractorsForMarker: array<Attractor>;
@group(0) @binding(1) var<storage, read> particlePosForMarker: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> markerRadius: f32; // NDC-space, same units as render.wgsl's own pointRadius

var<private> markerQuadOffsets: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct MarkerVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) offset: vec2<f32>,
};

@vertex
fn attractorMarkerVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> MarkerVOut {
  let particleIndex = attractorsForMarker[instanceIndex].particleIndex;
  // Same domain-[0,1]^2-to-NDC mapping as render.wgsl's own
  // particleVertex (no Y-flip — see that file's own comment for why).
  let center = particlePosForMarker[particleIndex] * 2.0 - vec2<f32>(1.0, 1.0);
  let offset = markerQuadOffsets[vertexIndex];
  var out: MarkerVOut;
  out.position = vec4<f32>(center + offset * markerRadius, 0.0, 1.0);
  out.offset = offset;
  return out;
}

@fragment
fn attractorMarkerFragment(in: MarkerVOut) -> @location(0) vec4<f32> {
  // A ring, not a filled disc — reads as "this one's targeted" without
  // hiding the particle (or its own real color) underneath it.
  let d2 = dot(in.offset, in.offset);
  if (d2 > 1.0 || d2 < 0.5) { discard; }
  return vec4<f32>(1.0, 0.95, 0.3, 1.0);
}

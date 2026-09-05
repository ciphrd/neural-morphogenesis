// Particle-particle repulsion via an inverse-distance-style density
// field — a cheap alternative to genuine O(particles^2) pairwise
// neighbor checks. Each particle deposits a Gaussian "splat" into a
// single-channel field (splatDensity, into densityAccum below, then
// baked into an r32float texture by densityToTexture); a particle then
// gets pushed away from wherever that field is locally highest by
// sampling the texture's own gradient (applyRepulsion) — a central
// finite difference of a BILINEARLY-FILTERED read of that field (see
// sampleDensityBilinear() below), not a raw per-texel one, so the
// resulting force varies smoothly as a particle moves WITHIN a texel,
// not just when it crosses one — this is what actually mattered for sim
// quality: nearest-texel sampling made the gradient (and so the push) a
// piecewise-constant step function, visible as a subtle per-texel jitter
// in how particles settle relative to each other. The filtering is
// manual (4 loadDensity() taps + mix()), not hardware textureSample —
// see loadDensity()'s own comment for why, and why that's also the
// right call for memory (no format change, no bigger texture).
//
// applyRepulsion writes ONLY to particleVel, not particlePos — this
// mechanism has gone through 3 real revisions, worth recording since the
// tradeoffs are genuinely subtle and easy to re-litigate by accident:
//
// 1. Direct particlePos write (original, ported from the sandbox).
//    Kinematic, not physical — bypasses the grid entirely, no momentum
//    conservation, no decay mechanism (nothing like gridUpdate.wgsl's
//    own Damping bleeds it off), and splatDensity's Gaussian never
//    reaches exactly zero, so the push never truly stopped, just got
//    smaller, compounding every substep.
// 2. Velocity-only, still a standalone pass running AFTER g2p (i.e.
//    feeding the NEXT substep's p2g). Routes through the normal
//    momentum-conserving grid transfer at last — but a per-particle
//    value that only reaches the grid via P2G's own scatter is subject
//    to gridUpdate.wgsl's own mass-weighted momentum average, which
//    CANCELS opposing per-particle contributions from particles sharing
//    a P2G stencil — exactly the case right after growth spawns a child
//    beside its own parent (core/agents.wgsl's own SPLIT_DISPLACEMENT).
// 3. THIS version: still velocity-only, but (a) sampled from THIS
//    substep's own density field at each particle's own exact position
//    (unchanged from revision 2 — see loadDensity() below, using
//    FIELD_N's own fine resolution, independent of the coarser physics
//    grid), and (b) dispatched BEFORE clearGrid/p2g/gridUpdate/g2p each
//    substep rather than after, so the push is routed through the SAME
//    substep's transfer immediately rather than sitting stale in
//    particleVel for one substep.
//
// A 4th revision was tried and reverted: moving the push into
// gridUpdate.wgsl itself as a per-GRID-NODE acceleration (alongside
// gravity), specifically to eliminate revision 2's cancellation problem
// by entering AFTER the mass-weighted average rather than before it.
// That does eliminate cancellation — but it also caps the push's
// effective SPATIAL resolution at the physics grid's own cell size
// (GRID_N=128, DX≈0.0078), coarser than the density field's own
// resolution (FIELD_N=256, texel≈0.0039) and, critically, coarser than
// core/agents.wgsl's own SPLIT_DISPLACEMENT (0.01): two particles that
// share the same 3x3 B-spline node stencil (any pair less than ~1
// GRID_N cell apart, which growth-spawned pairs always are) interpolate
// nearly IDENTICAL repulsion from that stencil and barely separate at
// all relative to each other, even though both get pushed. Confirmed
// empirically, not just in theory: two particles seeded 0.01 apart (
// SPLIT_DISPLACEMENT's own value) and run for 1500 substeps at a
// repulsion strength strong enough to make revision 3's own effect
// clearly visible stayed within 3% of their starting separation under
// the per-node design, while THIS revision (still not eliminating
// cancellation, only reducing it) reached 15-25% growth at the same
// strength over the same substep count, purely because it still samples
// the fine per-particle field rather than the coarse per-node one. In
// other words: the momentum-cancellation problem the per-node redesign
// set out to fix turned out to be the SMALLER of the two problems —
// losing sub-grid-cell spatial resolution entirely was worse in
// practice for the one thing this mechanism actually needs to do
// (separate freshly-spawned overlapping particles). Kept here as the
// record of why gridUpdate.wgsl does NOT carry a densityTex binding.
//
// Four passes total, each its own compute pass (not chained within one)
// for the same reason clearGrid/p2g/gridUpdate/g2p are separate passes:
// WebGPU doesn't guarantee one dispatch's writes are visible to the next
// dispatch *within* a single compute pass, only across pass boundaries.
//
// Independent copy of mls-mpm/src/gpu/repulsion.wgsl (this project's own
// sandbox), kept as a real, always-on part of core physics rather than
// sandbox-only — the sandbox's own history is explicit that this is
// meant to run every substep unconditionally (a variant that only ran it
// as a spawn-triggered burst was tried and reverted at the user's own
// request), and it matters directly for growth: it's what keeps
// particles from overlapping when new ones get placed right next to
// existing ones, which happens constantly under chemical-driven spawning.
// Only the display-only render pass (repulsionFieldVertex/Fragment, the
// sandbox's own "Field" dropdown visualization) is dropped — no headless
// trainer use for it, same reasoning that excludes field.wgsl from this
// core entirely.
//
// Deliberately its own field, NOT folded into the MPM grid
// (gridAccum/GRID_N in p2g.wgsl/gridUpdate.wgsl): this field's own
// resolution is independently configurable, whereas GRID_N is baked into
// DT via the elastic-wave CFL condition and can't be changed without
// re-deriving stability bounds. Keeping this fully separate also
// sidesteps a real bug hit earlier in the sandbox project: p2g's own
// compute stage once needed 11 storage buffers for a similar
// multi-channel scatter and blew WebGPU's 8-per-stage cap — every pass
// below is its OWN pipeline with its OWN small bind group, nowhere near
// that limit.
//
// ALWAYS ON, runs every substep. KNOWN LIMITATION: for particles closer
// together than roughly one grid cell (the growth-spawn case), the push
// is attenuated (not eliminated — see the revision-2-vs-3 comparison
// above) by gridUpdate.wgsl's own mass-weighted average; a fully
// cancellation-free fix would need a genuinely different transfer path
// for this one force (e.g. its own separate small grid, sized well
// below GRID_N, with its own P2G-style scatter), not attempted here.
// The current default strength/splat radius (see simulation_settings.py)
// are tuned empirically against this attenuated effective magnitude —
// expect to need noticeably higher REPULSION_STRENGTH values than the
// old direct-position-write revision ever needed for the same visible
// effect, since that revision's instantaneous kinematic correction had
// no grid-transfer attenuation at all.

const FIELD_N: u32 = __FIELD_N__u;
const TEXELS: u32 = FIELD_N * FIELD_N;
const DT: f32 = __DT__;
const CLEAR_WORKGROUP_SIZE: u32 = 64u;

// Fixed-point scale for the atomic scatter — same trick as p2g.wgsl's
// own SCALE (see that file's header). Splat weights are a Gaussian in
// [0,1] per tap, and MAX_KERNEL_RADIUS_TEXELS below bounds how many taps
// can land on one texel from a single particle's own splat (at most
// (2*MAX_KERNEL_RADIUS_TEXELS+1)^2), so headroom here only needs to
// survive that many particles' worth of near-1.0 contributions stacking
// on one texel — nowhere near the ~32768 this scale allows before i32
// overflow for any configuration this project's own sliders can reach.
const SCALE: f32 = 65536.0;

// Hard cap on the splat's own texel footprint, regardless of how large
// the splat radius or field resolution go — the whole point of a
// density-field approach over pairwise checks is O(particles *
// bounded_kernel), not O(particles^2); this is what keeps that bound
// genuinely constant. A large domain-space radius combined with a high
// field resolution WILL truncate the Gaussian's tail before its natural
// 3-sigma falloff once sigmaTexels exceeds roughly
// MAX_KERNEL_RADIUS_TEXELS/3 — clips beyond, doesn't fully resolve
// further, a deliberate bounded-cost tradeoff.
const MAX_KERNEL_RADIUS_TEXELS: i32 = 5;

@group(0) @binding(0) var<storage, read_write> densityAccum: array<atomic<i32>>;

@compute @workgroup_size(64)
fn clearDensity(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  let idx = gid.x + gid.y * workgroups.x * CLEAR_WORKGROUP_SIZE;
  if (idx >= TEXELS) { return; }
  atomicStore(&densityAccum[idx], 0);
}

// Live-adjustable, unlike FIELD_N above (a compile-time const sizing
// densityAccum/densityTexture — changing field resolution rebuilds this
// whole pipeline, not a live uniform write, since it changes buffer/
// texture SIZE) — a queue.writeBuffer, no pipeline recreation.
struct SplatParams {
  sigma: f32, // domain-space Gaussian sigma ([0,1] units, "splat radius")
  _padding0: f32,
  _padding1: f32,
  _padding2: f32,
}
@group(0) @binding(1) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> activeCount: u32;
@group(0) @binding(3) var<uniform> splatParams: SplatParams;

// Euclidean modulo, i32 in/out (textureLoad wants signed texel
// coordinates, unlike core/p2g.wgsl's/g2p.wgsl's own u32-returning
// wrapIndex() over grid nodes) — same wraparound idea, this field's own
// FIELD_N instead of the physics grid's GRID_N.
fn wrapFieldIndex(i: i32) -> i32 {
  let n = i32(FIELD_N);
  return ((i % n) + n) % n;
}

@compute @workgroup_size(64)
fn splatDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let texPos = pos * f32(FIELD_N); // continuous texel-space position
  let baseI = i32(floor(texPos.x));
  let baseJ = i32(floor(texPos.y));

  let sigmaTexels = max(splatParams.sigma * f32(FIELD_N), 1e-3);
  let sigma2 = sigmaTexels * sigmaTexels;
  // 3-sigma is where a Gaussian's contribution is already <1.1% of its
  // peak — truncating there (subject to MAX_KERNEL_RADIUS_TEXELS's own
  // hard cap, see that const's own comment) loses nothing visible.
  let kernelRadius = min(i32(ceil(3.0 * sigmaTexels)), MAX_KERNEL_RADIUS_TEXELS);

  // TOROIDAL: ti/tj (the *unwrapped* signed texel coords) still drive the
  // Gaussian falloff itself — texPos - texelCenter has to stay a genuine
  // continuous-space distance for the splat to look right across the
  // seam (a particle at the very edge splatting "through" it onto
  // texels on the opposite side, at their true short distance, not the
  // long way around) — only the *storage* index gets wrapped.
  for (var di = -kernelRadius; di <= kernelRadius; di = di + 1) {
    for (var dj = -kernelRadius; dj <= kernelRadius; dj = dj + 1) {
      let ti = baseI + di;
      let tj = baseJ + dj;

      let texelCenter = vec2<f32>(f32(ti) + 0.5, f32(tj) + 0.5);
      let delta = texPos - texelCenter;
      let d2 = dot(delta, delta);
      let idx = u32(wrapFieldIndex(ti)) * FIELD_N + u32(wrapFieldIndex(tj));
      let weight = exp(-d2 / (2.0 * sigma2));
      atomicAdd(&densityAccum[idx], i32(round(weight * SCALE)));
    }
  }
}

// --- densityAccum (fixed-point) -> densityTexture (r32float) ---
// No native float atomics on storage buffers, so the splat above has to
// go through a fixed-point buffer; this pass decodes it once into a real
// texture so applyRepulsion below can use ordinary texel lookups instead
// of manually re-deriving neighbor weights from raw buffer indices.

@group(0) @binding(1) var densityTex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16, 1)
fn densityToTexture(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= FIELD_N || j >= FIELD_N) { return; }
  let idx = i * FIELD_N + j;
  let value = f32(atomicLoad(&densityAccum[idx])) / SCALE;
  textureStore(densityTex, vec2<i32>(i32(i), i32(j)), vec4<f32>(value, 0.0, 0.0, 0.0));
}

// --- Gradient-descent repulsion: nudge each particle away from wherever
// the density field is locally highest, at that particle's own exact
// position (fine FIELD_N resolution — see this file's own module
// docstring for why that matters, and why this stays a per-particle
// pass rather than moving into gridUpdate.wgsl) ---

struct RepulsionParams {
  strength: f32,
  // Hard cap on the MAGNITUDE of one substep's own velocity delta
  // (`grad * strength * DT` below) — the missing piece that let
  // `strength` alone decide whether repulsion does anything or breaks
  // everything, with no usable middle: this project's own material
  // stiffness (trainer/simulation_settings.py's own MATERIAL_E) resists
  // repulsion's push continuously, every substep, from the very first
  // one (NOT something core/agents.wgsl's own growthJpRelief touches at
  // all — that only nudges Jp at discrete split events) — so `strength`
  // has to be pushed high enough to genuinely win against that
  // resistance to have any visible effect. But an UNCLAMPED delta at
  // that magnitude is exactly what produces a single-substep velocity
  // kick large enough to violate MLS-MPM's own implicit assumption that
  // a particle stays within its local 3x3 P2G/G2P stencil for one
  // substep — confirmed empirically earlier in this project's own
  // history: two particles seeded SPLIT_DISPLACEMENT apart, repulsion
  // strength >=100, blow up to NaN in a SINGLE substep, with zero growth
  // or elasticity involved at all. This clamp lets `strength` be turned
  // up as far as needed to beat elastic stiffness while keeping the
  // PER-SUBSTEP delta bounded to something MLS-MPM can actually resolve
  // — trainer/simulation_settings.py's own REPULSION_MAX_DELTA is the
  // starting value, live-tunable via PhysicsPanel same as strength
  // itself.
  maxDelta: f32,
  _padding: vec2<f32>,
}
@group(0) @binding(4) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(5) var densityTexSampled: texture_2d<f32>;
@group(0) @binding(7) var<uniform> repulsionParams: RepulsionParams;


// r32float textures are "unfilterable-float" by default in core WebGPU
// (no sampler/textureSample against them without the optional
// float32-filterable feature) — textureLoad below reads exact texels by
// integer coordinate instead, no sampler needed at all. Deliberately not
// requested even to unlock hardware filtering: this project's own
// gpu/device.ts requestDevice() call asks for exactly one requiredLimits
// bump and zero requiredFeatures, on principle (every optional WebGPU
// feature is a real chance of failing to acquire a device on some
// backend) — see sampleDensityBilinear() below for how filtering is done
// without needing this texture to become hardware-filterable at all.
fn loadDensity(texel: vec2<i32>) -> f32 {
  let wrapped = vec2<i32>(wrapFieldIndex(texel.x), wrapFieldIndex(texel.y));
  return textureLoad(densityTexSampled, wrapped, 0).r;
}

// Manual bilinear filtering — 4 loadDensity() taps + 2 mix()es — rather
// than a hardware-filtered textureSample(). Two independent reasons this
// beats switching densityTexture to a format that DOES filter natively
// (e.g. rgba16float): (1) no core-WebGPU single-channel format is both
// storage-texture-WRITABLE (needed by densityToTexture's own
// textureStore above) and natively filterable without an optional
// feature — r8unorm/r16float both need the "texture-formats-tier1"
// feature just to be storage-write targets, same "don't depend on
// optional features" reasoning as loadDensity()'s own comment; the only
// core-mandatory storage-writable single-channel format is exactly the
// r32float already in use. (2) even ignoring (1), the smallest format
// that WOULD hardware-filter without a feature is 4-channel (rgba16float
// or similar — no single/dual-channel core format is both storage-
// writable and filterable) which is strictly MORE memory than staying
// single-channel r32float and filtering by hand. So this keeps the exact
// same densityAccum/densityTexture byte footprint this file always had —
// the quality gain below is pure math, not a bigger buffer.
//
// Texel i's own sample point is at continuous coordinate i+0.5, matching
// splatDensity's own texelCenter convention above — hence the -0.5
// before flooring to find the 4 surrounding texel corners.
fn sampleDensityBilinear(domainPos: vec2<f32>) -> f32 {
  let texPos = domainPos * f32(FIELD_N) - vec2<f32>(0.5, 0.5);
  let base = vec2<i32>(floor(texPos));
  let f = texPos - vec2<f32>(base);
  let d00 = loadDensity(base);
  let d10 = loadDensity(base + vec2<i32>(1, 0));
  let d01 = loadDensity(base + vec2<i32>(0, 1));
  let d11 = loadDensity(base + vec2<i32>(1, 1));
  return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

@compute @workgroup_size(64)
fn applyRepulsion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  // 1 texel apart each side (2 texels total span, same as the old
  // texel-indexed version), in domain units — but now a central
  // difference of sampleDensityBilinear()'s own CONTINUOUS field rather
  // than loadDensity()'s texel-quantized one, so the gradient (and the
  // resulting push) varies smoothly as a particle moves within a texel,
  // not just when it crosses one.
  let eps = 1.0 / f32(FIELD_N);
  let dx = sampleDensityBilinear(pos + vec2<f32>(eps, 0.0)) - sampleDensityBilinear(pos - vec2<f32>(eps, 0.0));
  let dy = sampleDensityBilinear(pos + vec2<f32>(0.0, eps)) - sampleDensityBilinear(pos - vec2<f32>(0.0, eps));
  let grad = vec2<f32>(dx, dy) / (2.0 * eps);

  // Downhill on the density field, i.e. away from wherever local
  // crowding is highest — velocity only (see this file's own module
  // docstring for the 3-revision history of why, and the empirical
  // comparison against a per-node alternative). Dispatched BEFORE
  // clearGrid/p2g/gridUpdate/g2p each substep (see mpm_core.py's/
  // mpmCore.ts's own step()/encodeSteps() for the exact ordering), so
  // this reaches the grid through the SAME substep's own transfer,
  // subject to gridUpdate.wgsl's own Damping like any other velocity —
  // it decays on its own, no unbounded-drift risk the way the old
  // direct-position-write revision had.
  var delta = grad * (repulsionParams.strength * DT);
  // Clamp the delta's own MAGNITUDE, not each component independently —
  // preserves push DIRECTION exactly, only ever shrinks how far it goes
  // in one substep. See RepulsionParams.maxDelta's own comment for why
  // this exists at all.
  let deltaLen = length(delta);
  if (deltaLen > repulsionParams.maxDelta) {
    delta = delta * (repulsionParams.maxDelta / deltaLen);
  }
  particleVel[pi] = particleVel[pi] - delta;
}

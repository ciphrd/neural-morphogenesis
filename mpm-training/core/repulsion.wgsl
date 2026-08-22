// Particle-particle repulsion via an inverse-distance-style density
// field — a cheap alternative to genuine O(particles^2) pairwise
// neighbor checks. Each particle deposits a Gaussian "splat" into a
// single-channel field (splatDensity, into densityAccum below, then
// baked into an r32float texture by densityToTexture); a particle then
// gets pushed away from wherever that field is locally highest by
// sampling the texture's own gradient (applyRepulsion) — a central
// finite difference between the 2 neighboring texels each axis, using
// ordinary textureLoad rather than manually re-deriving Gaussian weights
// from the raw accumulator buffer. applyRepulsion writes that gradient
// directly to particlePos, not just particleVel — see that function's
// own comment for why a velocity-only nudge doesn't actually separate
// particles closer together than one MPM grid cell (gridUpdate.wgsl's
// own mass-weighted average cancels opposing velocities from particles
// sharing the same P2G stencil). Four passes, each its own compute pass
// (not chained within one) for the same reason clearGrid/p2g/gridUpdate/
// g2p are separate passes: WebGPU doesn't guarantee one dispatch's
// writes are visible to the next dispatch *within* a single compute
// pass, only across pass boundaries.
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
// ALWAYS ON, runs every substep. KNOWN LIMITATION, inherited as-is from
// the sandbox, not fixed by this extraction: applyRepulsion's direct
// position write has no decay mechanism the way a velocity nudge does
// (nothing like gridUpdate.wgsl's own Damping slider bleeds it off), and
// splatDensity's Gaussian never reaches exactly zero — so this push
// never truly stops, just gets smaller, and compounds every substep.
// Over enough substeps this can disperse even an already-settled world
// if pushed too hard — the default strength/splat radius (see
// constants.json) are tuned to keep that gentle at default settings, not
// to eliminate it (the actual fix would be a density threshold below
// which the force is exactly zero, not yet implemented, same as the
// sandbox).

const FIELD_N: u32 = __FIELD_N__u;
const TEXELS: u32 = FIELD_N * FIELD_N;
const DT: f32 = __DT__;

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
fn clearDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= TEXELS) { return; }
  atomicStore(&densityAccum[idx], 0);
}

// Live-adjustable, unlike FIELD_N above (a compile-time const sizing
// densityAccum/densityTexture — changing field resolution rebuilds this
// whole pipeline, not a live uniform write, since it changes buffer/
// texture SIZE) — a queue.writeBuffer, no pipeline recreation.
struct SplatParams {
  sigma: f32, // domain-space Gaussian sigma ([0,1] units, "splat radius")
}
// read_write, not read — splatDensity itself never writes through this
// (see below), but applyRepulsion's own bind group (which reuses this
// same declaration, per-pipeline bind groups being inferred
// independently regardless of shared WGSL variable names) needs write
// access, for reasons that function's own comment explains.
@group(0) @binding(1) var<storage, read_write> particlePos: array<vec2<f32>>;
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
      let weight = exp(-d2 / (2.0 * sigma2));

      let idx = u32(wrapFieldIndex(ti)) * FIELD_N + u32(wrapFieldIndex(tj));
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
// the density field is locally highest ---

struct RepulsionParams {
  strength: f32,
}
@group(0) @binding(4) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(5) var densityTexSampled: texture_2d<f32>;
@group(0) @binding(7) var<uniform> repulsionParams: RepulsionParams;

// r32float textures are "unfilterable-float" by default in core WebGPU
// (no sampler/textureSample against them without the optional
// float32-filterable feature) — textureLoad below reads exact texels by
// integer coordinate instead, no sampler needed at all.
const FIELD_N_I: i32 = i32(FIELD_N);

fn loadDensity(texel: vec2<i32>) -> f32 {
  let wrapped = vec2<i32>(wrapFieldIndex(texel.x), wrapFieldIndex(texel.y));
  return textureLoad(densityTexSampled, wrapped, 0).r;
}

@compute @workgroup_size(64)
fn applyRepulsion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let texel = vec2<i32>(pos * f32(FIELD_N));
  let dx = loadDensity(texel + vec2<i32>(1, 0)) - loadDensity(texel - vec2<i32>(1, 0));
  let dy = loadDensity(texel + vec2<i32>(0, 1)) - loadDensity(texel - vec2<i32>(0, 1));
  // 2 texels apart (i32(1)-i32(-1)) in domain units.
  let grad = vec2<f32>(dx, dy) / (2.0 / f32(FIELD_N));

  // Downhill on the density field, i.e. away from wherever local
  // crowding is highest. Written to BOTH particlePos and particleVel,
  // not just velocity — a velocity-only nudge here doesn't work: two
  // particles closer together than one MPM grid cell get nearly
  // identical P2G interpolation weights, so if repulsion gives them
  // opposing velocities, their momentum contributions land on the same
  // grid nodes and largely cancel in gridUpdate.wgsl's mass-weighted
  // average — g2p.wgsl then hands both particles back the same
  // (near-zero, averaged-away) velocity, erasing the separation in the
  // very substep it was computed. Writing directly to particlePos
  // bypasses the grid round-trip entirely for the positional effect.
  // This makes repulsion a kinematic position correction layered on top
  // of MPM, NOT a proper physical force the way gravity is — it doesn't
  // conserve momentum the way a grid-mediated force does. Acceptable
  // tradeoff for "don't let particles overlap," not a general substitute
  // for gravity's own grid-routed approach.
  //
  // KNOWN LIMITATION, inherited as-is from the sandbox: a direct
  // position write has no decay mechanism the way a velocity nudge does,
  // and splatDensity's Gaussian never reaches exactly zero — so this
  // push never truly stops either, just gets smaller, and compounds
  // every substep. Default strength/splat radius are picked to keep
  // that dispersal slow/gentle at default settings, not to stop it.
  let delta = grad * (repulsionParams.strength * DT);
  particleVel[pi] = particleVel[pi] - delta;
  // Toroidal: wrap into [0,1) rather than clamp against a wall — see
  // g2p.wgsl's own fract()-based position update for the same reasoning.
  // No margin needed either (this file's own former POS_MARGIN existed
  // solely to keep a particle inside P2G's valid stencil range for one
  // more substep — moot now that stencil indexing wraps unconditionally,
  // see p2g.wgsl's own wrapIndex()).
  particlePos[pi] = fract(pos - delta);
}

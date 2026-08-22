// Particle-particle repulsion via an inverse-distance-style density
// field — a cheap alternative to genuine O(particles^2) pairwise
// neighbor checks. Each particle deposits a Gaussian "splat" into a
// single-channel field (splatDensity, into densityAccum below, then
// baked into an r32float texture by densityToTexture); a particle then
// gets pushed away from wherever that field is locally highest by
// sampling the texture's own gradient (applyRepulsion) — a central
// finite difference between the 2 neighboring texels each axis, using
// ordinary textureLoad (see that function's own comment for why not
// filtered sampling) rather than manually re-deriving Gaussian weights
// from the raw accumulator buffer. applyRepulsion writes that gradient
// directly to particlePos, not just particleVel — see its own comment
// for why a velocity-only nudge doesn't actually separate particles
// closer together than one MPM grid cell (gridUpdate.wgsl's own mass-
// weighted average cancels opposing velocities from particles sharing
// the same P2G stencil). Four passes, each its own compute
// pass (not chained within one) for the same reason clearGrid/p2g/
// gridUpdate/g2p are separate passes — see gpu/mpm.ts's own class
// comment: WebGPU doesn't guarantee one dispatch's writes are visible to
// the next dispatch *within* a single compute pass, only across pass
// boundaries.
//
// Deliberately its own field, NOT folded into the MPM grid
// (gridAccum/GRID_N in p2g.wgsl/gridUpdate.wgsl): this field's own
// resolution is a live, independently-adjustable control (main.ts's
// "Field resolution"), whereas GRID_N is baked into DT via the elastic-
// wave CFL condition (see gpu/mpm.ts's own DT comment) and can't be
// changed without re-deriving stability bounds. Keeping this fully
// separate also sidesteps a real bug hit earlier in this project: p2g's
// own compute stage once needed 11 storage buffers for a similar
// multi-channel scatter and blew WebGPU's 8-per-stage cap (see that
// file's own gridAccum comment) — every pass below is its OWN pipeline
// with its OWN small bind group, nowhere near that limit.
//
// ALWAYS ON, runs every substep (see gpu/mpm.ts's own step()) — a
// variant that instead only ran this as a bounded burst triggered by
// MpmSimulation.addParticles() was tried and reverted at the user's own
// request (they want repulsion continuously active, not just a spawn-
// time declump pass). That variant existed specifically to route around
// this file's own known limitation, documented below because it's still
// real and still applies: a direct position write has no decay
// mechanism the way a velocity nudge does (nothing like gridUpdate.wgsl's
// own Damping slider bleeds it off), and splatDensity's Gaussian never
// reaches exactly zero — so this push never truly stops either, just
// gets smaller, and compounds every substep. Over enough substeps this
// can disperse even an already-settled world if pushed too hard —
// gpu/mpm.ts's DEFAULT_REPULSION_STRENGTH is tuned live to keep that
// gentle at default settings, not to eliminate it (the actual fix would
// be a density threshold below which the force is exactly zero, not yet
// implemented).

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
// overflow for any configuration this project's UI can reach.
const SCALE: f32 = 65536.0;

// Hard cap on the splat's own texel footprint, regardless of how large
// main.ts's "Splat radius" slider or "Field resolution" go — the whole
// point of a density-field approach over pairwise checks is
// O(particles * bounded_kernel), not O(particles^2); this is what keeps
// that bound genuinely constant. A large domain-space radius combined
// with a high field resolution WILL truncate the Gaussian's tail before
// its natural 3-sigma falloff once sigmaTexels exceeds roughly
// MAX_KERNEL_RADIUS_TEXELS/3 — same "clips beyond, doesn't fully resolve
// further" tradeoff field.wgsl's own DENSITY_MAX/SPEED_MAX already make,
// just here bounding compute cost directly rather than just display
// contrast.
const MAX_KERNEL_RADIUS_TEXELS: i32 = 5;

@group(0) @binding(0) var<storage, read_write> densityAccum: array<atomic<i32>>;

@compute @workgroup_size(64)
fn clearDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= TEXELS) { return; }
  atomicStore(&densityAccum[idx], 0);
}

// Live-adjustable, unlike FIELD_N above (a compile-time const sizing
// densityAccum/densityTexture — changing main.ts's "Field resolution"
// control rebuilds this whole pipeline, not a live uniform write, since
// it changes buffer/texture SIZE) — a queue.writeBuffer, no pipeline
// recreation.
struct SplatParams {
  sigma: f32, // domain-space Gaussian sigma ([0,1] units, main.ts's "Splat radius")
}
// read_write, not read — splatDensity itself never writes through this
// (see below), but applyRepulsion's own bind group (which reuses this
// same declaration, per-pipeline bind groups being inferred
// independently regardless of shared WGSL variable names) needs write
// access, for reasons that function's own comment explains.
@group(0) @binding(1) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> activeCount: u32;
@group(0) @binding(3) var<uniform> splatParams: SplatParams;

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

  for (var di = -kernelRadius; di <= kernelRadius; di = di + 1) {
    for (var dj = -kernelRadius; dj <= kernelRadius; dj = dj + 1) {
      let ti = baseI + di;
      let tj = baseJ + dj;
      if (ti < 0 || tj < 0 || ti >= i32(FIELD_N) || tj >= i32(FIELD_N)) { continue; }

      let texelCenter = vec2<f32>(f32(ti) + 0.5, f32(tj) + 0.5);
      let delta = texPos - texelCenter;
      let d2 = dot(delta, delta);
      let weight = exp(-d2 / (2.0 * sigma2));

      let idx = u32(ti) * FIELD_N + u32(tj);
      atomicAdd(&densityAccum[idx], i32(round(weight * SCALE)));
    }
  }
}

// --- densityAccum (fixed-point) -> densityTexture (r32float) ---
// Same split as field.wgsl's own colorizeField: no native float atomics
// on storage buffers, so the splat above has to go through a fixed-point
// buffer; this pass decodes it once into a real texture so
// applyRepulsion below (and main.ts's own "Distance field" display mode)
// can use ordinary texel lookups instead of manually re-deriving
// neighbor weights from raw buffer indices.

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
// (no `sampler`/textureSample against them without the optional
// float32-filterable feature, confirmed live as a real bind-group
// validation error, not a hypothetical) — textureLoad below reads exact
// texels by integer coordinate instead, no sampler needed at all. Cheap
// either way (this was never meant to lean on filtering hardware for
// smoothing, just to avoid a manual multi-tap blur), and keeps
// densityTexture genuinely single-channel r32float rather than needing a
// 4-channel filterable format as a workaround.
const FIELD_N_I: i32 = i32(FIELD_N);

// Defensive-only, same spirit as g2p.wgsl's own MIN_POS/MAX_POS (see
// that file's own comment on why a stranded-outside-the-domain particle
// is a real failure mode, not a hypothetical): applyRepulsion's own
// direct particlePos write below runs AFTER g2p.wgsl's own boundary
// clamp this substep, and BEFORE it again next substep (P2G runs first)
// — so an unclamped write here could hand P2G a position outside its
// valid stencil range for a full substep, silently dropping that
// particle's contribution everywhere (P2G's own bounds check just skips
// it, contributing to nothing). A fixed margin, not derived from GRID_N
// (this file doesn't otherwise know it — deliberately decoupled, see
// this file's own header), comfortably larger than this project's own
// GRID_N=64's DX (~0.0156) for headroom.
const POS_MARGIN: f32 = 0.02;

fn loadDensity(texel: vec2<i32>) -> f32 {
  let clamped = clamp(texel, vec2<i32>(0), vec2<i32>(FIELD_N_I - 1));
  return textureLoad(densityTexSampled, clamped, 0).r;
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
  // not just velocity — see this project's own discussion of why a
  // velocity-only nudge here turned out not to work: two particles
  // closer together than one MPM grid cell get nearly identical P2G
  // interpolation weights, so if repulsion gives them opposing
  // velocities, their momentum contributions land on the same grid nodes
  // and largely cancel in gridUpdate.wgsl's mass-weighted average —
  // g2p.wgsl then hands both particles back the same (near-zero,
  // averaged-away) velocity, erasing the separation in the very substep
  // it was computed. Writing directly to particlePos bypasses the grid
  // round-trip entirely for the positional effect: this displacement
  // survives regardless of what P2G/gridUpdate/G2P do with the
  // (still-updated, for downstream momentum/APIC-C consistency)
  // velocity. This makes repulsion a kinematic position correction
  // layered on top of MPM, NOT a proper physical force the way gravity/
  // mouse-force are (see gridUpdate.wgsl's own MODE_FORCE/MODE_MOVE
  // comment for that distinction) — it doesn't conserve momentum the way
  // a grid-mediated force does, a directly-moved particle didn't "earn"
  // its displacement through any resolved velocity. Acceptable tradeoff
  // for "don't let particles overlap," not represented as a general
  // substitute for gravity/mouse-force's own grid-routed approach.
  //
  // KNOWN LIMITATION, confirmed live, not yet fixed: a direct position
  // write has no decay mechanism the way a velocity nudge does (nothing
  // like gridUpdate.wgsl's own Damping slider bleeds it off), and
  // splatDensity's Gaussian never reaches exactly zero — so this push
  // never truly stops either, just gets smaller, and compounds every
  // substep. Over enough substeps this reliably disperses even an
  // already-settled world if pushed too hard — gpu/mpm.ts's
  // DEFAULT_REPULSION_STRENGTH/DEFAULT_SPLAT_RADIUS are picked to keep
  // that dispersal slow/gentle at default settings, not to stop it — the
  // actual fix is a density threshold below which `delta` is exactly
  // zero (e.g. multiply by smoothstep(0.0, THRESHOLD, localDensity)),
  // giving the force a real finite-support cutoff instead of an
  // asymptotic one.
  let delta = grad * (repulsionParams.strength * DT);
  particleVel[pi] = particleVel[pi] - delta;
  particlePos[pi] = clamp(pos - delta, vec2<f32>(POS_MARGIN), vec2<f32>(1.0 - POS_MARGIN));
}

// --- Distance-field display (main.ts's Field dropdown) — a full-screen
// quad sampling densityTexture directly, own copy of field.wgsl's own
// fieldVertex/fieldFragment pattern (WGSL has no #include, same
// duplication tradeoff documented elsewhere in this project, e.g.
// g2p.wgsl's own polarDecompose comment) since this field lives in its
// own, independently-sized texture rather than field.wgsl's shared
// GRID_N+1-resolution one. ---

var<private> quadPositions: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct RepulsionFieldVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var displayTex: texture_2d<f32>;

@vertex
fn repulsionFieldVertex(@builtin(vertex_index) vertexIndex: u32) -> RepulsionFieldVOut {
  let p = quadPositions[vertexIndex];
  var out: RepulsionFieldVOut;
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = (p + vec2<f32>(1.0)) * 0.5;
  return out;
}

// Reference's own background color, 0x112F41 — see field.wgsl's own BG
// comment for why every field ramp fades up from this same floor.
const BG: vec3<f32> = vec3<f32>(17.0 / 255.0, 47.0 / 255.0, 65.0 / 255.0);
// Normalization ceiling, picked by eye — raw density units (a sum of
// overlapping per-particle Gaussian weights, roughly "how many
// particle-radii of crowding overlap here"), not physical units. Same
// "clips beyond, doesn't need to be exact" tradeoff as field.wgsl's own
// DENSITY_MAX.
const DISPLAY_MAX: f32 = 3.0;

@fragment
fn repulsionFieldFragment(in: RepulsionFieldVOut) -> @location(0) vec4<f32> {
  // Nearest-texel lookup, not filtered sampling — see loadDensity's own
  // comment for why (r32float's default unfilterable-float sample
  // type). No visible cost at this project's own field resolutions (256
  // and up).
  let texel = clamp(vec2<i32>(in.uv * f32(FIELD_N)), vec2<i32>(0), vec2<i32>(FIELD_N_I - 1));
  let density = textureLoad(displayTex, texel, 0).r;
  let t = clamp(density / DISPLAY_MAX, 0.0, 1.0);
  // Magenta — distinct from every other field's own hue (density=cyan,
  // speed=orange, shear=green, deformation=red/blue, pressure=amber/
  // violet), so it reads unambiguously as its own thing at a glance.
  let color = mix(BG, vec3<f32>(1.0, 0.25, 0.75), t);
  return vec4<f32>(color, 1.0);
}

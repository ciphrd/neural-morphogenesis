// Visualizes one of the grid's own fields (mass/density, velocity
// magnitude) as a background heatmap behind the particles — the actual
// MPM state that drives everything else, but otherwise never rendered
// at all (gridMass/gridVel are pure scratch, overwritten every substep).
// Two stages: a compute pass turning the current grid state into an
// (GRID_N+1)^2 RGBA texture (colorizeField), and a full-screen quad
// sampling it (fieldVertex/fieldFragment) — same split responsibility as
// sibling project envnca's colorize.wgsl + present.wgsl, scaled down
// (fixed normalization constants below rather than an EMA-tracked
// display scale — this project's fields don't swing over the orders-of-
// magnitude range envnca's chemical substrate does).
//
// No Y-flip anywhere here, deliberately unlike envnca's own version of
// this: both the write (colorizeField, indexed by the same (i,j) =
// (x-cell, y-cell) convention every other shader in this project uses)
// and the read (fieldVertex, UV = raw NDC-to-[0,1] with no inversion)
// are authored together in this one file, so there's no externally-
// imposed row order to reconcile — texel (i,j) *is* domain position
// (i/GRID_N, j/GRID_N), and sampling with uv = domain position directly
// lands on exactly that texel. (Contrast: envnca's grid layout comes
// from environment.py's own (C,H,W) row-major convention, authored
// independently of its render code, which is why *that* project needs
// an explicit flip.)

const GRID_N: u32 = __GRID_N__u;
const NODES: u32 = GRID_N + 1u;
const SCALE: f32 = 65536.0; // must match p2g.wgsl/gridUpdate.wgsl's own SCALE

// gridAccum's channel layout — must match p2g.wgsl/clearGrid.wgsl/
// gridUpdate.wgsl's own copies (WGSL has no #include). CH_MOM_X/CH_MOM_Y
// aren't read here (gridVel, below, already carries the resolved
// velocity gridUpdate.wgsl computed from them last pass).
const CH_MASS: u32 = 2u;
const CH_J: u32 = 3u;
const CH_SHEAR: u32 = 4u;
const CH_PRESSURE: u32 = 5u;
const CHANNELS: u32 = 6u;
const PRESSURE_SCALE: f32 = 2.0; // must match p2g.wgsl's own PRESSURE_SCALE

const MODE_NONE: u32 = 0u;
const MODE_DENSITY: u32 = 1u;
const MODE_SPEED: u32 = 2u;
const MODE_DEFORMATION: u32 = 3u;
const MODE_PRESSURE: u32 = 4u;
const MODE_SHEAR: u32 = 5u;

// Fixed normalization ceilings, not dynamically tracked — picked by
// eye against this project's own default settings (gravity=200,
// 8 reference-scale blobs). A cell holding ~2x this project's typical
// settled packing (see scene.ts's own notes on packing density) or a
// node moving at ~half this project's own MAX-ish transient speed maps
// to the top of the ramp; either can still go higher, just clips to the
// ramp's brightest color instead of resolving further (same tradeoff
// envnca's own intensity control makes, just without a live slider).
const DENSITY_MAX: f32 = 10.0;
const SPEED_MAX: f32 = 4.0;
// DEFORMATION/SHEAR are kinematic (dimensionless, no stiffness in them —
// see p2g.wgsl's own J/shearMag comment), so one fixed ceiling works
// across every material setting. PRESSURE_MAX is NOT stiffness-
// independent (pressure = lambda*(J-1) — see p2g.wgsl) so this ceiling is
// only "right" at this project's own default Stiffness/Poisson slider
// values; a stiffer material saturates the ramp at a smaller-looking
// deformation, same tradeoff DENSITY_MAX already makes for particle count
// rather than a stiffness-normalized quantity.
//
// DEFORMATION_MAX is deliberately NOT |J-1|=1 (the theoretical max this
// project's yieldLow/yieldHigh clamp can ever produce) — confirmed live
// that a body under a real, active drag only ever reaches |J-1| in the
// ~0.05-0.15 range (a full 2x/0.5x stretch needs the SVD clamp fully
// pinned on BOTH axes at once, which ordinary manipulation never hits),
// so a ceiling picked for the theoretical extreme left the ramp looking
// almost entirely flat/invisible during normal use. 0.15 instead
// saturates at what a real drag actually produces. SHEAR_MAX below is
// still 1.0, not similarly retuned — shear (Frobenius norm of F-r,
// unbounded above unlike |J-1|) already showed visible color at that
// ceiling; only deformation needed this.
const DEFORMATION_MAX: f32 = 0.15;
const SHEAR_MAX: f32 = 1.0;
const PRESSURE_MAX: f32 = 4000.0;

// p2g.wgsl's own scatter target (mass + the field-visualize diagnostics
// J-sum/shear-sum/pressure-sum — see that file's own comment on why
// they're mass-weighted sums, not raw per-node values, needing a
// divide-by-mass below to become a true per-node average). Same single
// combined buffer clearGrid.wgsl/p2g.wgsl/gridUpdate.wgsl all share
// (WebGPU's 8-storage-buffers-per-stage cap — see p2g.wgsl's own comment
// for the validation error this replaced).
@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> gridVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> mode: u32;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

// Below this, a node's own mass-weighted J/shear/pressure average is
// meaningless (no particle actually contributed there) — same
// `mass <= 0.0` guard gridUpdate.wgsl already applies before using this
// node's momentum, just re-checked here since this is a separate pass
// reading the same underlying accumulator.
const MIN_MASS: f32 = 1e-6;

// Reference's own background color, 0x112F41 = rgb(17,47,65) — the
// "nothing here" floor every ramp below fades up from, so a field view
// still reads as an extension of the normal background rather than a
// jarring different palette.
const BG: vec3<f32> = vec3<f32>(17.0 / 255.0, 47.0 / 255.0, 65.0 / 255.0);

fn densityRamp(t: f32) -> vec3<f32> {
  return mix(BG, vec3<f32>(0.35, 0.85, 1.0), clamp(t, 0.0, 1.0));
}

fn speedRamp(t: f32) -> vec3<f32> {
  return mix(BG, vec3<f32>(1.0, 0.55, 0.15), clamp(t, 0.0, 1.0));
}

fn shearRamp(t: f32) -> vec3<f32> {
  return mix(BG, vec3<f32>(0.4, 1.0, 0.55), clamp(t, 0.0, 1.0));
}

// Diverging ramp shared by deformation/pressure — both have a genuine
// sign (compression vs. expansion, compression vs. tension) unlike
// density/speed/shear's plain magnitudes, so `t` is expected in [-1,1]
// (already-normalized-by-its-own-MAX) rather than [0,1]: negative fades
// up to `negColor`, positive to `posColor`, 0 stays at BG (undeformed
// reads identically to "no material here" — same principle as density's
// own mass=0 => BG).
fn divergingRamp(t: f32, negColor: vec3<f32>, posColor: vec3<f32>) -> vec3<f32> {
  let c = clamp(t, -1.0, 1.0);
  if (c < 0.0) {
    return mix(BG, negColor, -c);
  }
  return mix(BG, posColor, c);
}

// sqrt-shaped contrast boost, sign-preserving — steepens the ramp near
// 0 (e.g. a t of 0.1 reads as 0.32, 0.25 as 0.5) so a small deformation
// still shows up as a clearly visible tint instead of a barely-there one,
// while t=1 (already fully saturated) is untouched either way. Applied
// only to deformation below, not pressure/shear — the ask this was added
// for was specifically "deformation is hard to see."
fn boostContrast(t: f32) -> f32 {
  return sign(t) * sqrt(abs(t));
}

@compute @workgroup_size(16, 16, 1)
fn colorizeField(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= NODES || j >= NODES) { return; }
  let base = (i * NODES + j) * CHANNELS;

  let mass = f32(atomicLoad(&gridAccum[base + CH_MASS])) / SCALE;

  var color = BG;
  if (mode == MODE_DENSITY) {
    color = densityRamp(mass / DENSITY_MAX);
  } else if (mode == MODE_SPEED) {
    let speed = length(gridVel[i * NODES + j]);
    color = speedRamp(speed / SPEED_MAX);
  } else if (mode == MODE_DEFORMATION && mass > MIN_MASS) {
    let avgJ = (f32(atomicLoad(&gridAccum[base + CH_J])) / SCALE) / mass;
    // compression (J<1) = warm/red, expansion (J>1) = cool/blue —
    // arbitrary but consistent with pressure's own compression=warm
    // choice below (same physical direction, different quantity).
    color = divergingRamp(boostContrast((avgJ - 1.0) / DEFORMATION_MAX), vec3<f32>(1.0, 0.35, 0.35), vec3<f32>(0.45, 0.55, 1.0));
  } else if (mode == MODE_PRESSURE && mass > MIN_MASS) {
    let avgPressure = (f32(atomicLoad(&gridAccum[base + CH_PRESSURE])) / PRESSURE_SCALE) / mass;
    // positive (compressive) pressure = amber, negative (tensile) = violet.
    color = divergingRamp(avgPressure / PRESSURE_MAX, vec3<f32>(0.6, 0.35, 1.0), vec3<f32>(1.0, 0.75, 0.2));
  } else if (mode == MODE_SHEAR && mass > MIN_MASS) {
    let avgShear = (f32(atomicLoad(&gridAccum[base + CH_SHEAR])) / SCALE) / mass;
    color = shearRamp(avgShear / SHEAR_MAX);
  }
  textureStore(outputTex, vec2<i32>(i32(i), i32(j)), vec4<f32>(color, 1.0));
}

// --- Background quad sampling the texture colorizeField just wrote ---

var<private> quadPositions: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct FieldVOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var fieldTex: texture_2d<f32>;
@group(0) @binding(1) var fieldSampler: sampler;

@vertex
fn fieldVertex(@builtin(vertex_index) vertexIndex: u32) -> FieldVOut {
  let p = quadPositions[vertexIndex];
  var out: FieldVOut;
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = (p + vec2<f32>(1.0)) * 0.5;
  return out;
}

@fragment
fn fieldFragment(in: FieldVOut) -> @location(0) vec4<f32> {
  return textureSample(fieldTex, fieldSampler, in.uv);
}

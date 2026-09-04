// Field-visualize background — WGSL port of mls-mpm/src/gpu/field.wgsl's
// own colorizeField compute pass + full-screen-quad present. Originally
// scoped to just "density"/"speed" (the two modes computable from
// ../core/'s own gridAccum — see this file's own git history);
// "deformation"/"pressure"/"shear" now read a SEPARATE diagnostics
// buffer this project's own fieldDiagnostics.wgsl scatters (see that
// file's own module docstring for why those three live outside
// ../core/ rather than being added back to it), "substrate" reads
// the chemical field environment.wgsl already maintains for the NN policy's
// own sensing.
//
// Three coloring conventions, per this project's own dataviz choice:
//  - Single, non-negative *magnitude* fields (density, speed, shear —
//    all "how much," never signed) use batlow(), a perceptually-uniform
//    scientific sequential colormap (Fabio Crameri's, via the
//    cmcrameri Python package — see BATLOW's own comment for exactly
//    how this LUT was generated), faded up from BG at t=0 the same way
//    this file's original single-hue ramps did.
//  - Fields that can genuinely go negative (deformation = J-1,
//    pressure, and every one of substrate's own chemical channels) use
//    graypoint(): 0 maps to an exact 0.5 gray, growing toward black as
//    the value goes negative and toward white as it goes positive —
//    the same technique envnca/frontend/src/gpu/colorize.wgsl's own
//    substrate mode already uses (`clamp(v,-s,s)/(2*s)+0.5`), applied
//    here to every signed layer, not just substrate, per this project's
//    own request to keep that one technique consistent everywhere it's
//    needed. Distinct from "no material here" (which still fades to
//    BG, same as ever): 0.5 gray means "material present, genuinely at
//    zero," not "nothing measured."
// Also includes the "repulsion" background mode's own present shader —
// same full-screen-quad vertex shader, a separate fragment entry point
// sampling MpmCore's own r32float density texture directly via
// textureLoad (unfilterable by default, same reasoning
// core/repulsion.wgsl's own applyRepulsion pass already documents),
// mirroring mls-mpm/src/gpu/repulsion.wgsl's own repulsionFieldFragment.

const GRID_N: u32 = __GRID_N__u;
const NODES: u32 = GRID_N + 1u;

const CH_MASS: u32 = 2u; // must match core/p2g.wgsl's own channel layout
const SCALE: f32 = 4096.0; // must match core/p2g.wgsl's own fixed-point SCALE

const MODE_NONE: u32 = 0u;
const MODE_DENSITY: u32 = 1u;
const MODE_SPEED: u32 = 2u;
const MODE_DEFORMATION: u32 = 3u;
const MODE_PRESSURE: u32 = 4u;
const MODE_SHEAR: u32 = 5u;

// Same background as this project's own particle-render clear color (see
// render.wgsl) — kept identical so an "empty" cell under any field mode
// reads the same as the canvas's own clear, same reasoning mls-mpm's own
// field.wgsl gives for matching its clear color exactly.
const BG: vec3<f32> = vec3<f32>(0.02, 0.02, 0.02);

// Starting-guess scale constants (not derived from any actual data,
// same "not tuned yet" caveat this project's own training constants
// carry elsewhere) — a cell at or above this value saturates to the
// mode's own full color. DEFORMATION_MAX/PRESSURE_MAX/SHEAR_MAX mirror
// mls-mpm/src/gpu/field.wgsl's own values exactly ("implemented in the
// same way," per this feature's own ask) — see that file's own comments
// for the empirical reasoning behind each (DEFORMATION_MAX in
// particular is deliberately far below the theoretical |J-1| extreme;
// ordinary manipulation never reaches it).
const DENSITY_MAX: f32 = 10.0;
const SPEED_MAX: f32 = 4.0;
const DEFORMATION_MAX: f32 = 0.15;
const SHEAR_MAX: f32 = 1.0;
const PRESSURE_MAX: f32 = 4000.0;
const PRESSURE_SCALE: f32 = 2.0; // must match fieldDiagnostics.wgsl's own PRESSURE_SCALE

// Below this, a node's own mass-weighted average is meaningless (no
// particle actually contributed there) — same MIN_MASS guard mls-mpm's
// own field.wgsl applies before dividing.
const MIN_MASS: f32 = 1e-6;

// --- batlow: Fabio Crameri's scientific colormap (scientificcolour
// maps.net), chosen for single-scalar fields per this project's own
// dataviz convention — perceptually uniform and colorblind-safe, unlike
// this file's original single-hue BG->color ramps. This 32-stop LUT was
// generated once, offline, via:
//   from cmcrameri import cm; import numpy as np
//   cm.batlow(np.linspace(0, 1, 32))[:, :3]
// then pasted in below — not computed at runtime (no Python/matplotlib
// dependency in this WebGPU project), and not the full-resolution
// published LUT (a 32-stop table linearly interpolated in-shader is
// visually indistinguishable from the real thing at this project's own
// display resolution, same tradeoff a texture-based LUT would make at a
// coarser size).
const BATLOW_N: u32 = 32u;
const BATLOW: array<vec3<f32>, 32> = array<vec3<f32>, 32>(
  vec3<f32>(0.0052, 0.0982, 0.3498),
  vec3<f32>(0.0321, 0.1468, 0.3582),
  vec3<f32>(0.0494, 0.1911, 0.3658),
  vec3<f32>(0.0592, 0.2298, 0.3723),
  vec3<f32>(0.0679, 0.2671, 0.3782),
  vec3<f32>(0.0771, 0.2970, 0.3824),
  vec3<f32>(0.0903, 0.3257, 0.3849),
  vec3<f32>(0.1097, 0.3531, 0.3842),
  vec3<f32>(0.1413, 0.3812, 0.3771),
  vec3<f32>(0.1778, 0.4030, 0.3638),
  vec3<f32>(0.2201, 0.4219, 0.3443),
  vec3<f32>(0.2662, 0.4386, 0.3201),
  vec3<f32>(0.3208, 0.4560, 0.2898),
  vec3<f32>(0.3709, 0.4711, 0.2621),
  vec3<f32>(0.4225, 0.4861, 0.2345),
  vec3<f32>(0.4762, 0.5013, 0.2081),
  vec3<f32>(0.5402, 0.5186, 0.1831),
  vec3<f32>(0.6005, 0.5336, 0.1706),
  vec3<f32>(0.6627, 0.5475, 0.1740),
  vec3<f32>(0.7243, 0.5596, 0.1954),
  vec3<f32>(0.7906, 0.5714, 0.2362),
  vec3<f32>(0.8457, 0.5817, 0.2826),
  vec3<f32>(0.8958, 0.5938, 0.3379),
  vec3<f32>(0.9378, 0.6096, 0.4023),
  vec3<f32>(0.9708, 0.6324, 0.4833),
  vec3<f32>(0.9864, 0.6555, 0.5571),
  vec3<f32>(0.9923, 0.6790, 0.6283),
  vec3<f32>(0.9930, 0.7020, 0.6958),
  vec3<f32>(0.9914, 0.7276, 0.7703),
  vec3<f32>(0.9891, 0.7510, 0.8380),
  vec3<f32>(0.9860, 0.7753, 0.9084),
  vec3<f32>(0.9814, 0.8004, 0.9813),
);

fn batlow(t: f32) -> vec3<f32> {
  let c = clamp(t, 0.0, 1.0) * f32(BATLOW_N - 1u);
  let i0 = u32(floor(c));
  let i1 = min(i0 + 1u, BATLOW_N - 1u);
  return mix(BATLOW[i0], BATLOW[i1], fract(c));
}

// Live-adjustable "accent" — a [-2,2] exponential contrast curve applied
// to EVERY background mode's own normalized value before color-mapping
// (accentedMagnitude()/accentedSigned() below, which scalarColor()/
// graypoint() and the repulsion mode's own standalone code all
// funnel through — see each call site for exactly how), per this
// project's own explicit request: a single shared knob to make a mode's
// own effect "more visible" without touching that mode's own MAX/scale
// constant. accent=0 leaves every mode exactly as it already rendered
// (identity curve); increasing toward 2 exaggerates faint signal, while
// decreasing toward -2 suppresses submaximal values toward the neutral
// background/midpoint (see accentedMagnitude() for the exact curve).
// PhysicsPanel-style live uniform, not compile-time — plain buffer
// write, no pipeline rebuild (gpu/render.ts's own setAccent()).
@group(0) @binding(13) var<uniform> accent: f32;

// Exponential accent curve for an UNSIGNED magnitude already normalized
// to [0,1] (density/speed/shear via scalarColor() below, repulsion's
// own standalone t) — accent=0 -> exponent 1.0 (exp(-0), identity).
// Positive accent shrinks the exponent toward exp(-2)~=0.135, and
// pow(t, exponent<1) pushes ANY nonzero t rapidly up toward 1 (a value
// that would otherwise read as a faint tint saturates well before
// accent reaches its own max) — exactly the "accentuates the effects
// exponentially so they're more visible" behavior this slider was
// requested for. Negative accent raises the exponent above 1 (up to
// exp(2)~=7.39), pushing submaximal values toward 0 and reducing their
// impact on the resulting color.
fn accentedMagnitude(t: f32) -> f32 {
  return pow(clamp(t, 0.0, 1.0), exp(-accent));
}

// Same curve as accentedMagnitude() above, applied SYMMETRICALLY around
// 0 for a SIGNED value already normalized to [-1,1] (graypoint() below,
// growth's own standalone t) — sign() preserves which side of 0 the
// value was on; only its own magnitude gets accentuated, so 0 always
// stays exactly 0 regardless of accent (consistent with graypoint()'s
// own "0 is a real, measured value" convention — see this file's own
// module docstring).
fn accentedSigned(t: f32) -> f32 {
  let c = clamp(t, -1.0, 1.0);
  return sign(c) * pow(abs(c), exp(-accent));
}

// Single, non-negative magnitude fields: batlow, faded up from BG at
// t=0 the same shape this file's original ramps used (mix() at the
// SAME t that indexes into batlow, not just at t=0/1, so a barely-above-
// zero cell reads as a faint tint rather than snapping straight to
// batlow's own (fully-saturated-looking) t=0 color). `t` goes through
// accentedMagnitude() first — see that function's own comment.
fn scalarColor(t: f32) -> vec3<f32> {
  let c = accentedMagnitude(t);
  return mix(BG, batlow(c), c);
}

// Signed fields: 0 -> exact 0.5 gray, growing toward black (negative)
// or white (positive) — see this file's own module docstring for why
// this differs from BG-fade-out (0.5 gray means "measured, genuinely
// zero," distinct from "nothing measured here"). `value/scale` goes
// through accentedSigned() first — see that function's own comment.
fn graypoint(value: f32, scale: f32) -> f32 {
  let s = max(scale, 1e-6);
  return accentedSigned(value / s) * 0.5 + 0.5;
}

// read_write, not read — WGSL requires atomic<T> storage variables to be
// read_write regardless of whether a given entry point ever writes
// through them (this pass never does; same "declared permissive, used
// narrowly" convention core/repulsion.wgsl's own particlePos binding
// already documents).
@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> gridVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> mode: u32;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;
// fieldDiagnostics.wgsl's own accumulator (CH_J/CH_SHEAR/CH_PRESSURE/
// CH_MASS — see that file's own header) — plain (non-atomic) read here
// is legal even though it was written via atomics in that OTHER shader
// module: atomicity is a per-shader-module access declaration on a
// GPUBuffer resource, not a property of the buffer itself, and the pass
// boundary between fieldDiagnostics.wgsl's own scatter pass and this
// one already provides the memory-visibility guarantee.
@group(0) @binding(7) var<storage, read> diagnostics: array<i32>;

@compute @workgroup_size(16, 16)
fn colorizeField(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= NODES || j >= NODES) { return; }
  let idx = i * NODES + j;
  let diagBase = idx * 4u;

  var color = BG;
  if (mode == MODE_DENSITY) {
    let mass = f32(atomicLoad(&gridAccum[idx * 3u + CH_MASS])) / SCALE;
    color = scalarColor(mass / DENSITY_MAX);
  } else if (mode == MODE_SPEED) {
    let speed = length(gridVel[idx]);
    color = scalarColor(speed / SPEED_MAX);
  } else if (mode == MODE_DEFORMATION || mode == MODE_PRESSURE || mode == MODE_SHEAR) {
    let mass = f32(diagnostics[diagBase + 3u]) / SCALE;
    if (mass > MIN_MASS) {
      if (mode == MODE_DEFORMATION) {
        let avgJ = (f32(diagnostics[diagBase + 0u]) / SCALE) / mass;
        let g = graypoint(avgJ - 1.0, DEFORMATION_MAX);
        color = vec3<f32>(g, g, g);
      } else if (mode == MODE_PRESSURE) {
        let avgPressure = (f32(diagnostics[diagBase + 2u]) / PRESSURE_SCALE) / mass;
        let g = graypoint(avgPressure, PRESSURE_MAX);
        color = vec3<f32>(g, g, g);
      } else {
        let avgShear = (f32(diagnostics[diagBase + 1u]) / SCALE) / mass;
        color = scalarColor(avgShear / SHEAR_MAX);
      }
    }
  }

  textureStore(outputTex, vec2<i32>(i32(i), i32(j)), vec4<f32>(color, 1.0));
}

// --- Full-screen quad, shared by every present path below ---

const QUAD_POSITIONS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

struct QuadOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

// Shared with render.wgsl's particle camera. Applying zoom to this quad's
// vertices makes every diagnostic/chemical texture rasterize through the same
// view transform as particle geometry; it is not a post-process canvas scale.
@group(0) @binding(22) var<uniform> fieldViewZoom: f32;

@vertex
fn fieldVertex(@builtin(vertex_index) vertexIndex: u32) -> QuadOut {
  let p = QUAD_POSITIONS[vertexIndex];
  var out: QuadOut;
  out.position = vec4<f32>(p * max(fieldViewZoom, 1e-4), 0.0, 1.0);
  out.uv = (p + vec2<f32>(1.0)) * 0.5;
  return out;
}

// Binding numbers here (and for repulsionFragment/colorizeSubstrate/
// substrateFragment below) deliberately don't restart at 0 — every
// @group/@binding declaration in a WGSL module shares one namespace
// regardless of which entry point uses it (there's no per-entry-point
// scoping the way there is for pipeline *layouts*, which layout:"auto"
// does derive per entry point via reachability), so colliding with
// colorizeField's own bindings 0-3/7 above would be a compile error even
// though no single pipeline ever touches both sets at once.
@group(0) @binding(4) var fieldTex: texture_2d<f32>;
@group(0) @binding(5) var fieldSampler: sampler;

@fragment
fn fieldFragment(in: QuadOut) -> @location(0) vec4<f32> {
  return textureSample(fieldTex, fieldSampler, in.uv);
}

// --- Repulsion background: same quad, samples MpmCore's own r32float
// density texture directly (textureLoad — unfilterable by default) ---

const REPULSION_FIELD_N: u32 = __REPULSION_FIELD_N__u;
const REPULSION_DISPLAY_MAX: f32 = 3.0;

@group(0) @binding(6) var repulsionTex: texture_2d<f32>;

@fragment
fn repulsionFragment(in: QuadOut) -> @location(0) vec4<f32> {
  let texel = clamp(vec2<i32>(in.uv * f32(REPULSION_FIELD_N)), vec2<i32>(0), vec2<i32>(i32(REPULSION_FIELD_N) - 1));
  let density = textureLoad(repulsionTex, texel, 0).r;
  let t = accentedMagnitude(density / REPULSION_DISPLAY_MAX);
  let color = mix(BG, vec3<f32>(1.0, 0.75, 0.35), t);
  return vec4<f32>(color, 1.0);
}

// Exact policy morphology input: already Gaussian-blurred and normalized to
// [0,1] by core/morphology.wgsl. RGB packs the three sensed quantities:
// R=density gradient X, G=density gradient Y, B=density. Gradients use the
// same one-texel central difference core/agents.wgsl samples before rotating
// it into each agent's heading-relative frame. Signed components are encoded
// around 0.5: 0.5 means zero, below 0.5 negative, above 0.5 positive.
@group(0) @binding(19) var morphologyTex: texture_2d<f32>;
struct MorphologyDisplay {
  gradientEnabled: f32,
  densityEnabled: f32,
}
@group(0) @binding(20) var<uniform> morphologyDisplay: MorphologyDisplay;
const MORPHOLOGY_GRADIENT_DISPLAY_MAX: f32 = 0.05;

fn morphologyAt(p: vec2<i32>) -> f32 {
  let n = i32(REPULSION_FIELD_N);
  let wrapped = ((p % vec2<i32>(n)) + vec2<i32>(n)) % vec2<i32>(n);
  return textureLoad(morphologyTex, wrapped, 0).r;
}

@fragment
fn morphologyFragment(in: QuadOut) -> @location(0) vec4<f32> {
  let texel = clamp(vec2<i32>(in.uv * f32(REPULSION_FIELD_N)), vec2<i32>(0), vec2<i32>(i32(REPULSION_FIELD_N) - 1));
  let density = morphologyAt(texel);
  let gx = 0.5 * (morphologyAt(texel + vec2<i32>(1, 0)) - morphologyAt(texel - vec2<i32>(1, 0)));
  let gy = 0.5 * (morphologyAt(texel + vec2<i32>(0, 1)) - morphologyAt(texel - vec2<i32>(0, 1)));
  let red = morphologyDisplay.gradientEnabled * (accentedSigned(gx / MORPHOLOGY_GRADIENT_DISPLAY_MAX) * 0.5 + 0.5);
  let green = morphologyDisplay.gradientEnabled * (accentedSigned(gy / MORPHOLOGY_GRADIENT_DISPLAY_MAX) * 0.5 + 0.5);
  let blue = morphologyDisplay.densityEnabled * accentedMagnitude(density);
  return vec4<f32>(red, green, blue, 1.0);
}

// --- Substrate background: environment.wgsl's own chemical field
// (gpu/environment.ts's ping-pong buffers), first 3 channels -> RGB via
// graypoint() per channel independently — the exact technique
// envnca/frontend/src/gpu/colorize.wgsl's own MODE_SUBSTRATE uses
// (`clamp(v,-s,s)/(2*s)+0.5` per channel), just with a fixed scale
// constant here rather than that project's live EMA-tracked one (this
// project doesn't otherwise track a running per-channel magnitude
// anywhere — a fixed starting-guess ceiling, same convention as
// DENSITY_MAX/SPEED_MAX above, not a behavior regression from anything
// this project already had). A genuinely negative channel darkens
// toward black, positive lightens toward white, 0 is exact 0.5 gray —
// per-channel, so with 3 differing channels the result is visibly
// colored despite each channel individually being graypoint-mapped, not
// literal grayscale (contrast MODE_DEFORMATION/MODE_PRESSURE above,
// where the SAME one value feeds R=G=B, so those genuinely render
// grayscale).
//
// Own output texture (SUBSTRATE_WIDTH x SUBSTRATE_HEIGHT — the chemical
// field's own resolution, unrelated to GRID_N+1) and own compute pass,
// not folded into colorizeField above: environment.ts's grid is a
// different size than MpmCore's, and (unlike density/speed/deformation/
// pressure/shear, which all read from a FIXED buffer object) which
// buffer is "current" flips every macro step (see environment.ts's own
// parity) — render.ts picks the right one of two precomputed bind
// groups each frame, mirroring environment.ts's own parity-indexed
// bind-group-array convention exactly.
const SUBSTRATE_WIDTH: u32 = __FIELD_MAX_WIDTH__u;
const SUBSTRATE_HEIGHT: u32 = __FIELD_MAX_HEIGHT__u;
const SUBSTRATE_CHANNELS: u32 = __CHANNELS__u;
const SUBSTRATE_WIDTHS: array<u32, SUBSTRATE_CHANNELS> = __FIELD_WIDTHS__;
const SUBSTRATE_HEIGHTS: array<u32, SUBSTRATE_CHANNELS> = __FIELD_HEIGHTS__;
const SUBSTRATE_OFFSETS: array<u32, SUBSTRATE_CHANNELS> = __FIELD_OFFSETS__;
const SUBSTRATE_MAX: f32 = 2.0;

fn substrateIndex(c: u32, y: u32, x: u32) -> u32 {
  return SUBSTRATE_OFFSETS[c] + y * SUBSTRATE_WIDTHS[c] + x;
}

fn substrateValue(c: u32, outputX: u32, outputY: u32) -> f32 {
  let uv = (vec2<f32>(f32(outputX), f32(outputY)) + vec2<f32>(0.5))
    / vec2<f32>(f32(SUBSTRATE_WIDTH), f32(SUBSTRATE_HEIGHT));
  let width = SUBSTRATE_WIDTHS[c];
  let height = SUBSTRATE_HEIGHTS[c];
  let x = min(u32(floor(uv.x * f32(width))), width - 1u);
  let y = min(u32(floor(uv.y * f32(height))), height - 1u);
  return substrateGrid[substrateIndex(c, y, x)];
}

@group(0) @binding(8) var<storage, read> substrateGrid: array<f32>;
@group(0) @binding(9) var substrateOutputTex: texture_storage_2d<rgba8unorm, write>;
// x: first RGB-window channel, y: isolate the orientation channel as grayscale.
@group(0) @binding(21) var<uniform> substrateDisplay: vec4<u32>;
// x: substrate zero-is-black, y: boundary-gradient zero-is-black.
@group(0) @binding(23) var<uniform> backgroundZeroIsBlack: vec2<u32>;

fn substrateDisplayValue(value: f32) -> f32 {
  if (backgroundZeroIsBlack.x != 0u) {
    return max(accentedSigned(value / max(SUBSTRATE_MAX, 1e-6)), 0.0);
  }
  return graypoint(value, SUBSTRATE_MAX);
}

@compute @workgroup_size(16, 16)
fn colorizeSubstrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= SUBSTRATE_WIDTH || y >= SUBSTRATE_HEIGHT) { return; }

  let channelStart = substrateDisplay.x;
  let r = substrateValue(channelStart, x, y);
  let g = substrateValue(min(channelStart + 1u, SUBSTRATE_CHANNELS - 1u), x, y);
  let b = substrateValue(min(channelStart + 2u, SUBSTRATE_CHANNELS - 1u), x, y);
  var color = vec3<f32>(substrateDisplayValue(r), substrateDisplayValue(g), substrateDisplayValue(b));
  if (substrateDisplay.y != 0u) {
    // Must match core/agents.wgsl's HEADING_CHANNEL fallback exactly.
    let orientationChannel = min(7u, SUBSTRATE_CHANNELS - 1u);
    color = vec3<f32>(substrateDisplayValue(substrateValue(orientationChannel, x, y)));
  }

  textureStore(substrateOutputTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}

@group(0) @binding(10) var substrateTex: texture_2d<f32>;

@fragment
fn substrateFragment(in: QuadOut) -> @location(0) vec4<f32> {
  return textureSample(substrateTex, fieldSampler, in.uv);
}

// --- Gradient background: a first step toward a "shape boundary"
// visualization — the DIRECTION/magnitude of the REPULSION density
// field's own spatial gradient (core/repulsion.wgsl's own densityAccum,
// splatted from every active particle's own position — the SAME field
// the "repulsion" background mode's own repulsionFragment above already
// samples via repulsionTex, binding 6), not a chemical channel. Computed
// via a Sobel finite difference read off the BLURRED density (see
// blurDensity()/blurredDensityAt() below — raw per-particle density is
// grainy at the scale of individual particles, which a Sobel kernel
// picks up as noise everywhere, not just at the shape's own boundary;
// blurring first is what makes the gradient respond to the shape,
// not the grain) — unlike the chemical field, repulsion has no
// precomputed gradient anywhere upstream (environment.wgsl's own
// computeGradient is chemical-field-only), and computing one here, once
// per background redraw, is cheap enough not to need one.
//
// A flat region (uniformly empty, or uniformly dense) has zero gradient.
// The default signed view renders it at the 0.5 midpoint: R is gradient X,
// G is gradient Y, and B is fixed at 0.5. The optional zero-is-black view
// instead maps zero and negative components to black while positive X/Y use
// the full R/G range. Direction alone for now—no arrow/outline geometry yet.
//
// Own output texture, sized to REPULSION_FIELD_N (core/constants.json's
// own FIELD_N, via mpmCore.ts's own REPULSION_FIELD_N — see that
// constant's own comment for why this is a DIFFERENT, unrelated
// resolution from SUBSTRATE_WIDTH/HEIGHT above), not reused from
// substrate/growth's own gradientOutputTex/dispatch.
//
// A first guess, not yet measured against a real repulsion field's own
// gradient range — environment.wgsl's own sobelX/sobelY taps top out at
// 0.25 magnitude (same kernel, ported below), so a sharp, fully-dense-
// to-empty adjacent-cell edge produces a raw gradient around that same
// scale; retune once this mode's actually been seen live.
const GRADIENT_MAX: f32 = 0.25;

// Toroidal wrap, matching MpmCore's own domain (gridUpdate.wgsl's own
// module docstring — no walls) — repulsion density is splatted the same
// way, so a boundary right at the domain edge should read no differently
// than one in the middle.
fn densityAt(x: i32, y: i32) -> f32 {
  let n = i32(REPULSION_FIELD_N);
  let wrapped = vec2<i32>((x + n) % n, (y + n) % n);
  return textureLoad(repulsionTex, wrapped, 0).r;
}

// --- Blur pass: a cheap, bounded-radius Gaussian blur of the raw
// density field, run BEFORE colorizeGradient below reads it — this is
// what "Blur" (the frontend's own slider, gpu/render.ts's own setBlur())
// actually controls: raw repulsion density is a hard, particle-sized-
// grain splat (see core/repulsion.wgsl's own densityAccum), so its own
// Sobel gradient is noisy at the scale of individual particles, not just
// at the boundary of the shape they form together. Blurring first
// smooths that grain out, so the gradient below responds to the
// SHAPE's own boundary rather than every particle's own edge.
//
// A single, non-separable 2D pass (not two 1D passes, which would be
// cheaper for a large radius) — simpler for a bounded, viewer-only
// diagnostic max radius (BLUR_MAX_RADIUS), not worth a second
// intermediate texture/dispatch. blurSigma<=1e-4 (the slider's own
// default/minimum) skips the blur entirely — copies the raw density
// through unchanged, the exact behavior this feature didn't touch.
const BLUR_MAX_RADIUS: i32 = 6;

@group(0) @binding(14) var<uniform> blurSigma: f32;
@group(0) @binding(17) var blurredDensityOutputTex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn blurDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= REPULSION_FIELD_N || gid.y >= REPULSION_FIELD_N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);

  if (blurSigma <= 1e-4) {
    textureStore(blurredDensityOutputTex, vec2<i32>(x, y), vec4<f32>(densityAt(x, y), 0.0, 0.0, 0.0));
    return;
  }

  // 3-sigma radius, same "enough of the kernel to matter" convention a
  // truncated Gaussian normally uses, capped at BLUR_MAX_RADIUS so the
  // slider's own top end can't make this pass unboundedly expensive.
  let radius = min(BLUR_MAX_RADIUS, i32(ceil(blurSigma * 3.0)));
  var sum: f32 = 0.0;
  var weightSum: f32 = 0.0;
  for (var dy: i32 = -radius; dy <= radius; dy = dy + 1) {
    for (var dx: i32 = -radius; dx <= radius; dx = dx + 1) {
      let w = exp(-f32(dx * dx + dy * dy) / (2.0 * blurSigma * blurSigma));
      sum = sum + densityAt(x + dx, y + dy) * w;
      weightSum = weightSum + w;
    }
  }
  textureStore(blurredDensityOutputTex, vec2<i32>(x, y), vec4<f32>(sum / weightSum, 0.0, 0.0, 0.0));
}

@group(0) @binding(18) var blurredDensityTex: texture_2d<f32>;

// Same toroidal-wrap reasoning as densityAt() above, reading the BLURRED
// texture blurDensity() just wrote instead of raw repulsionTex.
fn blurredDensityAt(x: i32, y: i32) -> f32 {
  let n = i32(REPULSION_FIELD_N);
  let wrapped = vec2<i32>((x + n) % n, (y + n) % n);
  return textureLoad(blurredDensityTex, wrapped, 0).r;
}

// Matches environment.wgsl's own sobelX/sobelY exactly — [[-1,0,1],
// [-2,0,2],[-1,0,1]]/8 (and its transpose for Y) — same kernel, just
// sampling blurredDensityAt() instead of a chemical channel.
fn sobelX(dy: i32, dx: i32) -> f32 {
  if (dx == 0) { return 0.0; }
  let mag = select(0.125, 0.25, dy == 0);
  return select(-mag, mag, dx > 0);
}
fn sobelY(dy: i32, dx: i32) -> f32 {
  return sobelX(dx, dy);
}

@group(0) @binding(15) var gradientOutputTex: texture_storage_2d<rgba8unorm, write>;

// "Gradient exponent" (the frontend's own slider, gpu/render.ts's own
// setGradientExponent()) — a dedicated power curve for THIS mode
// specifically, separate from the shared "Accent" slider every other
// background mode reaches through graypoint()/accentedSigned(). Applied
// to the gradient vector's own MAGNITUDE, not each of gx/gy
// independently — reshaping magnitude alone preserves the vector's own
// true DIRECTION (an independent per-component power curve would distort
// the angle, which matters here since this mode's whole point is "which
// way does the boundary face," not just "how strong is it"). >1
// suppresses weak/noisy gradients and sharpens onto strong, real edges
// (useful when Blur alone doesn't fully clean up residual per-particle
// grain); <1 brings out faint boundaries the linear mapping would
// otherwise show as barely-there. 1 = identity, exactly today's linear-
// normalized mapping, unchanged from before this knob existed.
@group(0) @binding(19) var<uniform> gradientExponent: f32;

@compute @workgroup_size(16, 16)
fn colorizeGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= REPULSION_FIELD_N || gid.y >= REPULSION_FIELD_N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);

  var gx: f32 = 0.0;
  var gy: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let v = blurredDensityAt(x + dx, y + dy);
      gx = gx + v * sobelX(dy, dx);
      gy = gy + v * sobelY(dy, dx);
    }
  }

  let normalized = clamp(vec2<f32>(gx, gy) / GRADIENT_MAX, vec2<f32>(-1.0), vec2<f32>(1.0));
  let mag = length(normalized);
  let shapedMag = pow(min(mag, 1.0), gradientExponent);
  var dir = vec2<f32>(0.0, 0.0);
  if (mag > 1e-6) {
    dir = normalized / mag;
  }
  let shaped = dir * shapedMag;

  var color = vec3<f32>(shaped.x * 0.5 + 0.5, shaped.y * 0.5 + 0.5, 0.5);
  if (backgroundZeroIsBlack.y != 0u) {
    color = vec3<f32>(max(shaped.x, 0.0), max(shaped.y, 0.0), 0.0);
  }
  textureStore(gradientOutputTex, vec2<i32>(x, y), vec4<f32>(color, 1.0));
}

@group(0) @binding(16) var gradientTex: texture_2d<f32>;

@fragment
fn gradientFragment(in: QuadOut) -> @location(0) vec4<f32> {
  return textureSample(gradientTex, fieldSampler, in.uv);
}

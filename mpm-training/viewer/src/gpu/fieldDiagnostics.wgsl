// Viewer-only diagnostic P2G scatter — computes the same per-particle
// deformation/shear/pressure quantities mls-mpm/src/gpu/p2g.wgsl scatters
// into its own CH_J/CH_SHEAR/CH_PRESSURE channels, for field.wgsl's
// "Deformation"/"Pressure"/"Shear" background modes. Deliberately NOT
// added to ../core/p2g.wgsl itself: that file was stripped of exactly
// these diagnostics on purpose when this project's shared physics core
// was extracted from mls-mpm's sandbox (see its own module docstring) —
// core/ is reused verbatim by trainer/mpm_core.py's headless training
// path, where these extra atomic scatters would be pure wasted per-
// particle cost paid by every candidate of every generation, for a value
// training itself never reads. Keeping this a separate, viewer-owned
// pass means training stays exactly as cheap as before, and the viewer
// gets the same visualization anyway.
//
// Runs once per rendered frame (see render.ts's own render(), alongside
// colorizeField's existing density/speed dispatch), not once per physics
// substep the way core/p2g.wgsl's own scatter does — using whichever
// particle state (positions/F/Jp) is current *after* all of this frame's
// substeps have already run. This is deliberately self-contained rather
// than trying to average against ../core/'s own gridAccum mass channel:
// that channel's own last write reflects the P2G scatter from *before*
// the final substep's g2p update (one substep stale relative to the
// positions/F this pass reads), so reusing it here would silently
// mismatch "where mass is" against "what J/shear/pressure value is
// shown there." Scattering this pass's own mass-weight alongside J/
// shear/pressure keeps every value in this file mutually consistent,
// at the cost of one extra atomic add per stencil cell over what
// mls-mpm's own combined pass needs.
//
// Stencil/wraparound math (quadraticWeights, wrapIndex) is duplicated
// from ../core/p2g.wgsl rather than shared — WGSL has no #include, the
// same duplication tradeoff every other file in this project's own
// core/viewer split already accepts (see e.g. core/g2p.wgsl's own
// Material struct comment). This project's domain is toroidal (no
// walls — see core/p2g.wgsl's own docstring), so wrapIndex() is used
// here too, NOT mls-mpm's own bounds-check-and-skip (that project's
// domain is walled).

const GRID_N: u32 = __GRID_N__u;
const DX: f32 = __DX__;
const INV_DX: f32 = __INV_DX__;

// Matches core/p2g.wgsl's own SCALE — same headroom reasoning (bounded
// by a handful of nearby particles' worth of contribution, nowhere near
// i32 overflow). PRESSURE_SCALE separately matches mls-mpm/src/gpu/
// p2g.wgsl's own value: pressure's own magnitude (up to PRESSURE_CLAMP)
// is orders larger than J/shear's ~0-2 range, so it needs a much smaller
// fixed-point scale to stay in range.
const SCALE: f32 = 4096.0;
const PRESSURE_SCALE: f32 = 2.0;
const PRESSURE_CLAMP: f32 = 1.0e6;

// This pass's own accumulator layout — unrelated to (and NOT the same
// buffer as) core/'s own gridAccum (CH_MOM_X/CH_MOM_Y/CH_MASS only).
const CH_J: u32 = 0u;
const CH_SHEAR: u32 = 1u;
const CH_PRESSURE: u32 = 2u;
const CH_MASS: u32 = 3u;
// Four channels are allocated/read by render.ts/field.wgsl. Keeping this
// stride at 3 made each node overlap the next one while colorizeField read
// four-wide records, producing the characteristic vertical-strip corruption.
const CHANNELS: u32 = 4u;

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> particleF: array<vec4<f32>>;
// Per-particle rest-state bookkeeping (growthF / jp / cycleActive) — see
// ../../../core/agents.wgsl's own ParticleRest struct for the full docs.
// Duplicated here, same convention this file's own Material struct and
// quadraticWeights()/wrapIndex() already follow (WGSL has no #include).
struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  cycleActive: f32,
  growthAngle: f32,
  growthAnisotropy: f32,
  divisionBias: f32,
  growthFrameHeading: f32,
  appearanceScale: f32,
  _padding: f32,
}
@group(0) @binding(2) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(3) var<storage, read_write> diagnostics: array<atomic<i32>>;

struct Material {
  mu0: f32,
  lambda0: f32,
  hardening: f32,
  yieldLow: f32,
  yieldHigh: f32,
  growthRate: f32,
  growthMax: f32,
  growthAnisotropy: f32,
  particleMass: f32,
  particleVolume: f32,
  growthCompressionStart: f32,
  growthCompressionStop: f32,
  growthCompressionFeedback: f32,
}
@group(0) @binding(4) var<uniform> material: Material;
@group(0) @binding(5) var<uniform> activeCount: u32;

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z,
    a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z,
    a.z * b.y + a.w * b.w
  );
}

fn matTranspose(m: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(m.x, m.z, m.y, m.w);
}

fn matDet(m: vec4<f32>) -> f32 {
  return m.x * m.w - m.y * m.z;
}

fn matInverse(m: vec4<f32>) -> vec4<f32> {
  let det = matDet(m);
  if (abs(det) < 1e-8) {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
  }
  return vec4<f32>(m.w, -m.y, -m.z, m.x) / det;
}

struct Polar {
  r: vec4<f32>,
  s: vec4<f32>,
};

fn polarDecompose(m: vec4<f32>) -> Polar {
  let x = m.x + m.w;
  let y = m.z - m.y;
  let d = sqrt(x * x + y * y);
  var r: vec4<f32>;
  if (d < 1e-6) {
    r = vec4<f32>(1.0, 0.0, 0.0, 1.0);
  } else {
    let c = x / d;
    let s = y / d;
    r = vec4<f32>(c, -s, s, c);
  }
  var out: Polar;
  out.r = r;
  out.s = matMul(matTranspose(r), m);
  return out;
}

fn wrapIndex(i: i32) -> u32 {
  let n = i32(GRID_N);
  return u32(((i % n) + n) % n);
}

fn quadraticWeights(fx: vec2<f32>) -> array<vec2<f32>, 3> {
  var w: array<vec2<f32>, 3>;
  let a = vec2<f32>(1.5) - fx;
  let b = fx - vec2<f32>(1.0);
  let c = fx - vec2<f32>(0.5);
  w[0] = 0.5 * a * a;
  w[1] = vec2<f32>(0.75) - b * b;
  w[2] = 0.5 * c * c;
  return w;
}

@compute @workgroup_size(16, 16)
fn clearDiagnostics(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i > GRID_N || j > GRID_N) { return; }
  let base = (i * (GRID_N + 1u) + j) * CHANNELS;
  atomicStore(&diagnostics[base + CH_J], 0);
  atomicStore(&diagnostics[base + CH_SHEAR], 0);
  atomicStore(&diagnostics[base + CH_PRESSURE], 0);
  atomicStore(&diagnostics[base + CH_MASS], 0);
}

@compute @workgroup_size(64)
fn scatterDiagnostics(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let F = particleF[pi];
  let rest = particleRest[pi];
  let Jp = rest.jp;

  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  let e = exp(material.hardening * (1.0 - Jp));
  let lambda = material.lambda0 * e;

  // Diagnostics must use the same elastic deformation as P2G. Raw F also
  // contains stress-free growth and would falsely display grown tissue as
  // strained/pressurized.
  let g = max(matDet(rest.growthF), 1e-6);
  let Fe = matMul(F, matInverse(rest.growthF));
  let J = matDet(Fe);
  let polar = polarDecompose(Fe);
  let r = polar.r;

  // Same two diagnostics mls-mpm/src/gpu/p2g.wgsl computes — see that
  // file's own comment for the physical meaning of each:
  //  - shearMag: Frobenius norm of (F - r), how far F is from a pure
  //    rotation, 0 = undeformed.
  //  - pressure: isotropic part of the elastic stress, sign-flipped so
  //    compression (J<1) reads positive.
  let FminusR = Fe - r;
  let shearMag = sqrt(FminusR.x * FminusR.x + FminusR.y * FminusR.y + FminusR.z * FminusR.z + FminusR.w * FminusR.w);
  let pressure = clamp(-lambda * (J - 1.0), -PRESSURE_CLAMP, PRESSURE_CLAMP);

  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let ni = wrapIndex(base.x + i32(i));
      let nj = wrapIndex(base.y + i32(j));

      let wgt = w[i].x * w[j].y;
      // Same grown mass weighting as core/p2g.wgsl.
      let massContribution = wgt * material.particleMass * g;

      let nodeIndex = (ni * (GRID_N + 1u) + nj) * CHANNELS;
      atomicAdd(&diagnostics[nodeIndex + CH_J], i32(round(massContribution * J * SCALE)));
      atomicAdd(&diagnostics[nodeIndex + CH_SHEAR], i32(round(massContribution * shearMag * SCALE)));
      atomicAdd(&diagnostics[nodeIndex + CH_PRESSURE], i32(round(massContribution * pressure * PRESSURE_SCALE)));
      atomicAdd(&diagnostics[nodeIndex + CH_MASS], i32(round(massContribution * SCALE)));
    }
  }
}

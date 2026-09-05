// Particle-to-grid transfer — direct port of the P2G half of
// mls-mpm88-explained.cpp's advance() (see mls-mpm/README.md for the
// reference this was checked line-by-line against).
//
// Independent copy of mls-mpm/src/gpu/p2g.wgsl (this project's own
// sandbox), stripped of that file's field-visualize diagnostic channels
// (CH_J/CH_SHEAR/CH_PRESSURE, and the shearMag/pressure computation that
// only fed them) — this core has no display layer to feed, only
// CH_MOM_X/CH_MOM_Y/CH_MASS are scattered.
//
// TOROIDAL: the reference (and mls-mpm's own sandbox) is a walled
// domain, so its stencil loop below just drops a node index that lands
// outside [0,GRID_N] (defensive only there — the wall keeps particles
// well clear in practice). This project's own domain has no walls
// (see gridUpdate.wgsl's own module docstring), so a particle right at
// the edge legitimately needs its 3x3 stencil to wrap onto the opposite
// edge — wrapIndex() below replaces the old bounds-check-and-skip with
// modulo wraparound, applied to exactly GRID_N distinct node indices per
// axis ([0,GRID_N)); index GRID_N of this shader's own (GRID_N+1)-sized
// grid buffers is therefore never written (nor ever read — g2p.wgsl's
// own gather wraps identically) — a harmless one-node-wide dead strip on
// two edges of the allocated buffer, not a bug, kept only because
// resizing every buffer down to GRID_N nodes wasn't worth the churn for
// what's otherwise a pure indexing change.
//
// 2x2 matrices (F, C, stress, ...) are NOT WGSL's built-in mat2x2<f32> —
// that type is column-major (m[col][row]), and every formula below is
// transcribed straight from the reference's row-major math notation.
// Using the built-in type would mean silently transposing every formula
// or renaming every access; a plain vec4<f32> holding (m00, m01, m10,
// m11) in the same order the reference writes it sidesteps that
// entirely, at the cost of spelling out matMul/matTranspose/matDet by
// hand below.
//
// No native float atomics on storage buffers in core WebGPU, so the P2G
// scatter-add (momentum-x, momentum-y, mass — accumulated from however
// many particles land near a given grid node) goes through a fixed-point
// i32 buffer, decoded back to float by gridUpdate.wgsl next.

const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);

const DX: f32 = __DX__;
const INV_DX: f32 = __INV_DX__;
const DT: f32 = __DT__;

// Matches g2p's shared SCALE — headroom: total mass/momentum landing on
// 4096 retains sub-per-mille transfer precision while providing 16x the
// momentum headroom of the old 65536 scale. The old value was measured
// reaching 2.13e9 raw units during an ordinary growing rollout, close
// enough to i32 overflow that atomic ordering made collapse intermittent.
const SCALE: f32 = 4096.0;

// gridAccum's per-node channel layout — must match clearGrid.wgsl/
// gridUpdate.wgsl's own copies of these exact same constants (WGSL has
// no #include, same duplication tradeoff as the Material struct below).
const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CHANNELS: u32 = 3u;

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> particleC: array<vec4<f32>>;
// Per-particle rest-state bookkeeping (growthF / jp / cycleActive) — see
// core/agents.wgsl's own ParticleRest struct for the full field-by-field
// docs. Declared identically
// here (WGSL has no #include, same duplication tradeoff as the Material
// struct above).
struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  cycleActive: f32,
  growthAngle: f32,
  growthAnisotropy: f32,
  divisionBias: f32, // Original world area (legacy ABI name).
  growthFrameAngle: f32,
  appearanceScale: f32,
  quadratureWeight: f32,
  // Transported world-space half edges, row major. Independent of plastic F.
  domain: vec4<f32>,
}
@group(0) @binding(4) var<storage, read> particleRest: array<ParticleRest>;
@group(0) @binding(5) var<storage, read_write> gridAccum: array<atomic<i32>>;

// Live-adjustable, unlike GRID_N/DX/... above (those are compile-time
// consts sizing arrays/dispatches) — the material params push new values
// here on every tick via a cheap queue.writeBuffer, no pipeline
// recreation. mu0/lambda0 are the already-Lamé-converted form of Young's
// modulus (E) and Poisson's ratio (nu) — see mls-mpm/src/gpu/mpm.ts's
// lameParams() for the conversion formula (mu0 = E/(2*(1+nu)), lambda0 =
// E*nu/((1+nu)*(1-2*nu))), done host-side so this shader only ever
// multiplies, never divides by a value close to zero.
// yieldLow/yieldHigh (g2p.wgsl's own plasticity clamp bounds) live in
// this same struct/buffer too, even though this shader never reads
// them — one Material concept, one buffer; p2g.wgsl and g2p.wgsl each
// just read the subset of fields they need. Declared here identically to
// g2p.wgsl's own copy (WGSL has no #include) so both shaders agree on
// the byte layout regardless of which fields either one actually uses.
struct Material {
  mu0: f32,
  lambda0: f32,
  hardening: f32,
  yieldLow: f32,
  yieldHigh: f32,
  // Growth params — read by core/g2p.wgsl only (that's where growth is
  // actually advanced); declared here purely so both shaders agree on
  // this uniform's byte layout, the same "one Material concept, one
  // buffer; each shader reads the subset it needs" convention
  // yieldLow/yieldHigh above already follow in the other direction (this
  // shader reads THOSE, g2p reads these).
  growthRate: f32,
  growthMax: f32,
  growthAnisotropy: f32,
  // Base mass and rest area represented by one g=1 sampling particle.
  // Runtime values let density vary between rollouts without rebuilding
  // pipelines; both scale together so physical material density is fixed.
  particleMass: f32,
  particleVolume: f32,
  growthCompressionStart: f32,
  growthCompressionStop: f32,
  growthCompressionFeedback: f32,
  // Per-physics-substep fraction of elastic shear that relaxes. 0 retains
  // the legacy solid response exactly; 1 removes all deviatoric strain.
  fluidity: f32,
}
@group(0) @binding(6) var<uniform> material: Material;

// Live particle count, unlike the compile-time GRID_N/... consts above —
// a fixed-capacity buffer with a live count is what lets a scene switch
// (different particle counts) or particle injection at runtime happen
// without rebuilding this pipeline.
@group(0) @binding(7) var<uniform> activeCount: u32;

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

fn matAddScaledIdentity(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(m.x + s, m.y, m.z, m.w + s);
}

struct Polar {
  r: vec4<f32>,
  s: vec4<f32>,
};

// Closed-form 2x2 polar decomposition (M = R*S, R a proper rotation, S
// symmetric positive semi-definite) — the standard formula: for
// x=m00+m11, y=m10-m01, d=sqrt(x^2+y^2), R = [[x,-y],[y,x]]/d.
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

// Euclidean modulo (always non-negative, unlike WGSL's own `%` on
// negative operands, which follows truncated-division sign — matches
// the dividend) — the wraparound this file's own module docstring
// describes. Adding `n` before the final `%` guards the negative case
// without a branch: for i in [-n, n), (i%n)+n is in [0, 2n), so one more
// `%n` always lands in [0,n).
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

__DOMAIN_FUNCTIONS__

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let vel = particleVel[pi];
  let F = particleF[pi];
  let C = particleC[pi];
  let rest = particleRest[pi];
  let Jp = rest.jp;

  // base = floor(y - 0.5), fx = y - base — fx therefore lands in
  // [0.5, 1.5), NOT [0,1). quadraticWeights() below is only a valid
  // non-negative partition of unity for fx in [0.5, 1.5), the range this
  // particular base/fx relationship (one cell *before* the nearest node,
  // per http://mpm.graphics Eqn. 123) produces.
  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  // Current Lamé parameters — http://mpm.graphics Eqn. 86.
  let e = exp(material.hardening * (1.0 - Jp));
  let mu = material.mu0 * e;
  let lambda = material.lambda0 * e;
  // MULTIPLICATIVE GROWTH DECOMPOSITION: F = Fe*Fg, where Fg is a
  // stress-free change of this particle's own REST configuration and Fe
  // is the only part elasticity is allowed to see. Fg is stored as a full
  // row-major 2x2 tensor and can accumulate directional rest deformation.
  // THIS SUBSTITUTION IS THE WHOLE POINT: evaluating the
  // constitutive law below on Fe rather than raw F means grown volume
  // costs zero stress by construction, so elasticity resists only
  // deviation from the newly-grown rest state instead of forever trying
  // to restore the original one. See core/g2p.wgsl for where Fg is
  // advanced, and core/agents.wgsl's own ParticleRest.growthF field
  // comment for the full rationale.
  let Fg = rest.growthF;
  let g = max(matDet(Fg), 1e-6); // det(Fg): grown rest-area ratio
  let q = max(rest.quadratureWeight, 1e-6);
  let Fe = matMul(F, matInverse(Fg));
  let Je = matDet(Fe);
  let polar = polarDecompose(Fe);
  let r = polar.r;

  let volEff = material.particleVolume * q * g;
  let massEff = material.particleMass * q * g;

  let PF = matAddScaledIdentity(2.0 * mu * matMul(Fe - r, matTranspose(Fe)), lambda * (Je - 1.0) * Je);
  // Domain-integrated basis gradients derive from elastic virtual work.
  // APIC uses the affine velocity about the domain center at each GRID node.
  for (var k = 0u; k < domainQuadratureCount(rest.domain); k++) {
    let quadrature = domainQuadrature(rest.domain, k);
    let samplePos = pos + quadrature.xy;
    let sampleBase = vec2<i32>(floor(samplePos * INV_DX - vec2<f32>(0.5)));
    let sampleFx = samplePos * INV_DX - vec2<f32>(sampleBase);
    let sampleW = quadraticWeights(sampleFx);
    for (var i = 0u; i < 3u; i++) {
      for (var j = 0u; j < 3u; j++) {
        let node = sampleBase + vec2<i32>(i32(i), i32(j));
        let dpos = vec2<f32>(node) * DX - pos;
        let wgt = quadrature.z * sampleW[i].x * sampleW[j].y;
        let gradient = quadrature.z * domainBasisGradient(sampleFx, i, j, INV_DX);
        let affineVelocity = vel + vec2<f32>(C.x*dpos.x+C.y*dpos.y, C.z*dpos.x+C.w*dpos.y);
        let force = -volEff * vec2<f32>(PF.x*gradient.x+PF.y*gradient.y, PF.z*gradient.x+PF.w*gradient.y);
        let momentum = massEff * wgt * affineVelocity + DT * force;
        let nodeIndex = (wrapIndex(node.x) * (GRID_N+1u) + wrapIndex(node.y)) * CHANNELS;
        atomicAdd(&gridAccum[nodeIndex + CH_MOM_X], i32(round(momentum.x * SCALE)));
        atomicAdd(&gridAccum[nodeIndex + CH_MOM_Y], i32(round(momentum.y * SCALE)));
        atomicAdd(&gridAccum[nodeIndex + CH_MASS], i32(round(massEff * wgt * SCALE)));
      }
    }
  }
}

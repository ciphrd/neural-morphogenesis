// Particle-to-grid transfer — direct port of the P2G half of
// mls-mpm88-explained.cpp's advance() (see that file, and this project's
// README, for the reference this was checked line-by-line against).
//
// 2x2 matrices (F, C, stress, ...) are NOT WGSL's built-in mat2x2<f32> —
// that type is column-major (m[col][row]), and every formula below is
// transcribed straight from the reference's row-major math notation.
// Using the built-in type would mean silently transposing every formula
// or renaming every access; a plain vec4<f32> holding (m00, m01, m10,
// m11) in the same order the reference writes it sidesteps that
// entirely, at the cost of spelling out matMul/matTranspose/matDet by
// hand below (same tradeoff this project's sibling envnca made for
// rotations via explicit cos/sin scalars instead of a rotation matrix
// type).
//
// No native float atomics on storage buffers in core WebGPU, so the P2G
// scatter-add (momentum-x, momentum-y, mass, plus the field-visualize
// diagnostics J-sum/shear-sum/pressure-sum below — accumulated from
// however many particles land near a given grid node) goes through a
// fixed-point i32 buffer — same SCALE-and-round trick envnca's
// agents.wgsl uses for its own deposit scatter, decoded back to float by
// gridUpdate.wgsl/field.wgsl next. All 6 channels share ONE buffer
// (gridAccum below, indexed nodeIndex*CHANNELS+channel) rather than one
// binding per channel: WebGPU caps compute stages at 8 storage buffers,
// and this shader's 5 read-only particle buffers already leave room for
// only a few more — 6 separate bindings would have blown that limit the
// moment the field-visualize diagnostics were added (confirmed live: a
// WebGPU validation error, "storage buffers (11) ... exceeds the maximum
// per-stage limit (8)", the first time this was tried as 6 separate
// bindings instead).

const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);

const DX: f32 = __DX__;
const INV_DX: f32 = __INV_DX__;
const DT: f32 = __DT__;
const PARTICLE_MASS: f32 = __PARTICLE_MASS__;
const VOL: f32 = __VOL__;

// Matches p2g/g2p's shared SCALE — headroom: total mass/momentum landing
// on one node is bounded by however many of the (few thousand) particles
// are within a 2-cell radius of it, each contributing at most
// PARTICLE_MASS — nowhere near the ~32768 this scale allows before i32
// overflow for any configuration this project's UI can reach.
const SCALE: f32 = 65536.0;

// gridAccum's per-node channel layout — must match clearGrid.wgsl/
// gridUpdate.wgsl/field.wgsl's own copies of these exact same 6 constants
// (WGSL has no #include, same duplication tradeoff as the Material struct
// below). CH_J/CH_SHEAR/CH_PRESSURE reuse SCALE directly (same fixed-
// point encoding as momentum/mass): J and shear magnitude are both in
// roughly the same order of magnitude as PARTICLE_MASS itself (J in
// ~[0,8], shear magnitude in ~[0,4] even under adversarial elasticity
// settings — see g2p.wgsl's own yieldLow/yieldHigh), so the same headroom
// argument as gridMass's own SCALE comment (below) still holds. Pressure
// can't share it: pressure = lambda*(J-1) (see p2g() body below), and
// lambda alone can reach ~3e5 at this project's own slider extremes
// (stiffness=20000, poisson=0.45, hardening=4, Jp clamped to 0.6) —
// SCALE=65536 would overflow i32 from a *single* particle's contribution.
// PRESSURE_SCALE/PRESSURE_CLAMP are sized for that instead: clamping
// first bounds the worst case, and a much smaller scale keeps the
// (clamp * scale * up-to-a-few-dozen-particles) sum nowhere near i32's
// ~2.1e9 range. Sub-integer precision on pressure doesn't matter for a
// color ramp anyway (field.wgsl normalizes/clamps it again for display).
const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CH_J: u32 = 3u;
const CH_SHEAR: u32 = 4u;
const CH_PRESSURE: u32 = 5u;
const CHANNELS: u32 = 6u;
const PRESSURE_SCALE: f32 = 2.0;
const PRESSURE_CLAMP: f32 = 1.0e6;

@group(0) @binding(0) var<storage, read> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> particleC: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> particleJp: array<f32>;
// Headroom: total mass/momentum landing on one node is bounded by however
// many of the (few thousand) particles are within a 2-cell radius of it,
// each contributing at most PARTICLE_MASS — nowhere near the ~32768 this
// scale allows before i32 overflow for any configuration this project's
// UI can reach.
@group(0) @binding(5) var<storage, read_write> gridAccum: array<atomic<i32>>;

// Live-adjustable, unlike GRID_N/DX/... above (those are compile-time
// consts sizing arrays/dispatches) — the "Material" panel's sliders push
// new values here on every tick via a cheap queue.writeBuffer, no
// pipeline recreation. mu0/lambda0 are the already-Lamé-converted form
// of Young's modulus (E) and Poisson's ratio (nu) — see gpu/mpm.ts's
// setMaterial(), which does that conversion host-side (same formula the
// reference computes once at startup: mu0 = E/(2*(1+nu)), lambda0 =
// E*nu/((1+nu)*(1-2*nu))) so this shader only ever multiplies, never
// divides by a slider-controlled value close to zero.
// yieldLow/yieldHigh (g2p.wgsl's own plasticity clamp bounds) live in
// this same struct/buffer too, even though this shader never reads
// them — one Material concept per world, one setMaterial() call, one
// buffer; p2g.wgsl and g2p.wgsl each just read the subset of fields
// they need. Declared here identically to g2p.wgsl's own copy (WGSL has
// no #include) so both shaders agree on the byte layout regardless of
// which fields either one actually uses.
struct Material {
  mu0: f32,
  lambda0: f32,
  hardening: f32,
  yieldLow: f32,
  yieldHigh: f32,
}
@group(0) @binding(6) var<uniform> material: Material;

// Live particle count, unlike the compile-time GRID_N/... consts above —
// see gpu/mpm.ts's own activeCount comment for why: a fixed-capacity
// buffer (MAX_PARTICLES) with a live count is what lets main.ts switch
// worlds (different particle counts) without rebuilding this pipeline,
// and is the foundation for injecting particles at runtime later.
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

fn matAddScaledIdentity(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(m.x + s, m.y, m.z, m.w + s);
}

struct Polar {
  r: vec4<f32>,
  s: vec4<f32>,
};

// Closed-form 2x2 polar decomposition (M = R*S, R a proper rotation, S
// symmetric positive semi-definite) — the standard formula: for
// x=m00+m11, y=m10-m01, d=sqrt(x^2+y^2), R = [[x,-y],[y,x]]/d. Verified
// against the identity (R=I) and pure-rotation (R=M) cases by hand
// before porting.
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

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let vel = particleVel[pi];
  let F = particleF[pi];
  let C = particleC[pi];
  let Jp = particleJp[pi];

  // base = floor(y - 0.5), fx = y - base — fx therefore lands in
  // [0.5, 1.5), NOT [0,1). Easy to get wrong (fx = frac(y-0.5) is the
  // natural-looking formula, and IS in [0,1) — but it's a different,
  // invalid quantity here: quadraticWeights() below is only a valid
  // non-negative partition of unity for fx in [0.5, 1.5), the range this
  // particular base/fx relationship (one cell *before* the nearest
  // node, per http://mpm.graphics Eqn. 123) produces. Confirmed by hand
  // before fixing: fx=0 gives a *negative* w[1], impossible for a real
  // B-spline weight.
  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  // Current Lamé parameters — http://mpm.graphics Eqn. 86.
  let e = exp(material.hardening * (1.0 - Jp));
  let mu = material.mu0 * e;
  let lambda = material.lambda0 * e;

  let J = matDet(F);
  let polar = polarDecompose(F);
  let r = polar.r;

  // Deformation-field diagnostics (main.ts's "Field" dropdown — see
  // field.wgsl) — not otherwise used by the physics itself, computed
  // here because F/r/mu/lambda are already at hand:
  //  - shearMag: Frobenius norm of (F - r), i.e. how far F is from a pure
  //    rotation — the SAME quantity the elastic stress term above already
  //    penalizes (matMul(F-r, F^T)), just reduced to one scalar for
  //    display. 0 = locally undeformed/pure rotation, growing with shape
  //    distortion regardless of sign.
  //  - pressure: the isotropic (volumetric) part of the same PK stress
  //    computed below, sign-flipped so compression (J<1) reads positive
  //    and tension/expansion (J>1) reads negative — the conventional
  //    sense of "pressure." Clamped defensively (see PRESSURE_CLAMP's own
  //    comment above) before scatter, not because this ever needs to
  //    happen at reasonable slider settings, just because an atomic i32
  //    accumulator has no clamp of its own once it overflows.
  let FminusR = F - r;
  let shearMag = sqrt(FminusR.x * FminusR.x + FminusR.y * FminusR.y + FminusR.z * FminusR.z + FminusR.w * FminusR.w);
  let pressure = clamp(-lambda * (J - 1.0), -PRESSURE_CLAMP, PRESSURE_CLAMP);

  // http://mpm.graphics Paragraph after Eqn. 176 / Eqn. 52.
  let Dinv = 4.0 * INV_DX * INV_DX;

  let PF = matAddScaledIdentity(2.0 * mu * matMul(F - r, matTranspose(F)), lambda * (J - 1.0) * J);
  let stress = -(DT * VOL * Dinv) * PF;
  // Fused APIC momentum + MLS-MPM stress contribution (taichi MLS-MPM/CPIC
  // notes, Eqn. 29).
  let affine = stress + PARTICLE_MASS * C;

  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let ni = base.x + i32(i);
      let nj = base.y + i32(j);
      // Defensive only — see this file's own note on why the reference
      // never needs this in practice (the sticky boundary keeps
      // particles well clear of the domain edge every step).
      if (ni < 0 || nj < 0 || ni > i32(GRID_N) || nj > i32(GRID_N)) { continue; }

      let dpos = (vec2<f32>(f32(i), f32(j)) - fx) * DX;
      let wgt = w[i].x * w[j].y;
      let affineDpos = vec2<f32>(affine.x * dpos.x + affine.y * dpos.y, affine.z * dpos.x + affine.w * dpos.y);
      let momentum = wgt * (vel * PARTICLE_MASS + affineDpos);
      let massContribution = wgt * PARTICLE_MASS;

      let nodeIndex = (u32(ni) * (GRID_N + 1u) + u32(nj)) * CHANNELS;
      atomicAdd(&gridAccum[nodeIndex + CH_MOM_X], i32(round(momentum.x * SCALE)));
      atomicAdd(&gridAccum[nodeIndex + CH_MOM_Y], i32(round(momentum.y * SCALE)));
      atomicAdd(&gridAccum[nodeIndex + CH_MASS], i32(round(massContribution * SCALE)));
      atomicAdd(&gridAccum[nodeIndex + CH_J], i32(round(massContribution * J * SCALE)));
      atomicAdd(&gridAccum[nodeIndex + CH_SHEAR], i32(round(massContribution * shearMag * SCALE)));
      atomicAdd(&gridAccum[nodeIndex + CH_PRESSURE], i32(round(massContribution * pressure * PRESSURE_SCALE)));
    }
  }
}

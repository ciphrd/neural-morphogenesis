// Grid-to-particle transfer + the MLS-MPM deformation-gradient update +
// snow plasticity clamp — direct port of the G2P half of
// mls-mpm88-explained.cpp's advance(). See p2g.wgsl's own header for why
// 2x2 matrices are plain vec4<f32>(m00,m01,m10,m11) rather than WGSL's
// column-major mat2x2<f32>.
//
// Independent copy of mls-mpm/src/gpu/g2p.wgsl (this project's own
// sandbox) — this file was already pure physics, no mouse/repulsion/
// attract coupling of any kind; the one functional change from that
// walled reference is TOROIDAL wraparound (see wrapIndex() and the
// fract()-based position update below) — this project's own domain has
// no walls (see gridUpdate.wgsl's own module docstring).
//
// The reference's `plastic` flag is a compile-time `true` for every
// particle — there is no UI here to disable it, so the clamp below is
// unconditional rather than porting the reference's
// `for (i<2*int(plastic))` loop literally.

const GRID_N: u32 = __GRID_N__u;
const INV_DX: f32 = __INV_DX__;
const DT: f32 = __DT__;

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particleC: array<vec4<f32>>;
// Per-particle rest-state bookkeeping (growthF / jp / cycleActive) — see
// core/agents.wgsl's own ParticleRest struct for the full field-by-field
// docs. THIS SHADER advances `jp` (the plastic clamp below) and `growthF`
// (the substrate-driven growth law below). `cycleActive` and
// `growthDirection` are owned by core/agents.wgsl and must be PRESERVED on write — the
// element used to be a bare f32 this shader could overwrite wholesale,
// and it no longer is.
struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  cycleActive: f32,
  growthDirection: vec2<f32>,
  growthControls: vec2<f32>,
}
@group(0) @binding(4) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(5) var<storage, read> gridVel: array<vec2<f32>>;

// Live particle count — see p2g.wgsl's own comment on why (a fixed-
// capacity buffer with a live count, not a compile-time PARTICLE_COUNT).
@group(0) @binding(6) var<uniform> activeCount: u32;

// Same Material struct/buffer p2g.wgsl binds (see that file's own
// comment on why one shared struct) — this shader reads
// yieldLow/yieldHigh (the plasticity clamp bounds just below) and the
// growth params (the growth relaxation just below that), but never
// mu0/lambda0/hardening.
struct Material {
  mu0: f32,
  lambda0: f32,
  hardening: f32,
  yieldLow: f32,
  yieldHigh: f32,
  // Exponential area-growth rate while the substrate-triggered cell cycle
  // is active. 0 disables growth.
  growthRate: f32,
  // Legacy uniform slot retained for host-layout compatibility. Division
  // is fixed at area ratio 2 below because any other value would make one
  // parent -> two baseline daughters non-conservative.
  growthMax: f32,
  // Compression reference for continuous mechanical inhibition. Below
  // this Je, growth is slowed in proportion to Je/reference rather than
  // stopped. 0 disables mechanical inhibition entirely.
  growthThreshold: f32,
  // Global multiplier on the policy's per-particle anisotropy. The trainer
  // uses 1; the viewer exposes [0,1] as a live blob-vs-tendril bias.
  growthAnisotropy: f32,
}
@group(0) @binding(7) var<uniform> material: Material;

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

fn identityPlusScaled(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(1.0 + s * m.x, s * m.y, s * m.z, 1.0 + s * m.w);
}

struct Polar {
  r: vec4<f32>,
  s: vec4<f32>,
};

// Same closed-form 2x2 polar decomposition as p2g.wgsl — duplicated
// rather than shared (WGSL has no #include), see this project's own
// design notes on why a little duplication across small, self-contained
// shader files beats introducing a build-time concatenation step.
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

struct Svd {
  u: vec4<f32>,
  sigma: vec2<f32>,
  v: vec4<f32>,
};

// 2x2 SVD built on top of polarDecompose: M = R*S (R rotation, S
// symmetric PSD) => S's eigendecomposition S = V*Sigma*V^T gives
// Sigma's diagonal as M's singular values (S=sqrt(M^T M) is a standard
// polar-decomposition identity) and U = R*V. S is symmetric, so its
// eigenvectors are automatically orthogonal — v2 is built as v1 rotated
// 90° rather than solved for separately, which also guarantees V stays
// a proper rotation (det=+1), matching R's own convention.
fn svd2(m: vec4<f32>) -> Svd {
  let polar = polarDecompose(m);
  let r = polar.r;
  let s = polar.s;
  let a = s.x;
  // Average the two off-diagonal entries — S is symmetric by
  // construction (S = R^T*M with R from polarDecompose), so s.y and s.z
  // should already be equal; this only guards float roundoff.
  let b = 0.5 * (s.y + s.z);
  let d = s.w;

  let tr = a + d;
  let diff = a - d;
  let radius = sqrt(diff * diff * 0.25 + b * b);
  let lambda1 = tr * 0.5 + radius;
  let lambda2 = tr * 0.5 - radius;

  // Eigenvector for lambda1, solved from (S-lambda1*I)v=0's second row
  // (b*vx + (d-lambda1)*vy = 0 => v=(lambda1-d, b)); falls back to the
  // first row's form, then to (1,0), for the degenerate cases where one
  // or both formulas vanish (S already diagonal).
  var v1 = vec2<f32>(lambda1 - d, b);
  if (length(v1) < 1e-6) {
    v1 = vec2<f32>(b, lambda1 - a);
  }
  if (length(v1) < 1e-6) {
    v1 = vec2<f32>(1.0, 0.0);
  }
  v1 = normalize(v1);
  let v2 = vec2<f32>(-v1.y, v1.x);

  var out: Svd;
  out.v = vec4<f32>(v1.x, v2.x, v1.y, v2.y);
  out.sigma = vec2<f32>(lambda1, lambda2);
  out.u = matMul(r, out.v);
  return out;
}

// Euclidean modulo — see p2g.wgsl's own copy of this exact function for
// why (must agree with it exactly: G2P has to gather from the same
// wrapped stencil P2G scattered into this same step).
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

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  let pos = particlePos[pi];
  let F0 = particleF[pi];
  let rest0 = particleRest[pi];
  let Jp0 = rest0.jp;
  let Fg0 = rest0.growthF;
  let g0 = max(matDet(Fg0), 1e-6); // det(Fg): grown rest-area ratio

  // base = floor(y - 0.5), fx = y - base, landing in [0.5, 1.5) — must
  // match p2g.wgsl's own note on this exactly (G2P has to gather from
  // the same 3x3 stencil P2G scattered into this same step).
  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  var v = vec2<f32>(0.0, 0.0);
  var C = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let ni = wrapIndex(base.x + i32(i));
      let nj = wrapIndex(base.y + i32(j));

      // NOT scaled by dx here — unlike p2g's dpos, matches the reference
      // exactly (the 4*inv_dx below already carries the right units).
      let dpos = vec2<f32>(f32(i), f32(j)) - fx;
      let wgt = w[i].x * w[j].y;
      let nodeIndex = ni * (GRID_N + 1u) + nj;
      let gv = gridVel[nodeIndex];
      let wgv = wgt * gv;

      v = v + wgv;
      // APIC affine velocity field: C += 4*inv_dx * outer(w*grid_v, dpos).
      C = C + (4.0 * INV_DX) * vec4<f32>(wgv.x * dpos.x, wgv.x * dpos.y, wgv.y * dpos.x, wgv.y * dpos.y);
    }
  }

  // Toroidal: wrap into [0,1) rather than clamp against a wall — fract()
  // is WGSL's own always-non-negative x-floor(x), so this is correct
  // wraparound even for a newPos that undershot below 0.0, not just one
  // that overshot past 1.0. No velocity zeroing either (there was one
  // here, in this project's own earlier walled version, matching a
  // sticky-wall's own "stop dead at the boundary" semantics) — nothing
  // here is a boundary to stop at anymore, a particle keeps whatever
  // velocity it arrived with straight through the wrap.
  let newPos = fract(pos + DT * v);
  let newVel = v;

  var F = matMul(identityPlusScaled(C, DT), F0);

  let FeTrial = matMul(F, matInverse(Fg0));
  let svd = svd2(FeTrial);
  // Bounds are the world's own Material.yieldLow/yieldHigh — how much of
  // a stretch/compression the corotated elastic term is allowed to fully
  // recover from versus how much gets baked in as permanent (plastic)
  // deformation via Jp below.
  //
  // Yield is a property of ELASTIC strain, so the clamp belongs directly
  // on Fe's singular values, never raw F's. This matters a lot: otherwise
  // accumulated growth would eventually push total F's singular values
  // past the yield bounds all on its own and get
  // silently converted into plastic deformation, which is precisely the
  // "elasticity fights growth" failure the whole decomposition exists to
  // eliminate.
  let sigma = clamp(
    svd.sigma,
    vec2<f32>(material.yieldLow),
    vec2<f32>(material.yieldHigh)
  );

  let oldJe = matDet(FeTrial);
  let sigmaMat = vec4<f32>(sigma.x, 0.0, 0.0, sigma.y);
  let Fe = matMul(matMul(svd.u, sigmaMat), matTranspose(svd.v));
  F = matMul(Fe, Fg0);
  let newJe = matDet(Fe);

  let JpNew = clamp(Jp0 * oldJe / newJe, 0.6, 20.0);

  // SUBSTRATE-DRIVEN GROWTH. agentStep probabilistically starts a cell
  // cycle from the last substrate channel and latches cycleActive. Once
  // active, Fg grows independently of pre-existing dilation. Holding F
  // fixed while g increases makes Fe compressive; the constitutive force
  // then expands the compatible body. This is growth followed by elastic
  // relaxation, rather than the old circular repulsion -> dilation ->
  // growth ratchet.
  let je = oldJe;
  var gNew = g0;
  var FgNew = Fg0;
  // Continuous feedback, not a checkpoint. At and above the reference,
  // growth runs at its full configured rate; compression below it slows
  // the rate smoothly. Any physically valid Je > 0 retains a positive
  // growth rate, so a cycle cannot become permanently latched merely by
  // crossing a threshold. The max(..., 0) guard prevents an inverted F
  // from turning growth into exponential shrinkage.
  var compressionScale = 1.0;
  if (material.growthThreshold > 0.0) {
    compressionScale = clamp(max(je, 0.0) / material.growthThreshold, 0.0, 1.0);
  }
  if (rest0.cycleActive > 0.5 && material.growthRate > 0.0) {
    gNew = min(
      g0 * exp(material.growthRate * compressionScale * DT),
      2.0
    );
    let areaFactor = gNew / g0;
    let directionMagnitude = length(rest0.growthDirection);
    let signalStrength = clamp(
      rest0.growthControls.x * material.growthAnisotropy,
      0.0,
      1.0
    );
    if (directionMagnitude < 1e-6 || signalStrength < 1e-6) {
      // Exact scalar-model equivalence when the network emits no axis.
      FgNew = Fg0 * sqrt(areaFactor);
    } else {
      // The policy direction is expressed in the spatial/world frame.
      // Pull it back through Fe's polar rotation so H acts objectively in
      // Fg's intermediate configuration: rotating the whole body and its
      // heading rotates the result rather than changing its material law.
      let r = polarDecompose(FeTrial).r;
      let worldDir = rest0.growthDirection / directionMagnitude;
      let n = vec2<f32>(
        r.x * worldDir.x + r.z * worldDir.y,
        r.y * worldDir.x + r.w * worldDir.y
      );
      // det(H)=areaFactor exactly. signalStrength=0 is isotropic;
      // signalStrength=1 places the full area increment along n while the
      // perpendicular rest stretch stays unchanged.
      let logArea = log(max(areaFactor, 1.0));
      let parallel = exp(0.5 * logArea * (1.0 + signalStrength));
      let perpendicular = exp(0.5 * logArea * (1.0 - signalStrength));
      let delta = parallel - perpendicular;
      let H = vec4<f32>(
        perpendicular + delta * n.x * n.x,
        delta * n.x * n.y,
        delta * n.y * n.x,
        perpendicular + delta * n.y * n.y
      );
      FgNew = matMul(H, Fg0);
    }
  }

  particlePos[pi] = newPos;
  particleVel[pi] = newVel;
  particleF[pi] = F;
  particleC[pi] = C;
  // cycleActive carried through untouched — owned by core/agents.wgsl (see
  // this file's own ParticleRest comment); this element is shared now,
  // not a bare f32 to overwrite wholesale.
  particleRest[pi] = ParticleRest(FgNew, JpNew, rest0.cycleActive, rest0.growthDirection, rest0.growthControls);
}

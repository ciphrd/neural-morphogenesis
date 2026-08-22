// Grid-to-particle transfer + the MLS-MPM deformation-gradient update +
// snow plasticity clamp — direct port of the G2P half of
// mls-mpm88-explained.cpp's advance(). See p2g.wgsl's own header for why
// 2x2 matrices are plain vec4<f32>(m00,m01,m10,m11) rather than WGSL's
// column-major mat2x2<f32>.
//
// The reference's `plastic` flag is a compile-time `true` for every
// particle (every world's particles share one material — see
// worlds/types.ts) — this project has no UI to disable it, so the clamp
// below is unconditional rather than porting the reference's
// `for (i<2*int(plastic))` loop literally.

const GRID_N: u32 = __GRID_N__u;
const INV_DX: f32 = __INV_DX__;
const DT: f32 = __DT__;
const DX: f32 = 1.0 / INV_DX;

// Tightest margin that keeps base=floor(pos*inv_dx-0.5) in [0, GRID_N-2]
// for every particle, always — i.e. the 3x3 stencil P2G/G2P both index
// from `base` never needs a node outside [0, GRID_N]. The reference's
// own sticky boundary (gridUpdate.wgsl, 0.05 thick) is *meant* to keep
// material this far from the edge always, by zeroing approach velocity
// there — but that only clamps the *grid's* velocity, one step late: a
// particle that gathers a large enough velocity from cells still well
// inside the domain can advect (pos += dt*v) straight past a 0.05-thick
// wall in a single dt=1e-4 step if |v| is large enough, before the wall
// ever gets a chance to stop it. Confirmed this actually happens (not
// just theoretically): a plain-JS CPU port of this exact algorithm,
// run for 3000+ frames, shows isolated particles spontaneously spiking
// to several-x their neighbors' speed out of an otherwise fully-settled
// pile (a known MPM artifact — a nearly-static particle crossing a grid
// cell boundary discretely reassigns which stencil it belongs to,
// briefly desynchronizing its stress estimate) — rare, usually small,
// but occasionally large enough to jump the wall in one step. Without
// this clamp, that particle lands outside the valid stencil range,
// P2G/G2P's own defensive bounds check (`if (ni < 0 || ...) continue`)
// then gives it zero grid interaction on every subsequent step (no
// gravity, no collisions, nothing) — permanently stranded outside the
// domain forever, not just visually wrong but actually inert. This
// clamp is what makes the sticky boundary an *actual* wall (nothing
// can ever end up outside it) instead of a strong deterrent with a
// timestep-sized gap.
const MIN_POS: f32 = DX;
const MAX_POS: f32 = 1.0 - DX;

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particleC: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> particleJp: array<f32>;
@group(0) @binding(5) var<storage, read> gridVel: array<vec2<f32>>;

// Live particle count — see p2g.wgsl's own comment on why (a fixed-
// capacity buffer with a live count, not a compile-time PARTICLE_COUNT).
@group(0) @binding(6) var<uniform> activeCount: u32;

// Same Material struct/buffer p2g.wgsl binds (see that file's own
// comment on why one shared struct) — this shader only reads
// yieldLow/yieldHigh, the plasticity clamp bounds just below.
struct Material {
  mu0: f32,
  lambda0: f32,
  hardening: f32,
  yieldLow: f32,
  yieldHigh: f32,
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
  // or both formulas vanish (S already diagonal) — see this project's
  // design notes for the worked-through case analysis.
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
  let Jp0 = particleJp[pi];

  // base = floor(y - 0.5), fx = y - base, landing in [0.5, 1.5) — see
  // p2g.wgsl's own note on this (must match exactly, since G2P has to
  // gather from the same 3x3 stencil P2G scattered into this same
  // step).
  let y = pos * INV_DX;
  let base = vec2<i32>(floor(y - vec2<f32>(0.5)));
  let fx = y - vec2<f32>(base);
  let w = quadraticWeights(fx);

  var v = vec2<f32>(0.0, 0.0);
  var C = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  for (var i: u32 = 0u; i < 3u; i = i + 1u) {
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      let ni = base.x + i32(i);
      let nj = base.y + i32(j);
      if (ni < 0 || nj < 0 || ni > i32(GRID_N) || nj > i32(GRID_N)) { continue; }

      // NOT scaled by dx here — unlike p2g's dpos, matches the reference
      // exactly (the 4*inv_dx below already carries the right units).
      let dpos = vec2<f32>(f32(i), f32(j)) - fx;
      let wgt = w[i].x * w[j].y;
      let nodeIndex = u32(ni) * (GRID_N + 1u) + u32(nj);
      let gv = gridVel[nodeIndex];
      let wgv = wgt * gv;

      v = v + wgv;
      // APIC affine velocity field: C += 4*inv_dx * outer(w*grid_v, dpos).
      C = C + (4.0 * INV_DX) * vec4<f32>(wgv.x * dpos.x, wgv.x * dpos.y, wgv.y * dpos.x, wgv.y * dpos.y);
    }
  }

  var newPos = pos + DT * v;
  var newVel = v;
  // See MIN_POS/MAX_POS's own comment: this is a hard safety net, not
  // normal-path behavior — the grid's own sticky boundary should always
  // stop a particle well before this clamp would ever engage. Zeroing
  // the overshooting velocity component (not just clamping position)
  // matches that same sticky-wall semantics applied here directly at
  // the particle level, so a caught particle doesn't just immediately
  // try to re-overshoot next step with the same velocity it arrived
  // with.
  if (newPos.x < MIN_POS || newPos.x > MAX_POS) { newVel.x = 0.0; }
  if (newPos.y < MIN_POS || newPos.y > MAX_POS) { newVel.y = 0.0; }
  newPos = clamp(newPos, vec2<f32>(MIN_POS), vec2<f32>(MAX_POS));

  var F = matMul(identityPlusScaled(C, DT), F0);

  let svd = svd2(F);
  // Bounds are the world's own Material.yieldLow/yieldHigh (main.ts's
  // "Elasticity" slider), not a hardcoded snow-tight literal — see
  // gpu/mpm.ts's yieldBounds() for what the range actually controls: how
  // much of a stretch/compression the corotated elastic term is allowed
  // to fully recover from versus how much gets baked in as permanent
  // (plastic) deformation via Jp below.
  let sigma = clamp(svd.sigma, vec2<f32>(material.yieldLow), vec2<f32>(material.yieldHigh));

  let oldJ = matDet(F);
  let sigmaMat = vec4<f32>(sigma.x, 0.0, 0.0, sigma.y);
  F = matMul(matMul(svd.u, sigmaMat), matTranspose(svd.v));
  let newJ = matDet(F);

  let JpNew = clamp(Jp0 * oldJ / newJ, 0.6, 20.0);

  particlePos[pi] = newPos;
  particleVel[pi] = newVel;
  particleF[pi] = F;
  particleC[pi] = C;
  particleJp[pi] = JpNew;
}

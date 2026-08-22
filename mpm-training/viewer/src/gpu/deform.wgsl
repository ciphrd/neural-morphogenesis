// Viewer-only "Deform" tool — while held, injects a RADIAL push/pull
// into every particle within RADIUS of the pointer (each particle's own
// direction is computed per-particle, straight toward or away from the
// click point — NOT a single uniform vector applied to every particle
// the way an earlier revision of this tool worked, see this file's own
// git history), scaled by params.strength, in one of two modes
// (params.mode, a uniform not a compile-time const so the UI's own mode
// checkbox never needs a pipeline rebuild):
//
//  - MODE_VELOCITY: adds to velocity, letting the existing
//    P2G -> gridUpdate -> G2P momentum transfer produce a real,
//    physically-propagating deformation over subsequent substeps — same
//    "route it through the normal physics" reasoning interact.wgsl's own
//    applyDrag experiment already explored for the "Move" tool. Safe
//    (can't corrupt material state) but a force, not an instant reshape.
//  - MODE_DEFORMATION: a direct, immediate edit to the deformation
//    gradient F itself — a LEFT-multiplicative stretch along each
//    particle's own radial direction, composed the EXACT same way
//    core/g2p.wgsl's own G2P integrator composes its own F updates
//    (F_new = (I + s*C) * F_old): F_new = (I + s*dir⊗dir) * F_old.
//    Bypasses physics propagation entirely for an instant reshape — this
//    is "the deformation" in the literal MLS-MPM sense (an actual
//    radial expansion/contraction of material), not a force that
//    produces one.
//
// params.outward selects which way along that per-particle radial line
// the push goes: outward>=0.5 pushes each particle AWAY from the click
// point (an "explosion" — MODE_DEFORMATION here reads as local
// expansion/stretching outward), otherwise it pulls each particle TOWARD
// the click point (an "implosion" — MODE_DEFORMATION here reads as local
// compression). A particle sitting exactly at the click point (dist~0)
// has no well-defined radial direction — see injectDeform()'s own
// dist>1e-6 guard, which leaves it unaffected rather than producing a
// NaN from normalizing a zero vector.
//
// Lives outside ../core/ for the same reason interact.wgsl does (see
// that file's own module docstring): purely interactive tooling the
// headless training path never needs to pay for.
//
// Dispatched once per RENDERED FRAME while the tool is held (see
// render/GridCanvas.tsx's own RAF loop) — same cadence interact.wgsl's
// own applyDrag runs at, NOT a one-shot click the way this used to work.
// Each dispatch is a full, independent injection: MODE_VELOCITY's own
// contribution simply ADDS every frame (ordinary impulse accumulation,
// same as any other force applied over time), but MODE_DEFORMATION's
// own left-multiplicative stretch COMPOUNDS — holding for N frames
// stretches by roughly (1+s)^N along the radial direction, not N*s the
// way velocity would, so the same strength that felt gentle as a single
// click can escalate fast once held; there's no per-frame rescaling
// here to compensate, since "how strong per frame" is exactly what the
// existing strength slider already controls; retune it live if holding
// feels too aggressive at whatever value clicking alone was tuned to.
// Smoothstep falloff from full strength at the pointer to EXACTLY zero
// at params.radius — deliberately bounded there (not an unbounded
// Gaussian the way core/repulsion.wgsl's own splat is) so the effect's
// own true extent matches the UI's own preview circle exactly, not just
// approximately.
// Toroidal-aware: uses the same minimum-image (shortest distance across
// the domain seam) convention core/repulsion.wgsl's own splatDensity()
// already documents, since MpmCore's own domain wraps.
//
// matMul/identityPlusScaled below are a small, deliberate DUPLICATE of
// core/g2p.wgsl's own versions — same convention that file itself
// already follows relative to core/p2g.wgsl (WGSL has no cross-module
// share mechanism, and these are cheap enough not to bother routing
// through a shared binding scheme just to avoid four lines of
// repetition).

const MODE_VELOCITY: u32 = 0u;
const MODE_DEFORMATION: u32 = 1u;

// Starting-guess scale factors, NOT tuned — params.strength needs a very
// different effective magnitude depending on which mode it drives
// (velocity routinely reaches several units in this project's own
// domain — see trainer/simulation_settings.py's own MAX_STRAFE — while a
// deformation-gradient stretch visibly misbehaves well before O(1)), so
// each mode scales the same user-facing strength scalar by its own
// constant rather than asking the UI to expose two different numeric
// ranges behind one slider. Expect to re-tune both live.
const VELOCITY_SCALE: f32 = 40.0;
const DEFORMATION_SCALE: f32 = 1.5;

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> activeCount: u32;
@group(0) @binding(2) var<uniform> clickPos: vec2<f32>;
@group(0) @binding(3) var<storage, read_write> particleVel: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> particleF: array<vec4<f32>>;

struct DeformParams {
  strength: f32,
  radius: f32,
  // Uniform floats (not separate bindings) purely to keep this struct
  // one tidy 16-byte block — both compared against a 0.5 threshold
  // below, deform.ts's own inject() writes 0.0/1.0 for each.
  outward: f32,
  mode: f32,
}
@group(0) @binding(5) var<uniform> params: DeformParams;

fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.x + a.y * b.z,
    a.x * b.y + a.y * b.w,
    a.z * b.x + a.w * b.z,
    a.z * b.y + a.w * b.w
  );
}

fn identityPlusScaled(m: vec4<f32>, s: f32) -> vec4<f32> {
  return vec4<f32>(1.0 + s * m.x, s * m.y, s * m.z, 1.0 + s * m.w);
}

@compute @workgroup_size(64)
fn injectDeform(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi = gid.x;
  if (pi >= activeCount) { return; }

  var delta = particlePos[pi] - clickPos;
  delta.x = delta.x - round(delta.x);
  delta.y = delta.y - round(delta.y);
  let dist = length(delta);
  if (dist > params.radius) { return; }

  let t = clamp(1.0 - dist / max(params.radius, 1e-6), 0.0, 1.0);
  let falloff = t * t * (3.0 - 2.0 * t);

  // Per-particle radial direction — straight away from the click point
  // (delta/dist), or negated for an inward pull. dist~0 (a particle
  // sitting right at the click point) has no defined radial direction —
  // left at the zero vector rather than dividing by ~0, so that particle
  // simply isn't pushed (its own contribution to any visible effect is
  // negligible either way, at the exact singular center of the tool).
  var dir = vec2<f32>(0.0, 0.0);
  if (dist > 1e-6) {
    dir = delta / dist;
  }
  if (params.outward < 0.5) {
    dir = -dir;
  }

  if (params.mode < 0.5) {
    particleVel[pi] = particleVel[pi] + dir * params.strength * VELOCITY_SCALE * falloff;
  } else {
    let stretch = identityPlusScaled(
      vec4<f32>(dir.x * dir.x, dir.x * dir.y, dir.y * dir.x, dir.y * dir.y),
      params.strength * DEFORMATION_SCALE * falloff
    );
    particleF[pi] = matMul(stretch, particleF[pi]);
  }
}

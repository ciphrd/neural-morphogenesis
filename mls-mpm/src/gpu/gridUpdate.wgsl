// Converts p2g.wgsl's accumulated (fixed-point) momentum/mass into a
// plain float velocity per grid node, applies gravity, and applies the
// reference's two boundary rules — direct port of advance()'s "For all
// grid nodes" loop in mls-mpm88-explained.cpp.

const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);
const DT: f32 = __DT__;
const SCALE: f32 = 65536.0; // must match p2g.wgsl's own SCALE
const BOUNDARY: f32 = 0.05; // boundary thickness, in [0,1] domain units

// gridAccum's channel layout — must match p2g.wgsl/clearGrid.wgsl/
// field.wgsl's own copies (WGSL has no #include). This shader only ever
// reads CH_MOM_X/CH_MOM_Y/CH_MASS; the field-visualize channels
// (CH_J/CH_SHEAR/CH_PRESSURE) are field.wgsl's own to read.
const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CHANNELS: u32 = 6u;

// Not in the reference (which has zero numerical dissipation anywhere —
// see g2p.wgsl's own MIN_POS/MAX_POS comment for what that costs it at
// rest: isolated particles spontaneously popping out of an otherwise
// fully-settled pile, a known artifact of quadratic-B-spline MPM at
// grid-cell-crossing events, confirmed happening in this exact port via
// a long CPU-side run). A small, deliberate deviation to drain that
// residual energy before it accumulates into a visible pop, rather than
// letting it circulate indefinitely.
//
// Live-adjustable (main.ts's "Damping" slider), unlike GRID_N/DT above —
// a uniform, not a const, gpu/mpm.ts's setDamping() converts the
// slider's own per-*rendered-frame* percentage into this per-*substep*
// multiplier (see that function's docstring for why the conversion has
// to happen there, not here). NOT a free stability improvement at any
// value: confirmed live that pushing this well past its own default to
// calm a body's at-rest jitter also measurably destabilizes it under
// active manipulation (a dragged region's velocity is set directly by
// the Move tool regardless of this value — see MODE_MOVE below — so
// more damping here only slows how fast the *rest* of the body can
// mechanically follow, widening the velocity mismatch, and therefore
// the internal stress, right at the drag boundary where stress is
// already highest). Strain-rate-proportional (Rayleigh-style) damping in
// the stress term was tried as an alternative and removed — it damps
// hardest exactly at that same drag-boundary discontinuity (a real local
// strain rate, not noise), making it structurally worse there, not
// better. This slider is the cheap, immediate lever, not a complete fix.
@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> gridVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> gravity: f32;
@group(0) @binding(4) var<uniform> damping: f32;

// Mouse interaction — every grid node within `radius` of the cursor's
// own domain position is affected, falling off linearly to 0 at the
// radius edge (`falloff`, computed once below and shared by both modes).
// Two distinct interaction modes, not just a signed strength, because
// they're fundamentally different operations on `v`, not two signs of
// the same one:
//  - MODE_FORCE (1): the original attract/repel — a genuine force
//    (scaled by DT, same as gravity above), ADDED to `v`, signed
//    `strength` picking pull-toward (+) vs push-away (-). Composes with
//    gravity/momentum rather than fighting them, but that's also why it
//    reads as "abrupt": a large constant force (main.ts's MOUSE_STRENGTH)
//    dumped in all at once on mousedown, independent of how the mouse is
//    actually moving.
//  - MODE_MOVE (2): a kinematic drag — `v` is blended (mix, by falloff)
//    *toward* `mouse.vel` instead of having something added to it.
//    main.ts computes `mouse.vel` from the cursor's own actual on-screen
//    motion since the last frame (zero on the very first frame after
//    mousedown, so grabbing never itself causes a jump), so grabbed
//    material can only ever move as fast as the cursor is actually
//    moving — self-limiting, no separate strength constant that can be
//    "too strong" for a given grab. At the falloff=1 point (right at the
//    cursor) this fully overrides whatever v already was (gravity,
//    residual momentum, ...), so grabbed material tracks the cursor
//    precisely rather than fighting its own inertia.
const MODE_OFF: f32 = 0.0;
const MODE_FORCE: f32 = 1.0;
const MODE_MOVE: f32 = 2.0;

struct Mouse {
  pos: vec2<f32>,
  vel: vec2<f32>,
  strength: f32,
  radius: f32,
  mode: f32,
}
@group(0) @binding(3) var<uniform> mouse: Mouse;

@compute @workgroup_size(64)
fn gridUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NODE_COUNT) { return; }

  let base = idx * CHANNELS;
  let mass = f32(atomicLoad(&gridAccum[base + CH_MASS])) / SCALE;
  if (mass <= 0.0) {
    gridVel[idx] = vec2<f32>(0.0, 0.0);
    return;
  }

  let momX = f32(atomicLoad(&gridAccum[base + CH_MOM_X])) / SCALE;
  let momY = f32(atomicLoad(&gridAccum[base + CH_MOM_Y])) / SCALE;
  var v = (vec2<f32>(momX, momY) / mass) * damping;
  v.y = v.y - DT * gravity;

  // nodeIndex = i*(GRID_N+1) + j, matching p2g.wgsl's own flat-index
  // convention — i tracks the position's x-axis cell, j the y-axis.
  let i = idx / (GRID_N + 1u);
  let j = idx % (GRID_N + 1u);
  let x = f32(i) / f32(GRID_N);
  let y = f32(j) / f32(GRID_N);

  if (mouse.mode != MODE_OFF) {
    let toMouse = mouse.pos - vec2<f32>(x, y);
    let dist = length(toMouse);
    if (dist < mouse.radius) {
      let falloff = 1.0 - dist / mouse.radius;
      if (mouse.mode == MODE_FORCE && dist > 1e-5) {
        v = v + (toMouse / dist) * (mouse.strength * falloff * DT);
      } else if (mouse.mode == MODE_MOVE) {
        v = mix(v, mouse.vel, falloff);
      }
    }
  }

  // Sticky boundary: fully zero velocity near 3 of the 4 walls.
  if (x < BOUNDARY || x > 1.0 - BOUNDARY || y > 1.0 - BOUNDARY) {
    v = vec2<f32>(0.0, 0.0);
  }
  // Separate boundary (the floor): may leave the floor (v.y>0) or sit at
  // rest, but never sink into it (v.y<0 gets clamped to 0) — the one
  // asymmetric wall, matching the reference's `g[1]=max(0,g[1])` exactly.
  if (y < BOUNDARY) {
    v.y = max(0.0, v.y);
  }

  gridVel[idx] = v;
}

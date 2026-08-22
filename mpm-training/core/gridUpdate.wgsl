// Converts p2g.wgsl's accumulated (fixed-point) momentum/mass into a
// plain float velocity per grid node and applies gravity — direct port
// of advance()'s "For all grid nodes" loop in mls-mpm88-explained.cpp,
// minus that reference's own two wall/floor boundary rules: this domain
// is toroidal (see p2g.wgsl's/g2p.wgsl's own module docstrings for the
// wraparound stencil indexing that makes that true), so there is no wall
// for a grid node to sit near in the first place — nothing here needs to
// special-case "close to an edge" the way the reference (and this
// project's own earlier, walled version) did.
//
// Independent copy of mls-mpm/src/gpu/gridUpdate.wgsl (this project's own
// sandbox), stripped of that file's Mouse-driven interaction (MODE_FORCE/
// MODE_MOVE) and its own wall/floor boundary — there is no mouse here,
// and no walls, only gravity/damping.

const GRID_N: u32 = __GRID_N__u;
const NODE_COUNT: u32 = (GRID_N + 1u) * (GRID_N + 1u);
const DT: f32 = __DT__;
const SCALE: f32 = 65536.0; // must match p2g.wgsl's own SCALE

// gridAccum's channel layout — must match p2g.wgsl/clearGrid.wgsl's own
// copies (WGSL has no #include).
const CH_MOM_X: u32 = 0u;
const CH_MOM_Y: u32 = 1u;
const CH_MASS: u32 = 2u;
const CHANNELS: u32 = 3u;

// Not in the reference (which has zero numerical dissipation anywhere) —
// a small, deliberate deviation to drain residual energy before it
// accumulates into a visible pop (a known artifact of quadratic-B-spline
// MPM at grid-cell-crossing events), rather than letting it circulate
// indefinitely. Live-adjustable, unlike GRID_N/DT above — a uniform, not
// a const.
@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> gridVel: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> gravity: f32;
@group(0) @binding(3) var<uniform> damping: f32;

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

  gridVel[idx] = v;
}

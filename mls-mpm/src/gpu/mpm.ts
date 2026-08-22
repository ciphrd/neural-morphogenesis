import type { SceneData } from "../worlds/types"
import attractSrc from "./attract.wgsl?raw"
import clearGridSrc from "./clearGrid.wgsl?raw"
import g2pSrc from "./g2p.wgsl?raw"
import { ceilDiv, writeBuffer } from "./gpuUtil"
import gridUpdateSrc from "./gridUpdate.wgsl?raw"
import p2gSrc from "./p2g.wgsl?raw"
import repulsionSrc from "./repulsion.wgsl?raw"
import { templateShader } from "./shaderTemplate"

// Grid is GRID_N cells per axis, i.e. (GRID_N+1)^2 *nodes*. The
// reference's own value is 80 — this project's is now 512 (bumped by
// hand, not re-derived from anything below).
export const GRID_N = 64

// DT is NOT the reference's own 1e-4 held constant across that GRID_N
// change — it can't be. p2g.wgsl's stress term scales with
// Dinv=4*inv_dx^2, i.e. with 1/dx^2; explicit-time-integration stability
// for that term needs dt to shrink *linearly* with dx (the standard
// elastic-wave CFL condition, dt <~ dx/wave_speed — wave_speed is a
// material property, independent of grid resolution, so a smaller dx
// directly demands a smaller dt to keep the same margin). Holding DT at
// the reference's 1e-4 while only bumping GRID_N (dx shrank 512/80=6.4x)
// is NOT a milder version of the same simulation, it's a qualitatively
// different, unstable one — confirmed live: at GRID_N=512/DT=1e-4, even
// worlds/blocks.ts's own already-verified-stable configuration explodes
// into a scattered gas within ~2 seconds instead of settling. Scaling DT
// down by that same 6.4x (BASELINE_GRID_N/GRID_N) restores the original
// ratio this project's whole stability history (hardening, damping,
// position-clamp — see g2p.wgsl/gridUpdate.wgsl's own comments) was
// actually verified against. DEFAULT_SUBSTEPS is scaled up by the same
// factor so a rendered frame still advances the same amount of
// *simulated* time as before (substeps*DT held ~constant at 1e-3),
// rather than everything just moving in slow motion at the new DT.
const BASELINE_GRID_N = 80
const BASELINE_DT = 1e-4
export const DT = BASELINE_DT * (BASELINE_GRID_N / GRID_N)
export const DX = 1 / GRID_N
export const INV_DX = GRID_N
export const PARTICLE_MASS = 1.0
export const VOL = 1.0

// Fixed GPU buffer capacity, decoupled from any one world's own particle
// count (see worlds/types.ts's own SceneData.count) — this is what lets
// main.ts switch worlds (or, later, inject new particles at runtime)
// without ever touching a GPUBuffer's size or recreating a pipeline,
// only a small activeCount uniform (see p2g.wgsl/g2p.wgsl's own comment
// on it) and a writeBuffer into the head of each particle buffer. Sized
// generously past every world's own count defined so far (worlds/*.ts),
// with headroom for particle injection.
export const MAX_PARTICLES = 200_000

export const DEFAULT_GRAVITY = 0
// reference's frame_dt/dt = 1e-3/1e-4 = 10; scaled by the same
// BASELINE_GRID_N/GRID_N factor DT itself was, so substeps*DT (one
// rendered frame's worth of simulated time) stays ~1e-3 regardless of
// GRID_N — see DT's own comment above for why both had to move together.
export const DEFAULT_SUBSTEPS = Math.round(10 * (GRID_N / BASELINE_GRID_N))
export const DEFAULT_E = 1e4 // Young's modulus — stiffness
export const DEFAULT_NU = 0.2 // Poisson's ratio — 0 = freely compressible, ->0.5 = incompressible
// The reference's own value (10.0) is NOT this project's default — see
// index.html's hardening slider comment for the empirical investigation
// behind why: at this scene's own particle density (worlds/blocks.ts —
// several times the reference's own, spread across more, closer-packed
// blobs), hardening=10 lets isolated settled particles periodically pop
// with a sudden burst of speed out of an otherwise dead-calm pile
// (confirmed via a long CPU-side port of this exact algorithm — a real
// numerical artifact, not a rendering glitch). 3.0 stayed clean through
// the same test; 5.0 already popped. This is a genuine behavior change
// from the reference (material hardens far less under compression now),
// not just a stability tweak with no visible cost — chosen because a
// silently-corrupting simulation is worse than a less dramatic snow
// hardening effect.
export const DEFAULT_HARDENING = 3.0

// Grid-velocity damping — see gridUpdate.wgsl's own comment on the
// mechanism and its known tradeoff (calms at-rest jitter, but also
// measurably destabilizes active manipulation, confirmed live — this is
// a real lever, not a strictly-better-at-any-value fix). Expressed here
// as a fraction of velocity lost per *rendered frame* (0..~1), not the
// raw per-*substep* multiplier gridUpdate.wgsl actually wants — a
// per-substep value like 0.999 is meaningless on its own (its real
// effect depends entirely on how many substeps run per frame, which
// main.ts's own Substeps slider can change independently), whereas "6%
// of velocity lost per frame" means the same thing regardless. DEFAULT
// is computed FROM the project's original hardcoded per-substep
// constant (0.999) at DEFAULT_SUBSTEPS, not chosen freshly — so worlds
// that don't override it (worlds/blocks.ts, disc.ts) reproduce that
// exact original behavior, bit for bit, rather than an approximation.
export const DEFAULT_DAMPING = 1 - 0.999 ** DEFAULT_SUBSTEPS

/** Converts the slider's own per-rendered-frame loss fraction into the
 * per-substep multiplier gridUpdate.wgsl's `damping` uniform actually
 * wants: `perSubstep^substeps` must equal `1 - lossFraction`, so
 * `perSubstep = (1 - lossFraction)^(1/substeps)`. Takes `substeps`
 * explicitly (not read from anywhere internal) because main.ts's own
 * Substeps slider can change independently of this one — recomputing
 * from both live values on every call is what keeps "X% lost per frame"
 * actually meaning X% per frame regardless of the current substep count. */
export function perSubstepDamping(
  lossFraction: number,
  substeps: number
): number {
  const clamped = Math.min(Math.max(lossFraction, 0), 0.999)
  return (1 - clamped) ** (1 / Math.max(substeps, 1))
}

// Elastic yield range — how far a particle's local stretch/compression
// (an SVD singular value of F, ~1 = undeformed) can go before g2p.wgsl's
// plasticity clamp kicks in and bakes the excess in permanently (via Jp)
// instead of letting the corotated elastic stress term pull it back. The
// reference's own bounds (sigma in [0.975, 1.0075]) are snow-tight — the
// clamp engages almost immediately, which is exactly right for snow
// (worlds/blocks.ts keeps this at elasticity=0, unchanged behavior) but
// wrong for anything meant to deform and *spring back*: at that range,
// ordinary manipulation (worlds/organism.ts's whole reason for existing)
// gets treated as plastic overflow and never recovers. `elasticity`
// (0..1) is the one slider-facing knob for this — 0 reproduces the
// reference's own bounds exactly, 1 widens them to [0.5, 2.0] (half to
// double length along a principal axis before the safety clamp engages
// at all) — linear in between. Still a real clamp, not fully disabled,
// at either end: some yield bound is what keeps a WGSL SVD-based
// corotated model from a truly unbounded stress blowup under extreme,
// deliberately-adversarial stretching (see g2p.wgsl's own MIN_POS/
// MAX_POS comment for this project's general stance on "keep a real
// safety net even where the normal path should never need it").
export const DEFAULT_ELASTICITY = 0
const SNOW_YIELD_LOW = 1.0 - 2.5e-2
const SNOW_YIELD_HIGH = 1.0 + 7.5e-3
const WIDE_YIELD_LOW = 0.5
const WIDE_YIELD_HIGH = 2.0
export function yieldBounds(elasticity: number): {
  yieldLow: number
  yieldHigh: number
} {
  const t = Math.min(Math.max(elasticity, 0), 1)
  return {
    yieldLow: SNOW_YIELD_LOW + t * (WIDE_YIELD_LOW - SNOW_YIELD_LOW),
    yieldHigh: SNOW_YIELD_HIGH + t * (WIDE_YIELD_HIGH - SNOW_YIELD_HIGH),
  }
}

// Mouse interaction defaults — see gridUpdate.wgsl's own Mouse struct
// comment for MODE_FORCE vs MODE_MOVE's very different semantics.
// FORCE_STRENGTH is well above gravity's own magnitude (200) specifically
// so the cursor reads as clearly dominant over gravity while active —
// dt*STRENGTH per substep at the cursor's own position (falloff=1) is a
// 0.5 velocity-unit nudge, versus gravity's dt*200=0.02. MOVE_RADIUS/
// FORCE_RADIUS are separate constants (not one shared MOUSE_RADIUS)
// since the two tools have different visual footprints in practice: a
// force strong enough to matter at FORCE_RADIUS's edge would be far too
// strong right at the cursor, whereas MODE_MOVE's mix()-based falloff
// has no such tension (see that struct's own comment on why).
export const MOUSE_FORCE_STRENGTH = 5000
export const MOUSE_FORCE_RADIUS = 0.15
export const MOUSE_MOVE_RADIUS = 0.08

// Particle-particle repulsion — see repulsion.wgsl's own header for the
// splat -> texture -> gradient mechanism. Deliberately its own field,
// decoupled from GRID_N (see that file's own comment on why): its
// resolution is this project's first ever LIVE-changeable grid size —
// unlike GRID_N, which is a build-time constant nothing rebuilds,
// changing this triggers MpmSimulation.setFieldResolution() to destroy
// and recreate the densityAccum buffer/texture/pipelines/bind groups at
// the new size, plus a matching MpmRenderer.rebuildRepulsionDisplay()
// call (render.ts's own display bind group goes stale otherwise) — see
// both methods' own docstrings. A rare, deliberate "rebuild," not a
// per-frame operation, same tradeoff every other resize-driven
// recreation in WebGPU makes.
export const FIELD_RESOLUTIONS = [64, 128, 256, 512, 1024] as const
export type FieldResolution = (typeof FIELD_RESOLUTIONS)[number]
export const DEFAULT_FIELD_RESOLUTION: FieldResolution = 256
// Domain-space Gaussian sigma ([0,1] units) for each particle's own
// splat — see repulsion.wgsl's own splatDensity comment for the texel-
// space conversion and its MAX_KERNEL_RADIUS_TEXELS clamp.
export const DEFAULT_SPLAT_RADIUS = 0.006
// Force-scale multiplier on the density field's own gradient, written
// directly to particlePos as well as particleVel (see repulsion.wgsl's
// own applyRepulsion for why: a velocity-only nudge gets erased by
// gridUpdate.wgsl's own mass-weighted averaging for particles closer
// together than one MPM grid cell). That direct position write has NO
// decay mechanism of its own, unlike a velocity nudge (which
// gridUpdate.wgsl's Damping slider bleeds off every substep) — a
// splatDensity Gaussian never reaches exactly zero, so without an
// explicit cutoff the resulting push never fully stops either, and
// compounds every substep. Runs every substep, always on — an
// insert-triggered-burst-only variant was tried and reverted at the
// user's own request (they want this continuously active, not just a
// spawn-time declump pass). Tuned live at that cadence: 0.1 (with
// index.html's own 0-0.15 slider range) keeps an already-settled world's
// own dispersal slow/gentle, not stopped — see repulsion.wgsl's own
// applyRepulsion comment for the actual fix (a density threshold below
// which the force is exactly zero), not yet implemented.
export const DEFAULT_REPULSION_STRENGTH = 0.1

// "Attract to Point" tool (tools/types.ts) — see attract.wgsl's own
// header for the 2-click pick/highlight/commit/apply mechanism. A fixed,
// small capacity (not a slider-adjustable one — this project's own
// MAX_PARTICLES isn't either) since each entry only ever comes from a
// deliberate 2-click gesture, not something that grows on its own the
// way particle count does; 32 simultaneous attractors is generous for
// interactive use. Once full, MpmSimulation.commitAttractorAt() silently
// no-ops on further picks — same "truncate, don't crash" stance as
// addParticles() takes at MAX_PARTICLES.
export const MAX_ATTRACTORS = 32
// Rate constant for attract.wgsl's own applyAttraction — see that
// function's own comment for why it's an exponential approach (a direct
// particlePos write closing a fraction of the remaining distance every
// substep, fraction = attractStrength*DT) rather than a constant-speed
// pull; that shape is what makes it decelerate and settle smoothly with
// no separate damping term needed, regardless of this constant's size.
// 4 (this constant's first tuned value) turned out too weak in a dense/
// cohesive world (worlds/blocks.ts): confirmed live the picked particle
// barely inched toward its target in 10+ seconds, needing dozens of
// neighbors' own elastic resistance to be overcome one tiny step at a
// time. Retuned live to 30 — still a clean, non-oscillating settle
// (attractStrength*DT stays far below the ~1 point where a single step
// would overshoot the target and start bouncing), just fast enough
// (~1s) to visibly win against a stiff block's own cohesion instead of
// reading as "stuck."
export const DEFAULT_ATTRACT_STRENGTH = 30

/** mu0/lambda0 — the Lamé parameters p2g.wgsl's Material uniform actually
 * wants — computed here, host-side, from the sliders' own Young's-
 * modulus/Poisson's-ratio units (matching the reference's one-time
 * startup computation) specifically so the shader itself never divides
 * by (1-2*nu): doing that conversion here, once on change, is both
 * safer and cheaper than every one of MAX_PARTICLES threads redoing an
 * increasingly ill-conditioned division every substep as a
 * slider-controlled nu approaches 0.5 (main.ts caps its range short of
 * that, but still close enough for (1-2*nu) to get small). */
export function lameParams(
  E: number,
  nu: number
): { mu0: number; lambda0: number } {
  return {
    mu0: E / (2 * (1 + nu)),
    lambda0: (E * nu) / ((1 + nu) * (1 - 2 * nu)),
  }
}

const NODE_COUNT = (GRID_N + 1) * (GRID_N + 1)
const WORKGROUP = 64
// momX, momY, mass, J-sum, shear-sum, pressure-sum — must match
// clearGrid.wgsl/p2g.wgsl/gridUpdate.wgsl/field.wgsl's own CHANNELS.
const GRID_ACCUM_CHANNELS = 6
// repulsion.wgsl's own densityToTexture workgroup_size(16,16,1).
const FIELD_WORKGROUP = 16

/** Owns every GPU resource for the MLS-MPM simulation itself (not
 * rendering — see render.ts) and the one `step(substeps)` entry point
 * that runs `substeps` full advance() iterations in a single submitted
 * command buffer. Mirrors mls-mpm88-explained.cpp's advance(): clear
 * grid -> P2G -> grid update (velocity + gravity + boundary + mouse
 * tool) -> G2P (velocity/affine-C gather, advection, F-update,
 * plasticity), each its own compute pass — NOT chained within one pass.
 * That distinction matters: WebGPU does not guarantee visibility of one
 * dispatch's storage-buffer writes to the next dispatch *within the same
 * compute pass* the way it does across pass boundaries — sibling project
 * envnca's gpu/simulation.ts hit exactly this as a real, silent-
 * corruption bug (P2G's atomic writes racing clearGrid's), so every
 * stage below gets its own begin/end pass from the start.
 *
 * Particle buffers are sized to MAX_PARTICLES (fixed capacity), not to
 * whatever world is currently loaded — `loadScene()` (constructor and
 * world-switch/reset both funnel through it) only ever writes into the
 * *head* of each buffer and updates the small `activeCount` uniform
 * p2g.wgsl/g2p.wgsl gate their per-particle work on, so switching worlds
 * (even to one with a different particle count) never touches a
 * GPUBuffer's size or recreates a pipeline. */
export class MpmSimulation {
  private readonly device: GPUDevice

  readonly positions: GPUBuffer
  readonly velocities: GPUBuffer
  readonly colors: GPUBuffer
  private readonly F: GPUBuffer
  private readonly C: GPUBuffer
  private readonly Jp: GPUBuffer

  // Momentum-x/y + mass + the field-visualize diagnostics (J-sum/shear-
  // sum/pressure-sum — main.ts's "Field" dropdown) all packed into ONE
  // fixed-point buffer, channel-indexed (nodeIndex*6+channel) rather than
  // one GPUBuffer per channel — see p2g.wgsl's own comment for why: 6
  // separate storage-buffer bindings would exceed WebGPU's 8-per-stage
  // cap once stacked against p2g.wgsl's 5 read-only particle buffers
  // (confirmed live as a real validation error, not a hypothetical).
  // Not private (pure P2G scratch otherwise) — render.ts's field-
  // visualize colorize pass reads it directly (this still holds this
  // step's post-P2G fixed-point values, right up until the next
  // substep's clearGrid() zeroes it again).
  readonly gridAccum: GPUBuffer
  readonly gridVel: GPUBuffer
  private readonly gravityUniform: GPUBuffer
  private readonly materialUniform: GPUBuffer
  private readonly mouseUniform: GPUBuffer
  private readonly activeCountUniform: GPUBuffer
  private readonly dampingUniform: GPUBuffer

  private readonly clearGridPipeline: GPUComputePipeline
  private readonly p2gPipeline: GPUComputePipeline
  private readonly gridUpdatePipeline: GPUComputePipeline
  private readonly g2pPipeline: GPUComputePipeline

  private readonly clearGridBindGroup: GPUBindGroup
  private readonly p2gBindGroup: GPUBindGroup
  private readonly gridUpdateBindGroup: GPUBindGroup
  private readonly g2pBindGroup: GPUBindGroup

  private readonly gridDispatch: number
  // How many of MAX_PARTICLES are actually live — see loadScene(). Read
  // by render.ts's draw call (see the `activeCount` getter) and by this
  // class's own step() to size the P2G/G2P dispatch.
  private activeParticleCount = 0

  // Repulsion (see repulsion.wgsl's own header) — everything below is
  // reassigned, not readonly, by setFieldResolution()'s rebuild (see that
  // method's own docstring for why: FIELD_N is baked into the shader
  // module at compile time, same as GRID_N elsewhere, so a resolution
  // change needs fresh pipelines/bind groups, not just a buffer resize).
  // densityTexture is NOT private — render.ts's own "Distance field"
  // display quad samples it directly (same reasoning as gridAccum/
  // gridVel above), and needs to re-read it after every rebuild via its
  // own rebuildRepulsionDisplay().
  private densityAccum!: GPUBuffer
  densityTexture!: GPUTexture
  private splatParamsUniform!: GPUBuffer
  private repulsionParamsUniform!: GPUBuffer
  private clearDensityPipeline!: GPUComputePipeline
  private splatDensityPipeline!: GPUComputePipeline
  private densityToTexturePipeline!: GPUComputePipeline
  private applyRepulsionPipeline!: GPUComputePipeline
  private clearDensityBindGroup!: GPUBindGroup
  private splatDensityBindGroup!: GPUBindGroup
  private densityToTextureBindGroup!: GPUBindGroup
  private applyRepulsionBindGroup!: GPUBindGroup
  private densityClearDispatch = 0
  private densityTextureDispatch: readonly [number, number] = [0, 0]
  private fieldResolution: FieldResolution = DEFAULT_FIELD_RESOLUTION

  // "Attract to Point" tool (see attract.wgsl's own header). Unlike
  // repulsion's own resources above, these never need rebuilding — their
  // sizes (MAX_ATTRACTORS) are fixed constants, not a live control the
  // way Field Resolution is — so they're created once in the constructor
  // and left readonly.
  private readonly pickResultBuffer: GPUBuffer
  // Not private — render.ts's own marker-overlay pass (see that file's
  // own comment on why it exists: with no depth buffer, a highlighted
  // particle's own color can be invisibly overdrawn by later-indexed
  // neighbors in a dense pack) reads each committed attractor's own
  // particleIndex directly, every frame, to look up its CURRENT position.
  readonly attractorsBuffer: GPUBuffer
  private readonly pickPosUniform: GPUBuffer
  private readonly commitTargetUniform: GPUBuffer
  private readonly attractorSlotUniform: GPUBuffer
  private readonly attractorCountUniform: GPUBuffer
  private readonly attractStrengthUniform: GPUBuffer
  private readonly pickParticlePipeline: GPUComputePipeline
  private readonly highlightPickedPipeline: GPUComputePipeline
  private readonly commitAttractorPipeline: GPUComputePipeline
  private readonly applyAttractionPipeline: GPUComputePipeline
  private readonly pickParticleBindGroup: GPUBindGroup
  private readonly highlightPickedBindGroup: GPUBindGroup
  private readonly commitAttractorBindGroup: GPUBindGroup
  private readonly applyAttractionBindGroup: GPUBindGroup
  // How many of MAX_ATTRACTORS slots are actually committed — tracked
  // host-side (not read back from the GPU) since every change to it
  // comes from a discrete, JS-visible mouse click (pickParticleAt/
  // commitAttractorAt below), never from anything computed on the GPU
  // itself. Sizes applyAttraction's own dispatch in step() and gates
  // commitAttractorAt() once MAX_ATTRACTORS is reached. Backing field for
  // the public attractorCount getter below (render.ts's own marker-
  // overlay draw call needs to read this too, to size its own instance
  // count).
  private _attractorCount = 0

  get attractorCount(): number {
    return this._attractorCount
  }

  constructor(device: GPUDevice, scene: SceneData) {
    this.device = device

    this.positions = device.createBuffer({
      size: MAX_PARTICLES * 2 * 4,
      // COPY_SRC added for main.ts's own TEMP DEBUG HOOK
      // (window.__mpmDebug.readPositions()) — nothing else reads this
      // buffer back to the host normally, so this was never needed
      // before. Purely additive (an extra allowed usage costs nothing at
      // runtime); revert alongside that hook once the cross-check is done.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.velocities = device.createBuffer({
      size: MAX_PARTICLES * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.F = device.createBuffer({
      size: MAX_PARTICLES * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.C = device.createBuffer({
      size: MAX_PARTICLES * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.Jp = device.createBuffer({
      size: MAX_PARTICLES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.colors = device.createBuffer({
      size: MAX_PARTICLES * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    this.gridAccum = device.createBuffer({
      size: NODE_COUNT * GRID_ACCUM_CHANNELS * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    this.gridVel = device.createBuffer({
      size: NODE_COUNT * 2 * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    this.gravityUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setGravity(DEFAULT_GRAVITY)
    // Material: mu0 + lambda0 + hardening + yieldLow + yieldHigh = 5
    // floats / 20 bytes — see p2g.wgsl's and g2p.wgsl's own (identical)
    // Material struct declarations for the WGSL-side layout this must
    // match exactly (p2g reads mu0/lambda0/hardening, g2p only
    // yieldLow/yieldHigh — see setMaterial()'s own docstring for why they
    // still share one buffer/struct rather than two separate ones).
    this.materialUniform = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setMaterial(DEFAULT_E, DEFAULT_NU, DEFAULT_HARDENING, DEFAULT_ELASTICITY)
    // Mouse: pos(vec2) + vel(vec2) + strength(f32) + radius(f32) +
    // mode(f32) + pad(f32) = 8 floats / 32 bytes — see gridUpdate.wgsl's
    // own Mouse struct for the WGSL-side layout this must match exactly.
    this.mouseUniform = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setMouse({
      x: 0,
      y: 0,
      velX: 0,
      velY: 0,
      strength: 0,
      radius: 0,
      mode: 0,
    })
    this.activeCountUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.dampingUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setDamping(DEFAULT_DAMPING, DEFAULT_SUBSTEPS)

    const templateVars = { GRID_N, DX, INV_DX, DT, PARTICLE_MASS, VOL }

    const clearGridModule = device.createShaderModule({
      code: templateShader(clearGridSrc, { GRID_N }),
    })
    this.clearGridPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: clearGridModule, entryPoint: "clearGrid" },
    })
    this.clearGridBindGroup = device.createBindGroup({
      layout: this.clearGridPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.gridAccum } }],
    })

    const p2gModule = device.createShaderModule({
      code: templateShader(p2gSrc, templateVars),
    })
    this.p2gPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: p2gModule, entryPoint: "p2g" },
    })
    this.p2gBindGroup = device.createBindGroup({
      layout: this.p2gPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.positions } },
        { binding: 1, resource: { buffer: this.velocities } },
        { binding: 2, resource: { buffer: this.F } },
        { binding: 3, resource: { buffer: this.C } },
        { binding: 4, resource: { buffer: this.Jp } },
        { binding: 5, resource: { buffer: this.gridAccum } },
        { binding: 6, resource: { buffer: this.materialUniform } },
        { binding: 7, resource: { buffer: this.activeCountUniform } },
      ],
    })

    const gridUpdateModule = device.createShaderModule({
      code: templateShader(gridUpdateSrc, { GRID_N, DT }),
    })
    this.gridUpdatePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: gridUpdateModule, entryPoint: "gridUpdate" },
    })
    this.gridUpdateBindGroup = device.createBindGroup({
      layout: this.gridUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gridAccum } },
        { binding: 1, resource: { buffer: this.gridVel } },
        { binding: 2, resource: { buffer: this.gravityUniform } },
        { binding: 3, resource: { buffer: this.mouseUniform } },
        { binding: 4, resource: { buffer: this.dampingUniform } },
      ],
    })

    const g2pModule = device.createShaderModule({
      code: templateShader(g2pSrc, { GRID_N, INV_DX, DT }),
    })
    this.g2pPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: g2pModule, entryPoint: "g2p" },
    })
    this.g2pBindGroup = device.createBindGroup({
      layout: this.g2pPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.positions } },
        { binding: 1, resource: { buffer: this.velocities } },
        { binding: 2, resource: { buffer: this.F } },
        { binding: 3, resource: { buffer: this.C } },
        { binding: 4, resource: { buffer: this.Jp } },
        { binding: 5, resource: { buffer: this.gridVel } },
        { binding: 6, resource: { buffer: this.activeCountUniform } },
        { binding: 7, resource: { buffer: this.materialUniform } },
      ],
    })

    this.gridDispatch = ceilDiv(NODE_COUNT, WORKGROUP)

    this.splatParamsUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setSplatRadius(DEFAULT_SPLAT_RADIUS)
    this.repulsionParamsUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setRepulsionStrength(DEFAULT_REPULSION_STRENGTH)
    this.buildRepulsionResources(DEFAULT_FIELD_RESOLUTION)

    // --- "Attract to Point" tool (see attract.wgsl's own header) ---
    this.pickResultBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    // Attractor: particleIndex(u32) + targetX(f32) + targetY(f32) = 12
    // bytes/entry — see attract.wgsl's own struct declaration for the
    // WGSL-side layout this must match exactly.
    this.attractorsBuffer = device.createBuffer({
      size: MAX_ATTRACTORS * 12,
      usage: GPUBufferUsage.STORAGE,
    })
    this.pickPosUniform = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.commitTargetUniform = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.attractorSlotUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.attractorCountUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    writeBuffer(device, this.attractorCountUniform, 0, new Uint32Array([0]))
    this.attractStrengthUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setAttractStrength(DEFAULT_ATTRACT_STRENGTH)

    const attractModule = device.createShaderModule({
      code: templateShader(attractSrc, { DT }),
    })

    this.pickParticlePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: attractModule, entryPoint: "pickParticle" },
    })
    this.pickParticleBindGroup = device.createBindGroup({
      layout: this.pickParticlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.positions } },
        { binding: 1, resource: { buffer: this.activeCountUniform } },
        { binding: 2, resource: { buffer: this.pickPosUniform } },
        { binding: 3, resource: { buffer: this.pickResultBuffer } },
      ],
    })

    this.highlightPickedPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: attractModule, entryPoint: "highlightPicked" },
    })
    this.highlightPickedBindGroup = device.createBindGroup({
      layout: this.highlightPickedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: this.pickResultBuffer } },
        { binding: 4, resource: { buffer: this.colors } },
      ],
    })

    this.commitAttractorPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: attractModule, entryPoint: "commitAttractor" },
    })
    this.commitAttractorBindGroup = device.createBindGroup({
      layout: this.commitAttractorPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: this.pickResultBuffer } },
        { binding: 5, resource: { buffer: this.commitTargetUniform } },
        { binding: 6, resource: { buffer: this.attractorSlotUniform } },
        { binding: 7, resource: { buffer: this.attractorsBuffer } },
      ],
    })

    this.applyAttractionPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: attractModule, entryPoint: "applyAttraction" },
    })
    this.applyAttractionBindGroup = device.createBindGroup({
      layout: this.applyAttractionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.attractorsBuffer } },
        { binding: 1, resource: { buffer: this.attractorCountUniform } },
        { binding: 2, resource: { buffer: this.positions } },
        { binding: 3, resource: { buffer: this.velocities } },
        { binding: 4, resource: { buffer: this.activeCountUniform } },
        { binding: 5, resource: { buffer: this.attractStrengthUniform } },
      ],
    })

    this.loadScene(scene)
  }

  /** (Re)builds every resolution-dependent repulsion resource — the
   * densityAccum buffer, densityTexture, and all 4 repulsion.wgsl
   * pipelines/bind groups (their shader modules have FIELD_N baked in at
   * compile time, same as GRID_N elsewhere, so a resolution change needs
   * fresh ones, not just a buffer resize). Called once from the
   * constructor (`repulsionBuilt` false, nothing to destroy yet) and
   * again from setFieldResolution() on every later change (destroys the
   * previous resolution's resources first — GPUBuffer/GPUTexture.destroy()
   * is safe to call even mid-frame here since this only ever runs between
   * rendered frames, never inside step()'s own command encoding). */
  private repulsionBuilt = false
  private buildRepulsionResources(resolution: FieldResolution): void {
    const device = this.device
    if (this.repulsionBuilt) {
      this.densityAccum.destroy()
      this.densityTexture.destroy()
    }
    this.repulsionBuilt = true
    this.fieldResolution = resolution

    const texelCount = resolution * resolution
    this.densityAccum = device.createBuffer({
      size: texelCount * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    this.densityTexture = device.createTexture({
      size: [resolution, resolution],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const densityTextureView = this.densityTexture.createView()

    const repulsionModule = device.createShaderModule({
      code: templateShader(repulsionSrc, { FIELD_N: resolution, DT }),
    })

    this.clearDensityPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: repulsionModule, entryPoint: "clearDensity" },
    })
    this.clearDensityBindGroup = device.createBindGroup({
      layout: this.clearDensityPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.densityAccum } }],
    })

    this.splatDensityPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: repulsionModule, entryPoint: "splatDensity" },
    })
    this.splatDensityBindGroup = device.createBindGroup({
      layout: this.splatDensityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityAccum } },
        { binding: 1, resource: { buffer: this.positions } },
        { binding: 2, resource: { buffer: this.activeCountUniform } },
        { binding: 3, resource: { buffer: this.splatParamsUniform } },
      ],
    })

    this.densityToTexturePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: repulsionModule, entryPoint: "densityToTexture" },
    })
    this.densityToTextureBindGroup = device.createBindGroup({
      layout: this.densityToTexturePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityAccum } },
        { binding: 1, resource: densityTextureView },
      ],
    })

    this.applyRepulsionPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: repulsionModule, entryPoint: "applyRepulsion" },
    })
    this.applyRepulsionBindGroup = device.createBindGroup({
      layout: this.applyRepulsionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.positions } },
        { binding: 2, resource: { buffer: this.activeCountUniform } },
        { binding: 4, resource: { buffer: this.velocities } },
        { binding: 5, resource: densityTextureView },
        { binding: 7, resource: { buffer: this.repulsionParamsUniform } },
      ],
    })

    this.densityClearDispatch = ceilDiv(texelCount, WORKGROUP)
    this.densityTextureDispatch = [
      ceilDiv(resolution, FIELD_WORKGROUP),
      ceilDiv(resolution, FIELD_WORKGROUP),
    ]
  }

  get activeCount(): number {
    return this.activeParticleCount
  }

  /** Writes `scene` into the head of every particle buffer and updates
   * activeCount — the one path both the constructor and every later
   * world-switch/Reset funnel through. `scene.count` must not exceed
   * MAX_PARTICLES (every worlds/*.ts file is expected to stay well under
   * that on its own; this doesn't re-check it per call). Whatever
   * garbage is left in the buffers past `scene.count` (a previous,
   * possibly-larger world's tail) is inert: P2G/G2P's activeCount bound
   * means nothing beyond it is ever read or written, and the draw call
   * (render.ts) only ever requests `activeCount` instances. */
  loadScene(scene: SceneData): void {
    writeBuffer(this.device, this.positions, 0, scene.positions)
    writeBuffer(this.device, this.velocities, 0, scene.velocities)
    writeBuffer(this.device, this.F, 0, scene.F)
    writeBuffer(this.device, this.C, 0, scene.C)
    writeBuffer(this.device, this.Jp, 0, scene.Jp)
    writeBuffer(this.device, this.colors, 0, scene.colors)
    this.activeParticleCount = scene.count
    writeBuffer(
      this.device,
      this.activeCountUniform,
      0,
      new Uint32Array([scene.count])
    )
    // A fresh scene means every OLD particle index any committed
    // attractor (see attract.wgsl's own header) refers to is meaningless
    // — clear rather than leave them silently pointing at whatever
    // particle happens to occupy that same slot in the new scene.
    this._attractorCount = 0
    writeBuffer(this.device, this.attractorCountUniform, 0, new Uint32Array([0]))
  }

  /** Appends `scene`'s particles right after the current activeCount,
   * instead of loadScene()'s overwrite-from-zero — the one path that lets
   * a world grow at runtime (main.ts's "Add Particles" tool) rather than
   * only ever being reset back to a fixed initial condition. Silently
   * truncates to whatever headroom is left under MAX_PARTICLES (this
   * class's own fixed buffer capacity) rather than erroring — a spawn
   * tool hitting the cap mid-drag should just stop adding particles, not
   * crash the frame loop — and returns how many actually got appended so
   * a caller can tell the cap was hit if it cares. */
  addParticles(scene: SceneData): number {
    const offset = this.activeParticleCount
    const toAdd = Math.min(scene.count, MAX_PARTICLES - offset)
    if (toAdd <= 0) return 0

    writeBuffer(this.device, this.positions, offset * 2 * 4, scene.positions.subarray(0, toAdd * 2))
    writeBuffer(this.device, this.velocities, offset * 2 * 4, scene.velocities.subarray(0, toAdd * 2))
    writeBuffer(this.device, this.F, offset * 4 * 4, scene.F.subarray(0, toAdd * 4))
    writeBuffer(this.device, this.C, offset * 4 * 4, scene.C.subarray(0, toAdd * 4))
    writeBuffer(this.device, this.Jp, offset * 4, scene.Jp.subarray(0, toAdd))
    writeBuffer(this.device, this.colors, offset * 4 * 4, scene.colors.subarray(0, toAdd * 4))

    this.activeParticleCount += toAdd
    writeBuffer(this.device, this.activeCountUniform, 0, new Uint32Array([this.activeParticleCount]))
    return toAdd
  }

  setGravity(gravity: number): void {
    writeBuffer(
      this.device,
      this.gravityUniform,
      0,
      new Float32Array([gravity])
    )
  }

  /** Live-updates gridUpdate.wgsl's `damping` uniform — a plain buffer
   * write, no pipeline recreation, safe to call every rendered frame or
   * on every tick of a slider. `lossFraction`/`substeps` are the same
   * two live values perSubstepDamping() itself takes — see that
   * function's own docstring for why both matter (not just
   * lossFraction) and why the conversion happens here/there rather than
   * in the shader. */
  setDamping(lossFraction: number, substeps: number): void {
    writeBuffer(
      this.device,
      this.dampingUniform,
      0,
      new Float32Array([perSubstepDamping(lossFraction, substeps)])
    )
  }

  /** Live-updates the Material uniform (mu0/lambda0/hardening, read by
   * p2g.wgsl; yieldLow/yieldHigh, read by g2p.wgsl — see yieldBounds()'s
   * own docstring for what `elasticity` means) — a plain buffer write, no
   * pipeline recreation, so this is safe to call on every tick of a
   * slider without disturbing the rollout in flight (same reasoning/
   * pattern as setGravity() above). `E`/`nu` are converted to Lamé
   * parameters here — see lameParams()'s own docstring for why that
   * conversion belongs on this side, not in the shader. */
  setMaterial(E: number, nu: number, hardening: number, elasticity: number): void {
    const { mu0, lambda0 } = lameParams(E, nu)
    const { yieldLow, yieldHigh } = yieldBounds(elasticity)
    writeBuffer(
      this.device,
      this.materialUniform,
      0,
      new Float32Array([mu0, lambda0, hardening, yieldLow, yieldHigh])
    )
  }

  /** Live-updates repulsion.wgsl's SplatParams uniform (its own
   * `sigma`) — a plain buffer write, no pipeline recreation, safe to
   * call on every tick of main.ts's "Splat radius" slider. Unlike
   * setFieldResolution() below, this never needs a rebuild: sigma is
   * read at runtime by splatDensity, not baked into the shader module
   * the way FIELD_N is. */
  setSplatRadius(sigma: number): void {
    writeBuffer(this.device, this.splatParamsUniform, 0, new Float32Array([sigma]))
  }

  /** Live-updates repulsion.wgsl's RepulsionParams uniform (its own
   * `strength`) — same plain-buffer-write, no-rebuild reasoning as
   * setSplatRadius() above. */
  setRepulsionStrength(strength: number): void {
    writeBuffer(this.device, this.repulsionParamsUniform, 0, new Float32Array([strength]))
  }

  /** Rebuilds every repulsion resource at a new field resolution (see
   * buildRepulsionResources()'s own docstring for why this can't be a
   * live uniform write like setSplatRadius/setRepulsionStrength above —
   * FIELD_N is compile-time-baked into repulsion.wgsl's shader module).
   * Callers MUST also call the matching MpmRenderer method
   * (rebuildRepulsionDisplay()) right after this — this class has no
   * reference back to the renderer, so it can't do that itself; see
   * main.ts's own field-resolution change handler for the pairing. */
  setFieldResolution(resolution: FieldResolution): void {
    if (resolution === this.fieldResolution) return
    this.buildRepulsionResources(resolution)
  }

  /** Step 1 of the "Attract to Point" tool's own 2-click gesture (see
   * attract.wgsl's own header) — finds whichever active particle is
   * nearest `(x, y)` and highlights it, via one submitted command buffer
   * running pickParticle then highlightPicked. The result (attract.wgsl's
   * own pickResult buffer) stays valid until the NEXT call to this
   * method — commitAttractorAt() below reads whatever this last wrote,
   * however much later the second click actually happens; no same-
   * submission ordering concern between the two, they're driven by two
   * separate mouse clicks arbitrarily far apart in time, and WebGPU's own
   * queue already serializes separately-submitted command buffers. */
  pickParticleAt(x: number, y: number): void {
    writeBuffer(this.device, this.pickPosUniform, 0, new Float32Array([x, y]))
    writeBuffer(this.device, this.pickResultBuffer, 0, new Uint32Array([0xffffffff]))

    const encoder = this.device.createCommandEncoder()
    const pickPass = encoder.beginComputePass()
    pickPass.setPipeline(this.pickParticlePipeline)
    pickPass.setBindGroup(0, this.pickParticleBindGroup)
    pickPass.dispatchWorkgroups(ceilDiv(this.activeParticleCount, WORKGROUP))
    pickPass.end()

    const highlightPass = encoder.beginComputePass()
    highlightPass.setPipeline(this.highlightPickedPipeline)
    highlightPass.setBindGroup(0, this.highlightPickedBindGroup)
    highlightPass.dispatchWorkgroups(1)
    highlightPass.end()

    this.device.queue.submit([encoder.finish()])
  }

  /** Step 2 of the "Attract to Point" tool's own 2-click gesture — commits
   * whichever particle pickParticleAt() last found as an attractor
   * targeting `(x, y)`, then grows attractorCount so step()'s own
   * applyAttraction dispatch picks it up starting next substep. No-ops
   * once MAX_ATTRACTORS is reached (see that const's own docstring) —
   * silently, same "truncate, don't crash" stance as addParticles() takes
   * at MAX_PARTICLES, rather than erroring on an unremarkable UI edge
   * case. */
  commitAttractorAt(x: number, y: number): void {
    if (this._attractorCount >= MAX_ATTRACTORS) return
    writeBuffer(this.device, this.commitTargetUniform, 0, new Float32Array([x, y]))
    writeBuffer(this.device, this.attractorSlotUniform, 0, new Uint32Array([this._attractorCount]))

    const encoder = this.device.createCommandEncoder()
    const commitPass = encoder.beginComputePass()
    commitPass.setPipeline(this.commitAttractorPipeline)
    commitPass.setBindGroup(0, this.commitAttractorBindGroup)
    commitPass.dispatchWorkgroups(1)
    commitPass.end()
    this.device.queue.submit([encoder.finish()])

    this._attractorCount += 1
    writeBuffer(this.device, this.attractorCountUniform, 0, new Uint32Array([this._attractorCount]))
  }

  /** Live-updates attract.wgsl's own `attractStrength` uniform — a plain
   * buffer write, no pipeline recreation, safe to call on every tick of
   * main.ts's "Attraction strength" slider. */
  setAttractStrength(strength: number): void {
    writeBuffer(this.device, this.attractStrengthUniform, 0, new Float32Array([strength]))
  }

  /** Live-updates gridUpdate.wgsl's Mouse uniform — a plain buffer
   * write, no pipeline recreation, safe to call every rendered frame.
   * `mode` 0=off (every other field ignored by the shader), 1=force
   * (attract/repel via signed `strength`, `vel` ignored), 2=move
   * (kinematic drag via `vel`, `strength` ignored) — see
   * gridUpdate.wgsl's own MODE_FORCE/MODE_MOVE comment for what each
   * actually does. `pos`/`vel` are both in the same [0,1]^2 domain
   * coordinates as everything else here (main.ts owns the canvas-
   * pixels-to-domain conversion, including the CSS-pixels-are-Y-down-
   * but-the-domain-is-Y-up flip). A single struct argument, not 6
   * positional numbers — that signature is one an easy mistake to call
   * with arguments swapped, and this is called every frame. */
  setMouse(state: {
    x: number
    y: number
    velX: number
    velY: number
    strength: number
    radius: number
    mode: 0 | 1 | 2
  }): void {
    writeBuffer(
      this.device,
      this.mouseUniform,
      0,
      new Float32Array([
        state.x,
        state.y,
        state.velX,
        state.velY,
        state.strength,
        state.radius,
        state.mode,
        0,
      ])
    )
  }

  /** Runs `substeps` full advance() iterations in one submitted command
   * buffer — matches the reference's own frame_dt/dt ratio (10 tiny,
   * numerically-stable dt=1e-4 steps per visible frame) rather than one
   * large step, which the explicit MPM update here is not stable under.
   * Dispatch size is recomputed from activeParticleCount every call
   * (not cached) — cheap, and it's the only thing that can change it
   * (loadScene()) between calls. */
  step(substeps: number): void {
    const particleDispatch = ceilDiv(this.activeParticleCount, WORKGROUP)
    const encoder = this.device.createCommandEncoder()
    for (let s = 0; s < substeps; s++) {
      const clearPass = encoder.beginComputePass()
      clearPass.setPipeline(this.clearGridPipeline)
      clearPass.setBindGroup(0, this.clearGridBindGroup)
      clearPass.dispatchWorkgroups(this.gridDispatch)
      clearPass.end()

      const p2gPass = encoder.beginComputePass()
      p2gPass.setPipeline(this.p2gPipeline)
      p2gPass.setBindGroup(0, this.p2gBindGroup)
      p2gPass.dispatchWorkgroups(particleDispatch)
      p2gPass.end()

      const gridPass = encoder.beginComputePass()
      gridPass.setPipeline(this.gridUpdatePipeline)
      gridPass.setBindGroup(0, this.gridUpdateBindGroup)
      gridPass.dispatchWorkgroups(this.gridDispatch)
      gridPass.end()

      const g2pPass = encoder.beginComputePass()
      g2pPass.setPipeline(this.g2pPipeline)
      g2pPass.setBindGroup(0, this.g2pBindGroup)
      g2pPass.dispatchWorkgroups(particleDispatch)
      g2pPass.end()

      // Repulsion (see repulsion.wgsl's own header) — runs every substep,
      // same cadence as gravity/mouse-force above, right after G2P has
      // resolved this substep's own particleVel: the nudge added here
      // feeds into next substep's P2G as ordinary momentum, no separate
      // grid-node pass needed (this field is particle-to-particle, not
      // grid-based, unlike gravity/mouse-force).
      const clearDensityPass = encoder.beginComputePass()
      clearDensityPass.setPipeline(this.clearDensityPipeline)
      clearDensityPass.setBindGroup(0, this.clearDensityBindGroup)
      clearDensityPass.dispatchWorkgroups(this.densityClearDispatch)
      clearDensityPass.end()

      const splatDensityPass = encoder.beginComputePass()
      splatDensityPass.setPipeline(this.splatDensityPipeline)
      splatDensityPass.setBindGroup(0, this.splatDensityBindGroup)
      splatDensityPass.dispatchWorkgroups(particleDispatch)
      splatDensityPass.end()

      const densityToTexturePass = encoder.beginComputePass()
      densityToTexturePass.setPipeline(this.densityToTexturePipeline)
      densityToTexturePass.setBindGroup(0, this.densityToTextureBindGroup)
      densityToTexturePass.dispatchWorkgroups(
        this.densityTextureDispatch[0],
        this.densityTextureDispatch[1]
      )
      densityToTexturePass.end()

      const applyRepulsionPass = encoder.beginComputePass()
      applyRepulsionPass.setPipeline(this.applyRepulsionPipeline)
      applyRepulsionPass.setBindGroup(0, this.applyRepulsionBindGroup)
      applyRepulsionPass.dispatchWorkgroups(particleDispatch)
      applyRepulsionPass.end()

      // "Attract to Point" (see attract.wgsl's own header) — skipped
      // entirely while nothing's been committed, rather than dispatched
      // at size 0 every substep forever for the common case (most
      // sessions never touch this tool).
      if (this._attractorCount > 0) {
        const attractPass = encoder.beginComputePass()
        attractPass.setPipeline(this.applyAttractionPipeline)
        attractPass.setBindGroup(0, this.applyAttractionBindGroup)
        attractPass.dispatchWorkgroups(ceilDiv(this._attractorCount, WORKGROUP))
        attractPass.end()
      }
    }
    this.device.queue.submit([encoder.finish()])
  }

  destroy(): void {
    this.positions.destroy()
    this.velocities.destroy()
    this.F.destroy()
    this.C.destroy()
    this.Jp.destroy()
    this.colors.destroy()
    this.gridAccum.destroy()
    this.gridVel.destroy()
    this.gravityUniform.destroy()
    this.materialUniform.destroy()
    this.mouseUniform.destroy()
    this.activeCountUniform.destroy()
    this.dampingUniform.destroy()
    this.densityAccum.destroy()
    this.densityTexture.destroy()
    this.splatParamsUniform.destroy()
    this.repulsionParamsUniform.destroy()
    this.pickResultBuffer.destroy()
    this.attractorsBuffer.destroy()
    this.pickPosUniform.destroy()
    this.commitTargetUniform.destroy()
    this.attractorSlotUniform.destroy()
    this.attractorCountUniform.destroy()
    this.attractStrengthUniform.destroy()
  }
}

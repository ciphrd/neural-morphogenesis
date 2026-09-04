// Browser port of trainer/mpm_core.py's own MpmCore — same buffer
// layout, bind groups, and pass ordering (clearDensity -> splatDensity
// -> densityToTexture -> applyRepulsion -> clearGrid -> p2g ->
// gridUpdate -> g2p), scoped to exactly ../core/'s passes, same as the
// Python headless wrapper. The WGSL itself isn't re-typed here at all —
// it's loaded straight out of ../core/*.wgsl via Vite's `?raw` imports,
// the same single-source-of-truth convention trainer/shader_template.py's
// load_core_shader() uses on the Python side (see vite.config.ts's own
// comment on why that path needs server.fs.allow).
//
// Repulsion (clearDensity/splatDensity/densityToTexture/applyRepulsion,
// all from ../../../core/repulsion.wgsl) runs FIRST each substep —
// applyRepulsion nudges velocity from THIS substep's own freshly-built
// density field, at each particle's own exact position, so the push
// reaches the grid through the very same substep's own P2G->gridUpdate
// ->G2P transfer immediately. See that file's own module docstring for
// the full revision history, including why a per-grid-node variant
// (inside gridUpdate.wgsl) was tried and reverted: it avoided P2G's own
// momentum-cancellation for overlapping particles, but at the cost of
// capping the push's spatial resolution at the physics grid's own cell
// size — confirmed empirically to barely separate particles at the
// distance core/agents.wgsl's own growth spawns them at, which is
// exactly the case this mechanism exists for.
//
// Unlike the Python wrapper, step() does NOT need to chunk large substep
// counts across multiple command encoders with an intermediate host sync
// — that limit (a hard cap on outstanding command buffers) is specific to
// wgpu-native's Metal backend running headlessly outside the browser;
// mls-mpm/src/gpu/mpm.ts's own step() (the browser sandbox this project's
// shaders were extracted from) confirms the browser's own Dawn/tint
// backend doesn't hit it at the substep counts this project ever calls
// per frame. One command encoder, one submit, regardless of substep count.
//
// No DEFAULT_* material/damping/repulsion constants live here (an
// earlier version hardcoded its own copy, independent of trainer/
// simulation_settings.py's own values — a real, silent-drift risk).
// gravity/material/damping/splatRadius/repulsionStrength are left
// un-set at construction (WebGPU zero-initializes storage/uniform
// buffers) — callers (gpu/simulation.ts) always apply
// SimulationConfig's own broadcast values via the set*() methods below
// immediately after construction, before any step() ever runs.

import clearGridSrc from "../../../core/clearGrid.wgsl?raw";
import p2gSrc from "../../../core/p2g.wgsl?raw";
import gridUpdateSrc from "../../../core/gridUpdate.wgsl?raw";
import g2pSrc from "../../../core/g2p.wgsl?raw";
import repulsionSrc from "../../../core/repulsion.wgsl?raw";
import morphologySrc from "../../../core/morphology.wgsl?raw";
import coreConstants from "../../../core/constants.json";
import { templateShader } from "./shaderTemplate";
import { ceilDiv, flatDispatch2D, writeFloat32 } from "./gpuUtil";
import type { SceneData } from "./types";

export const GRID_N: number = coreConstants.GRID_N;
export const DX: number = coreConstants.DX;
// Exported — gpu/fieldDiagnostics.wgsl's own scatterDiagnostics pass
// needs both to reproduce ../core/p2g.wgsl's exact same stencil/mass
// math (see that file's own module docstring).
export const INV_DX: number = coreConstants.INV_DX;
const DT: number = coreConstants.DT;
export const PARTICLE_MASS: number = coreConstants.PARTICLE_MASS;
export const PARTICLE_VOLUME: number = coreConstants.VOL;
export const MAX_PARTICLES: number = coreConstants.MAX_PARTICLES;
// core/constants.json's own FIELD_N is the repulsion density texture's
// resolution — renamed here to avoid collision with the chemical field's
// own (unrelated) FIELD_N in gpu/environment.ts.
export const REPULSION_FIELD_N: number = coreConstants.FIELD_N;

const NODE_COUNT = (GRID_N + 1) * (GRID_N + 1);
const WORKGROUP = 64;
const FIELD_WORKGROUP = 16;
const GRID_ACCUM_CHANNELS = 3;
// growthF(4), jp, cycleActive, growthAngle, growthAnisotropy, divisionBias,
// growthFrameAngle, appearanceScale, padding — 48 bytes.
const REST_FIELDS = 12;

/** Expands a flat (count,) Jp array into ParticleRest's own
 * tensor-rest layout, defaulting growthF=I and cycleActive=0.
 * Mirrors trainer/mpm_core.py's own
 * _pack_rest() exactly — exists so loadScene()/resetGrowthBuffers() keep
 * their original scalar-Jp signatures and rng.ts's own seedBlob() never
 * has to know the underlying buffer grew two siblings. */
function packRest(jp: Float32Array): Float32Array {
  const packed = new Float32Array(jp.length * REST_FIELDS);
  for (let i = 0; i < jp.length; i++) {
    packed[i * REST_FIELDS] = 1;
    packed[i * REST_FIELDS + 3] = 1;
    packed[i * REST_FIELDS + 4] = jp[i];
    packed[i * REST_FIELDS + 10] = 1;
  }
  return packed;
}

function perSubstepDamping(lossFraction: number, substeps: number): number {
  const clamped = Math.min(Math.max(lossFraction, 0.0), 0.999);
  return (1 - clamped) ** (1 / Math.max(substeps, 1));
}

function lameParams(e: number, nu: number): [number, number] {
  const mu0 = e / (2 * (1 + nu));
  const lambda0 = (e * nu) / ((1 + nu) * (1 - 2 * nu));
  return [mu0, lambda0];
}

const SNOW_YIELD_LOW = 1.0 - 2.5e-2;
const SNOW_YIELD_HIGH = 1.0 + 7.5e-3;
const WIDE_YIELD_LOW = 0.5;
const WIDE_YIELD_HIGH = 2.0;

function yieldBounds(elasticity: number): [number, number] {
  const t = Math.min(Math.max(elasticity, 0.0), 1.0);
  const yieldLow = SNOW_YIELD_LOW + t * (WIDE_YIELD_LOW - SNOW_YIELD_LOW);
  const yieldHigh = SNOW_YIELD_HIGH + t * (WIDE_YIELD_HIGH - SNOW_YIELD_HIGH);
  return [yieldLow, yieldHigh];
}

export class MpmCore {
  readonly device: GPUDevice;
  readonly positions: GPUBuffer;
  readonly velocities: GPUBuffer;
  // Public — gpu/fieldDiagnostics.wgsl's own scatterDiagnostics pass
  // (render.ts's "Deformation"/"Pressure"/"Shear" background modes)
  // reads F/Jp directly, the same particle-state buffers ../core/'s own
  // p2g.wgsl already reads each substep — see that file's own module
  // docstring for why this is a separate, viewer-owned pass rather than
  // an addition to core/p2g.wgsl itself. C was private until
  // core/agents.wgsl's own growth-inherits-parent-state change needed it
  // too (gpu/agents.ts's own bind group — see core/agents.wgsl's own
  // module docstring for why).
  readonly F: GPUBuffer;
  readonly C: GPUBuffer;
  readonly rest: GPUBuffer;
  // Public — the render layer's own field-visualize pass (gpu/render.ts's
  // "Density"/"Speed" background modes) reads these directly: gridAccum
  // for CH_MASS, gridVel for the already grid-update.wgsl-resolved
  // velocity. Neither is written by anything outside MpmCore itself.
  readonly gridAccum: GPUBuffer;
  readonly gridVel: GPUBuffer;

  private readonly gravityUniform: GPUBuffer;
  // Public — fieldDiagnostics.wgsl's own scatterDiagnostics pass needs
  // lambda0/hardening to compute the same pressure value core/p2g.wgsl's
  // own physics substeps already do, from the exact same live-adjustable
  // Material this buffer already holds (see setMaterial() below) — no
  // separate copy, so a Stiffness/Poisson/Hardening slider tick stays in
  // sync with the diagnostic display automatically.
  readonly materialUniform: GPUBuffer;
  // Public — agents.ts's own agentStep pass shares this exact buffer
  // (its own activeCount binding), not a separate copy, so the NN
  // forward pass and the physics substeps always gate on the same count.
  readonly activeCountUniform: GPUBuffer;
  private readonly dampingUniform: GPUBuffer;
  private readonly splatParamsUniform: GPUBuffer;
  private readonly repulsionParamsUniform: GPUBuffer;
  private readonly morphologyParamsUniform: GPUBuffer;

  private readonly densityAccum: GPUBuffer;
  // Public — gpu/render.ts's own "Repulsion" background mode samples this
  // r32float texture directly (textureLoad, same as core/repulsion.wgsl's
  // own applyRepulsion pass does internally).
  readonly densityTexture: GPUTexture;
  /** Bounded, blurred particle occupancy sampled by the neural policy. */
  readonly morphologyTexture: GPUTexture;
  private readonly morphologyBlurTexture: GPUTexture;

  private readonly clearGridPipeline: GPUComputePipeline;
  private readonly clearGridBindGroup: GPUBindGroup;
  private readonly p2gPipeline: GPUComputePipeline;
  private readonly p2gBindGroup: GPUBindGroup;
  private readonly gridUpdatePipeline: GPUComputePipeline;
  private readonly gridUpdateBindGroup: GPUBindGroup;
  private readonly g2pPipeline: GPUComputePipeline;
  private g2pBindGroup: GPUBindGroup;
  private readonly chemicalStateFallback: GPUBuffer;

  private readonly clearDensityPipeline: GPUComputePipeline;
  private readonly clearDensityBindGroup: GPUBindGroup;
  private readonly splatDensityPipeline: GPUComputePipeline;
  private readonly splatDensityBindGroup: GPUBindGroup;
  private readonly densityToTexturePipeline: GPUComputePipeline;
  private readonly densityToTextureBindGroup: GPUBindGroup;
  private readonly applyRepulsionPipeline: GPUComputePipeline;
  private readonly applyRepulsionBindGroup: GPUBindGroup;
  private readonly morphologyHorizontalPipeline: GPUComputePipeline;
  private readonly morphologyVerticalPipeline: GPUComputePipeline;
  private readonly morphologyHorizontalBindGroup: GPUBindGroup;
  private readonly morphologyVerticalBindGroup: GPUBindGroup;

  private readonly gridDispatch: number;
  private readonly densityClearDispatch: [number, number];
  private readonly densityTextureDispatch: [number, number];

  private _activeCount = 0;

  get activeCount(): number {
    return this._activeCount;
  }

  constructor(device: GPUDevice) {
    this.device = device;
    const f32 = 4;

    this.positions = device.createBuffer({ size: MAX_PARTICLES * 2 * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.velocities = device.createBuffer({ size: MAX_PARTICLES * 2 * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.F = device.createBuffer({ size: MAX_PARTICLES * 4 * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.C = device.createBuffer({ size: MAX_PARTICLES * 4 * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rest = device.createBuffer({ size: MAX_PARTICLES * REST_FIELDS * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.chemicalStateFallback = device.createBuffer({
      size: 256 + MAX_PARTICLES * 112,
      usage: GPUBufferUsage.STORAGE,
    });

    this.gridAccum = device.createBuffer({ size: NODE_COUNT * GRID_ACCUM_CHANNELS * f32, usage: GPUBufferUsage.STORAGE });
    this.gridVel = device.createBuffer({ size: NODE_COUNT * 2 * f32, usage: GPUBufferUsage.STORAGE });

    this.gravityUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // 64 bytes — fourteen material/growth/fluidity floats plus padding,
    // matching ../../../core/p2g.wgsl's and g2p.wgsl's identical Material.
    this.materialUniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.activeCountUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dampingUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const templateVars = { GRID_N, DX, INV_DX, DT, CHEMICAL_CHANNELS: 8 };

    const clearGridModule = device.createShaderModule({ code: templateShader(clearGridSrc, { GRID_N }) });
    this.clearGridPipeline = device.createComputePipeline({ layout: "auto", compute: { module: clearGridModule, entryPoint: "clearGrid" } });
    this.clearGridBindGroup = device.createBindGroup({
      layout: this.clearGridPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.gridAccum } }],
    });

    const p2gModule = device.createShaderModule({ code: templateShader(p2gSrc, templateVars) });
    this.p2gPipeline = device.createComputePipeline({ layout: "auto", compute: { module: p2gModule, entryPoint: "p2g" } });
    this.p2gBindGroup = device.createBindGroup({
      layout: this.p2gPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.positions } },
        { binding: 1, resource: { buffer: this.velocities } },
        { binding: 2, resource: { buffer: this.F } },
        { binding: 3, resource: { buffer: this.C } },
        { binding: 4, resource: { buffer: this.rest } },
        { binding: 5, resource: { buffer: this.gridAccum } },
        { binding: 6, resource: { buffer: this.materialUniform } },
        { binding: 7, resource: { buffer: this.activeCountUniform } },
      ],
    });

    const gridUpdateModule = device.createShaderModule({ code: templateShader(gridUpdateSrc, { GRID_N, DT }) });
    this.gridUpdatePipeline = device.createComputePipeline({ layout: "auto", compute: { module: gridUpdateModule, entryPoint: "gridUpdate" } });
    this.gridUpdateBindGroup = device.createBindGroup({
      layout: this.gridUpdatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gridAccum } },
        { binding: 1, resource: { buffer: this.gridVel } },
        { binding: 2, resource: { buffer: this.gravityUniform } },
        { binding: 3, resource: { buffer: this.dampingUniform } },
      ],
    });

    const g2pModule = device.createShaderModule({
      code: templateShader(g2pSrc, { GRID_N, INV_DX, DT, CHEMICAL_CHANNELS: 8 }),
    });
    this.g2pPipeline = device.createComputePipeline({ layout: "auto", compute: { module: g2pModule, entryPoint: "g2p" } });
    this.g2pBindGroup = device.createBindGroup({
      layout: this.g2pPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.positions } },
        { binding: 1, resource: { buffer: this.velocities } },
        { binding: 2, resource: { buffer: this.F } },
        { binding: 3, resource: { buffer: this.C } },
        { binding: 4, resource: { buffer: this.rest } },
        { binding: 5, resource: { buffer: this.gridVel } },
        { binding: 6, resource: { buffer: this.activeCountUniform } },
        { binding: 7, resource: { buffer: this.materialUniform } },
        { binding: 8, resource: { buffer: this.chemicalStateFallback } },
      ],
    });

    this.gridDispatch = ceilDiv(NODE_COUNT, WORKGROUP);

    // --- Repulsion ---
    const texelCount = REPULSION_FIELD_N * REPULSION_FIELD_N;
    this.densityAccum = device.createBuffer({ size: texelCount * f32, usage: GPUBufferUsage.STORAGE });
    this.densityTexture = device.createTexture({
      size: [REPULSION_FIELD_N, REPULSION_FIELD_N, 1],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const densityTextureView = this.densityTexture.createView();
    this.morphologyTexture = device.createTexture({
      size: [REPULSION_FIELD_N, REPULSION_FIELD_N, 1],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.morphologyBlurTexture = device.createTexture({
      size: [REPULSION_FIELD_N, REPULSION_FIELD_N, 1],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.morphologyParamsUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.splatParamsUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.repulsionParamsUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const repulsionModule = device.createShaderModule({ code: templateShader(repulsionSrc, { FIELD_N: REPULSION_FIELD_N, DT }) });

    this.clearDensityPipeline = device.createComputePipeline({ layout: "auto", compute: { module: repulsionModule, entryPoint: "clearDensity" } });
    this.clearDensityBindGroup = device.createBindGroup({
      layout: this.clearDensityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityAccum } },
      ],
    });

    this.splatDensityPipeline = device.createComputePipeline({ layout: "auto", compute: { module: repulsionModule, entryPoint: "splatDensity" } });
    this.splatDensityBindGroup = device.createBindGroup({
      layout: this.splatDensityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityAccum } },
        { binding: 1, resource: { buffer: this.positions } },
        { binding: 2, resource: { buffer: this.activeCountUniform } },
        { binding: 3, resource: { buffer: this.splatParamsUniform } },
      ],
    });

    this.densityToTexturePipeline = device.createComputePipeline({ layout: "auto", compute: { module: repulsionModule, entryPoint: "densityToTexture" } });
    this.densityToTextureBindGroup = device.createBindGroup({
      layout: this.densityToTexturePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityAccum } },
        { binding: 1, resource: densityTextureView },
      ],
    });

    this.applyRepulsionPipeline = device.createComputePipeline({ layout: "auto", compute: { module: repulsionModule, entryPoint: "applyRepulsion" } });
    this.applyRepulsionBindGroup = device.createBindGroup({
      layout: this.applyRepulsionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.positions } },
        { binding: 2, resource: { buffer: this.activeCountUniform } },
        { binding: 4, resource: { buffer: this.velocities } },
        { binding: 5, resource: densityTextureView },
        { binding: 7, resource: { buffer: this.repulsionParamsUniform } },
      ],
    });

    this.densityClearDispatch = flatDispatch2D(
      texelCount,
      WORKGROUP,
      device.limits.maxComputeWorkgroupsPerDimension,
    );
    this.densityTextureDispatch = [ceilDiv(REPULSION_FIELD_N, FIELD_WORKGROUP), ceilDiv(REPULSION_FIELD_N, FIELD_WORKGROUP)];

    const morphologyModule = device.createShaderModule({ code: templateShader(morphologySrc, { FIELD_N: REPULSION_FIELD_N }) });
    this.morphologyHorizontalPipeline = device.createComputePipeline({ layout: "auto", compute: { module: morphologyModule, entryPoint: "blurHorizontal" } });
    this.morphologyVerticalPipeline = device.createComputePipeline({ layout: "auto", compute: { module: morphologyModule, entryPoint: "blurVerticalAndNormalize" } });
    this.morphologyHorizontalBindGroup = device.createBindGroup({
      layout: this.morphologyHorizontalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: densityTextureView },
        { binding: 1, resource: this.morphologyBlurTexture.createView() },
        { binding: 2, resource: { buffer: this.morphologyParamsUniform } },
      ],
    });
    this.morphologyVerticalBindGroup = device.createBindGroup({
      layout: this.morphologyVerticalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.morphologyBlurTexture.createView() },
        { binding: 1, resource: this.morphologyTexture.createView() },
        { binding: 2, resource: { buffer: this.morphologyParamsUniform } },
      ],
    });

  }

  loadScene(scene: SceneData): void {
    if (scene.count > MAX_PARTICLES) throw new Error(`scene.count (${scene.count}) exceeds MAX_PARTICLES (${MAX_PARTICLES})`);
    writeFloat32(this.device, this.positions, 0, scene.positions);
    writeFloat32(this.device, this.velocities, 0, scene.velocities);
    writeFloat32(this.device, this.F, 0, scene.F);
    writeFloat32(this.device, this.C, 0, scene.C);
    // Scene API deliberately unchanged: callers still hand over a flat
    // (count,) Jp array (rng.ts's own seedBlob()). Expanded here into
    // ParticleRest's tensor layout — growthF=I and cycleActive=0 are
    // exactly right for
    // genuinely-seeded particles, which have no ramp to serve.
    writeFloat32(this.device, this.rest, 0, packRest(scene.Jp));
    this.setActiveCount(scene.count);
  }

  /** Updates both the JS-side count (dispatch sizing — see encodeSteps()'s
   * own particleDispatch) and the shared GPU uniform (p2g/gridUpdate-
   * adjacent/g2p/repulsion, and gpu/agents.ts's own Agents — all bind
   * this EXACT buffer, not a copy, so a single write here reaches every
   * one of them at once). loadScene()/addParticleAt() are the usual
   * callers, but growth needs this too — gpu/simulation.ts's own step()
   * calls this after reading back core/agents.wgsl's own atomic growth
   * counter, since GPU-side splitting changes the true count without
   * this class ever finding out on its own (see that module's own
   * module docstring for the full readback/propagation design). */
  setActiveCount(count: number): void {
    this._activeCount = count;
    writeFloat32(this.device, this.activeCountUniform, 0, new Uint32Array([count]));
  }

  /** Diagnostic-only active position readback, used once per sample sweep. */
  async readPositions(): Promise<Float32Array> {
    const byteLength = this._activeCount * 2 * 4;
    if (byteLength === 0) return new Float32Array();
    const staging = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.positions, 0, staging, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  /** Read an evenly distributed subset of active positions for inexpensive
   * viewer diagnostics such as auto-framing. */
  async readPositionSamples(maxSamples: number): Promise<Float32Array> {
    const activeCount = this._activeCount;
    const sampleCount = Math.min(activeCount, Math.max(1, Math.floor(maxSamples)));
    if (sampleCount === 0) return new Float32Array();

    const bytesPerPosition = 2 * 4;
    const staging = this.device.createBuffer({
      size: sampleCount * bytesPerPosition,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const particleIndex = sampleCount === 1
        ? 0
        : Math.round((sample * (activeCount - 1)) / (sampleCount - 1));
      encoder.copyBufferToBuffer(
        this.positions,
        particleIndex * bytesPerPosition,
        staging,
        sample * bytesPerPosition,
        bytesPerPosition,
      );
    }
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  /** Zero/identity-fills velocities/F/C/ParticleRest for [0, maxActive) — call once
   * per rollout, after loadScene(). Slots beyond this rollout's own
   * particle count are destined to become real particles via growth
   * (core/agents.wgsl's own agentStep() — see that file's own module
   * docstring for why every claimable slot must be initialized before
   * division overwrites it with inherited live state), and need to
   * start from the exact same fresh MPM state seedBlob() already gives
   * every genuinely-seeded particle — WITHOUT this, a slot THIS
   * rollout's own growth later claims could inherit a PREVIOUS rollout's
   * stale, possibly heavily-deformed state instead (loadScene() only
   * ever writes the HEAD of each buffer, up to that rollout's own
   * particle count, never the tail a previous rollout's growth may have
   * touched — and unlike the Python trainer, this object is reused
   * across every rollout a session ever plays, not rebuilt). Safe
   * (idempotent) to run over indices loadScene() ALSO just wrote —
   * seedBlob()'s own velocity/F/C/ParticleRest defaults are identical —
   * so this can unconditionally cover the whole [0, maxActive) range
   * rather than needing to carefully skip the already-real particles. */
  resetGrowthBuffers(maxActive: number): void {
    writeFloat32(this.device, this.velocities, 0, new Float32Array(maxActive * 2));
    const identityF = new Float32Array(maxActive * 4);
    for (let i = 0; i < maxActive; i++) {
      identityF[i * 4] = 1;
      identityF[i * 4 + 3] = 1;
    }
    writeFloat32(this.device, this.F, 0, identityF);
    writeFloat32(this.device, this.C, 0, new Float32Array(maxActive * 4));
    writeFloat32(this.device, this.rest, 0, packRest(new Float32Array(maxActive).fill(1)));
  }

  /** "Add Particle" tool (gpu/interact.wgsl's own module docstring — this
   * one needs no WGSL at all, just plain buffer writes at the next free
   * slot). `(x, y)`: MpmCore's own [0,1]^2 domain coords. A fresh
   * particle starts at rest (velocity 0), with an undeformed identity F
   * and Jp=1 — the same rest state this project's own scene seeding
   * (rng.ts's seedBlob()) gives every particle initially. Returns false
   * (a no-op) at MAX_PARTICLES capacity rather than throwing — this is a
   * live, interactive tool, not a one-shot scene load, so silently
   * refusing once full is the right "truncate, don't crash" stance (same
   * one loadScene() itself takes on an oversized scene, and
   * mls-mpm/src/gpu/mpm.ts's own addParticles() takes at its own
   * MAX_PARTICLES). */
  addParticleAt(x: number, y: number): boolean {
    if (this._activeCount >= MAX_PARTICLES) return false;
    const i = this._activeCount;
    writeFloat32(this.device, this.positions, i * 2 * 4, new Float32Array([x, y]));
    writeFloat32(this.device, this.velocities, i * 2 * 4, new Float32Array([0, 0]));
    writeFloat32(this.device, this.F, i * 4 * 4, new Float32Array([1, 0, 0, 1]));
    writeFloat32(this.device, this.C, i * 4 * 4, new Float32Array([0, 0, 0, 0]));
    writeFloat32(this.device, this.rest, i * REST_FIELDS * 4, new Float32Array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]));
    this.setActiveCount(this._activeCount + 1);
    return true;
  }

  setGravity(gravity: number): void {
    writeFloat32(this.device, this.gravityUniform, 0, new Float32Array([gravity]));
  }

  setDamping(lossFraction: number, substeps: number): void {
    writeFloat32(this.device, this.dampingUniform, 0, new Float32Array([perSubstepDamping(lossFraction, substeps)]));
  }

  /** The growth params drive ../../../core/g2p.wgsl's own growth
   * relaxation (the multiplicative F = Fe*Fg decomposition) — see that
   * file's own Material struct for what each does. */
  setMaterial(
    e: number,
    nu: number,
    hardening: number,
    elasticity: number,
    growthDuration: number,
    growthMax: number,
    growthAnisotropy: number,
    substepsPerMacro: number,
    particleMass: number = PARTICLE_MASS,
    particleVolume: number = PARTICLE_VOLUME,
    growthCompressionStart: number = 0.10,
    growthCompressionStop: number = 0.10,
    growthCompressionFeedback: number = 1.0,
    fluidity: number = 0,
  ): void {
    if (!(particleMass > 0) || !(particleVolume > 0)) {
      throw new Error("particle mass and volume must be positive");
    }
    if (!(growthCompressionStart >= 0) || !(growthCompressionStop >= growthCompressionStart)) {
      throw new Error("growth compression thresholds require 0 <= start <= stop");
    }
    if (!(growthCompressionFeedback >= 0 && growthCompressionFeedback <= 1)) {
      throw new Error("growth compression feedback must be in [0, 1]");
    }
    const [mu0, lambda0] = lameParams(e, nu);
    const [yieldLow, yieldHigh] = yieldBounds(elasticity);
    // The shader integrates exponential growth once per physics substep.
    // Deriving its low-level rate here makes one doubling consume the same
    // number of neural/chemical ticks at every numerical substep count.
    const growthRate = growthDuration > 0
      ? Math.log(2) / (growthDuration * Math.max(substepsPerMacro, 1) * DT)
      : 0;
    writeFloat32(
      this.device,
      this.materialUniform,
      0,
      new Float32Array([
        mu0, lambda0, hardening, yieldLow,
        yieldHigh, growthRate, growthMax, growthAnisotropy,
        particleMass, particleVolume,
        growthCompressionStart, growthCompressionStop, growthCompressionFeedback,
        fluidity,
      ])
    );
  }

  /** Supplies the persistent per-cell chemistry used by fluidity. */
  setChemicalStateBuffer(buffer: GPUBuffer): void {
    this.g2pBindGroup = this.device.createBindGroup({
      layout: this.g2pPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.positions } },
        { binding: 1, resource: { buffer: this.velocities } },
        { binding: 2, resource: { buffer: this.F } },
        { binding: 3, resource: { buffer: this.C } },
        { binding: 4, resource: { buffer: this.rest } },
        { binding: 5, resource: { buffer: this.gridVel } },
        { binding: 6, resource: { buffer: this.activeCountUniform } },
        { binding: 7, resource: { buffer: this.materialUniform } },
        { binding: 8, resource: { buffer } },
      ],
    });
  }

  setSplatRadius(sigma: number): void {
    writeFloat32(this.device, this.splatParamsUniform, 0, new Float32Array([sigma, 0, 0, 0]));
  }

  /** `maxDelta` is core/repulsion.wgsl's own RepulsionParams.maxDelta —
   * see that field's own comment for what it bounds and why. */
  setRepulsionStrength(strength: number, maxDelta: number): void {
    writeFloat32(this.device, this.repulsionParamsUniform, 0, new Float32Array([
      strength, maxDelta, 0, 0,
    ]));
  }

  setMorphology(sigmaDomain: number, densityReference: number): void {
    writeFloat32(this.device, this.morphologyParamsUniform, 0, new Float32Array([sigmaDomain, densityReference, 0, 0]));
  }

  /** Rebuilds policy occupancy from current positions once per controller tick. */
  encodeMorphology(encoder: GPUCommandEncoder): void {
    const particleDispatch = ceilDiv(this._activeCount, WORKGROUP);
    const passes: [GPUComputePipeline, GPUBindGroup, [number, number?]][] = [
      [this.clearDensityPipeline, this.clearDensityBindGroup, this.densityClearDispatch],
      [this.splatDensityPipeline, this.splatDensityBindGroup, [particleDispatch]],
      [this.densityToTexturePipeline, this.densityToTextureBindGroup, this.densityTextureDispatch],
      [this.morphologyHorizontalPipeline, this.morphologyHorizontalBindGroup, this.densityTextureDispatch],
      [this.morphologyVerticalPipeline, this.morphologyVerticalBindGroup, this.densityTextureDispatch],
    ];
    for (const [pipeline, bindGroup, dispatch] of passes) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(dispatch[0], dispatch[1]);
      pass.end();
    }
  }

  /** Encodes `substeps` full advance() iterations into `encoder` — does
   * NOT submit; callers (simulation.ts) fold this into a larger per-
   * macro-step encoder alongside the chemical-field/NN passes, one
   * submit per macro step, not one per physics substep. */
  encodeSteps(encoder: GPUCommandEncoder, substeps: number): void {
    const particleDispatch = ceilDiv(this._activeCount, WORKGROUP);
    for (let i = 0; i < substeps; i++) {
      let pass = encoder.beginComputePass();
      pass.setPipeline(this.clearDensityPipeline);
      pass.setBindGroup(0, this.clearDensityBindGroup);
      pass.dispatchWorkgroups(...this.densityClearDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.splatDensityPipeline);
      pass.setBindGroup(0, this.splatDensityBindGroup);
      pass.dispatchWorkgroups(particleDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.densityToTexturePipeline);
      pass.setBindGroup(0, this.densityToTextureBindGroup);
      pass.dispatchWorkgroups(...this.densityTextureDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.applyRepulsionPipeline);
      pass.setBindGroup(0, this.applyRepulsionBindGroup);
      pass.dispatchWorkgroups(particleDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.clearGridPipeline);
      pass.setBindGroup(0, this.clearGridBindGroup);
      pass.dispatchWorkgroups(this.gridDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.p2gPipeline);
      pass.setBindGroup(0, this.p2gBindGroup);
      pass.dispatchWorkgroups(particleDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.gridUpdatePipeline);
      pass.setBindGroup(0, this.gridUpdateBindGroup);
      pass.dispatchWorkgroups(this.gridDispatch);
      pass.end();

      pass = encoder.beginComputePass();
      pass.setPipeline(this.g2pPipeline);
      pass.setBindGroup(0, this.g2pBindGroup);
      pass.dispatchWorkgroups(particleDispatch);
      pass.end();
    }
  }

  destroy(): void {
    this.positions.destroy();
    this.velocities.destroy();
    this.F.destroy();
    this.C.destroy();
    this.rest.destroy();
    this.gridAccum.destroy();
    this.gridVel.destroy();
    this.gravityUniform.destroy();
    this.materialUniform.destroy();
    this.activeCountUniform.destroy();
    this.dampingUniform.destroy();
    this.splatParamsUniform.destroy();
    this.repulsionParamsUniform.destroy();
    this.morphologyParamsUniform.destroy();
    this.densityAccum.destroy();
    this.densityTexture.destroy();
    this.morphologyTexture.destroy();
    this.morphologyBlurTexture.destroy();
  }
}

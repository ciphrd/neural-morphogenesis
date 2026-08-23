// Canvas rendering — particles (render.wgsl) and mls-mpm's own
// field-visualize background system (field.wgsl), ported with the same
// options mls-mpm/src/gpu/render.ts
// exposes: a Field-mode background dropdown and particle-size control,
// plus particle render modes for white dots, growth-neuron-colored dots,
// translucent activation dots, and signed directional-growth arrows.
//
// Field modes: "none" | "density" | "speed" | "deformation" | "pressure"
// | "shear" | "repulsion" | "substrate" | "growth" | "gradient" — the
// same set mls-mpm/src/gpu/render.ts exposes, plus "substrate"/"growth"/
// "gradient" (this project's own chemical field/repulsion density, mls-
// mpm has no equivalent of). "deformation"/"pressure"/"shear" read
// gpu/fieldDiagnostics.wgsl's own scatter pass, NOT core/p2g.wgsl's
// gridAccum (that shared physics core was deliberately stripped of these
// diagnostic channels when it was extracted from mls-mpm's sandbox — see
// fieldDiagnostics.wgsl's own module docstring for why this project
// keeps them in a separate, viewer-owned pass instead of adding them
// back). "growth" reads the SAME chemical field "substrate" does, just
// one specific channel (the LAST one — this project's own growth
// probability substrate, see core/agents.wgsl's own module docstring)
// through a cividis colormap over its own clamped [-1,1] range instead
// of substrate's 3-channel RGB composite — see field.wgsl's own
// colorizeGrowth() comment. "gradient" reads the REPULSION density
// field's own spatial gradient instead (mpmCore.densityTexture — the
// SAME field "repulsion" mode's own repulsionFragment samples, NOT the
// chemical field), computed on the fly via a Sobel finite difference
// (there's no precomputed gradient for repulsion the way
// environment.wgsl's own computeGradient gives the chemical field one) —
// a first step toward a "shape boundary" background, see field.wgsl's
// own colorizeGradient() comment for the full reasoning.

import fieldSrc from "./field.wgsl?raw";
import fieldDiagnosticsSrc from "./fieldDiagnostics.wgsl?raw";
import renderSrc from "./render.wgsl?raw";
import { writeFloat32, ceilDiv } from "./gpuUtil";
import type { Environment } from "./environment";
import { DX, GRID_N, INV_DX, PARTICLE_MASS, REPULSION_FIELD_N, type MpmCore } from "./mpmCore";
import { templateShader } from "./shaderTemplate";

export type FieldMode = "none" | "density" | "speed" | "deformation" | "pressure" | "shear" | "repulsion" | "substrate" | "growth" | "gradient";
export type ParticleRenderMode = "dots-white" | "dots-activation" | "dots-activation-translucent" | "directional-arrows";

const FIELD_MODE_CODE: Record<Exclude<FieldMode, "repulsion" | "substrate" | "growth" | "gradient">, number> = {
  none: 0,
  density: 1,
  speed: 2,
  deformation: 3,
  pressure: 4,
  shear: 5,
};
// Modes whose own colorizeField dispatch (this file's own render(), the
// grid-node-diagnostics branch below) needs fieldDiagnostics.wgsl's
// clear+scatter passes to have run first this frame — density/speed
// don't (they read ../core/'s own gridAccum/gridVel directly, already
// kept current by MpmCore.encodeSteps()).
const DIAGNOSTIC_MODES: ReadonlySet<FieldMode> = new Set(["deformation", "pressure", "shear"]);
const GRID_FIELD_MODES: ReadonlySet<FieldMode> = new Set(["density", "speed", "deformation", "pressure", "shear"]);

const PARTICLE_COLOR = [1, 1, 1, 1]; // white — matches debug_images.py's GROWN_COLOR
const TARGET_COLOR = [0.95, 0.4, 0.25, 0.8]; // warm accent, alpha-blended under the particles
const GROWTH_AXIS_COLOR = [0.2, 0.95, 0.85, 0.95]; // cyan-green, distinct from particles/target

// mls-mpm/src/gpu/render.ts's own DEFAULT_POINT_RADIUS_PX is 1 — this
// project's particle counts run smaller by default (hundreds, not
// thousands), so 2px reads better at a typical viewport size; still a
// starting guess, same as that project's own, not derived from anything.
const DEFAULT_PARTICLE_RADIUS_PX = 2.0;
const TARGET_RADIUS_PX = 1.75;

function alphaBlend(): GPUBlendState {
  return {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
}

export class Renderer {
  private readonly device: GPUDevice;

  // --- particles/target (render.wgsl) ---
  private readonly pointLayout: GPUBindGroupLayout;
  private readonly circlePipeline: GPURenderPipeline;

  private readonly particleRadiusUniform: GPUBuffer;
  private readonly particleColorUniform: GPUBuffer;
  private readonly circleParticleBindGroup: GPUBindGroup;
  private readonly activationParticlePipeline: GPURenderPipeline;
  private readonly activationParticleBindGroup: GPUBindGroup;
  private readonly activationAlphaUniform: GPUBuffer;
  private readonly growthAxisPipeline: GPURenderPipeline;
  private readonly growthAxisBindGroup: GPUBindGroup;
  private readonly growthAxisStyleUniform: GPUBuffer;
  private readonly growthAxisColorUniform: GPUBuffer;

  private readonly targetRadiusUniform: GPUBuffer;
  private readonly targetColorUniform: GPUBuffer;
  private targetPositions: GPUBuffer | null = null;
  private targetBindGroup: GPUBindGroup | null = null;
  private targetCount = 0;
  private targetVisible = true;

  private particleRenderMode: ParticleRenderMode = "dots-white";
  private whiteDotsAlpha = 1.0;
  private activationAlpha = 0.2;
  private particleRadiusPx = DEFAULT_PARTICLE_RADIUS_PX;
  private growthAxisLengthPx = 24;
  private canvasMinDimPx = 512;

  // --- field-visualize background (field.wgsl) ---
  private readonly colorizePipeline: GPUComputePipeline;
  private readonly colorizeBindGroup: GPUBindGroup;
  private readonly fieldModeUniform: GPUBuffer;
  // field.wgsl's own accent uniform (binding 13) — shared across every
  // background mode's own color-computing pass (colorizeField,
  // repulsionFragment, colorizeSubstrate, colorizeGrowth all reach it,
  // transitively, via accentedMagnitude()/accentedSigned() or directly
  // — see that file's own comment), so it's threaded into each of THEIR
  // bind groups below, not just this class's own field one.
  private readonly accentUniform: GPUBuffer;
  private readonly fieldTexture: GPUTexture;
  private readonly fieldPresentPipeline: GPURenderPipeline;
  private readonly fieldPresentBindGroup: GPUBindGroup;
  private readonly repulsionPresentPipeline: GPURenderPipeline;
  private readonly repulsionPresentBindGroup: GPUBindGroup;
  private readonly fieldDispatch: [number, number];

  // --- deformation/pressure/shear diagnostics (fieldDiagnostics.wgsl) ---
  // See that file's own module docstring for why this is a separate
  // pass/buffer from ../core/'s own gridAccum, run once per rendered
  // frame (not per physics substep) from this class's own render().
  private readonly diagnosticsBuffer: GPUBuffer;
  private readonly clearDiagnosticsPipeline: GPUComputePipeline;
  private readonly clearDiagnosticsBindGroup: GPUBindGroup;
  private readonly scatterDiagnosticsPipeline: GPUComputePipeline;
  private readonly scatterDiagnosticsBindGroup: GPUBindGroup;

  // --- substrate background (field.wgsl's own colorizeSubstrate) ---
  // Two precomputed bind groups, indexed by environment.parity — that
  // buffer ping-pongs every macro step (see environment.ts's own
  // docstring), so which one is "current" can't be baked in once at
  // construction the way density/speed/deformation/pressure/shear's own
  // (non-ping-ponging) source buffers can.
  private readonly environment: Environment;
  private readonly substrateTexture: GPUTexture;
  private readonly substrateColorizePipeline: GPUComputePipeline;
  private readonly substrateColorizeBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly substratePresentPipeline: GPURenderPipeline;
  private readonly substratePresentBindGroup: GPUBindGroup;
  private readonly substrateDispatch: [number, number];

  // --- growth background (field.wgsl's own colorizeGrowth) ---
  // Same parity-indexed bind-group-array shape as substrate above (same
  // reasoning, same source buffers — colorizeGrowth reads the LAST
  // channel of that SAME environment.buffers[p], not a separate buffer),
  // own output texture/present pipeline since it's a genuinely different
  // image (cividis over one channel's own clamped [-1,1] range, not
  // substrate's 3-channel RGB composite).
  private readonly growthTexture: GPUTexture;
  private readonly growthColorizePipeline: GPUComputePipeline;
  private readonly growthColorizeBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly growthPresentPipeline: GPURenderPipeline;
  private readonly growthPresentBindGroup: GPUBindGroup;

  // --- gradient background (field.wgsl's own blurDensity/colorizeGradient) ---
  // Reads the REPULSION density field (mpmCore.densityTexture — the same
  // one "repulsion" mode's own repulsionFragment samples), not the
  // chemical field, so this has its own resolution (REPULSION_FIELD_N,
  // NOT environment.width/height — see mpmCore.ts's own REPULSION_FIELD_N
  // comment for why the two are unrelated) and its own dispatch, not
  // substrate/growth's. Single bind groups throughout, not parity-
  // indexed — mpmCore.densityTexture is one texture, rewritten in place
  // every frame (MpmCore's own densityToTexture pass), never swapped.
  //
  // blurDensity runs FIRST each frame this mode is active, writing a
  // blurred copy of the raw density into blurredDensityTexture;
  // colorizeGradient then reads THAT (not repulsionTex directly) — see
  // field.wgsl's own blurDensity() comment for why (raw per-particle
  // density is too grainy for a clean shape-boundary gradient). setBlur()
  // below controls blurSigmaUniform, the one knob between them.
  private readonly blurredDensityTexture: GPUTexture;
  private readonly blurDensityPipeline: GPUComputePipeline;
  private readonly blurDensityBindGroup: GPUBindGroup;
  private readonly blurSigmaUniform: GPUBuffer;
  private readonly gradientTexture: GPUTexture;
  private readonly gradientColorizePipeline: GPUComputePipeline;
  private readonly gradientColorizeBindGroup: GPUBindGroup;
  private readonly gradientPresentPipeline: GPURenderPipeline;
  private readonly gradientPresentBindGroup: GPUBindGroup;
  // colorizeGradient's own dedicated power curve (field.wgsl's own
  // gradientExponent uniform) — NOT the shared accentUniform every other
  // mode reaches through graypoint()/accentedSigned(); this mode stopped
  // calling graypoint() entirely once it needed a magnitude-preserving-
  // direction curve instead (see that pass' own comment), so accent
  // isn't even reachable from its own auto-derived bind group layout
  // anymore — setGradientExponent() below is the only knob this mode has.
  private readonly gradientExponentUniform: GPUBuffer;
  private readonly gradientDispatch: [number, number];

  private fieldMode: FieldMode = "none";

  constructor(device: GPUDevice, format: GPUTextureFormat, mpmCore: MpmCore, environment: Environment) {
    this.device = device;
    this.environment = environment;
    const renderModule = device.createShaderModule({ code: renderSrc });

    // --- particles/target ---
    this.pointLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const pointLayoutPipeline = device.createPipelineLayout({ bindGroupLayouts: [this.pointLayout] });

    this.circlePipeline = device.createRenderPipeline({
      layout: pointLayoutPipeline,
      vertex: { module: renderModule, entryPoint: "particleVertex" },
      fragment: { module: renderModule, entryPoint: "particleFragment", targets: [{ format, blend: alphaBlend() }] },
      primitive: { topology: "triangle-list" },
    });

    this.particleRadiusUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.particleColorUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.particleColorUniform, 0, new Float32Array(PARTICLE_COLOR));
    this.circleParticleBindGroup = device.createBindGroup({
      layout: this.pointLayout,
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: this.particleRadiusUniform } },
        { binding: 2, resource: { buffer: this.particleColorUniform } },
      ],
    });
    const activationLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.activationParticlePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [activationLayout] }),
      vertex: { module: renderModule, entryPoint: "activationParticleVertex" },
      fragment: { module: renderModule, entryPoint: "activationParticleFragment", targets: [{ format, blend: alphaBlend() }] },
      primitive: { topology: "triangle-list" },
    });
    this.activationAlphaUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.activationAlphaUniform, 0, new Float32Array([1]));
    this.activationParticleBindGroup = device.createBindGroup({
      layout: activationLayout,
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: this.particleRadiusUniform } },
        { binding: 4, resource: { buffer: mpmCore.rest } },
        { binding: 6, resource: { buffer: this.activationAlphaUniform } },
      ],
    });

    // Live directional-growth overlay. It reads MpmCore.rest directly,
    // so the glyph is exactly the signal g2p consumes rather than a
    // reconstructed NN preview. Binding numbers 0/2 reuse the positions
    // and color declarations in render.wgsl; 4/5 are overlay-specific.
    const growthAxisLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    this.growthAxisPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [growthAxisLayout] }),
      vertex: { module: renderModule, entryPoint: "growthAxisVertex" },
      fragment: { module: renderModule, entryPoint: "growthAxisFragment", targets: [{ format, blend: alphaBlend() }] },
      primitive: { topology: "triangle-list" },
    });
    this.growthAxisStyleUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.growthAxisColorUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.growthAxisColorUniform, 0, new Float32Array(GROWTH_AXIS_COLOR));
    this.growthAxisBindGroup = device.createBindGroup({
      layout: growthAxisLayout,
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 2, resource: { buffer: this.growthAxisColorUniform } },
        { binding: 4, resource: { buffer: mpmCore.rest } },
        { binding: 5, resource: { buffer: this.growthAxisStyleUniform } },
      ],
    });

    this.targetRadiusUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.targetColorUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.targetColorUniform, 0, new Float32Array(TARGET_COLOR));

    // --- field-visualize background ---
    const fieldModule = device.createShaderModule({
      code: templateShader(fieldSrc, {
        GRID_N,
        REPULSION_FIELD_N,
        SUBSTRATE_WIDTH: environment.width,
        SUBSTRATE_HEIGHT: environment.height,
        CHANNELS: environment.channels,
      }),
    });
    const nodes = GRID_N + 1;

    this.fieldTexture = device.createTexture({
      size: [nodes, nodes, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fieldModeUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.fieldModeUniform, 0, new Uint32Array([FIELD_MODE_CODE.none]));
    // 0 = identity (every background mode renders exactly as it did
    // before this knob existed) — see field.wgsl's own accent uniform
    // comment.
    this.accentUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.accentUniform, 0, new Float32Array([0]));

    // fieldDiagnostics.wgsl's own accumulator — 4 channels (J, shear,
    // pressure, mass — see that file's own header), i32-per-channel,
    // (GRID_N+1)^2 nodes. Cleared and re-scattered fresh every rendered
    // frame that needs it (see render()'s own DIAGNOSTIC_MODES branch),
    // not incrementally maintained.
    this.diagnosticsBuffer = device.createBuffer({ size: nodes * nodes * 4 * 4, usage: GPUBufferUsage.STORAGE });
    const diagnosticsModule = device.createShaderModule({
      code: templateShader(fieldDiagnosticsSrc, { GRID_N, DX, INV_DX, PARTICLE_MASS }),
    });
    this.clearDiagnosticsPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: diagnosticsModule, entryPoint: "clearDiagnostics" },
    });
    this.clearDiagnosticsBindGroup = device.createBindGroup({
      layout: this.clearDiagnosticsPipeline.getBindGroupLayout(0),
      entries: [{ binding: 3, resource: { buffer: this.diagnosticsBuffer } }],
    });
    this.scatterDiagnosticsPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: diagnosticsModule, entryPoint: "scatterDiagnostics" },
    });
    this.scatterDiagnosticsBindGroup = device.createBindGroup({
      layout: this.scatterDiagnosticsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: mpmCore.F } },
        { binding: 2, resource: { buffer: mpmCore.rest } },
        { binding: 3, resource: { buffer: this.diagnosticsBuffer } },
        { binding: 4, resource: { buffer: mpmCore.materialUniform } },
        { binding: 5, resource: { buffer: mpmCore.activeCountUniform } },
      ],
    });

    this.colorizePipeline = device.createComputePipeline({ layout: "auto", compute: { module: fieldModule, entryPoint: "colorizeField" } });
    this.colorizeBindGroup = device.createBindGroup({
      layout: this.colorizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mpmCore.gridAccum } },
        { binding: 1, resource: { buffer: mpmCore.gridVel } },
        { binding: 2, resource: { buffer: this.fieldModeUniform } },
        { binding: 3, resource: this.fieldTexture.createView() },
        { binding: 7, resource: { buffer: this.diagnosticsBuffer } },
        { binding: 13, resource: { buffer: this.accentUniform } },
      ],
    });
    this.fieldDispatch = [Math.ceil(nodes / 16), Math.ceil(nodes / 16)];

    const fieldSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.fieldPresentPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "fieldVertex" },
      fragment: { module: fieldModule, entryPoint: "fieldFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.fieldPresentBindGroup = device.createBindGroup({
      layout: this.fieldPresentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 4, resource: this.fieldTexture.createView() },
        { binding: 5, resource: fieldSampler },
      ],
    });

    this.repulsionPresentPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "fieldVertex" },
      fragment: { module: fieldModule, entryPoint: "repulsionFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.repulsionPresentBindGroup = device.createBindGroup({
      layout: this.repulsionPresentPipeline.getBindGroupLayout(0),
      // repulsionFragment computes its own color inline (no separate
      // colorize compute pass the way field/substrate/growth have —
      // see field.wgsl's own module docstring), so it reaches `accent`
      // directly rather than through an intermediate colorize pass'
      // own bind group.
      entries: [
        { binding: 6, resource: mpmCore.densityTexture.createView() },
        { binding: 13, resource: { buffer: this.accentUniform } },
      ],
    });

    // --- substrate background ---
    this.substrateTexture = device.createTexture({
      size: [environment.width, environment.height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.substrateColorizePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: fieldModule, entryPoint: "colorizeSubstrate" },
    });
    this.substrateColorizeBindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.substrateColorizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 8, resource: { buffer: environment.buffers[p] } },
          { binding: 9, resource: this.substrateTexture.createView() },
          { binding: 13, resource: { buffer: this.accentUniform } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];
    this.substrateDispatch = [ceilDiv(environment.width, 16), ceilDiv(environment.height, 16)];

    this.substratePresentPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "fieldVertex" },
      fragment: { module: fieldModule, entryPoint: "substrateFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.substratePresentBindGroup = device.createBindGroup({
      layout: this.substratePresentPipeline.getBindGroupLayout(0),
      // substrateFragment reuses fieldFragment's own fieldSampler
      // (binding 5, declared once at module scope — see field.wgsl's
      // own comment on why) — layout:"auto" pulls it into this
      // pipeline's own bind group layout too since it's reachable from
      // substrateFragment, so it has to be supplied here as well, not
      // just binding 10.
      entries: [
        { binding: 5, resource: fieldSampler },
        { binding: 10, resource: this.substrateTexture.createView() },
      ],
    });

    // --- growth background ---
    this.growthTexture = device.createTexture({
      size: [environment.width, environment.height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.growthColorizePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: fieldModule, entryPoint: "colorizeGrowth" },
    });
    this.growthColorizeBindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.growthColorizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 8, resource: { buffer: environment.buffers[p] } },
          { binding: 11, resource: this.growthTexture.createView() },
          { binding: 13, resource: { buffer: this.accentUniform } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];

    this.growthPresentPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "fieldVertex" },
      fragment: { module: fieldModule, entryPoint: "growthFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.growthPresentBindGroup = device.createBindGroup({
      layout: this.growthPresentPipeline.getBindGroupLayout(0),
      // growthFragment reuses fieldFragment's own fieldSampler (binding
      // 5) too — same reasoning substratePresentBindGroup's own comment
      // gives.
      entries: [
        { binding: 5, resource: fieldSampler },
        { binding: 12, resource: this.growthTexture.createView() },
      ],
    });

    // --- gradient background ---
    this.blurredDensityTexture = device.createTexture({
      size: [REPULSION_FIELD_N, REPULSION_FIELD_N, 1],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.blurSigmaUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.blurSigmaUniform, 0, new Float32Array([0]));
    this.blurDensityPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: fieldModule, entryPoint: "blurDensity" },
    });
    this.blurDensityBindGroup = device.createBindGroup({
      layout: this.blurDensityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 6, resource: mpmCore.densityTexture.createView() },
        { binding: 14, resource: { buffer: this.blurSigmaUniform } },
        { binding: 17, resource: this.blurredDensityTexture.createView() },
      ],
    });

    this.gradientTexture = device.createTexture({
      size: [REPULSION_FIELD_N, REPULSION_FIELD_N, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.gradientExponentUniform = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(device, this.gradientExponentUniform, 0, new Float32Array([1]));
    this.gradientColorizePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: fieldModule, entryPoint: "colorizeGradient" },
    });
    this.gradientColorizeBindGroup = device.createBindGroup({
      layout: this.gradientColorizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 18, resource: this.blurredDensityTexture.createView() },
        { binding: 15, resource: this.gradientTexture.createView() },
        { binding: 19, resource: { buffer: this.gradientExponentUniform } },
      ],
    });
    this.gradientDispatch = [ceilDiv(REPULSION_FIELD_N, 16), ceilDiv(REPULSION_FIELD_N, 16)];

    this.gradientPresentPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "fieldVertex" },
      fragment: { module: fieldModule, entryPoint: "gradientFragment", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.gradientPresentBindGroup = device.createBindGroup({
      layout: this.gradientPresentPipeline.getBindGroupLayout(0),
      // gradientFragment reuses fieldFragment's own fieldSampler (binding
      // 5) too — same reasoning substratePresentBindGroup's own comment
      // gives.
      entries: [
        { binding: 5, resource: fieldSampler },
        { binding: 16, resource: this.gradientTexture.createView() },
      ],
    });

    this.setCanvasSizePx(512, 512);
  }

  setFieldMode(mode: FieldMode): void {
    this.fieldMode = mode;
    if (mode !== "repulsion" && mode !== "substrate" && mode !== "growth" && mode !== "gradient") {
      writeFloat32(this.device, this.fieldModeUniform, 0, new Uint32Array([FIELD_MODE_CODE[mode]]));
    }
  }

  /** [-2,2] — see field.wgsl's accent uniform/accentedMagnitude()/
   * accentedSigned() comments for the exact exponential curve. Negative
   * suppresses submaximal values; positive accentuates them. Applies to
   * every background mode at once (one shared uniform, read by
   * whichever mode's own colorize pass currently runs); a plain buffer
   * write, no pipeline rebuild, same as setFieldMode() above's own
   * fieldModeUniform write — safe to call every frame off a live slider. */
  setAccent(accent: number): void {
    writeFloat32(this.device, this.accentUniform, 0, new Float32Array([accent]));
  }

  /** Gaussian sigma (texels, [0, 2] — see field.wgsl's own BLUR_MAX_RADIUS
   * for why this saturates rather than growing unboundedly) for the
   * "gradient" mode's own blur pass — see that pass' own comment for why
   * blurring the density field before differentiating it matters. 0 (the
   * default) skips the blur entirely. A plain buffer write, no pipeline
   * rebuild, same as setAccent() above — safe to call every frame off a
   * live slider. */
  setBlur(sigma: number): void {
    writeFloat32(this.device, this.blurSigmaUniform, 0, new Float32Array([sigma]));
  }

  /** Power curve applied to the "gradient" mode's own gradient
   * MAGNITUDE (direction is preserved exactly — see field.wgsl's own
   * colorizeGradient() comment for why magnitude, not each of gx/gy
   * independently). 1 = identity, the linear-normalized mapping this
   * mode used before this knob existed; >1 sharpens onto strong edges,
   * <1 brings out faint ones. A plain buffer write, no pipeline rebuild,
   * same as setAccent()/setBlur() above — safe to call every frame off a
   * live slider. */
  setGradientExponent(exponent: number): void {
    writeFloat32(this.device, this.gradientExponentUniform, 0, new Float32Array([exponent]));
  }

  setParticleRenderMode(mode: ParticleRenderMode): void {
    this.particleRenderMode = mode;
    if (mode === "dots-activation" || mode === "dots-activation-translucent") {
      writeFloat32(this.device, this.activationAlphaUniform, 0, new Float32Array([
        mode === "dots-activation-translucent" ? this.activationAlpha : 1.0,
      ]));
    }
  }

  setActivationAlpha(alpha: number): void {
    this.activationAlpha = Math.min(1, Math.max(0, alpha));
    if (this.particleRenderMode === "dots-activation-translucent") {
      writeFloat32(this.device, this.activationAlphaUniform, 0, new Float32Array([this.activationAlpha]));
    }
  }

  setWhiteDotsAlpha(alpha: number): void {
    this.whiteDotsAlpha = Math.min(1, Math.max(0, alpha));
    writeFloat32(this.device, this.particleColorUniform, 0, new Float32Array([1, 1, 1, this.whiteDotsAlpha]));
  }

  /** Total device-pixel length of a full-strength signed growth-polarity
   * arrow. Signal magnitude scales this length linearly. */
  setGrowthAxisLengthPx(px: number): void {
    this.growthAxisLengthPx = px;
    this.writeGrowthAxisStyle();
  }

  /** Device-pixel particle radius — mirrors mls-mpm/src/gpu/render.ts's
   * own setPointRadiusPx()/particleSize slider. Independent of canvas
   * resize (setCanvasSizePx() re-derives the NDC radius from whichever
   * of these two was set most recently). */
  setPointRadiusPx(px: number): void {
    this.particleRadiusPx = px;
    this.writeParticleRadius();
  }

  setCanvasSizePx(widthPx: number, heightPx: number): void {
    this.canvasMinDimPx = Math.max(1, Math.min(widthPx, heightPx));
    this.writeParticleRadius();
    this.writeGrowthAxisStyle();
    writeFloat32(this.device, this.targetRadiusUniform, 0, new Float32Array([(TARGET_RADIUS_PX * 2) / this.canvasMinDimPx]));
  }

  private writeParticleRadius(): void {
    writeFloat32(this.device, this.particleRadiusUniform, 0, new Float32Array([(this.particleRadiusPx * 2) / this.canvasMinDimPx]));
  }

  private writeGrowthAxisStyle(): void {
    const pxToNdc = 2 / this.canvasMinDimPx;
    writeFloat32(this.device, this.growthAxisStyleUniform, 0, new Float32Array([
      this.growthAxisLengthPx * 0.5 * pxToNdc,
      0.8 * pxToNdc,
      4.0 * pxToNdc,
      2.7 * pxToNdc,
    ]));
  }

  /** `points`: flat [x0,y0,x1,y1,...] in MpmCore's own [0,1]^2 domain
   * (see net/images.ts-adjacent target loading — the /target/points
   * endpoint already returns domain-space coordinates, no rescaling
   * needed here). Pass an empty array to clear the overlay. */
  setTargetPoints(points: Float32Array): void {
    this.targetCount = points.length / 2;
    this.targetPositions?.destroy();
    this.targetBindGroup = null;
    if (this.targetCount === 0) return;

    this.targetPositions = this.device.createBuffer({
      size: points.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    writeFloat32(this.device, this.targetPositions, 0, points);
    this.targetBindGroup = this.device.createBindGroup({
      layout: this.pointLayout,
      entries: [
        { binding: 0, resource: { buffer: this.targetPositions } },
        { binding: 1, resource: { buffer: this.targetRadiusUniform } },
        { binding: 2, resource: { buffer: this.targetColorUniform } },
      ],
    });
  }

  /** Rendering-only visibility switch. Target points remain uploaded so
   * toggling the overlay back on is immediate and does not rebuild or
   * restart the simulation. */
  setTargetVisible(visible: boolean): void {
    this.targetVisible = visible;
  }

  render(context: GPUCanvasContext, activeCount: number): void {
    const encoder = this.device.createCommandEncoder();

    if (GRID_FIELD_MODES.has(this.fieldMode)) {
      if (DIAGNOSTIC_MODES.has(this.fieldMode)) {
        // Fresh every frame this mode is active — see fieldDiagnostics.
        // wgsl's own module docstring for why this isn't incrementally
        // maintained the way ../core/'s own gridAccum is.
        const clearPass = encoder.beginComputePass();
        clearPass.setPipeline(this.clearDiagnosticsPipeline);
        clearPass.setBindGroup(0, this.clearDiagnosticsBindGroup);
        clearPass.dispatchWorkgroups(...this.fieldDispatch);
        clearPass.end();

        const scatterPass = encoder.beginComputePass();
        scatterPass.setPipeline(this.scatterDiagnosticsPipeline);
        scatterPass.setBindGroup(0, this.scatterDiagnosticsBindGroup);
        scatterPass.dispatchWorkgroups(ceilDiv(activeCount, 64));
        scatterPass.end();
      }

      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.colorizePipeline);
      computePass.setBindGroup(0, this.colorizeBindGroup);
      computePass.dispatchWorkgroups(...this.fieldDispatch);
      computePass.end();
    } else if (this.fieldMode === "substrate") {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.substrateColorizePipeline);
      computePass.setBindGroup(0, this.substrateColorizeBindGroups[this.environment.parity]);
      computePass.dispatchWorkgroups(...this.substrateDispatch);
      computePass.end();
    } else if (this.fieldMode === "growth") {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.growthColorizePipeline);
      computePass.setBindGroup(0, this.growthColorizeBindGroups[this.environment.parity]);
      computePass.dispatchWorkgroups(...this.substrateDispatch);
      computePass.end();
    } else if (this.fieldMode === "gradient") {
      // blurDensity MUST run first — colorizeGradient reads its own
      // output (blurredDensityTexture), not repulsionTex directly — see
      // field.wgsl's own blurDensity() comment.
      const blurPass = encoder.beginComputePass();
      blurPass.setPipeline(this.blurDensityPipeline);
      blurPass.setBindGroup(0, this.blurDensityBindGroup);
      blurPass.dispatchWorkgroups(...this.gradientDispatch);
      blurPass.end();

      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.gradientColorizePipeline);
      computePass.setBindGroup(0, this.gradientColorizeBindGroup);
      computePass.dispatchWorkgroups(...this.gradientDispatch);
      computePass.end();
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.02, g: 0.02, b: 0.02, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    if (GRID_FIELD_MODES.has(this.fieldMode)) {
      pass.setPipeline(this.fieldPresentPipeline);
      pass.setBindGroup(0, this.fieldPresentBindGroup);
      pass.draw(6);
    } else if (this.fieldMode === "repulsion") {
      pass.setPipeline(this.repulsionPresentPipeline);
      pass.setBindGroup(0, this.repulsionPresentBindGroup);
      pass.draw(6);
    } else if (this.fieldMode === "substrate") {
      pass.setPipeline(this.substratePresentPipeline);
      pass.setBindGroup(0, this.substratePresentBindGroup);
      pass.draw(6);
    } else if (this.fieldMode === "growth") {
      pass.setPipeline(this.growthPresentPipeline);
      pass.setBindGroup(0, this.growthPresentBindGroup);
      pass.draw(6);
    } else if (this.fieldMode === "gradient") {
      pass.setPipeline(this.gradientPresentPipeline);
      pass.setBindGroup(0, this.gradientPresentBindGroup);
      pass.draw(6);
    }

    if (this.targetVisible && this.targetBindGroup && this.targetCount > 0) {
      pass.setPipeline(this.circlePipeline);
      pass.setBindGroup(0, this.targetBindGroup);
      pass.draw(6, this.targetCount);
    }

    if (activeCount > 0) {
      if (this.particleRenderMode === "dots-white") {
        pass.setPipeline(this.circlePipeline);
        pass.setBindGroup(0, this.circleParticleBindGroup);
        pass.draw(6, activeCount);
      } else if (this.particleRenderMode === "dots-activation" || this.particleRenderMode === "dots-activation-translucent") {
        pass.setPipeline(this.activationParticlePipeline);
        pass.setBindGroup(0, this.activationParticleBindGroup);
        pass.draw(6, activeCount);
      } else {
        pass.setPipeline(this.growthAxisPipeline);
        pass.setBindGroup(0, this.growthAxisBindGroup);
        pass.draw(9, activeCount);
      }
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.particleRadiusUniform.destroy();
    this.particleColorUniform.destroy();
    this.activationAlphaUniform.destroy();
    this.growthAxisStyleUniform.destroy();
    this.growthAxisColorUniform.destroy();
    this.targetRadiusUniform.destroy();
    this.targetColorUniform.destroy();
    this.targetPositions?.destroy();
    this.fieldModeUniform.destroy();
    this.accentUniform.destroy();
    this.fieldTexture.destroy();
    this.diagnosticsBuffer.destroy();
    this.substrateTexture.destroy();
    this.growthTexture.destroy();
    this.blurredDensityTexture.destroy();
    this.blurSigmaUniform.destroy();
    this.gradientTexture.destroy();
    this.gradientExponentUniform.destroy();
  }
}

import shaderSrc from "./repulsion.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import type { GpuAgents } from "./agents";

// Independent of (and much coarser than, by design) the main (C,H,W)
// grid's own resolution — see repulsion.wgsl's own header comment and
// repulsion.py's module docstring for the full O(N)-not-O(N^2)
// reasoning. Structural (fixes buffer sizes), not live-adjustable —
// changing it needs a hard rebuild, same footing as CHANNELS/WIDTH/
// HEIGHT. Must match constants.py's own REPULSION_RESOLUTION for a
// replayed rollout to feel like the training run it came from (unlike
// sigma/strength, this one isn't broadcast — see train_server.py's own
// per-generation message for what is).
export const DEFAULT_REPULSION_RESOLUTION = 128;

export interface RepulsionConfig {
  resolution: number;
  gridSize: number; // main grid's own world-pixel span (assumes square: gridWidth === gridHeight)
  agentCount: number;
}

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

/** Owns the dedicated, independent-resolution density field behind
 * envnca's cheap O(N) repulsion approximation — mirrors repulsion.py
 * exactly (see that module's own docstring). Two-phase construction,
 * not one: the constructor only builds this field's own buffers/
 * pipelines (clearRepulsionScratch, mergeRepulsionDensity,
 * computeRepulsionGradient — none of which need anything from
 * GpuAgents), because `gradient` (built here) needs to exist *before*
 * GpuAgents is constructed (agentStep binds it to sample repulsion
 * force), while splatRepulsion's own bind group needs GpuAgents'
 * `positions`/`physicsUniform` buffers, which only exist *after*
 * GpuAgents is constructed. bindAgents() below closes that loop once
 * both sides exist — see gpu/simulation.ts's rebuild() for the exact
 * construction order this requires. */
export class GpuRepulsion {
  readonly scratch: GPUBuffer;
  readonly density: GPUBuffer;
  readonly gradient: GPUBuffer;

  readonly clearScratchPipeline: GPUComputePipeline;
  readonly mergeDensityPipeline: GPUComputePipeline;
  readonly computeGradientPipeline: GPUComputePipeline;
  private splatPipeline: GPUComputePipeline;

  readonly clearScratchBindGroup: GPUBindGroup;
  readonly mergeDensityBindGroup: GPUBindGroup;
  readonly computeGradientBindGroup: GPUBindGroup;
  // Only exists once bindAgents() has been called — see this class's
  // own docstring for why that can't happen at construction time.
  private splatBindGroup: GPUBindGroup | null = null;

  readonly clearScratchGroups: number;
  readonly fieldSquareGroups: readonly [number, number];
  readonly splatGroups: number;
  // Public so gpu/render.ts can template its own colorize shader against
  // this field's resolution when visualizing it as a background mode.
  readonly resolution: number;

  private readonly device: GPUDevice;
  private readonly module: GPUShaderModule;

  constructor(device: GPUDevice, config: RepulsionConfig) {
    this.device = device;
    this.resolution = config.resolution;
    const { resolution, gridSize, agentCount } = config;
    const fieldSize = resolution * resolution;

    this.scratch = device.createBuffer({ size: fieldSize * 4, usage: GPUBufferUsage.STORAGE });
    this.density = device.createBuffer({ size: fieldSize * 4, usage: GPUBufferUsage.STORAGE });
    this.gradient = device.createBuffer({ size: fieldSize * 2 * 4, usage: GPUBufferUsage.STORAGE });

    const source = templateShader(shaderSrc, {
      RESOLUTION: resolution,
      // Forced decimal point — WGSL const-decl float literals need one
      // (or an exponent) to parse as f32 rather than AbstractInt; a
      // plain "256" substituted in here would be a type mismatch against
      // `const GRID_SIZE: f32`.
      GRID_SIZE: gridSize.toFixed(1),
      AGENT_COUNT: agentCount,
    });
    this.module = device.createShaderModule({ code: source });

    this.clearScratchPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.module, entryPoint: "clearRepulsionScratch" },
    });
    this.mergeDensityPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.module, entryPoint: "mergeRepulsionDensity" },
    });
    this.computeGradientPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.module, entryPoint: "computeRepulsionGradient" },
    });
    // Pipeline only — bind group deferred to bindAgents() (needs
    // GpuAgents' own buffers, which don't exist yet at this point).
    this.splatPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: this.module, entryPoint: "splatRepulsion" },
    });

    this.clearScratchBindGroup = device.createBindGroup({
      layout: this.clearScratchPipeline.getBindGroupLayout(0),
      entries: [{ binding: 1, resource: { buffer: this.scratch } }],
    });
    this.mergeDensityBindGroup = device.createBindGroup({
      layout: this.mergeDensityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.scratch } },
        { binding: 2, resource: { buffer: this.density } },
      ],
    });
    this.computeGradientBindGroup = device.createBindGroup({
      layout: this.computeGradientPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: this.density } },
        { binding: 3, resource: { buffer: this.gradient } },
      ],
    });

    this.clearScratchGroups = ceilDiv(fieldSize, 256);
    this.fieldSquareGroups = [ceilDiv(resolution, 16), ceilDiv(resolution, 16)];
    this.splatGroups = ceilDiv(agentCount, 64);
  }

  /** Closes the construction-order loop described in this class's own
   * docstring — call once, right after constructing the GpuAgents this
   * repulsion field is meant to push around, before the first step(). */
  bindAgents(agents: GpuAgents): void {
    this.splatBindGroup = this.device.createBindGroup({
      layout: this.splatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: agents.positions } },
        { binding: 1, resource: { buffer: this.scratch } },
        { binding: 4, resource: { buffer: agents.physicsUniform } },
      ],
    });
  }

  /** The splatRepulsion pass specifically — kept separate from the
   * other three (clear/merge/gradient, which gpu/simulation.ts's step()
   * calls directly against their own public pipeline/bind-group pairs)
   * only because this one's bind group isn't available until
   * bindAgents() has run; asserting that here catches a wrong call order
   * immediately instead of a confusing null-buffer WebGPU validation
   * error. */
  encodeSplat(pass: GPUComputePassEncoder): void {
    if (!this.splatBindGroup) {
      throw new Error("GpuRepulsion.bindAgents() must be called before encodeSplat()");
    }
    pass.setPipeline(this.splatPipeline);
    pass.setBindGroup(0, this.splatBindGroup);
    pass.dispatchWorkgroups(this.splatGroups);
  }

  destroy(): void {
    this.scratch.destroy();
    this.density.destroy();
    this.gradient.destroy();
  }
}

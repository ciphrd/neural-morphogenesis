// TS wrapper around environment.wgsl. The field is a transient projection of
// cell-owned chemistry: clear atomic splats, materialize them into buffer 0,
// then compute one shared Sobel gradient. Buffer 1 remains allocated only to
// preserve the renderer-facing buffer/parity interface.

import environmentSrc from "../../../core/environment.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import { ceilDiv, writeFloat32 } from "./gpuUtil";

export interface EnvironmentConfig {
  channels: number;
  width: number;
  height: number;
  // Legacy run-metadata fields; transient fields do not use either value.
  decay: number;
  depositRate: number;
}

const CLEAR_WORKGROUP = 256;
const GRID_WORKGROUP = 16;

export class Environment {
  readonly channels: number;
  readonly width: number;
  readonly height: number;
  // Public so agents.ts can build its own parity-indexed bind group
  // variants against the exact same buffers, kept in lockstep by the one
  // parity counter this class owns.
  readonly buffers: [GPUBuffer, GPUBuffer];
  readonly gradient: GPUBuffer;
  readonly depositScratch: GPUBuffer;

  private readonly device: GPUDevice;

  private readonly clearScratchPipeline: GPUComputePipeline;
  private readonly clearScratchBindGroup: GPUBindGroup;
  private readonly materializeSplatPipeline: GPUComputePipeline;
  private readonly materializeSplatBindGroup: GPUBindGroup;
  private readonly computeGradientPipeline: GPUComputePipeline;
  private readonly computeGradientBindGroups: [GPUBindGroup, GPUBindGroup];

  private readonly clearDispatch: number;
  private readonly gridDispatch: [number, number, number];

  private _parity = 0;
  get parity(): number {
    return this._parity;
  }

  constructor(device: GPUDevice, config: EnvironmentConfig) {
    this.device = device;
    this.channels = config.channels;
    this.width = config.width;
    this.height = config.height;

    const total = config.width * config.height * config.channels;
    const f32 = 4;

    this.buffers = [
      device.createBuffer({ size: total * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      device.createBuffer({ size: total * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    ];
    this.gradient = device.createBuffer({ size: total * 2 * f32, usage: GPUBufferUsage.STORAGE });
    this.depositScratch = device.createBuffer({ size: total * f32, usage: GPUBufferUsage.STORAGE });

    const module = device.createShaderModule({
      code: templateShader(environmentSrc, { CHANNELS: config.channels, WIDTH: config.width, HEIGHT: config.height }),
    });

    this.clearScratchPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "clearScratch" } });
    this.clearScratchBindGroup = device.createBindGroup({
      layout: this.clearScratchPipeline.getBindGroupLayout(0),
      entries: [{ binding: 2, resource: { buffer: this.depositScratch } }],
    });

    this.materializeSplatPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "materializeSplat" } });
    this.materializeSplatBindGroup = device.createBindGroup({
      layout: this.materializeSplatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers[0] } },
        { binding: 2, resource: { buffer: this.depositScratch } },
      ],
    });

    this.computeGradientPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "computeGradient" } });

    this.computeGradientBindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.computeGradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers[p] } },
          { binding: 1, resource: { buffer: this.gradient } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];

    this.clearDispatch = ceilDiv(total, CLEAR_WORKGROUP);
    this.gridDispatch = [ceilDiv(config.width, GRID_WORKGROUP), ceilDiv(config.height, GRID_WORKGROUP), config.channels];
  }

  /** Zeroes both grid buffers and resets parity to 0 — call at the start
   * of every rollout (a fresh chemical field, same as a fresh
   * trainer/environment.py Environment instance each Python-side
   * rollout). */
  reset(): void {
    const total = this.width * this.height * this.channels;
    const zeros = new Float32Array(total);
    writeFloat32(this.device, this.buffers[0], 0, zeros);
    writeFloat32(this.device, this.buffers[1], 0, zeros);
    this._parity = 0;
  }

  /** Starts a communication round by removing every previous splat. */
  encodeClear(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.clearScratchPipeline);
    pass.setBindGroup(0, this.clearScratchBindGroup);
    pass.dispatchWorkgroups(this.clearDispatch);
    pass.end();
  }

  /** Materializes the just-published cell states, then computes the field
   * gradient shared by every brain invocation in this round. */
  encodeSense(encoder: GPUCommandEncoder): void {
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.materializeSplatPipeline);
    pass.setBindGroup(0, this.materializeSplatBindGroup);
    pass.dispatchWorkgroups(this.clearDispatch);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.computeGradientPipeline);
    pass.setBindGroup(0, this.computeGradientBindGroups[0]);
    pass.dispatchWorkgroups(...this.gridDispatch);
    pass.end();
  }

  destroy(): void {
    this.buffers[0].destroy();
    this.buffers[1].destroy();
    this.gradient.destroy();
    this.depositScratch.destroy();
  }
}

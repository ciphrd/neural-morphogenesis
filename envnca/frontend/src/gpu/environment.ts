import shaderSrc from "./environment.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import { writeFloat32 } from "./gpuUtil";

export interface EnvironmentConfig {
  width: number;
  height: number;
  channels: number;
  decay: number;
}

const WORKGROUP_1D = 256;
const WORKGROUP_2D = 16;

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

/** Owns the GPU-resident chemical grid — gradient (Sobel), deposit-merge,
 * and diffuse+decay passes — mirroring environment.py. Ping-pong grid
 * buffers (gridA/gridB) with both parity variants of every buffer-
 * dependent bind group precomputed once here (not per frame) — see
 * environment.wgsl's own comment for why only diffuseDecay actually
 * needs the second buffer. gpu/simulation.ts picks which parity's
 * pipelines/bind groups to encode each step. */
export class GpuEnvironment {
  readonly gridA: GPUBuffer;
  readonly gridB: GPUBuffer;
  readonly gradient: GPUBuffer;
  readonly depositScratch: GPUBuffer;

  readonly clearScratchPipeline: GPUComputePipeline;
  readonly computeGradientPipeline: GPUComputePipeline;
  readonly mergeDepositPipeline: GPUComputePipeline;
  readonly diffuseDecayPipeline: GPUComputePipeline;

  readonly clearScratchBindGroup: GPUBindGroup;
  // index 0: current=gridA (next=gridB). index 1: current=gridB (next=gridA).
  readonly computeGradientBindGroups: [GPUBindGroup, GPUBindGroup];
  readonly mergeDepositBindGroups: [GPUBindGroup, GPUBindGroup];
  readonly diffuseDecayBindGroups: [GPUBindGroup, GPUBindGroup];

  readonly clearScratchGroups: number;
  readonly grid2DGroups: readonly [number, number];

  private readonly device: GPUDevice;
  private readonly zeroPlane: Float32Array;

  constructor(device: GPUDevice, config: EnvironmentConfig) {
    this.device = device;
    const { width, height, channels, decay } = config;
    const planeSize = channels * height * width;
    this.zeroPlane = new Float32Array(planeSize);

    this.gridA = device.createBuffer({
      size: planeSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.gridB = device.createBuffer({
      size: planeSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.gradient = device.createBuffer({
      size: planeSize * 2 * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.depositScratch = device.createBuffer({
      size: planeSize * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    const source = templateShader(shaderSrc, { CHANNELS: channels, WIDTH: width, HEIGHT: height, DECAY: decay });
    const module = device.createShaderModule({ code: source });

    this.clearScratchPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "clearScratch" } });
    this.computeGradientPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "computeGradient" } });
    this.mergeDepositPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "mergeDeposit" } });
    this.diffuseDecayPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "diffuseDecay" } });

    this.clearScratchBindGroup = device.createBindGroup({
      layout: this.clearScratchPipeline.getBindGroupLayout(0),
      entries: [{ binding: 2, resource: { buffer: this.depositScratch } }],
    });

    this.computeGradientBindGroups = [
      device.createBindGroup({
        layout: this.computeGradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridA } },
          { binding: 1, resource: { buffer: this.gradient } },
        ],
      }),
      device.createBindGroup({
        layout: this.computeGradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridB } },
          { binding: 1, resource: { buffer: this.gradient } },
        ],
      }),
    ];

    this.mergeDepositBindGroups = [
      device.createBindGroup({
        layout: this.mergeDepositPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridA } },
          { binding: 2, resource: { buffer: this.depositScratch } },
        ],
      }),
      device.createBindGroup({
        layout: this.mergeDepositPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridB } },
          { binding: 2, resource: { buffer: this.depositScratch } },
        ],
      }),
    ];

    this.diffuseDecayBindGroups = [
      device.createBindGroup({
        layout: this.diffuseDecayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridA } },
          { binding: 3, resource: { buffer: this.gridB } },
        ],
      }),
      device.createBindGroup({
        layout: this.diffuseDecayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridB } },
          { binding: 3, resource: { buffer: this.gridA } },
        ],
      }),
    ];

    this.clearScratchGroups = ceilDiv(planeSize, WORKGROUP_1D);
    this.grid2DGroups = [ceilDiv(width, WORKGROUP_2D), ceilDiv(height, WORKGROUP_2D)];
  }

  gridBuffer(parity: number): GPUBuffer {
    return parity === 0 ? this.gridA : this.gridB;
  }

  /** Zeroes both grid buffers — used when a replay restarts (loops back
   * to step 0 with a fresh seed). Not needed on first construction:
   * WebGPU buffers are zero-initialized on creation, matching
   * environment.py's `torch.zeros(...)`. */
  reset(): void {
    writeFloat32(this.device, this.gridA, 0, this.zeroPlane);
    writeFloat32(this.device, this.gridB, 0, this.zeroPlane);
  }

  destroy(): void {
    this.gridA.destroy();
    this.gridB.destroy();
    this.gradient.destroy();
    this.depositScratch.destroy();
  }
}

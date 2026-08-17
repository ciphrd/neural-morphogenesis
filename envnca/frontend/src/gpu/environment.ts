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
  // decay, live-adjustable — see setDecay() and environment.wgsl's own
  // EnvPhysics comment for why this is a real uniform buffer rather than
  // a templateShader() const like width/height/channels above.
  readonly physicsUniform: GPUBuffer;

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
    this.physicsUniform = device.createBuffer({
      size: 4, // EnvPhysics: 1x f32 (decay)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    writeFloat32(device, this.physicsUniform, 0, new Float32Array([decay]));

    const source = templateShader(shaderSrc, { CHANNELS: channels, WIDTH: width, HEIGHT: height });
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
          { binding: 4, resource: { buffer: this.physicsUniform } },
        ],
      }),
      device.createBindGroup({
        layout: this.diffuseDecayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gridB } },
          { binding: 3, resource: { buffer: this.gridA } },
          { binding: 4, resource: { buffer: this.physicsUniform } },
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

  /** Live-updates the EnvPhysics uniform — a plain buffer write, no
   * pipeline recreation and no effect on the grid's current contents, so
   * this is safe to call on every tick of a "Physics" panel slider
   * without disturbing the rollout currently in flight. */
  setDecay(decay: number): void {
    writeFloat32(this.device, this.physicsUniform, 0, new Float32Array([decay]));
  }

  destroy(): void {
    this.gridA.destroy();
    this.gridB.destroy();
    this.gradient.destroy();
    this.depositScratch.destroy();
    this.physicsUniform.destroy();
  }
}

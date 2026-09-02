// TS wrapper around environment.wgsl's two chemical lifecycles. Cell-owned
// projection materializes per-cell state into buffer 0 each round; persistent
// environment ping-pongs a spatial field through diffusion/decay and merges
// direct policy deposits. Both expose the same parity-indexed sensing buffers.

import environmentSrc from "../../../core/environment.wgsl?raw";
import {
  homogeneousChemicalChannelProfiles,
  packChemicalChannelLayout,
  type ChemicalChannelProfile,
  type PackedChemicalLayout,
} from "./chemicalChannels";
import { templateShader } from "./shaderTemplate";
import { ceilDiv, flatDispatch2D, writeFloat32 } from "./gpuUtil";
import type { ChemicalCommunicationArchitecture } from "./types";
import { GRID_N } from "./mpmCore";

export interface EnvironmentConfig {
  channels: number;
  width: number;
  height: number;
  // Legacy run-metadata fields; transient fields do not use either value.
  decay: number;
  depositRate: number;
  normalizeDepositsByLocalDensity?: boolean;
  depositDensityReference?: number;
  advectionDt?: number;
  chemicalCommunicationArchitecture?: ChemicalCommunicationArchitecture;
  channelProfiles?: readonly ChemicalChannelProfile[];
}

const CLEAR_WORKGROUP = 256;
const GRID_WORKGROUP = 16;

export class Environment {
  readonly chemicalCommunicationArchitecture: ChemicalCommunicationArchitecture;
  readonly channels: number;
  readonly width: number;
  readonly height: number;
  readonly layout: PackedChemicalLayout;
  readonly maxWidth: number;
  readonly maxHeight: number;
  // Public so agents.ts can build its own parity-indexed bind group
  // variants against the exact same buffers, kept in lockstep by the one
  // parity counter this class owns.
  readonly buffers: [GPUBuffer, GPUBuffer];
  readonly gradient: GPUBuffer;
  readonly depositScratch: GPUBuffer;

  private readonly device: GPUDevice;
  private baseDecay: number;
  private baseDepositRate: number;
  private normalizeDepositsByLocalDensity: boolean;
  private depositDensityReference: number;
  private advectionDt: number;
  private readonly physicsUniform: GPUBuffer;

  private readonly clearScratchPipeline: GPUComputePipeline;
  private readonly clearScratchBindGroup: GPUBindGroup;
  private readonly materializeSplatPipeline: GPUComputePipeline;
  private readonly materializeSplatBindGroup: GPUBindGroup;
  private readonly computeGradientPipeline: GPUComputePipeline;
  private readonly computeGradientBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly mergeDepositPipeline: GPUComputePipeline;
  private readonly diffuseDecayPipeline: GPUComputePipeline;
  private readonly mergeDepositBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly diffuseDecayBindGroups: [GPUBindGroup, GPUBindGroup];

  private readonly clearDispatch: [number, number];
  private readonly gridDispatch: [number, number, number];

  private _parity = 0;
  get parity(): number {
    return this._parity;
  }

  constructor(device: GPUDevice, config: EnvironmentConfig, mpmGridVelocity: GPUBuffer) {
    this.device = device;
    this.channels = config.channels;
    this.width = config.width;
    this.height = config.height;
    this.layout = packChemicalChannelLayout(
      config.width,
      config.height,
      config.channelProfiles ?? homogeneousChemicalChannelProfiles(config.channels),
    );
    this.maxWidth = this.layout.maxWidth;
    this.maxHeight = this.layout.maxHeight;
    this.chemicalCommunicationArchitecture = config.chemicalCommunicationArchitecture ?? "cell-owned-projection";
    this.baseDecay = config.decay;
    this.baseDepositRate = config.depositRate;
    this.normalizeDepositsByLocalDensity = config.normalizeDepositsByLocalDensity ?? false;
    this.depositDensityReference = Math.max(0, config.depositDensityReference ?? 1.0);
    this.advectionDt = Math.max(0, config.advectionDt ?? 0);

    const total = this.layout.total;
    const scratchTotal = total * 2;
    const f32 = 4;

    this.buffers = [
      device.createBuffer({ size: total * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      device.createBuffer({ size: total * f32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    ];
    this.gradient = device.createBuffer({ size: total * 2 * f32, usage: GPUBufferUsage.STORAGE });
    this.depositScratch = device.createBuffer({ size: scratchTotal * f32, usage: GPUBufferUsage.STORAGE });
    this.physicsUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setCommunicationTimestep(1, 1);

    const module = device.createShaderModule({
      code: templateShader(environmentSrc, {
        CHANNELS: config.channels,
        GRID_N,
        ...this.layout.shaderConstants,
      }),
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
        { binding: 4, resource: { buffer: this.physicsUniform } },
      ],
    });

    this.computeGradientPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "computeGradient" } });
    this.mergeDepositPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "mergeDeposit" } });
    this.diffuseDecayPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "diffuseDecay" } });

    this.computeGradientBindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.computeGradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers[p] } },
          { binding: 1, resource: { buffer: this.gradient } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];
    this.mergeDepositBindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.mergeDepositPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers[p] } },
          { binding: 2, resource: { buffer: this.depositScratch } },
          { binding: 4, resource: { buffer: this.physicsUniform } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];
    this.diffuseDecayBindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.diffuseDecayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers[p] } },
          { binding: 3, resource: { buffer: this.buffers[1 - p] } },
          { binding: 4, resource: { buffer: this.physicsUniform } },
          { binding: 5, resource: { buffer: mpmGridVelocity } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];

    this.clearDispatch = flatDispatch2D(
      scratchTotal,
      CLEAR_WORKGROUP,
      device.limits.maxComputeWorkgroupsPerDimension,
    );
    this.gridDispatch = [
      ceilDiv(this.maxWidth, GRID_WORKGROUP),
      ceilDiv(this.maxHeight, GRID_WORKGROUP),
      config.channels,
    ];
  }

  /** Configure one persistent-field evolution per macro tick, while returning
   * the smaller dt used by each of that tick's neural deliberation rounds. */
  setCommunicationTimestep(rounds: number, speed: number): number {
    const macroDt = Math.max(0, speed);
    const neuralDt = macroDt / Math.max(1, Math.round(rounds));
    const decay = Math.pow(Math.max(0, Math.min(1, this.baseDecay)), macroDt);
    writeFloat32(this.device, this.physicsUniform, 0, new Float32Array([
      decay,
      this.baseDepositRate * macroDt,
      Math.min(macroDt, 1),
      this.normalizeDepositsByLocalDensity ? 1 : 0,
      this.depositDensityReference,
      this.advectionDt,
      0, 0,
    ]));
    return neuralDt;
  }

  setPhysics(
    decay: number,
    depositRate: number,
    rounds: number,
    speed: number,
    normalizeDepositsByLocalDensity = false,
    depositDensityReference = 1.0,
  ): number {
    this.baseDecay = decay;
    this.baseDepositRate = depositRate;
    this.normalizeDepositsByLocalDensity = normalizeDepositsByLocalDensity;
    this.depositDensityReference = Math.max(0, depositDensityReference);
    return this.setCommunicationTimestep(rounds, speed);
  }

  setAdvectionTimestep(dt: number): void {
    this.advectionDt = Math.max(0, dt);
    writeFloat32(this.device, this.physicsUniform, 5 * 4, new Float32Array([this.advectionDt]));
  }

  /** Zeroes both grid buffers and resets parity to 0 — call at the start
   * of every rollout (a fresh chemical field, same as a fresh
   * trainer/environment.py Environment instance each Python-side
   * rollout). */
  reset(): void {
    const zeros = new Float32Array(this.layout.total);
    writeFloat32(this.device, this.buffers[0], 0, zeros);
    writeFloat32(this.device, this.buffers[1], 0, zeros);
    this._parity = 0;
  }

  /** Starts a communication round by removing every previous splat. */
  encodeClear(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.clearScratchPipeline);
    pass.setBindGroup(0, this.clearScratchBindGroup);
    pass.dispatchWorkgroups(...this.clearDispatch);
    pass.end();
  }

  /** Materializes the just-published cell states, then computes the field
   * gradient shared by every brain invocation in this round. */
  encodeSense(encoder: GPUCommandEncoder): void {
    let pass: GPUComputePassEncoder;
    if (this.chemicalCommunicationArchitecture === "cell-owned-projection") {
      pass = encoder.beginComputePass();
      pass.setPipeline(this.materializeSplatPipeline);
      pass.setBindGroup(0, this.materializeSplatBindGroup);
      pass.dispatchWorkgroups(...this.clearDispatch);
      pass.end();
    }

    pass = encoder.beginComputePass();
    pass.setPipeline(this.computeGradientPipeline);
    pass.setBindGroup(0, this.computeGradientBindGroups[this._parity]);
    pass.dispatchWorkgroups(...this.gridDispatch);
    pass.end();
  }

  /** Bring persistent substrate forward through the previous MPM motion
   * before the policy senses it, including divergent growth flow. */
  encodePreparePersistent(encoder: GPUCommandEncoder): void {
    if (this.chemicalCommunicationArchitecture !== "persistent-environment") return;
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.diffuseDecayPipeline);
    pass.setBindGroup(0, this.diffuseDecayBindGroups[this._parity]);
    pass.dispatchWorkgroups(...this.gridDispatch);
    pass.end();
    this._parity = 1 - this._parity;
  }

  /** Add the final neural round's direct writes to the prepared field. */
  encodeMergePersistent(encoder: GPUCommandEncoder): void {
    if (this.chemicalCommunicationArchitecture !== "persistent-environment") return;
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.mergeDepositPipeline);
    pass.setBindGroup(0, this.mergeDepositBindGroups[this._parity]);
    pass.dispatchWorkgroups(...this.clearDispatch);
    pass.end();
  }

  destroy(): void {
    this.buffers[0].destroy();
    this.buffers[1].destroy();
    this.gradient.destroy();
    this.depositScratch.destroy();
    this.physicsUniform.destroy();
  }
}

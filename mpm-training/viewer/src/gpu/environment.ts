// TS wrapper around environment.wgsl — owns the ping-pong grid buffers,
// the gradient/deposit-scratch buffers, and the 4 compute pipelines
// (clearScratch/computeGradient/mergeDeposit/diffuseDecay). Mirrors
// envnca/frontend/src/gpu/environment.ts's own shape: every bind group
// this needs is precomputed once at construction (both ping-pong
// variants, indexed by parity), never rebuilt per frame — simulation.ts
// just picks the right index each macro step.

import environmentSrc from "../../../core/environment.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import { ceilDiv, writeFloat32 } from "./gpuUtil";

export interface EnvironmentConfig {
  channels: number;
  width: number;
  height: number;
  decay: number;
  // Multiplier on this macro step's own accumulated deposits, applied
  // right before they're folded into the field — see
  // core/environment.wgsl's own EnvPhysics/mergeDeposit comments.
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
  private readonly physicsUniform: GPUBuffer;

  private readonly clearScratchPipeline: GPUComputePipeline;
  private readonly clearScratchBindGroup: GPUBindGroup;
  private readonly computeGradientPipeline: GPUComputePipeline;
  private readonly mergeDepositPipeline: GPUComputePipeline;
  private readonly diffuseDecayPipeline: GPUComputePipeline;
  private readonly computeGradientBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly mergeDepositBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly diffuseDecayBindGroups: [GPUBindGroup, GPUBindGroup];

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
    this.physicsUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setPhysics({ decay: config.decay, depositRate: config.depositRate });

    const module = device.createShaderModule({
      code: templateShader(environmentSrc, { CHANNELS: config.channels, WIDTH: config.width, HEIGHT: config.height }),
    });

    this.clearScratchPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "clearScratch" } });
    this.clearScratchBindGroup = device.createBindGroup({
      layout: this.clearScratchPipeline.getBindGroupLayout(0),
      entries: [{ binding: 2, resource: { buffer: this.depositScratch } }],
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
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];

    this.clearDispatch = ceilDiv(total, CLEAR_WORKGROUP);
    this.gridDispatch = [ceilDiv(config.width, GRID_WORKGROUP), ceilDiv(config.height, GRID_WORKGROUP), config.channels];
  }

  /** Writes the timestep-scaled EnvPhysics fields together — safe to call on
   * every tick of either the Decay or Deposit rate PhysicsPanel slider,
   * same "plain buffer write, no pipeline recreation" contract every
   * other live-adjustable setter in this project's own gpu/ layer has. */
  setPhysics(physics: { decay: number; depositRate: number; diffusionStep?: number }): void {
    writeFloat32(this.device, this.physicsUniform, 0, new Float32Array([
      physics.decay,
      physics.depositRate,
      physics.diffusionStep ?? 1.0,
      0.0,
    ]));
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

  /** Sense: clearScratch + computeGradient over the current grid. Call
   * once per macro step, before the NN forward pass reads it. Encodes
   * into `encoder`, does not submit. */
  encodeSense(encoder: GPUCommandEncoder): void {
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.clearScratchPipeline);
    pass.setBindGroup(0, this.clearScratchBindGroup);
    pass.dispatchWorkgroups(this.clearDispatch);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.computeGradientPipeline);
    pass.setBindGroup(0, this.computeGradientBindGroups[this._parity]);
    pass.dispatchWorkgroups(...this.gridDispatch);
    pass.end();
  }

  /** Diffuse+decay the CURRENT grid (as left by the previous macro step,
   * before this step's own deposit touches anything) into the other
   * buffer, then merge the NN forward pass's fresh deposit directly on
   * top of that already-decayed result — deliberately decay-THEN-
   * deposit, not deposit-then-decay (this used to run in the opposite
   * order, decaying a step's own brand-new deposit before it was ever
   * sensed by anyone, so a deposit's own value never actually reached
   * its own full depositRate*value magnitude at any sensed step). Flips
   * parity at the end, same as before. Call once per macro step, after
   * the NN forward pass has written into depositScratch. Encodes into
   * `encoder`, does not submit.
   *
   * Both passes' own WGSL bodies (core/environment.wgsl's own
   * mergeDeposit()/diffuseDecay()) are UNCHANGED — mergeDeposit's own
   * binding 0 doesn't know or care which physical buffer it's bound to,
   * it just adds scratch onto whatever's there. The entire reordering
   * lives here: diffuseDecay dispatches FIRST now (reading
   * buffers[this._parity], the pre-deposit "current" grid, writing the
   * decayed+blurred result into buffers[1-this._parity]), then
   * mergeDeposit dispatches SECOND, using
   * mergeDepositBindGroups[1-this._parity] — NOT this._parity — so its
   * own binding 0 targets that same just-decayed buffer (what
   * diffuseDecay just wrote), adding this step's own deposit on top of
   * it, undecayed. */
  encodeMergeAndDecay(encoder: GPUCommandEncoder): void {
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.diffuseDecayPipeline);
    pass.setBindGroup(0, this.diffuseDecayBindGroups[this._parity]);
    pass.dispatchWorkgroups(...this.gridDispatch);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.mergeDepositPipeline);
    pass.setBindGroup(0, this.mergeDepositBindGroups[1 - this._parity]);
    pass.dispatchWorkgroups(this.clearDispatch);
    pass.end();

    this._parity = 1 - this._parity;
  }

  destroy(): void {
    this.buffers[0].destroy();
    this.buffers[1].destroy();
    this.gradient.destroy();
    this.depositScratch.destroy();
    this.physicsUniform.destroy();
  }
}

// TS wrapper around gpu/nnProbe.wgsl — see that file's own module
// docstring for why it duplicates (rather than shares) core/agents.wgsl's
// own sensing/evalPolicy code. Built once per rebuild() (gpu/simulation.ts),
// same lifetime as every other GPU object there — binds directly against
// Agents'/Environment's/MpmCore's own live buffers (Agents.weightsState/
// physicsState/headingState getters, MpmCore.positions, Environment's own
// public buffers/gradient), so a probe always reflects whatever's
// currently loaded, "Randomize weights" included (Agents.randomizeWeights()
// is a plain write to the exact same buffer this binds).
//
// Two bind group variants, indexed by Environment's own ping-pong parity
// (exactly Environment's/Agents' own pattern for the same reason: which
// of environment.buffers[0]/[1] is "current" flips every macro step, and
// building a fresh bind group per probe() call just to pick the right one
// would be wasteful for something called on its own timer, independent of
// step()'s own cadence).

import nnProbeSrc from "./nnProbe.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import type { Agents } from "./agents";
import type { Environment } from "./environment";
import type { MpmCore } from "./mpmCore";

export interface NetworkProbe {
  channels: number;
  hiddenDim: number;
  /** IN_DIM = channels*3+2 — [value×channels, gradForward×channels, gradLateral×channels, (x,y) spawn-center-relative position]. */
  input: Float32Array;
  /** hiddenDim sin activations from the PRIMARY (non-chirality-mirrored) pass — see nnProbe.wgsl's own comment. */
  hidden: Float32Array;
  /** channels*4, spot-major (front,left,back,right), channels within each spot — chirality-averaged, matches what agentStep() actually deposits. */
  envWrite: Float32Array;
  angularAccel: number;
  strafe: [number, number];
  /** The last channel's own sensed value, clamped to [0,1] — this step's growth/split probability (not a network output). */
  splitProb: number;
}

function probeLayout(channels: number, hiddenDim: number) {
  // +2 — must match nnProbe.wgsl's own IN_DIM (and core/agents.wgsl's)
  // exactly: the agent's own spawn-center-relative (x,y) position,
  // appended after the per-channel value/grad_forward/grad_lateral
  // triples.
  const inDim = channels * 3 + 2;
  const envWriteDim = channels * 4;
  const inputOffset = 0;
  const hiddenOffset = inputOffset + inDim;
  const envWriteOffset = hiddenOffset + hiddenDim;
  const angularOffset = envWriteOffset + envWriteDim;
  const strafeOffset = angularOffset + 1;
  const splitProbOffset = strafeOffset + 2;
  const totalFloats = splitProbOffset + 1;
  return { inDim, envWriteDim, inputOffset, hiddenOffset, envWriteOffset, angularOffset, strafeOffset, splitProbOffset, totalFloats };
}

export class NnProbe {
  private readonly device: GPUDevice;
  private readonly channels: number;
  private readonly hiddenDim: number;
  private readonly outBuffer: GPUBuffer;
  private readonly staging: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroups: [GPUBindGroup, GPUBindGroup];
  // Guards against overlapping calls — a caller on its own timer, not
  // gated by step()'s own cadence, could otherwise fire a new probe
  // before a previous mapAsync() resolves (GPUBuffer.mapAsync() throws if
  // called while a previous mapping on the same buffer is still pending).
  private busy = false;

  constructor(device: GPUDevice, mpmCore: MpmCore, environment: Environment, agents: Agents, config: { channels: number; hiddenDim: number; chirality: boolean }) {
    this.device = device;
    this.channels = config.channels;
    this.hiddenDim = config.hiddenDim;

    const { totalFloats } = probeLayout(config.channels, config.hiddenDim);
    this.outBuffer = device.createBuffer({ size: totalFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // MAP_READ can't be combined with STORAGE — same reason
    // Agents.grownCountStaging exists separately from growthCountBuffer.
    this.staging = device.createBuffer({ size: totalFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const module = device.createShaderModule({
      code: templateShader(nnProbeSrc, {
        CHANNELS: config.channels,
        HIDDEN_DIM: config.hiddenDim,
        FIELD_WIDTH: environment.width,
        FIELD_HEIGHT: environment.height,
        CHIRALITY: config.chirality ? "true" : "false",
      }),
    });
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "probe" } });
    this.bindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: agents.weightsState } },
          { binding: 1, resource: { buffer: mpmCore.positions } },
          { binding: 2, resource: { buffer: agents.headingState } },
          { binding: 3, resource: { buffer: environment.buffers[p] } },
          { binding: 4, resource: { buffer: environment.gradient } },
          { binding: 5, resource: { buffer: agents.physicsState } },
          { binding: 6, resource: { buffer: this.outBuffer } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];
  }

  /** Encodes+submits the probe pass against whichever of
   * environment.buffers is CURRENT (`parity`, caller-supplied — same
   * value Environment.parity itself reports at call time) and
   * asynchronously reads back the result. Returns null while a previous
   * call is still resolving (see `busy`'s own comment) — the caller's
   * next scheduled tick picks it back up rather than this method queuing
   * or awaiting the in-flight one. */
  async probe(parity: number): Promise<NetworkProbe | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      const layout = probeLayout(this.channels, this.hiddenDim);

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroups[parity]);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(this.outBuffer, 0, this.staging, 0, layout.totalFloats * 4);
      this.device.queue.submit([encoder.finish()]);

      await this.staging.mapAsync(GPUMapMode.READ);
      // Copy out of the mapped range before unmap() — that ArrayBuffer
      // is invalidated the instant unmap() runs (same pattern
      // Agents.readGrownCount() already follows for the same reason).
      const raw = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();

      return {
        channels: this.channels,
        hiddenDim: this.hiddenDim,
        input: raw.slice(layout.inputOffset, layout.inputOffset + layout.inDim),
        hidden: raw.slice(layout.hiddenOffset, layout.hiddenOffset + this.hiddenDim),
        envWrite: raw.slice(layout.envWriteOffset, layout.envWriteOffset + layout.envWriteDim),
        angularAccel: raw[layout.angularOffset],
        strafe: [raw[layout.strafeOffset], raw[layout.strafeOffset + 1]],
        splitProb: raw[layout.splitProbOffset],
      };
    } finally {
      this.busy = false;
    }
  }

  destroy(): void {
    this.outBuffer.destroy();
    this.staging.destroy();
  }
}

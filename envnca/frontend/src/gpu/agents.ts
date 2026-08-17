import shaderSrc from "./agents.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import { seedAgentPositions, type SpawnDistribution } from "./rng";
import { writeFloat32 } from "./gpuUtil";
import type { UpdateRuleWeights } from "./types";
import type { GpuEnvironment } from "./environment";

export interface AgentsConfig {
  width: number;
  height: number;
  channels: number;
  hiddenDim: number;
  agentCount: number;
  spawnSpread: number;
  maxSpeed: number;
  maxAccel: number;
  maxStrafe: number;
}

interface WeightLayout {
  fc1wOffset: number;
  fc1bOffset: number;
  fc2wOffset: number;
  fc2bOffset: number;
  totalFloats: number;
}

function computeWeightLayout(channels: number, hiddenDim: number): WeightLayout {
  const inDim = 3 * channels + 2; // value + grad_forward + grad_lateral + heading(cos,sin)
  const outDim = channels + 4;
  const fc1wOffset = 0;
  const fc1bOffset = fc1wOffset + hiddenDim * inDim;
  const fc2wOffset = fc1bOffset + hiddenDim;
  const fc2bOffset = fc2wOffset + outDim * hiddenDim;
  const totalFloats = fc2bOffset + outDim;
  return { fc1wOffset, fc1bOffset, fc2wOffset, fc2bOffset, totalFloats };
}

function flattenWeights(weights: UpdateRuleWeights, layout: WeightLayout): Float32Array {
  const data = new Float32Array(layout.totalFloats);
  let idx = layout.fc1wOffset;
  for (const row of weights.fc1w) for (const v of row) data[idx++] = v;
  idx = layout.fc1bOffset;
  for (const v of weights.fc1b) data[idx++] = v;
  idx = layout.fc2wOffset;
  for (const row of weights.fc2w) for (const v of row) data[idx++] = v;
  idx = layout.fc2bOffset;
  for (const v of weights.fc2b) data[idx++] = v;
  return data;
}

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

/** Owns per-agent state (positions/velocity) and the network weights
 * buffer, plus the single fused sense->MLP->move->deposit-scatter
 * compute pass — mirroring agent_state.py + update_rule.py +
 * simulation.py::step()'s per-agent half. Two bind-group variants are
 * precomputed (one per grid ping-pong parity, see GpuEnvironment), since
 * this pass reads whichever grid buffer is "current" for the step it's
 * running in. */
export class GpuAgents {
  readonly positions: GPUBuffer;
  readonly velocity: GPUBuffer;
  readonly weights: GPUBuffer;
  // maxSpeed/maxAccel/maxStrafe, live-adjustable — see setPhysics() and
  // agents.wgsl's own AgentPhysics comment for why this is a real
  // uniform buffer rather than a templateShader() const like
  // width/height/hiddenDim above.
  readonly physicsUniform: GPUBuffer;

  readonly pipeline: GPUComputePipeline;
  // index 0: gridCurrent=environment.gridA. index 1: gridCurrent=environment.gridB.
  readonly bindGroups: readonly [GPUBindGroup, GPUBindGroup];
  readonly dispatchGroups: number;
  // Distinct from dispatchGroups (a workgroup count) — this is the raw
  // agent count, needed by gpu/render.ts's instanced-quad draw call.
  readonly agentCount: number;

  private readonly device: GPUDevice;
  private readonly config: AgentsConfig;
  private readonly weightLayout: WeightLayout;

  constructor(device: GPUDevice, environment: GpuEnvironment, config: AgentsConfig) {
    this.device = device;
    this.config = config;
    const { width, height, channels, hiddenDim, agentCount, maxSpeed, maxAccel, maxStrafe } = config;

    this.positions = device.createBuffer({
      size: agentCount * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.velocity = device.createBuffer({
      size: agentCount * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.weightLayout = computeWeightLayout(channels, hiddenDim);
    this.weights = device.createBuffer({
      size: this.weightLayout.totalFloats * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.physicsUniform = device.createBuffer({
      size: 12, // AgentPhysics: 3x f32 (maxSpeed, maxAccel, maxStrafe)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    writeFloat32(device, this.physicsUniform, 0, new Float32Array([maxSpeed, maxAccel, maxStrafe]));

    const source = templateShader(shaderSrc, {
      CHANNELS: channels,
      WIDTH: width,
      HEIGHT: height,
      HIDDEN: hiddenDim,
      AGENT_COUNT: agentCount,
      FC1W_OFFSET: this.weightLayout.fc1wOffset,
      FC1B_OFFSET: this.weightLayout.fc1bOffset,
      FC2W_OFFSET: this.weightLayout.fc2wOffset,
      FC2B_OFFSET: this.weightLayout.fc2bOffset,
    });
    const module = device.createShaderModule({ code: source });

    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "agentStep" } });

    const makeBindGroup = (gridCurrent: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gridCurrent } },
          { binding: 1, resource: { buffer: environment.gradient } },
          { binding: 2, resource: { buffer: environment.depositScratch } },
          { binding: 3, resource: { buffer: this.weights } },
          { binding: 4, resource: { buffer: this.positions } },
          { binding: 5, resource: { buffer: this.velocity } },
          { binding: 6, resource: { buffer: this.physicsUniform } },
        ],
      });

    this.bindGroups = [makeBindGroup(environment.gridA), makeBindGroup(environment.gridB)];
    this.dispatchGroups = ceilDiv(agentCount, 64);
    this.agentCount = agentCount;
  }

  loadWeights(weights: UpdateRuleWeights): void {
    writeFloat32(this.device, this.weights, 0, flattenWeights(weights, this.weightLayout));
  }

  /** Live-updates the AgentPhysics uniform — a plain buffer write, no
   * pipeline recreation and no effect on positions/velocity, so this is
   * safe to call on every tick of a "Physics" panel slider without
   * disturbing the rollout currently in flight. */
  setPhysics(maxSpeed: number, maxAccel: number, maxStrafe: number): void {
    writeFloat32(this.device, this.physicsUniform, 0, new Float32Array([maxSpeed, maxAccel, maxStrafe]));
  }

  /** Re-seeds agent positions/velocity for a fresh rollout — jitter is a
   * deterministic (but not torch-bit-exact — see gpu/rng.ts) PRNG seeded
   * with `seed`. `distribution` picks the initial-spread shape (viewer-
   * only — see rng.ts's SpawnDistribution docstring), defaulting to the
   * same tight jitter training itself uses. */
  resetAgents(gridWidth: number, gridHeight: number, seed: number, distribution: SpawnDistribution = "default"): void {
    const { agentCount, spawnSpread } = this.config;
    writeFloat32(
      this.device,
      this.positions,
      0,
      seedAgentPositions(agentCount, gridWidth, gridHeight, spawnSpread, seed, distribution)
    );
    writeFloat32(this.device, this.velocity, 0, new Float32Array(agentCount * 2));
  }

  destroy(): void {
    this.positions.destroy();
    this.velocity.destroy();
    this.weights.destroy();
    this.physicsUniform.destroy();
  }
}

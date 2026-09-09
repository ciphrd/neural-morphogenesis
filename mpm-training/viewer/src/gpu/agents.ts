
import agentsSrc from "../../../core/agents.wgsl?raw";
import growthFieldSrc from "../../../core/growthField.wgsl?raw";
import coreConstantsConfig from "../../../core/config.json";
const coreConstants = coreConstantsConfig.simulation;
import { templateShader } from "./shaderTemplate";
import { ceilDiv, writeFloat32 } from "./gpuUtil";
import type { Environment } from "./environment";
import { GROWTH_FIELD_CHANNELS, GRID_N, INV_DX, MAX_PARTICLES, NODE_COUNT, REPULSION_FIELD_N, type MpmCore } from "./mpmCore";
import { policyHasRecurrence, type ChemicalCommunicationArchitecture, type PolicyArchitecture, type UpdateRuleWeights } from "./types";
import policyParametersConfig from "../../../core/config.json";
const policyParameters = policyParametersConfig.policy;
import { policyWeightsShapeError } from "./policyEval";

const WORKGROUP = 64;

// core/agents.wgsl's own ParticleMeta struct. The two-float alignment cache
// occupies the former heading/turn-state bytes, preserving its packed ABI.
const particleMetaStride = (channels: number) => Math.ceil((60 + channels * 4) / 16) * 16;
// AgentState places its runtime ParticleMeta array after one atomic u32 and
// 63 padding u32s. 256 is also a legal standalone storage-binding offset.
export const PARTICLE_META_BUFFER_OFFSET = 256;

export interface AgentsConfig {
  channels: number;
  hiddenDim: number;
  policyArchitecture: PolicyArchitecture;
  chemicalCommunicationArchitecture: ChemicalCommunicationArchitecture;
  maxEnvWrite: number;
  sampleSpacing: number;
  friction: number;
  growthEnabled: number;
  maxActiveParticles: number;
  spawnX: number;
  spawnY: number;
  elasticStrainScale: number;
  elasticStrainInputsEnabled: boolean;
  chemicalValueInputMultiplier: number;
  chemicalGradientInputScale: number;
}

function weightLayout(channels: number, hiddenDim: number, architecture: PolicyArchitecture = "stateless-128") {
  // core/agents.wgsl's IN_DIM: value + heading-forward gradient + lateral
  // gradient per channel, with no positional inputs.
  const stateful = policyHasRecurrence(architecture);
  const inDim = channels * 3 + 6 + (stateful ? 8 : 0);
  // Chemical deltas plus a two-component local growth vector and either
  // private-state updates or RGB.
  const outDim = channels + (stateful ? 21 : 5);
  const fc1wOffset = 0;
  const fc1bOffset = fc1wOffset + hiddenDim * inDim;
  const fc2wOffset = fc1bOffset + hiddenDim;
  const fc2bOffset = fc2wOffset + outDim * hiddenDim;
  const totalFloats = fc2bOffset + outDim;
  return { inDim, outDim, fc1wOffset, fc1bOffset, fc2wOffset, fc2bOffset, totalFloats };
}

/** Flattens {fc1w,fc1b,fc2w,fc2b} (nn.Linear's own (out,in) orientation —
 * UpdateRule.export_weights()'s exact shape) into one Float32Array in
 * the fc1w/fc1b/fc2w/fc2b order agents.wgsl's own FC1W_OFFSET/etc.
 * consts expect. */
function flattenWeights(weights: UpdateRuleWeights, channels: number, hiddenDim: number, architecture: PolicyArchitecture): Float32Array {
  const { totalFloats } = weightLayout(channels, hiddenDim, architecture);
  const shapeError = policyWeightsShapeError(weights, channels, hiddenDim, architecture);
  if (shapeError) throw new Error(shapeError);
  const out = new Float32Array(totalFloats);
  let i = 0;
  for (const row of weights.fc1w) for (const v of row) out[i++] = v;
  for (const v of weights.fc1b) out[i++] = v;
  for (const row of weights.fc2w) for (const v of row) out[i++] = v;
  for (const v of weights.fc2b) out[i++] = v;
  return out;
}

function policyRandom(seed: number): () => number {
  if (seed === undefined) return Math.random;
  let state = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let x = state;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/** Samples a fresh viewer-side policy seed. Passing the previous seed makes
 * the caller's "random" action guarantee a different value even in the
 * extremely unlikely event that the random source repeats itself. */
export function randomPolicySeed(previous?: number): number {
  const values = new Uint32Array(1);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(values);
  } else {
    values[0] = Math.floor(Math.random() * 0x100000000) >>> 0;
  }
  return values[0] === previous ? (values[0] + 1) >>> 0 : values[0];
}

function randomSymmetric(bound: number, random: () => number): number {
  return (random() * 2 - 1) * bound;
}

function randomHead(
  outDim: number,
  inDim: number,
  config: { weightGain: number; biasCenter: number[]; biasJitter: number },
  random: () => number,
): { w: number[][]; b: number[] } {
  const bound = config.weightGain * Math.sqrt(6 / (inDim + outDim));
  const w = Array.from({ length: outDim }, () => Array.from({ length: inDim }, () => randomSymmetric(bound, random)));
  const centers = config.biasCenter.length === 1
    ? Array.from({ length: outDim }, () => config.biasCenter[0])
    : config.biasCenter;
  if (centers.length !== outDim) throw new Error(`policy head has ${centers.length} bias centers for ${outDim} outputs`);
  const b = centers.map((center) => center + randomSymmetric(config.biasJitter, random));
  return { w, b };
}

/** Exported for net/trainingSocket.ts's own placeholder GenerationRecord
 * (random weights to render SOMETHING while generation 0 is still being
 * evaluated — see that hook's own comment) — the exact same generator
 * Agents.randomizeWeights() below uses for the "Randomize weights"
 * button, just also reachable without an Agents instance to call it on. */
export function randomWeights(
  channels: number,
  hiddenDim: number,
  architecture: PolicyArchitecture = "stateless-128",
  seed?: number,
): UpdateRuleWeights {
  const random = policyRandom(seed ?? randomPolicySeed());
  const { inDim } = weightLayout(channels, hiddenDim, architecture);
  const trunk = policyParameters.trunk;
  const trunkBound = trunk.weightGain * Math.sqrt(6 / (inDim + hiddenDim));
  const fc1w = Array.from({ length: hiddenDim }, () =>
    Array.from({ length: inDim }, () => randomSymmetric(trunkBound, random))
  );
  const fc1b = Array.from({ length: hiddenDim }, () => randomSymmetric(trunk.biasJitter, random));
  const common = [
    [channels, policyParameters.heads.chemical],
    [2, policyParameters.heads.growthVector],
  ] as const;
  const specs = policyHasRecurrence(architecture)
    ? [...common, [8, policyParameters.heads.stateDelta] as const, [8, policyParameters.heads.stateGate] as const, [3, policyParameters.heads.color] as const]
    : [...common, [3, policyParameters.heads.color] as const];
  const initialized = specs.map(([size, config]) => randomHead(size, hiddenDim, config, random));
  return {
    fc1w,
    fc1b,
    fc2w: initialized.flatMap((head) => head.w),
    fc2b: initialized.flatMap((head) => head.b),
  };
}

export class Agents {
  unresolvedSamples = 0;
  capacityBlocked = false;
  private refinementRounds = 0;
  private readonly device: GPUDevice;
  private readonly channels: number;
  private readonly hiddenDim: number;
  private readonly policyArchitecture: PolicyArchitecture;

  private readonly weightsBuffer: GPUBuffer;
  private readonly physicsUniform: GPUBuffer;
  private readonly agentStateBuffer: GPUBuffer;
  private readonly sampleCountStaging: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly splatPipeline: GPUComputePipeline;
  private readonly splatBindGroup: GPUBindGroup;
  private readonly particleMetaStride: number;
  private readonly communicationBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly commitBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly stepModeUniforms: [GPUBuffer, GPUBuffer];
  private readonly growthField: GPUBuffer;
  private readonly refinement: GPUBuffer;
  private readonly growthEntries: readonly string[];
  private readonly growthPipelines: readonly GPUComputePipeline[];
  private readonly growthBindGroup: GPUBindGroup;
  private readonly growthDispatches: readonly (number | null)[];
  private forcedGrowthFieldOverride = false;
  // Assigned via setActiveCount() in the constructor (also growth's own
  // baseline write), not directly — `!` tells TS's definite-assignment
  // check that's fine, it just can't see through the method call itself.
  private dispatch!: number;

  constructor(device: GPUDevice, mpmCore: MpmCore, environment: Environment, config: AgentsConfig) {
    this.device = device;
    this.channels = config.channels;
    this.hiddenDim = config.hiddenDim;
    this.policyArchitecture = config.policyArchitecture;
    this.particleMetaStride = particleMetaStride(config.channels);

    const layout = weightLayout(config.channels, config.hiddenDim, this.policyArchitecture);
    const { totalFloats } = layout;
    this.weightsBuffer = device.createBuffer({ size: totalFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // 128 bytes — original layout plus runtime neural-input controls and
    // trailing uniform-alignment padding.
    // The final spawnX/spawnY/maxActiveParticles fields are
    // NOT written by setPhysics() below; see setSpawnCenter()'s own
    // docstring for why those get a separate setter into this same
    // buffer instead.
    this.physicsUniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setPhysics({
      maxEnvWrite: config.maxEnvWrite,
      sampleSpacing: config.sampleSpacing,
      friction: config.friction,
      growthEnabled: config.growthEnabled,
    });
    this.setSpawnCenter(config.spawnX, config.spawnY);
    this.setMaxActiveParticles(config.maxActiveParticles);
    this.setElasticStrainScale(config.elasticStrainScale);
    this.setChemicalValueInputMultiplier(config.chemicalValueInputMultiplier);

    this.setChemicalGradientInputScale(config.chemicalGradientInputScale);


    this.setForcedGrowthControl(null, [1, 0], false);
    this.setForcedGrowthFieldOverride(null);

    this.agentStateBuffer = device.createBuffer({
      size: PARTICLE_META_BUFFER_OFFSET + MAX_PARTICLES * this.particleMetaStride,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // A separate, MAP_READ-capable buffer agentStateBuffer itself can't
    // be (STORAGE and MAP_READ are mutually exclusive usages in WebGPU)
    // — encodeReadSampleCount() copies into this every macro step,
    // readSampleCount() maps/reads/unmaps it asynchronously afterward.
    this.sampleCountStaging = device.createBuffer({ size: 12, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // Rollouts start with their configured initial particle count and grow
    // via splitting from there (simulation.ts's own restartRollout()
    // sets that count every rollout — this is just
    // a construction-time placeholder so there's a sane dispatch size
    // before the first rollout ever starts) — see types.ts's own
    // SimulationConfig.particles docstring for why that's now a CAP, not
    // a fixed starting count.
    this.setActiveCount(1);

    const filterableMorphology = device.features.has("float32-filterable");
    const morphologySampler = filterableMorphology
      ? device.createSampler({
          addressModeU: "repeat",
          addressModeV: "repeat",
          minFilter: "linear",
          magFilter: "linear",
        })
      : null;
    const module = device.createShaderModule({
      code: templateShader(agentsSrc, {
        CHANNELS: config.channels,
        HIDDEN_DIM: config.hiddenDim,
        IN_DIM: layout.inDim,
        OUT_DIM: layout.outDim,
        ...environment.layout.shaderConstants,
        MORPHOLOGY_FIELD_N: REPULSION_FIELD_N,
        MORPHOLOGY_GRADIENT_INPUT_SCALE: coreConstants.MORPHOLOGY_GRADIENT_INPUT_SCALE,
        // WGSL wants lowercase `true`/`false` — a raw JS boolean would
        // template-substitute as "true"/"false" too via String(), so
        // this one actually works either way, but spelled out for
        // parity with agents_gpu.py's own version of this same
        // gotcha (Python's str(bool) gives "True"/"False", invalid
        // WGSL, so that side needs the explicit conversion).
        STATEFUL: policyHasRecurrence(this.policyArchitecture) ? "true" : "false",
        CELL_OWNED_CHEMISTRY: (config.chemicalCommunicationArchitecture) === "cell-owned-projection" ? "true" : "false",
        PRIVATE_STATE_INPUTS: policyHasRecurrence(this.policyArchitecture)
          ? "for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) { inputVec[3u * CHANNELS + 6u + s] = tanh(agentState.particleMeta[pi].privateState[s]); }"
          : "",
        POLICY_TAIL_DECODE: policyHasRecurrence(this.policyArchitecture)
          ? "out.color = vec3<f32>(safeSigmoid(outVec[ENV_WRITE_DIM + 18u]), safeSigmoid(outVec[ENV_WRITE_DIM + 19u]), safeSigmoid(outVec[ENV_WRITE_DIM + 20u])); for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) { out.stateDelta[s] = outVec[ENV_WRITE_DIM + 2u + s]; out.stateGate[s] = safeSigmoid(outVec[ENV_WRITE_DIM + 2u + PRIVATE_STATE_DIM + s]); }"
          : "out.color = vec3<f32>(safeSigmoid(outVec[ENV_WRITE_DIM + 2u]), safeSigmoid(outVec[ENV_WRITE_DIM + 3u]), safeSigmoid(outVec[ENV_WRITE_DIM + 4u])); for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) { out.stateDelta[s] = 0.0; out.stateGate[s] = 0.0; }",
        ELASTIC_STRAIN_INPUTS_ENABLED: config.elasticStrainInputsEnabled ? "true" : "false",
        MORPHOLOGY_SAMPLER_DECLARATION: filterableMorphology
          ? "@group(0) @binding(14) var morphologySampler: sampler;"
          : "",
        MORPHOLOGY_SAMPLE_BODY: filterableMorphology
          ? "return textureSampleLevel(morphologyTexture, morphologySampler, (p + vec2<f32>(0.5)) / f32(MORPHOLOGY_FIELD_N), 0.0).x;"
          : "let base = vec2<i32>(floor(p)); let f = fract(p); let a = mix(morphologyLoad(base), morphologyLoad(base + vec2<i32>(1, 0)), f.x); let b = mix(morphologyLoad(base + vec2<i32>(0, 1)), morphologyLoad(base + vec2<i32>(1, 1)), f.x); return mix(a, b, f.y);",
      }),
    });
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "agentStep" } });
    this.splatPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "splatChemicalState" } });
    this.splatBindGroup = device.createBindGroup({
      layout: this.splatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: mpmCore.positions } },
        { binding: 2, resource: { buffer: mpmCore.activeCountUniform } },
        { binding: 5, resource: { buffer: environment.depositScratch } },
        { binding: 6, resource: { buffer: this.physicsUniform } },
        { binding: 7, resource: { buffer: this.agentStateBuffer } },
        { binding: 11, resource: { buffer: mpmCore.rest } },
      ],
    });
    this.stepModeUniforms = [0, 1].map((commit) => {
      const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, new Uint32Array([commit]));
      writeFloat32(device, buffer, 4, new Float32Array([1.0]));
      writeFloat32(device, buffer, 8, new Float32Array([1.0]));
      writeFloat32(device, buffer, 12, new Float32Array([1.0]));
      return buffer;
    }) as [GPUBuffer, GPUBuffer];

    const bindGroups = (commitGrowth: 0 | 1) => [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.weightsBuffer } },
          { binding: 1, resource: { buffer: mpmCore.positions } },
          { binding: 2, resource: { buffer: mpmCore.activeCountUniform } },
          { binding: 3, resource: { buffer: environment.buffers[p] } },
          { binding: 4, resource: { buffer: environment.gradient } },
          { binding: 5, resource: { buffer: environment.depositScratch } },
          { binding: 6, resource: { buffer: this.physicsUniform } },
          { binding: 7, resource: { buffer: this.agentStateBuffer } },
          { binding: 9, resource: { buffer: mpmCore.velocities } },
          // mpmCore.F/rest — same buffers ../core/'s own p2g.wgsl/g2p.wgsl
          // already read/write every physics substep — see
          // ../../../core/agents.wgsl's own module docstring for why
          // agentStep() now needs them too (a freshly-claimed particle
          // inherits its parent's CURRENT deformation state at split
          // time, rather than starting undeformed with zero rest history). mpmCore.C is
          // binding 8 so the split preserves the complete APIC state.
          { binding: 10, resource: { buffer: mpmCore.F } },
          { binding: 11, resource: { buffer: mpmCore.rest } },
          { binding: 12, resource: mpmCore.morphologyTexture.createView() },
          { binding: 13, resource: { buffer: this.stepModeUniforms[commitGrowth] } },
          ...(morphologySampler ? [{ binding: 14, resource: morphologySampler }] : []),
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];
    this.communicationBindGroups = bindGroups(0);
    this.commitBindGroups = bindGroups(1);

    this.growthField = mpmCore.growthField;
    const refineHashSize = 2 ** Math.ceil(Math.log2(6 * MAX_PARTICLES));
    const refineWords = refineHashSize + 5 * MAX_PARTICLES + 2;
    this.refinement = device.createBuffer({
      label: "conforming refinement scratch", size: 4 * refineWords, usage: GPUBufferUsage.STORAGE,
    });
    const growthModule = device.createShaderModule({
      code: templateShader(growthFieldSrc, {
        CHANNELS: config.channels, GRID_N, INV_DX,
        MORPHOLOGY_FIELD_N: REPULSION_FIELD_N,
        REFINE_CAPACITY: MAX_PARTICLES, REFINE_HASH_SIZE: refineHashSize,
      }),
    });
    const stages: [string, number | null][] = [
      ["clearGrowthField", ceilDiv(GROWTH_FIELD_CHANNELS * NODE_COUNT, 256)],
      ["clearRefinement", ceilDiv(refineWords, 256)],
      ["indexRefinementEdges", null],
      ["scatterGrowthIntent", null], ["scatterGrowthBoundary", null],
      ["enforceGrowthField", ceilDiv(NODE_COUNT, 256)],
      ["linkRefinementEdges", null],
      ...Array.from({ length: Math.ceil(Math.log2(MAX_PARTICLES)) },
        (): [string, null] => ["propagateRefinement", null]),
      ["requestRefinement", null], ["reserveRefinement", null],
      ["commitResample", null], ["classifyPruning", null], ["pruneMaterial", 1],
      ["stopGrowthAtCapacity", ceilDiv(GROWTH_FIELD_CHANNELS * NODE_COUNT, 256)],
    ];
    // Keep one stable ABI for every growth pass. With `layout: "auto"`, WebGPU
    // infers a different layout per entry point and removes bindings that an
    // entry point does not reach. That made this host-side resource table
    // brittle whenever the optimizer eliminated a formerly-used binding.
    const growthBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const growthPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [growthBindGroupLayout],
    });
    const pipelines = new Map(stages.map(([entry]) => [entry, null as GPUComputePipeline | null]));
    for (const entryPoint of pipelines.keys()) {
      pipelines.set(entryPoint, device.createComputePipeline({
        layout: growthPipelineLayout, compute: { module: growthModule, entryPoint },
      }));
    }
    this.growthEntries = stages.map(([entry]) => entry);
    this.growthPipelines = stages.map(([entry]) => pipelines.get(entry)!);
    const resources = new Map<number, GPUBindingResource>([
      [0, { buffer: mpmCore.positions }],
      [1, { buffer: mpmCore.activeCountUniform }],
      [2, { buffer: mpmCore.rest }],
      [3, { buffer: this.agentStateBuffer }],
      [4, { buffer: mpmCore.C }],
      [5, { buffer: mpmCore.velocities }],
      [6, { buffer: mpmCore.F }],
      [7, { buffer: this.refinement }],
      [8, { buffer: this.growthField }],
      [9, { buffer: this.physicsUniform }],
    ]);
    this.growthBindGroup = device.createBindGroup({
      layout: growthBindGroupLayout,
      entries: Array.from(resources, ([binding, resource]) => ({ binding, resource })),
    });
    this.growthDispatches = stages.map(([, dispatch]) => dispatch);
  }

  /** Exposed so Renderer's own triangle-shape pipeline can point each
   * particle along its real heading state rather than (mis-)deriving one
   * from velocity — see render.wgsl's own triangleVertex. Returns the
   * COMBINED ParticleMeta buffer, not a standalone heading array — that
   * pipeline's own WGSL declares the same packed struct and reads its
   * heading and neural color fields. */
  get particleMetaState(): GPUBuffer {
    return this.agentStateBuffer;
  }

  /** Viewer-only access to the spatially integrated growth decision field.
   * The renderer reads it after the growth passes have completed; it never
   * mutates the field or participates in admission. */
  get integratedGrowthField(): GPUBuffer {
    return this.growthField;
  }

  loadWeights(weights: UpdateRuleWeights): void {
    writeFloat32(this.device, this.weightsBuffer, 0, flattenWeights(weights, this.channels, this.hiddenDim, this.policyArchitecture));
  }

  /** Overwrites the weights buffer with a fresh random init (see
   * randomWeights()'s own docstring) — same live buffer write as
   * loadWeights(), no rebuild. Callers generally want to follow this
   * with restartRollout() too (see GpuSimulation.randomizeWeights()) —
   * particles already grown under the old weights don't retroactively
   * un-grow just because the policy driving them changed. */
  randomizeWeights(): UpdateRuleWeights {
    const weights = randomWeights(this.channels, this.hiddenDim, this.policyArchitecture);
    this.loadWeights(weights);
    return weights;
  }

  setPhysics(settings: {
    maxEnvWrite: number;
    sampleSpacing: number;
    friction: number;
    growthEnabled: number;
  }): void {
    writeFloat32(
      this.device,
      this.physicsUniform,
      0,
      new Float32Array([
        settings.maxEnvWrite,
        settings.sampleSpacing,
        settings.friction,
        settings.growthEnabled,
      ])
    );
  }

  /** Gates entry into new cell cycles without cancelling cycles already
   * underway. AgentPhysics.growthEnabled is the twelfth f32 (byte 44). */
  setGrowthEnabled(enabled: boolean): void {
    writeFloat32(
      this.device,
      this.physicsUniform,
      12,
      new Float32Array([enabled ? 1.0 : 0.0])
    );
  }

  /** Neural evaluation timestep; shared by communication and commit modes. */
  setCommunicationTimestep(dt: number): void {
    const value = new Float32Array([Math.max(0, dt)]);
    for (const buffer of this.stepModeUniforms) {
      writeFloat32(this.device, buffer, 4, value);
    }
  }

  /** Multiplier for private-state residual integration only; 1 is baseline. */
  setInternalStateSpeed(speed: number): void {
    const value = new Float32Array([Math.max(0, speed)]);
    for (const buffer of this.stepModeUniforms) {
      writeFloat32(this.device, buffer, 8, value);
    }
  }

  /** Caps polarized daughter placement: 0 = symmetric, 1 = full policy. */

  setSpawnCenter(spawnX: number, spawnY: number): void {
    writeFloat32(this.device, this.physicsUniform, 16, new Float32Array([spawnX, spawnY]));
  }

  /** Runtime growth cap at AgentPhysics byte offset 56. */
  setMaxActiveParticles(maxActiveParticles: number): void {
    const cap = Math.max(1, Math.floor(maxActiveParticles));
    this.device.queue.writeBuffer(this.physicsUniform, 24, new Uint32Array([cap]));
  }

  /** AgentPhysics.elasticStrainScale at byte offset 60. */
  setElasticStrainScale(scale: number): void {
    writeFloat32(this.device, this.physicsUniform, 28, new Float32Array([Math.max(scale, 1e-6)]));
  }

  setChemicalGradientInputScale(scale: number): void {
    writeFloat32(this.device, this.physicsUniform, 32, new Float32Array([Math.max(scale, 1e-6)]));
  }

  /** Relative neural gain for chemical concentration values; zero ablates them. */
  setChemicalValueInputMultiplier(multiplier: number): void {
    writeFloat32(this.device, this.physicsUniform, 60, new Float32Array([Math.max(multiplier, 0)]));
  }

  /** Force growth along an explicit world-space direction for Lab scenarios. */
  setForcedGrowthControl(
    index: number | null,
    direction: readonly [number, number],
    forceMagnitude: boolean,
    particleCount = 1,
  ): void {
    const x = direction[0];
    const y = direction[1];
    const length = Math.hypot(x, y);
    if (!Number.isFinite(length) || length === 0) throw new Error("Forced growth requires a finite, nonzero direction");
    const value = index === null ? 0xffffffff : Math.max(0, Math.floor(index));
    const endValue = index === null
      ? 0xffffffff
      : value + Math.max(1, Math.floor(particleCount)) - 1;
    this.device.queue.writeBuffer(this.physicsUniform, 36, new Uint32Array([value]));
    this.device.queue.writeBuffer(this.physicsUniform, 40, new Uint32Array([forceMagnitude ? 1 : 0]));
    writeFloat32(this.device, this.physicsUniform, 48, new Float32Array([
      x / length,
      y / length,
    ]));
    this.device.queue.writeBuffer(this.physicsUniform, 56, new Uint32Array([endValue]));
  }

  setForcedGrowthFieldOverride(mode: "radial-inward" | null): void {
    this.forcedGrowthFieldOverride = mode === "radial-inward";
    this.device.queue.writeBuffer(
      this.physicsUniform,
      64,
      new Uint32Array([this.forcedGrowthFieldOverride ? 1 : 0]),
    );
  }

  /** Updates this class's own agentStep() dispatch size AND growth's own
   * atomic "next free slot" counter (core/agents.wgsl's own module
   * docstring), which always needs to start from the current
   * activeCount — called once per rollout (simulation.ts's own
   * restartRollout()) and again every macro step growth actually changes
   * the count (that module's own step(), after awaiting
   * readSampleCount()). Deliberately does NOT touch
   * mpmCore.activeCountUniform itself — MpmCore.setActiveCount() (a
   * distinct method, on a distinct object) owns that, since it's shared
   * with p2g/gridUpdate-adjacent/g2p/repulsion too, not just this
   * class's own dispatch. */
  setActiveCount(activeCount: number): void {
    this.refinementRounds = Math.ceil(Math.log2(Math.max(1, activeCount)));
    this.dispatch = ceilDiv(activeCount, WORKGROUP);
    writeFloat32(this.device, this.agentStateBuffer, 0, new Uint32Array([activeCount]));
  }

  /** Encodes the copy of growth's own atomic counter into a mappable
   * staging buffer — must be encoded AFTER encodeStep() in the SAME
   * submit (see gpu/simulation.ts's own step()), so the copied value
   * reflects whatever this macro step's own agentStep() pass just
   * claimed. Does not submit. */
  encodeReadSampleCount(encoder: GPUCommandEncoder): void {
    encoder.copyBufferToBuffer(this.agentStateBuffer, 0, this.sampleCountStaging, 0, 12);
  }

  /** Reads back growth's own atomic counter, via encodeReadSampleCount()'s
   * own staging copy from THIS macro step's submit — a real, deliberate
   * host round-trip, once per macro step (gpu/simulation.ts's own step()
   * is the only caller), needed because dispatch sizing for EVERY pass
   * (P2G/gridUpdate/G2P/repulsion, and this class's own next
   * agentStep()) is decided on the CPU/JS side, from a cached count
   * nothing else updates automatically when growth happens purely on the
   * GPU. Async — unlike the Python trainer's synchronous wgpu-py
   * read_buffer(), WebGPU's own buffer readback (mapAsync) has no
   * synchronous equivalent, which is why gpu/simulation.ts's own step()
   * is async too (see that module's own module docstring). */
  async readSampleCount(): Promise<number> {
    await this.sampleCountStaging.mapAsync(GPUMapMode.READ);
    const status = new Uint32Array(this.sampleCountStaging.getMappedRange());
    const value = status[0];
    this.unresolvedSamples = status[1];
    this.capacityBlocked = status[2] !== 0;
    this.sampleCountStaging.unmap();
    return value;
  }

  /** Clears all rollout-scoped agent state. Alignment is deliberately zero
   * here and is reconstructed from channel index 3's gradient by agentStep. */
  resetState(initial?: { chemistry: Float32Array; privateState: Float32Array }): void {
    this.unresolvedSamples = 0;
    this.capacityBlocked = false;
    this.device.queue.writeBuffer(this.agentStateBuffer, 4, new Uint32Array(2));
    const count = (this.agentStateBuffer.size - PARTICLE_META_BUFFER_OFFSET) / this.particleMetaStride;
    const buf = new ArrayBuffer(count * this.particleMetaStride);
    const view = new DataView(buf);
    for (let i = 0; i < count; i++) {
      const base = i * this.particleMetaStride;
      // Density model v3 uses this u32 as a lineage-generation counter.
      if (initial && i < initial.privateState.length / 8) {
        for (let k = 0; k < 8; k++) view.setFloat32(base + 28 + 4*k, initial.privateState[i*8+k], true);
        for (let k = 0; k < this.channels; k++) view.setFloat32(base + 60 + 4*k, initial.chemistry[i*this.channels+k], true);
      }
    }
    this.device.queue.writeBuffer(this.agentStateBuffer, PARTICLE_META_BUFFER_OFFSET, buf);
    this.device.queue.writeBuffer(this.growthField, 0, new Uint32Array(this.growthField.size / 4));

  }

  encodeStep(encoder: GPUCommandEncoder, parity: number, commitGrowth = true): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, (commitGrowth ? this.commitBindGroups : this.communicationBindGroups)[parity]);
    pass.dispatchWorkgroups(this.dispatch);
    pass.end();
    if (commitGrowth) this.encodeGrowthField(encoder);
  }

  /** Spatially average policy intent, then conservatively refine any
   * under-resolved grown or deformed material footprint. */
  private encodeGrowthField(encoder: GPUCommandEncoder): void {
    let propagationRound = 0;
    for (let i = 0; i < this.growthPipelines.length; i++) {
      // Limit pointer-jumping to the live sample count, independent of the
      // number of field-projection passes before refinement.
      if (this.growthEntries[i] === "propagateRefinement" &&
          ++propagationRound > this.refinementRounds) continue;
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.growthPipelines[i]);
      pass.setBindGroup(0, this.growthBindGroup);
      pass.dispatchWorkgroups(this.growthDispatches[i] ?? this.dispatch);
      pass.end();
    }
  }

  /** Publishes each cell's persistent chemical state into the cleared,
   * transient substrate before any brain in this round runs. */
  encodeSplatChemicalState(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.splatPipeline);
    pass.setBindGroup(0, this.splatBindGroup);
    pass.dispatchWorkgroups(this.dispatch);
    pass.end();
  }

  destroy(): void {
    this.refinement.destroy();
    this.weightsBuffer.destroy();
    this.physicsUniform.destroy();
    this.agentStateBuffer.destroy();
    this.sampleCountStaging.destroy();
    this.stepModeUniforms[0].destroy();
    this.stepModeUniforms[1].destroy();
  }
}

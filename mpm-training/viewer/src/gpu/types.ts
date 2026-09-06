import type { InitialConditionPreset } from "./initialConditions";
import type { ShapeStopSettings } from "./shapeMatch";
export interface UpdateRuleWeights {
  fc1w: number[][]; // (HIDDEN_DIM, 3*channels+6 [+ 8 private state])
  fc1b: number[]; // (HIDDEN_DIM,)
  fc2w: number[][]; // stateless: channels+10; stateful: channels+23
  fc2b: number[];
}
export type PolicyArchitecture = "stateless-128" | "stateful-64" | "stateful-128";
export type CellMemory = "none" | "recurrent";
export type ChemicalCommunicationArchitecture = "persistent-environment" | "cell-owned-projection";

// Mirrors train_server.py's own GET /settings response, field for field
// (camelCase on the wire, same JSON keys) — every simulation/search
// setting that's FIXED for a run's entire lifetime (target, particles,
// channels, decay, population, ...). Fetched ONCE per run (see
// net/trainingSocket.ts's own useTrainingSocket()), NOT resent on every
// generation the way it used to be (train_server.py's own
// SETTINGS_PATH global has the full "why" — real, avoidable duplication
// once weights were already the dominant per-message payload size, and
// more importantly: a run's settings exist from near the moment training
// starts, well before generation 0 has finished evaluating, so a browser
// tab connecting during that window can still build a channels/hiddenDim-
// correct GpuSimulation immediately instead of waiting on a generation
// that might still be minutes away — see gpu/agents.ts's own
// randomWeights() for how that gap gets filled with a placeholder
// rollout in the meantime).
export interface RunSettings extends ShapeStopSettings {
  rasterResolution: number;
  estimatedSampleCapacity?: number;
  // The growth CAP, not the starting count — rollouts start with
  // two half-weight triangles per initialParticleCount seed cell (see
  // restartRollout()) and grow via splitting from there. This is the
  // trained/default ceiling; the viewer may apply a different playback-only
  // cap through AgentPhysics.maxActiveParticles without changing it.
  particles: number;
  initialParticleCount: number;
  initialCondition: InitialConditionPreset;
  initialConditionStrength: number;
  initialConditionChannel: number;
  densityModelVersion: number;
  trainingDensityMultipliers: number[];
  densityAggregation: "worst" | "mean";
  particleCapacity: number;
  particleMass: number;
  particleVolume: number;
  chemicalValueInputMultiplier: number;
  chemicalGradientInputScale: number;
  macroSteps: number;
  // Optional time cutoff for starting new cycles. null/absent means
  // growth remains chemically controlled for the whole replay.
  growthSteps: number | null;
  substepsPerMacro: number;
  gravity: number;
  spawnX: number;
  spawnY: number;
  channels: number;
  baseResolution: number;
  /** Per-channel spatial/temporal transport profile, packed into shared buffers. */
  chemicalChannelProfiles: ChemicalChannelProfile[];
  morphologyBlurSigma: number;
  morphologyDensityReference: number;
  neuralUpdatesPerMacro: number;
  communicationSpeed: number;
  internalStateSpeed: number;
  /** Cap on policy-polarized daughter placement: 0 symmetric, 1 full. */

  elasticStrainScale: number;
  elasticStrainInputsEnabled: boolean;
  hiddenDim: number;

  hiddenLayers: number[];
  /** Behavioral memory selection, independent from network capacity. */
  cellMemory: CellMemory;
  policyArchitecture: PolicyArchitecture;
  /** Where chemical memory lives and how the policy's chemical head is used. */
  chemicalCommunicationArchitecture: ChemicalCommunicationArchitecture;
  /** Used by persistent-environment; ignored by cell-owned-projection. */
  decay: number;
  /** Used by persistent-environment; ignored by cell-owned-projection. */
  depositRate: number;
  normalizeDepositsByLocalDensity: boolean;
  /** Represented-material density at which overcrowding normalization begins. */
  /** Amplitude of the signed neural chemical delta rate. */
  maxEnvWrite: number;

  // Gaussian splat sigma in normalized [0,1] world-domain units. The shader
  // projects it onto each channel's native grid without changing its physical
  // footprint as channel resolution changes.
  sampleSpacing: number;
  friction: number;
  growthDuration: number;

  growthModelVersion: number;
  domainGeometry: "triangle-vertices";
  /** Blend integrated tensor growth from isotropic (0) to directional (1). */
  growthAnisotropy: number;
  /** Elastic areal-compression interval that smoothly suppresses growth. */
  growthCompressionStart: number;
  growthCompressionStop: number;
  /** 0 disables mechanical feedback; 1 applies the full compression gate. */
  growthCompressionFeedback: number;
  mpmEnabled: boolean;
  damping: number;
  materialE: number;
  materialNu: number;
  materialHardening: number;
  materialElasticity: number;

  splatRadius: number;
  repulsionStrength: number;
  // Hard cap on the magnitude of one physics substep's own repulsion
  // velocity delta — see core/repulsion.wgsl's own
  // RepulsionParams.maxDelta field comment for the full reasoning
  // (repulsionStrength alone has to be pushed high enough to beat
  // materialE's own continuous elastic resistance to have any visible
  // effect, but unclamped that's exactly what produces a single-substep
  // MLS-MPM stability violation). Live-tunable via PhysicsPanel.
  repulsionMaxDelta: number;
  target: string;
  population: number;
  /** Shared rollout-seed batch size used for candidate evaluation. */
  seedsPerCandidate: number;
  elites: number;
  mutationSigma: number;
  runSeed: number;
  totalGenerations: number;
  checkpointEvery: number;
}

export function chemicalCommunicationArchitectureFromConfig(
  config: Pick<RunSettings, "chemicalCommunicationArchitecture" | "decay">,
): ChemicalCommunicationArchitecture {
  return config.chemicalCommunicationArchitecture;
}

export function policyHasRecurrence(architecture: PolicyArchitecture): boolean {
  return architecture === "stateful-64" || architecture === "stateful-128";
}

export function policyArchitectureForCellMemory(cellMemory: CellMemory): PolicyArchitecture {
  return cellMemory === "recurrent" ? "stateful-128" : "stateless-128";
}

export function cellMemoryFromConfig(
  config: Pick<RunSettings, "cellMemory" | "policyArchitecture">,
): CellMemory {
  return config.cellMemory;
}

export function hiddenLayersFromConfig(
  config: Pick<RunSettings, "hiddenLayers" | "hiddenDim">,
): number[] {
  return [...config.hiddenLayers];
}

// Mirrors train_server.py's own trimmed "generation" broadcast message —
// ONLY what's specific to one generation's own winning rollout; anything
// fixed for the whole run lives in RunSettings above instead. `weights`
// is genuinely optional here (not just typed loosely) — the one place
// that manufactures a GenerationRecord without a server message at all
// is useTrainingSocket()'s own placeholder (see that hook's own
// comment): random init, before generation 0 exists.
export interface GenerationRecord {
  generation: number;
  best: number;
  mean: number;
  worst: number;
  allTimeBest: number;
  seed: number;
  /** Seeds shared by every candidate during this generation's evaluation. */
  evaluationSeeds?: number[];
  /** Representative worst-case sampling density for this winner replay. */
  particleDensityMultiplier?: number;
  densityFitnesses?: Record<string, number>;
  weights: UpdateRuleWeights;
}

// What GridCanvas/GpuSimulation actually need to build and run a
// rollout — this run's own fixed settings plus ONE generation's own
// weights/seed, merged. Never received pre-merged from the server
// anymore (see RunSettings'/GenerationRecord's own docstrings for why
// the two are fetched/broadcast separately) — always assembled by
// spreading {...settings, ...generationRecord} at the point of use
// (net/trainingSocket.ts's own applyGeneration()), so this type stays
// exactly what it was before the settings/generation split, and
// GridCanvas/GpuSimulation/PhysicsPanel/etc. need no changes at all.
export type SimulationConfig = RunSettings & GenerationRecord;

// The live-adjustable subset of SimulationConfig — everything else
// (particles/macroSteps/channels/baseResolution/hiddenDim/target/...) is baked
// into GPU buffer sizes or WGSL compile-time consts, and changing it
// mid-replay would need a full rebuild, not a live uniform write. Backed
// by uniform buffers in the GPU layer (MpmCore's own set*() methods,
// Agents.setPhysics()) — every field here has a live setter, so tweaking
// one never disturbs the rollout currently in flight.
export interface PhysicsSettings {
  gravity: number;
  damping: number;
  materialE: number;
  materialNu: number;
  materialHardening: number;
  materialElasticity: number;
  particleMass: number;
  particleVolume: number;
  chemicalValueInputMultiplier: number;
  chemicalGradientInputScale: number;
  decay: number;
  depositRate: number;
  normalizeDepositsByLocalDensity: boolean;

  maxEnvWrite: number;
  sampleSpacing: number;
  friction: number;
  growthDuration: number;
  /** Playback multiplier on the configured material-growth rate. */
  growthSpeedMultiplier: number;
  neuralUpdatesPerMacro: number;
  communicationSpeed: number;
  internalStateSpeed: number;
  // Global cap on the neural per-particle anisotropy output.
  growthAnisotropy: number;
  growthCompressionStart: number;
  growthCompressionStop: number;
  growthCompressionFeedback: number;
  /** Gaussian sigma of the raw particle-density splat, in domain units. */
  splatRadius: number;
  /** Gaussian sigma of the policy morphology blur, in domain units. */
  morphologyBlurSigma: number;
  /** Half-saturation density used by rho/(rho + reference). */
  morphologyDensityReference: number;
  repulsionStrength: number;
  repulsionMaxDelta: number;
  mpmEnabled: boolean;
}

export function physicsSettingsFromConfig(config: SimulationConfig): PhysicsSettings {
  return {
    gravity: config.gravity,
    damping: config.damping,
    materialE: config.materialE,
    materialNu: config.materialNu,
    materialHardening: config.materialHardening,
    materialElasticity: config.materialElasticity,
    particleMass: config.particleMass,
    particleVolume: config.particleVolume,
    chemicalValueInputMultiplier: config.chemicalValueInputMultiplier,
    chemicalGradientInputScale: config.chemicalGradientInputScale,
    decay: config.decay,
    depositRate: config.depositRate,
    normalizeDepositsByLocalDensity: config.normalizeDepositsByLocalDensity,
    maxEnvWrite: config.maxEnvWrite,
    sampleSpacing: config.sampleSpacing,
    friction: config.friction,
    growthDuration: config.growthDuration,
    growthSpeedMultiplier: 1,
    neuralUpdatesPerMacro: config.neuralUpdatesPerMacro,
    communicationSpeed: config.communicationSpeed,
    internalStateSpeed: config.internalStateSpeed,
    growthAnisotropy: config.growthAnisotropy,
    growthCompressionStart: config.growthCompressionStart,
    growthCompressionStop: config.growthCompressionStop,
    growthCompressionFeedback: config.growthCompressionFeedback,
    splatRadius: config.splatRadius,
    morphologyBlurSigma: config.morphologyBlurSigma,
    morphologyDensityReference: config.morphologyDensityReference,
    repulsionStrength: config.repulsionStrength,
    repulsionMaxDelta: config.repulsionMaxDelta,
    mpmEnabled: config.mpmEnabled,
  };
}

// One scene's worth of particle state to seed MpmCore with — mirrors
// trainer/training_sim.py's own seed_blob() output shape.
export interface SceneData {
  domainGeometry?: "triangle-vertices";
  domain?: Float32Array;
  quadratureWeights?: Float32Array;
  count: number;
  positions: Float32Array; // (count,2)
  velocities: Float32Array; // (count,2), zero
  F: Float32Array; // (count,4), identity
  C: Float32Array; // (count,4), zero
  Jp: Float32Array; // (count,), ones
}
import type { ChemicalChannelProfile } from "./chemicalChannels";

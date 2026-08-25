export interface UpdateRuleWeights {
  fc1w: number[][]; // (HIDDEN_DIM, 3*channels+6 [+ 8 private state])
  fc1b: number[]; // (HIDDEN_DIM,)
  fc2w: number[][]; // stateless: channels+9; stateful: channels+22
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
export interface RunSettings {
  // The growth CAP, not the starting count — rollouts start with
  // initialParticleCount agents (see gpu/simulation.ts's own
  // restartRollout()) and grow via splitting from there. This is the
  // trained/default ceiling; the viewer may apply a different playback-only
  // cap through AgentPhysics.maxActiveParticles without changing it.
  particles: number;
  initialParticleCount?: number;
  densityModelVersion?: number;
  trainingDensityMultipliers?: number[];
  densityAggregation?: "worst" | "mean";
  particleCapacity?: number;
  particleMass?: number;
  particleVolume?: number;
  chemicalGradientInputScale?: number;
  macroSteps: number;
  // Optional time cutoff for starting new cycles. null/absent means
  // growth remains chemically controlled for the whole replay.
  growthSteps?: number | null;
  substepsPerMacro: number;
  gravity: number;
  spawnX: number;
  spawnY: number;
  spawnHalfWidth: number;
  channels: number;
  fieldN: number;
  morphologyBlurSigma?: number;
  morphologyDensityReference?: number;
  neuralUpdatesPerMacro?: number;
  communicationSpeed?: number;
  internalStateSpeed?: number;
  /** Cap on policy-polarized daughter placement: 0 symmetric, 1 full. */
  divisionDirectionality?: number;
  elasticStrainScale?: number;
  elasticStrainInputsEnabled?: boolean;
  hiddenDim: number;
  /** Ordered hidden-layer widths. Currently one layer; hiddenDim is its legacy alias. */
  hiddenLayers?: number[];
  /** Behavioral memory selection, independent from network capacity. */
  cellMemory?: CellMemory;
  policyArchitecture?: PolicyArchitecture;
  /** Where chemical memory lives and how the policy's chemical head is used. */
  chemicalCommunicationArchitecture?: ChemicalCommunicationArchitecture;
  /** Used by persistent-environment; ignored by cell-owned-projection. */
  decay: number;
  /** Used by persistent-environment; ignored by cell-owned-projection. */
  depositRate: number;
  maxAccel: number;
  maxStrafe: number;
  maxEnvWrite: number;
  maxAngularAccel: number;
  angularDamping: number;
  maxAngularVelocity: number;
  /** Legacy ABI field; centered deposits do not use an offset. */
  depositDistance: number;
  // Gaussian splat radius (sigma), field-pixel units — same convention as
  // core/agents.wgsl's own depositGaussian() reads this, replacing that
  // shader's old flat 4-corner bilinear deposit scatter. Live-tunable,
  // added specifically for testing this splat's own shape/spread via
  // PhysicsPanel.
  depositSigma: number;
  splitDisplacement: number;
  divisionCooldown: number;
  friction: number;
  // Legacy ABI/configuration field. Split mass is immediately conservative;
  // the newborn's visual size instead follows growthDuration.
  massRampMacroSteps: number;
  // Kinematic growth (the multiplicative decomposition F = Fe*Fg) — see
  // core/g2p.wgsl's own Material struct for what each of these does, and
  // core/agents.wgsl's own ParticleRest.growthF for what they accumulate
  // into. Duration is measured in neural/chemical controller ticks and is
  // independent of substepsPerMacro. 0 disables growth.
  growthDuration?: number;
  /** Legacy run field, converted to a duration when growthDuration is absent. */
  growthRate?: number;
  growthMax: number;
  /** Cap on policy-directed rest-growth anisotropy. */
  growthAnisotropy?: number;
  // Debug/testing toggle — off skips MpmCore's own physics substeps
  // entirely each macro step (gpu/simulation.ts's own step(), see that
  // method's own comment for exactly what stays running regardless:
  // sensing/deposit/growth/chirality), freezing positions in place
  // except where growth itself writes a brand-new child's own spawn
  // position. Broadcast value is trainer/simulation_settings.py's own
  // MPM_ENABLED constant (that constant's own comment has the full
  // "why," including the fitness-scoring caveat — with real physics
  // off, the ACTUAL worker-pool population evaluation this constant
  // also applies to would train against a close-to-meaningless shape-
  // matching fitness, so it's a real testing/debug run mode, not just a
  // cosmetic replay toggle) — only the STARTING value here, still
  // live-flippable in PhysicsPanel
  // afterward regardless of what the run itself was started with (that
  // live override never touches the actual training in progress, same
  // "playback-only" reasoning every other PhysicsSettings field has).
  mpmEnabled: boolean;
  // Compile-time (core/agents.wgsl's own CHIRALITY template const, not a
  // live uniform) — changes how many times the NN forward pass runs per
  // agent, not a physics knob, so it lives here (with channels/fieldN/
  // hiddenDim, all of which force a rebuild on change) rather than in
  // PhysicsSettings below.
  chirality: boolean;
  // MPM material (corotated elasticity, trainer/simulation_settings.py's
  // own MATERIAL_*) and damping/repulsion — never CLI-varied per run, but
  // broadcast here anyway (not just hardcoded to match on this side) so
  // the two can never silently drift apart the way they used to before
  // simulation_settings.py consolidated them.
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
  seedsPerCandidate?: number;
  elites: number;
  mutationSigma: number;
  runSeed: number;
  totalGenerations: number;
  checkpointEvery: number;
}

/** Resolve untagged legacy runs without changing the explicit modern default.
 * The pre-cell-owned runtime always had a decaying persistent field, whereas
 * the first cell-owned records disabled decay entirely. */
export function chemicalCommunicationArchitectureFromConfig(
  config: Pick<RunSettings, "chemicalCommunicationArchitecture" | "decay">,
): ChemicalCommunicationArchitecture {
  return config.chemicalCommunicationArchitecture
    ?? (config.decay > 0 ? "persistent-environment" : "cell-owned-projection");
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
  if (config.cellMemory) return config.cellMemory;
  return policyHasRecurrence(config.policyArchitecture ?? "stateless-128") ? "recurrent" : "none";
}

export function hiddenLayersFromConfig(
  config: Pick<RunSettings, "hiddenLayers" | "hiddenDim">,
): number[] {
  return config.hiddenLayers?.length ? [...config.hiddenLayers] : [config.hiddenDim];
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
// (particles/macroSteps/channels/fieldN/hiddenDim/target/...) is baked
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
  chemicalGradientInputScale: number;
  decay: number;
  depositRate: number;
  maxAccel: number;
  maxStrafe: number;
  maxEnvWrite: number;
  maxAngularAccel: number;
  angularDamping: number;
  maxAngularVelocity: number;
  depositDistance: number;
  depositSigma: number;
  splitDisplacement: number;
  divisionCooldown: number;
  friction: number;
  massRampMacroSteps: number;
  growthDuration: number;
  neuralUpdatesPerMacro: number;
  communicationSpeed: number;
  internalStateSpeed: number;
  divisionDirectionality: number;
  growthMax: number;
  // Global cap on the neural per-particle anisotropy output.
  growthAnisotropy: number;
  splatRadius: number;
  repulsionStrength: number;
  repulsionMaxDelta: number;
  mpmEnabled: boolean;
}

export function physicsSettingsFromConfig(config: SimulationConfig): PhysicsSettings {
  const legacyDuration = (config.growthRate ?? 0) > 0
    ? Math.log(2) / ((config.growthRate ?? 0) * coreConstants.DT * Math.max(config.substepsPerMacro, 1))
    : 0;
  return {
    gravity: config.gravity,
    damping: config.damping,
    materialE: config.materialE,
    materialNu: config.materialNu,
    materialHardening: config.materialHardening,
    materialElasticity: config.materialElasticity,
    particleMass: config.particleMass ?? coreConstants.PARTICLE_MASS,
    particleVolume: config.particleVolume ?? coreConstants.VOL,
    chemicalGradientInputScale: config.chemicalGradientInputScale ?? coreConstants.CHEMICAL_GRADIENT_INPUT_SCALE,
    decay: config.decay,
    // Falls back to 1.0 (= unchanged, matching this project's own
    // pre-depositRate behavior — see simulation_settings.py's own
    // DEPOSIT_RATE comment) for a `generation` message from a
    // train_server.py process still running pre-depositRate code, so the
    // PhysicsPanel slider doesn't crash on `undefined.toFixed()` before a
    // restart picks up the new field.
    depositRate: config.depositRate ?? 1.0,
    maxAccel: config.maxAccel,
    maxStrafe: config.maxStrafe,
    maxEnvWrite: config.maxEnvWrite,
    maxAngularAccel: config.maxAngularAccel,
    angularDamping: config.angularDamping,
    maxAngularVelocity: config.maxAngularVelocity,
    depositDistance: config.depositDistance,
    // Falls back to 0.6 (trainer/simulation_settings.py's own
    // DEPOSIT_SIGMA default) for a `generation` message from a
    // train_server.py process still running pre-depositSigma code, same
    // "don't crash on `undefined.toFixed()` before a restart picks up
    // the new field" reasoning depositRate's own fallback above gives.
    depositSigma: config.depositSigma ?? 0.6,
    splitDisplacement: config.splitDisplacement,
    divisionCooldown: config.divisionCooldown,
    friction: config.friction,
    // Falls back to 1.0 (= disabled, this project's own behavior before
    // this knob existed) for a `generation` message from a
    // train_server.py process still running pre-growth code,
    // same reasoning depositRate's/depositSigma's own fallbacks above
    // give.
    massRampMacroSteps: config.massRampMacroSteps ?? 20.0,
    growthDuration: config.growthDuration ?? legacyDuration,
    neuralUpdatesPerMacro: Math.max(1, Math.round(config.neuralUpdatesPerMacro ?? 1)),
    communicationSpeed: Math.max(0, config.communicationSpeed ?? 1.0),
    internalStateSpeed: Math.max(0, config.internalStateSpeed ?? 1.0),
    divisionDirectionality: Math.max(0, Math.min(1, config.divisionDirectionality ?? 1.0)),
    growthMax: config.growthMax ?? 2.0,
    growthAnisotropy: Math.max(0, Math.min(1, config.growthAnisotropy ?? 1.0)),
    splatRadius: config.splatRadius,
    repulsionStrength: config.repulsionStrength,
    // Falls back to 40.0 (trainer/simulation_settings.py's own
    // REPULSION_MAX_DELTA default) for a `generation` message from a
    // train_server.py process still running pre-repulsionMaxDelta code,
    // same reasoning depositSigma's own fallback above gives.
    repulsionMaxDelta: config.repulsionMaxDelta ?? 40.0,
    // Falls back to true (= normal physics, this project's own behavior
    // before this knob existed) for a `generation` message from a
    // train_server.py process still running pre-mpmEnabled code, same
    // reasoning depositRate's/depositSigma's own fallbacks above give.
    mpmEnabled: config.mpmEnabled ?? true,
  };
}

// One scene's worth of particle state to seed MpmCore with — mirrors
// trainer/training_sim.py's own seed_blob() output shape.
export interface SceneData {
  count: number;
  positions: Float32Array; // (count,2)
  velocities: Float32Array; // (count,2), zero
  F: Float32Array; // (count,4), identity
  C: Float32Array; // (count,4), zero
  Jp: Float32Array; // (count,), ones
}
import coreConstants from "../../../core/constants.json";

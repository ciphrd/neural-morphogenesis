export interface UpdateRuleWeights {
  fc1w: number[][]; // (HIDDEN_DIM, 3*channels+2) — +2 for the agent's own spawn-center-relative (x,y) position, see agents.wgsl's own IN_DIM
  fc1b: number[]; // (HIDDEN_DIM,)
  fc2w: number[][]; // (channels*4+5, HIDDEN_DIM) — *4 for the 4 deposit spots, see agents.wgsl's own SPOTS
  fc2b: number[]; // (channels*4+5,)
}

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
  // The growth CAP, not a fixed starting count — every rollout always
  // starts with exactly ONE particle (see gpu/simulation.ts's own
  // restartRollout()) and grows via splitting from there, up to this
  // many (core/agents.wgsl's own MAX_ACTIVE_PARTICLES template const —
  // see that file's own module docstring for the full growth design).
  // Rebuild-triggering (baked into GPU buffer sizes/WGSL compile-time
  // consts — see resetKeyFor()), same as channels/fieldN/hiddenDim below.
  particles: number;
  macroSteps: number;
  substepsPerMacro: number;
  gravity: number;
  spawnX: number;
  spawnY: number;
  spawnHalfWidth: number;
  channels: number;
  fieldN: number;
  hiddenDim: number;
  decay: number;
  // Multiplier on this macro step's own accumulated deposits, applied
  // right before they're folded into the field (core/environment.wgsl's
  // own EnvPhysics/mergeDeposit — see that file's own comment for why
  // this is a different knob from maxEnvWrite below, which caps each
  // agent's own per-deposit-spot magnitude before scatter rather than
  // scaling the whole step's already-accumulated total).
  depositRate: number;
  maxAccel: number;
  maxStrafe: number;
  maxEnvWrite: number;
  maxAngularAccel: number;
  angularDamping: number;
  maxAngularVelocity: number;
  depositDistance: number;
  // Gaussian splat radius (sigma), field-pixel units — same convention
  // depositDistance above already uses — core/agents.wgsl's own
  // depositGaussian() reads this (AgentPhysics uniform), replacing that
  // shader's old flat 4-corner bilinear deposit scatter. Live-tunable,
  // added specifically for testing this splat's own shape/spread via
  // PhysicsPanel.
  depositSigma: number;
  splitDisplacement: number;
  divisionCooldown: number;
  friction: number;
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
  target: string;
  population: number;
  elites: number;
  mutationSigma: number;
  runSeed: number;
  totalGenerations: number;
  checkpointEvery: number;
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
  splatRadius: number;
  repulsionStrength: number;
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
    splatRadius: config.splatRadius,
    repulsionStrength: config.repulsionStrength,
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

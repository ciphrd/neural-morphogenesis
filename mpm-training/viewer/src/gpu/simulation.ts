// Ties MpmCore + Environment + Agents + Renderer into one autonomous
// macro step — the browser analogue of trainer/training_sim.py's own
// TrainingRollout.macro_step(). GPU-resident for every data-related
// buffer (positions, velocities, F/C/ParticleRest, the chemical field, weights),
// matching envnca/frontend/src/gpu/simulation.ts's own "GPU-resident is
// the whole point" design — with ONE exception: step() is now async and
// reads back a 4-byte grown-particle count every macro step (see its own
// docstring, and agents.ts's own readGrownCount()), because growth
// (core/agents.wgsl's own agentStep() — see that file's own module
// docstring) can change activeCount purely on the GPU, and nothing else
// (P2G/gridUpdate/G2P/repulsion dispatch sizing, agents' own next
// dispatch) would ever find out otherwise. Unlike the Python trainer
// (trainer/agents_gpu.py's own read_grown_count(), a synchronous wgpu-py
// call), WebGPU's own buffer readback (mapAsync) has no synchronous
// equivalent — that's the one real architectural difference this class
// has from training_sim.py's own macro_step(), not a design choice.
//
// Per macro step, in order:
//   1. Clear the round's deposit scratch. Cell-owned mode then publishes each
//      cell's chemistry; persistent mode leaves it ready for direct NN writes.
//   2. environment.encodeSense() — materialize cell splats when applicable and
//      compute the shared gradient over the architecture's current field.
//   3. agents.encodeStep() — NN forward pass: reads that field and either
//      updates cell-owned chemistry or deposits directly into the environment.
//      It also writes the
//      desired growth vector into persistent ParticleRest.growthAngle and
//      relaxes the persistent anisotropy toward its sigmoid target
//      (optionally also physical acceleration through maxStrafe) —
//      may also grow activeCount (agents.wgsl's own agentStep()). Persistent
//      mode then diffuses/decays the old field and merges the fresh writes.
//   4. agents.encodeReadGrownCount()/readGrownCount() — copies growth's
//      own atomic counter out and awaits it (submit happens between
//      encode and await, see step()'s own body), propagating any change
//      to mpmCore/agents' own dispatch sizing before physics runs.
//   5. mpmCore.encodeSteps()          — substepsPerMacro physics
//      substeps, integrating the nudged velocity into position, using
//      the updated activeCount if growth changed it this step.
//
// loadGeneration()/rebuild() mirrors envnca/frontend/src/gpu/simulation.ts's
// own resetKey diff-check: only particles (the growth CAP now, not a
// starting count — see types.ts's own SimulationConfig.particles
// docstring)/channels/fieldN/hiddenDim force a full rebuild (they're
// baked into GPU buffer sizes and WGSL compile-time consts) — a new
// generation with the same
// shape is just a cheap loadWeights() call.

import { Agents } from "./agents";
import type { BloomSettings } from "./bloom";
import { Deform, type DeformDirection, type DeformMode } from "./deform";
import { NoiseDisplacement } from "./noiseDisplacement";
import { Environment } from "./environment";
import { Interact } from "./interact";
import { MAX_PARTICLES, MpmCore } from "./mpmCore";
import { Renderer, type FieldMode, type ParticleRenderMode } from "./render";
import { seedBlob, seedRows } from "./rng";
import { chemicalCommunicationArchitectureFromConfig, physicsSettingsFromConfig, type PhysicsSettings, type SimulationConfig, type UpdateRuleWeights } from "./types";
import coreConstants from "../../../core/constants.json";

export interface SimulationScenario {
  initialLayout: { kind: "rows"; rows: number; columns: number };
  events: Array<{
    step: number;
    type: "split";
    particleIndex: number;
    /** Number of contiguous particle slots participating in this event. */
    particleCount?: number;
    /** Fixed world axis. Omit to follow the local boundary tangent. */
    direction?: readonly [number, number];
  }>;
  suppressNaturalGrowth?: boolean;
}

export class GpuSimulation {
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;

  private mpmCore: MpmCore | null = null;
  private environment: Environment | null = null;
  private agents: Agents | null = null;
  private renderer: Renderer | null = null;
  // "Move Particles" tool's own pick/drag state (gpu/interact.ts) — a
  // fresh instance per rebuild(), same as every other GPU object here,
  // since it binds MpmCore's own (also freshly rebuilt) buffers.
  private interact: Interact | null = null;
  // "Deform" tool's own one-shot direction-injection (gpu/deform.ts) —
  // same "fresh instance per rebuild()" reasoning as interact above.
  private deform: Deform | null = null;
  private noiseDisplacement: NoiseDisplacement | null = null;

  private config: SimulationConfig | null = null;
  private resetKey: string | null = null;
  private pendingTargetPoints: Float32Array | null = null;
  // View-only display preferences — not simulation state, so they must
  // survive rebuild() destroying and recreating the Renderer (a new
  // generation with a different particle/channel/field/hidden-dim shape
  // gets a brand-new Renderer instance; the user's own render-option
  // choices shouldn't reset just because that happened).
  private pendingFieldMode: FieldMode = "none";
  private pendingSubstrateChannelStart = 0;
  private pendingParticleRenderMode: ParticleRenderMode = "dots-white";
  private pendingWhiteDotsAlpha = 1.0;
  private pendingActivationAlpha = 0.2;
  private pendingNeuralColorAlpha = 1.0;
  private pendingInternalStateAlpha = 1.0;
  private pendingInternalStateChannelStart = 0;
  private pendingChemicalMemoryOpponentSubtraction = 0;
  private pendingBoundaryGradientScale = 0.01;
  private pendingPointRadiusPx: number | null = null;
  private pendingGrowthAxisLengthPx = 24;
  // 0 = identity — see gpu/render.ts's own setAccent()/field.wgsl's own
  // accent uniform comment. Same "view-only, survives rebuild()" reasoning
  // pendingFieldMode above already has.
  private pendingAccent = 0;
  private pendingMorphologyGradientVisible = true;
  private pendingMorphologyDensityVisible = true;
  // 0 = no blur — see gpu/render.ts's own setBlur()/field.wgsl's own
  // blurDensity() comment. Same "view-only, survives rebuild()" reasoning
  // pendingAccent above already has.
  private pendingBlur = 0;
  // 1 = identity — see gpu/render.ts's own setGradientExponent()/
  // field.wgsl's own colorizeGradient() comment. Same "view-only,
  // survives rebuild()" reasoning pendingAccent above already has.
  private pendingGradientExponent = 1;
  private pendingParticleCap: number | null = null;
  private pendingInitialParticleCount: number | null = null;
  private particleCap = 2;
  private pendingTargetVisible = true;
  private neuralUpdatesPerMacro = 1;
  private growthDuration = 0;
  // Low rates still retire particles predictably: fractional expected deaths
  // carry across macro steps until they add up to one whole particle.
  private deathRate = 0;
  private deathAccumulator = 0;
  private scenario: SimulationScenario | null = null;
  // The canvas's own backing-store size in DEVICE pixels, as last
  // reported by GridCanvas's own applySquareSize()/ResizeObserver.
  // Same "view-only, survives rebuild()" reasoning as the pending
  // fields above — and it genuinely needs it: with no config loaded
  // yet (a fresh page load before the first `generation` message
  // arrives), rebuild() has never run, so there IS no Renderer for
  // setCanvasSizePx() to forward to, and both GridCanvas's own
  // device-acquisition call AND its rAF re-validation call land on
  // `this.renderer?` === null and are silently dropped. The Renderer
  // built later, when that first generation finally arrives, then
  // starts from its own constructor default (512x512 — see
  // render.ts's own canvasMinDimPx) instead of the canvas's real size,
  // making every device-pixel-sized draw (particle radius, the
  // target-point overlay dots) render oversized by exactly
  // realSize/512 — the "everything is drawn twice as big until
  // something jogs a resize" bug. null until the first report.
  private pendingCanvasSizePx: [number, number] | null = null;
  private pendingZoom = 1;
  private pendingBloom: BloomSettings = {
    enabled: true,
    intensity: 0.8,
    threshold: 0.65,
    radiusPx: 2.5,
    scatter: 0.8,
    levels: 6,
  };

  // Bumped by anything that invalidates in-flight GPU state (rebuild(),
  // restartRollout(), destroy()) — step() captures this at its own start
  // and checks it again after its own await (see that method's own
  // docstring for the exact race this guards against: growth's own
  // async readGrownCount() can resolve AFTER a user-triggered restart
  // (GridCanvas.tsx's own imperative restart() — NOT the RAF loop's own
  // sequential step()/restartRollout() calls, which can't race each
  // other) already reset activeCount back to 1, and blindly reapplying
  // that stale, pre-restart count would silently reinflate activeCount
  // right back up — while the particles that count now (once again)
  // claims as active still hold whatever stale positions the PREVIOUS
  // rollout's growth left behind, since restartRollout() only ever
  // rewrites position[0] (a genuinely new particle's own position is
  // only ever written at the moment IT is claimed by a real split, not
  // pre-filled — see MpmCore.resetGrowthBuffers()'s own docstring for
  // why velocities/F/C/ParticleRest get that treatment but positions doesn't).
  private epoch = 0;

  // Debug/testing toggle — see step()'s own comment for exactly what
  // this skips and why. Live-adjustable via PhysicsSettings.mpmEnabled
  // (applyPhysics() below sets this, same as every other physics knob —
  // NOT a standalone imperative setter), so it's part of a generation's
  // own broadcast config (train_server.py always sends `true`; the
  // backend has no equivalent, since disabling real physics during
  // actual training would break fitness scoring entirely — this is a
  // frontend-only viewing aid) and follows the same
  // isOverridden/PhysicsPanel/reset-to-trained lifecycle every other
  // physics setting already has.
  private mpmEnabled = true;

  private _currentStep = 0;
  get currentStep(): number {
    return this._currentStep;
  }
  get steps(): number {
    return this.config?.macroSteps ?? 0;
  }
  /** How many particles are live RIGHT NOW — grows as growth splits
   * (core/agents.wgsl's own agentStep()), so this changes every macro
   * step, unlike config.particles which is only the CAP. 0 before the
   * first rebuild(). */
  get particleCount(): number {
    return this.mpmCore?.activeCount ?? 0;
  }
  async readPositions(): Promise<Float32Array> {
    return this.mpmCore ? this.mpmCore.readPositions() : new Float32Array();
  }
  async readPositionSamples(maxSamples: number): Promise<Float32Array> {
    return this.mpmCore
      ? this.mpmCore.readPositionSamples(maxSamples)
      : new Float32Array();
  }
  private growthIsEnabled(): boolean {
    if (!this.config) return false;
    // No implicit horizon-based gate: null/absent means the chemical
    // field and population cap are the only controls. A finite cutoff is
    // an explicit per-run choice supplied by --growth-steps.
    const cutoff = this.config.growthSteps;
    return cutoff == null || this._currentStep < cutoff;
  }
  get ready(): boolean {
    return this.mpmCore !== null;
  }

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  private resetKeyFor(config: SimulationConfig): string {
    return [
      config.particles,
      config.channels,
      config.fieldN,
      config.hiddenDim,
      config.policyArchitecture ?? "stateless-128",
      chemicalCommunicationArchitectureFromConfig(config),
      config.chirality,
      config.elasticStrainScale ?? 0.15,
      config.elasticStrainInputsEnabled ?? false,
    ].join(":");
  }

  loadGeneration(config: SimulationConfig): void {
    const key = this.resetKeyFor(config);
    if (!this.mpmCore || key !== this.resetKey) {
      this.rebuild(config);
      this.resetKey = key;
    } else {
      this.agents!.loadWeights(config.weights);
      this.config = config;
      this.applyPhysics(physicsSettingsFromConfig(config));
    }
    this.restartRollout();
  }

  private rebuild(config: SimulationConfig): void {
    this.epoch++;
    this.destroySimObjects();
    this.particleCap = Math.min(MAX_PARTICLES, Math.max(2, Math.floor(this.pendingParticleCap ?? config.particles)));

    const mpmCore = new MpmCore(this.device);

    const environment = new Environment(this.device, {
      channels: config.channels,
      width: config.fieldN,
      height: config.fieldN,
      decay: config.decay,
      // ?? 1.0 (= unchanged) guards a `generation` message from a
      // train_server.py process still running pre-depositRate code —
      // see types.ts's own physicsSettingsFromConfig() for the matching
      // guard on the PhysicsPanel's own read of this same field.
      depositRate: config.depositRate ?? 1.0,
      chemicalCommunicationArchitecture: chemicalCommunicationArchitectureFromConfig(config),
    });

    const agents = new Agents(this.device, mpmCore, environment, {
      channels: config.channels,
      hiddenDim: config.hiddenDim,
      policyArchitecture: config.policyArchitecture ?? "stateless-128",
      chemicalCommunicationArchitecture: chemicalCommunicationArchitectureFromConfig(config),
      maxAccel: config.maxAccel,
      maxStrafe: config.maxStrafe,
      steeringStrength: config.steeringStrength ?? 0,
      maxEnvWrite: config.maxEnvWrite,
      maxAngularAccel: config.maxAngularAccel,
      angularDamping: config.angularDamping,
      maxAngularVelocity: config.maxAngularVelocity,
      chirality: config.chirality,
      depositDistance: config.depositDistance,
      // ?? 0.6 (trainer/simulation_settings.py's own DEPOSIT_SIGMA
      // default) guards a `generation` message from a train_server.py
      // process still running pre-depositSigma code — same reasoning
      // depositRate's own ?? 1.0 guard above gives (see that guard's own
      // comment): an unguarded `undefined` here would write NaN into
      // this uniform and silently corrupt every deposit from step one.
      depositSigma: config.depositSigma ?? 0.6,
      splitDisplacement: config.splitDisplacement,
      divisionCooldown: config.divisionCooldown,
      friction: config.friction,
      growthEnabled: 1.0,
      maxActiveParticles: this.particleCap,
      spawnX: config.spawnX,
      spawnY: config.spawnY,
      elasticStrainScale: config.elasticStrainScale ?? 0.15,
      elasticStrainInputsEnabled: config.elasticStrainInputsEnabled ?? false,
      chemicalGradientInputScale: config.chemicalGradientInputScale ?? coreConstants.CHEMICAL_GRADIENT_INPUT_SCALE,
      chemicalProjectionWeight: config.chemicalProjectionWeight ?? 1.0,
      boundaryTangentMinGradient: config.boundaryTangentMinGradient
        ?? coreConstants.BOUNDARY_TANGENT_MIN_GRADIENT,
      growthCompressionStart: config.growthCompressionStart ?? 0.10,
      growthCompressionStop: config.growthCompressionStop ?? 0.10,
      growthCompressionFeedback: config.growthCompressionFeedback ?? 0.0,
    });
    mpmCore.setChemicalStateBuffer(agents.particleMetaState);
    agents.loadWeights(config.weights);

    const renderer = new Renderer(this.device, this.format, mpmCore, environment, agents.particleMetaState);
    if (this.pendingCanvasSizePx) renderer.setCanvasSizePx(...this.pendingCanvasSizePx);
    if (this.pendingTargetPoints) renderer.setTargetPoints(this.pendingTargetPoints);
    renderer.setTargetVisible(this.pendingTargetVisible);
    renderer.setFieldMode(this.pendingFieldMode);
    renderer.setSubstrateChannelStart(this.pendingSubstrateChannelStart);
    renderer.setParticleRenderMode(this.pendingParticleRenderMode);
    renderer.setWhiteDotsAlpha(this.pendingWhiteDotsAlpha);
    renderer.setActivationAlpha(this.pendingActivationAlpha);
    renderer.setNeuralColorAlpha(this.pendingNeuralColorAlpha);
    renderer.setInternalStateAlpha(this.pendingInternalStateAlpha);
    renderer.setInternalStateChannelStart(this.pendingInternalStateChannelStart);
    renderer.setChemicalMemoryOpponentSubtraction(this.pendingChemicalMemoryOpponentSubtraction);
    renderer.setBoundaryGradientScale(this.pendingBoundaryGradientScale);
    if (this.pendingPointRadiusPx !== null) renderer.setPointRadiusPx(this.pendingPointRadiusPx);
    renderer.setGrowthAxisLengthPx(this.pendingGrowthAxisLengthPx);
    renderer.setAccent(this.pendingAccent);
    renderer.setMorphologyDisplay(this.pendingMorphologyGradientVisible, this.pendingMorphologyDensityVisible);
    renderer.setBlur(this.pendingBlur);
    renderer.setGradientExponent(this.pendingGradientExponent);
    renderer.setZoom(this.pendingZoom);
    renderer.setBloom(this.pendingBloom);

    const interact = new Interact(this.device, mpmCore);
    const deform = new Deform(this.device, mpmCore);
    const noiseDisplacement = new NoiseDisplacement(this.device, mpmCore);

    this.mpmCore = mpmCore;
    this.environment = environment;
    this.agents = agents;
    this.renderer = renderer;
    this.interact = interact;
    this.deform = deform;
    this.noiseDisplacement = noiseDisplacement;
    this.config = config;
    this.applyPhysics(physicsSettingsFromConfig(config));
  }

  /** Re-seeds particles from the current config's spawn params + winner
   * seed, zeroes the chemical field, and resets the step counter — a
   * fresh rollout of the *same* generation, same "restart" GridCanvas's
   * Playback controls and the RAF loop's own loop-at-trained-steps
   * behavior both call. */
  restartRollout(): void {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    this.epoch++;
    const initialCount = Math.min(
      this.particleCap,
      Math.max(1, Math.floor(
        this.pendingInitialParticleCount
        ?? this.config.initialParticleCount
        ?? coreConstants.INITIAL_PARTICLE_COUNT
      ))
    );
    const scene = this.scenario?.initialLayout.kind === "rows"
      ? seedRows({
          rows: this.scenario.initialLayout.rows,
          columns: this.scenario.initialLayout.columns,
          centerX: this.config.spawnX,
          centerY: this.config.spawnY,
          spacing: this.config.splitDisplacement,
        })
      : seedBlob({
          count: initialCount,
          centerX: this.config.spawnX,
          centerY: this.config.spawnY,
          spacing: this.config.splitDisplacement,
          seed: this.config.seed,
        });
    this.mpmCore.loadScene(scene);
    // Every slot beyond the genuinely seeded particles is destined to become
    // a real particle via growth, at some unknown point in this rollout
    // — see MpmCore.resetGrowthBuffers()'s own docstring for why this
    // has to run every rollout (not just once, ever) despite seedBlob()
    // already giving genuinely-seeded particles these exact same fresh
    // defaults.
    this.mpmCore.resetGrowthBuffers(this.particleCap);
    this.environment.reset();
    // Every rollout — same "run-constant in practice today, but a
    // rollout-scoped setter regardless" convention this method's own
    // seedBlob()/setActiveCount() calls already follow. See
    // The Agents uniform retains legacy spawn slots for wire compatibility,
    // although position is no longer a policy input.
    this.agents.setSpawnCenter(this.config.spawnX, this.config.spawnY);
    this.agents.setMaxActiveParticles(this.particleCap);
    this.agents.setActiveCount(scene.count);
    // Heading's own per-slot fill and growth's own seed are both bit-
    // exact via rng.ts's own spawnUniform01()/growthSeed() respectively
    // (two DIFFERENT hash domains, see spawnUniform01()'s own comment
    // for why they're safe to derive from the identical raw seed
    // without correlating) — see Agents.resetHeading()'s own docstring.
    this.agents.resetHeading(this.config.seed, scene.positions);
    this.deathAccumulator = 0;
    this._currentStep = 0;
  }

  /** Installs an isolated, deterministic lab scenario and restarts it. */
  setScenario(scenario: SimulationScenario | null): void {
    this.scenario = scenario;
    if (this.mpmCore) this.restartRollout();
  }

  /** Live-adjustable knobs only — see types.ts's own PhysicsSettings
   * docstring for why this is a strict subset of SimulationConfig. No
   * rebuild, just uniform writes — every field here has a live setter on
   * MpmCore/Environment/Agents, so this is safe to call on every
   * PhysicsPanel slider tick without disturbing the rollout in flight.
   * Run configs are normalized through physicsSettingsFromConfig() before
   * reaching this method, including legacy growth-rate conversion. Damping's
   * own substep count comes from
   * `this.config` (not `physics`), matching evolve.py's own rollout()
   * (which converts a run's damping loss-fraction using its own
   * --substeps-per-macro, not a fixed constant) — this.config must
   * already be set before this runs. */
  private applyPhysics(physics: PhysicsSettings): void {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    this.mpmCore.setGravity(physics.gravity);
    this.neuralUpdatesPerMacro = Math.max(1, Math.round(physics.neuralUpdatesPerMacro));
    const communicationDt = Math.max(0, physics.communicationSpeed ?? 1.0) / this.neuralUpdatesPerMacro;
    this.environment.setPhysics(
      physics.decay,
      physics.depositRate,
      this.neuralUpdatesPerMacro,
      physics.communicationSpeed ?? 1.0,
    );
    const growthCompressionStart = Math.max(0, physics.growthCompressionStart ?? 0.10);
    const growthCompressionStop = Math.max(
      growthCompressionStart,
      physics.growthCompressionStop ?? 0.10,
    );
    const growthCompressionFeedback = Math.max(
      0, Math.min(1, physics.growthCompressionFeedback ?? 1.0),
    );
    const growthDrive = Math.max(0, Math.min(1, physics.growthDrive ?? 0.5));
    // 0 pauses even already-active morphoelastic cycles; 0.5 preserves the
    // configured rate. Values above the midpoint keep the native mechanical
    // rate while progressively bypassing compression feedback, matching the
    // admission-side override in agents.wgsl.
    const nativeGrowthWeight = Math.min(1, growthDrive * 2);
    const drivenGrowthDuration = nativeGrowthWeight > 0
      ? physics.growthDuration / nativeGrowthWeight
      : 0;
    const forcedGrowthBias = Math.max(0, (growthDrive - 0.5) * 2);
    const drivenCompressionFeedback = growthCompressionFeedback * (1 - forcedGrowthBias);
    this.mpmCore.setMaterial(
      physics.materialE,
      physics.materialNu,
      physics.materialHardening,
      physics.materialElasticity,
      // Controller ticks per uncompressed area doubling. MpmCore derives
      // the shader's internal per-substep rate from this and the run cadence.
      drivenGrowthDuration,
      physics.growthMax ?? 2.0,
      physics.growthAnisotropy ?? 1.0,
      this.config.substepsPerMacro,
      physics.particleMass,
      physics.particleVolume,
      growthCompressionStart,
      growthCompressionStop,
      drivenCompressionFeedback,
      physics.materialFluidity,
    );
    this.growthDuration = physics.growthDuration;
    this.deathRate = Math.max(0, Math.min(1, physics.deathRate ?? 0));
    if (this.deathRate === 0) this.deathAccumulator = 0;
    this.mpmCore.setDamping(physics.damping, this.config.substepsPerMacro);
    this.mpmCore.setSplatRadius(physics.splatRadius);
    this.mpmCore.setMorphology(
      physics.morphologyBlurSigma,
      physics.morphologyDensityReference,
    );
    // ?? 40.0 (trainer/simulation_settings.py's own REPULSION_MAX_DELTA
    // default) guards a call to this with a raw SimulationConfig from a
    // train_server.py process still running pre-repulsionMaxDelta code,
    // same reasoning depositRate's own guard below gives.
    this.mpmCore.setRepulsionStrength(
      physics.repulsionStrength,
      physics.repulsionMaxDelta ?? 40.0,
    );
    this.agents.setCommunicationTimestep(communicationDt);
    this.agents.setInternalStateSpeed(physics.internalStateSpeed ?? 1.0);
    this.agents.setDivisionDirectionality(physics.divisionDirectionality ?? 1.0);
    this.agents.setGrowthDrive(growthDrive);
    this.agents.setChemicalGradientInputScale(physics.chemicalGradientInputScale);
    this.agents.setChemicalProjectionWeight(physics.chemicalProjectionWeight);
    this.agents.setBoundaryTangentMinGradient(physics.boundaryTangentMinGradient);
    this.agents.setGrowthCompressionFeedback(
      growthCompressionStart,
      growthCompressionStop,
      drivenCompressionFeedback,
    );
    this.agents.setPhysics({
      maxAccel: physics.maxAccel,
      steeringStrength: physics.steeringStrength ?? 0,
      maxStrafe: physics.maxStrafe,
      maxEnvWrite: physics.maxEnvWrite,
      maxAngularAccel: physics.maxAngularAccel,
      angularDamping: physics.angularDamping,
      maxAngularVelocity: physics.maxAngularVelocity,
      depositDistance: physics.depositDistance,
      // ?? 0.6 — same pre-depositSigma-broadcast guard reasoning
      // depositRate's own ?? 1.0 guard above gives.
      depositSigma: physics.depositSigma ?? 0.6,
      splitDisplacement: physics.splitDisplacement,
      divisionCooldown: physics.divisionCooldown,
      friction: physics.friction,
      growthEnabled: this.growthIsEnabled() ? 1.0 : 0.0,
    });
    // ?? true — same pre-broadcast guard reasoning depositRate's own
    // ?? 1.0 guard above gives, for a train_server.py process still
    // running pre-mpmEnabled code. Not a GPU uniform write (unlike every
    // setting above) — a plain JS field step() reads to decide whether
    // to skip mpmCore.encodeSteps() at all (see that method's own
    // comment).
    this.mpmEnabled = physics.mpmEnabled ?? true;
  }

  setPhysics(physics: PhysicsSettings): void {
    this.applyPhysics(physics);
  }

  /** Retires tail slots and returns the number of same-frame replacement
   * divisions to force. The temporary growth ceiling is the pre-death count,
   * so natural and forced divisions together can refill, but never exceed,
   * the population that entered this macro step. */
  private applyDeaths(): number {
    if (!this.mpmCore || !this.agents || this.deathRate <= 0) return 0;
    const availableToRetire = Math.max(0, this.mpmCore.activeCount - 1);
    if (availableToRetire === 0) {
      this.deathAccumulator = 0;
      return 0;
    }
    this.deathAccumulator += this.mpmCore.activeCount * this.deathRate;
    // One surviving invocation can create one replacement in this pass.
    // Capping turnover at half guarantees there are enough parents; any
    // fractional remainder stays queued for a later macro step.
    const maxReplaceable = Math.floor(this.mpmCore.activeCount / 2);
    const deaths = Math.min(availableToRetire, maxReplaceable, Math.floor(this.deathAccumulator));
    if (deaths === 0) return 0;
    this.deathAccumulator -= deaths;
    const populationBeforeDeaths = this.mpmCore.activeCount;
    const survivors = populationBeforeDeaths - deaths;
    this.agents.setMaxActiveParticles(populationBeforeDeaths);
    this.mpmCore.setActiveCount(survivors);
    this.agents.setActiveCount(survivors);
    return deaths;
  }

  /** Playback-only live growth cap. Lowering it below the current count
   * restarts the rollout instead of deleting already-materialized mass. */
  setParticleCap(maxParticles: number): void {
    const cap = Math.min(MAX_PARTICLES, Math.max(2, Math.floor(maxParticles)));
    this.pendingParticleCap = cap;
    this.particleCap = cap;
    if (this.pendingInitialParticleCount !== null) {
      this.pendingInitialParticleCount = Math.min(this.pendingInitialParticleCount, cap);
    }
    this.agents?.setMaxActiveParticles(cap);
    if (this.mpmCore && this.mpmCore.activeCount > cap) {
      this.restartRollout();
    }
  }

  /** Playback-only seeded agent count. Applying it restarts the current
   * generation so the new initial condition takes effect immediately. */
  setInitialParticleCount(initialParticles: number): void {
    const count = Math.min(this.particleCap, Math.max(1, Math.floor(initialParticles)));
    if (this.pendingInitialParticleCount === count) return;
    this.pendingInitialParticleCount = count;
    if (this.mpmCore) this.restartRollout();
  }

  /** Randomly retires a fraction of the live population. Agents compacts
   * randomly-selected victim holes with complete tail-agent copies before
   * this lowers the live prefix, so the newest agents are preserved rather
   * than being systematically discarded. Unlike deathRate this does not
   * issue replacement splits; vacated slots remain available to growth. */
  killFraction(fraction: number): number {
    if (!this.mpmCore || !this.agents) return 0;
    const activeCount = this.mpmCore.activeCount;
    if (activeCount <= 1) return 0;
    const clampedFraction = Math.max(0, Math.min(1, fraction));
    const killed = Math.min(
      activeCount - 1,
      Math.max(1, Math.floor(activeCount * clampedFraction)),
    );
    const survivors = activeCount - killed;
    // Invalidate an async step that may currently be awaiting growth-count
    // readback, preventing it from restoring the pre-kill population.
    this.epoch++;
    this.agents.compactRandomCull(activeCount, killed);
    this.mpmCore.setActiveCount(survivors);
    this.agents.setActiveCount(survivors);
    return killed;
  }

  /** `points`: flat [x0,y0,x1,y1,...] in MpmCore's own [0,1]^2 domain.
   * Cached (not just forwarded) since it can arrive before the first
   * rebuild() ever runs. */
  setTargetPoints(points: Float32Array): void {
    this.pendingTargetPoints = points;
    this.renderer?.setTargetPoints(points);
  }

  /** Controls only the target overlay; target data and simulation state
   * are left untouched. */
  setTargetVisible(visible: boolean): void {
    this.pendingTargetVisible = visible;
    this.renderer?.setTargetVisible(visible);
  }

  /** Async — see this module's own module docstring for why (WebGPU's
   * own buffer readback, needed for growth's own grown-count propagation,
   * has no synchronous equivalent the way trainer/training_sim.py's own
   * macro_step() gets from wgpu-py). Two submits, not one: sense/act/
   * deposit (+ the copy encodeReadGrownCount() adds) first, then —
   * *after* awaiting readGrownCount(), so the result is actually known —
   * mpmCore.encodeSteps()'s own physics substeps, sized off whatever
   * activeCount now is. Splitting into two submits like this costs
   * nothing extra beyond the readback itself already costs: WebGPU's
   * queue is a single in-order timeline, so the second submit correctly
   * sees the first one's positions/deposits/grown particles regardless
   * of how many submits that took.
   *
   * Captures `this.epoch` before the await and bails out (no activeCount
   * write, no physics submit, no currentStep bump) if it's changed by
   * the time readGrownCount() resolves — see that field's own comment
   * for the exact restart-vs-in-flight-step race this prevents. */
  async step(): Promise<void> {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    const deathReplacements = this.applyDeaths();
    const populationCeiling = deathReplacements > 0
      ? this.mpmCore.activeCount + deathReplacements
      : this.particleCap;
    const stepEpoch = this.epoch;
    const nextStep = this._currentStep + 1;
    const forcedLifecycle = this.scenario?.events.find((event) => {
      const admissionStep = Math.max(1, event.step - Math.ceil(this.growthDuration));
      return event.type === "split" && nextStep >= admissionStep && nextStep <= event.step;
    });
    const admissionStep = forcedLifecycle
      ? Math.max(1, forcedLifecycle.step - Math.ceil(this.growthDuration))
      : -1;
    this.agents.setForcedDivisionControl(
      forcedLifecycle?.particleIndex ?? null,
      forcedLifecycle?.direction ?? null,
      nextStep === admissionStep,
      forcedLifecycle?.particleCount ?? 1,
      deathReplacements,
    );
    this.agents.setGrowthEnabled(
      !this.scenario?.suppressNaturalGrowth && this.growthIsEnabled(),
    );
    const encoder = this.device.createCommandEncoder();
    this.mpmCore.encodeMorphology(encoder);
    for (let communicationRound = 0; communicationRound < this.neuralUpdatesPerMacro; communicationRound++) {
      this.environment.encodeClear(encoder);
      if (this.environment.chemicalCommunicationArchitecture === "cell-owned-projection") {
        this.agents.encodeSplatChemicalState(encoder);
      }
      this.environment.encodeSense(encoder);
      this.agents.encodeStep(
        encoder,
        this.environment.parity,
        communicationRound === this.neuralUpdatesPerMacro - 1
      );
      this.environment.encodeAdvancePersistent(encoder);
    }
    this.agents.encodeReadGrownCount(encoder);
    this.device.queue.submit([encoder.finish()]);

    // min(...) — growth's own atomic counter can overshoot particleCap
    // slightly (several agents claiming a slot the same step,
    // right at the cap — see core/agents.wgsl's own agentStep() comment
    // for why that's left unguarded rather than compare-exchanged away);
    // clamping the *reported* count here is what actually enforces the
    // cap, since agents.wgsl itself already refuses to WRITE a claimed
    // slot past that either way. A plain != check below, not
    // unconditional writes, so a macro step where nothing actually split
    // costs one 4-byte readback and nothing else.
    const grown = Math.min(await this.agents.readGrownCount(), populationCeiling);
    // Death replacement only narrows the cap for this one policy pass.
    this.agents.setMaxActiveParticles(this.particleCap);
    if (this.epoch !== stepEpoch) return;
    if (!this.mpmCore || !this.agents || !this.config) return;
    if (grown !== this.mpmCore.activeCount) {
      this.mpmCore.setActiveCount(grown);
      this.agents.setActiveCount(grown);
    }

    // Skippable via PhysicsSettings.mpmEnabled (applyPhysics() sets
    // this.mpmEnabled — see that method's own comment) — a debug/testing
    // toggle to isolate sensing/deposit/growth/chirality (everything
    // above, still fully run every step regardless) from MpmCore's own
    // elastic material response, gravity, and repulsion: with this off,
    // positions never advance except where growth itself writes a brand
    // new child's own spawn position (core/agents.wgsl's own
    // agentStep()), so a rollout effectively freezes in place otherwise.
    // Frontend-only — the Python trainer has no equivalent, since
    // disabling real physics during actual evolutionary training would
    // break fitness scoring entirely; this is purely a live-replay
    // viewing aid, same reasoning every other PhysicsSettings field
    // being "playback-only, doesn't affect training" already carries.
    if (this.mpmEnabled) {
      const physicsEncoder = this.device.createCommandEncoder();
      this.mpmCore.encodeSteps(physicsEncoder, this.config.substepsPerMacro);
      this.device.queue.submit([physicsEncoder.finish()]);
    }
    this._currentStep += 1;
  }

  render(context: GPUCanvasContext): void {
    if (!this.renderer || !this.mpmCore) return;
    this.renderer.render(context, this.mpmCore.activeCount);
  }

  setCanvasSizePx(widthPx: number, heightPx: number): void {
    this.pendingCanvasSizePx = [widthPx, heightPx];
    this.renderer?.setCanvasSizePx(widthPx, heightPx);
  }

  /** Field-visualize background — see gpu/render.ts's own module
   * docstring for the full set of modes and gpu/fieldDiagnostics.wgsl's
   * own docstring for how deformation/pressure/shear stay viewer-only
   * rather than extending core/'s shared physics shaders. */
  setFieldMode(mode: FieldMode): void {
    this.pendingFieldMode = mode;
    this.renderer?.setFieldMode(mode);
  }

  setSubstrateChannelStart(start: number): void {
    this.pendingSubstrateChannelStart = start;
    this.renderer?.setSubstrateChannelStart(start);
  }

  setParticleRenderMode(mode: ParticleRenderMode): void {
    this.pendingParticleRenderMode = mode;
    this.renderer?.setParticleRenderMode(mode);
  }

  setActivationAlpha(alpha: number): void {
    this.pendingActivationAlpha = alpha;
    this.renderer?.setActivationAlpha(alpha);
  }

  setWhiteDotsAlpha(alpha: number): void {
    this.pendingWhiteDotsAlpha = alpha;
    this.renderer?.setWhiteDotsAlpha(alpha);
  }

  setNeuralColorAlpha(alpha: number): void {
    this.pendingNeuralColorAlpha = alpha;
    this.renderer?.setNeuralColorAlpha(alpha);
  }

  setInternalStateAlpha(alpha: number): void {
    this.pendingInternalStateAlpha = alpha;
    this.renderer?.setInternalStateAlpha(alpha);
  }

  setInternalStateChannelStart(start: number): void {
    this.pendingInternalStateChannelStart = start;
    this.renderer?.setInternalStateChannelStart(start);
  }

  setChemicalMemoryOpponentSubtraction(amount: number): void {
    this.pendingChemicalMemoryOpponentSubtraction = amount;
    this.renderer?.setChemicalMemoryOpponentSubtraction(amount);
  }

  setBoundaryGradientScale(g0: number): void {
    this.pendingBoundaryGradientScale = g0;
    this.renderer?.setBoundaryGradientScale(g0);
  }

  setGrowthAxisLengthPx(px: number): void {
    this.pendingGrowthAxisLengthPx = px;
    this.renderer?.setGrowthAxisLengthPx(px);
  }

  setPointRadiusPx(px: number): void {
    this.pendingPointRadiusPx = px;
    this.renderer?.setPointRadiusPx(px);
  }

  /** [-2,2] — see gpu/render.ts's setAccent()/field.wgsl's accent curve.
   * Negative suppresses and positive accentuates every background mode. */
  setAccent(accent: number): void {
    this.pendingAccent = accent;
    this.renderer?.setAccent(accent);
  }

  setMorphologyDisplay(gradientVisible: boolean, densityVisible: boolean): void {
    this.pendingMorphologyGradientVisible = gradientVisible;
    this.pendingMorphologyDensityVisible = densityVisible;
    this.renderer?.setMorphologyDisplay(gradientVisible, densityVisible);
  }

  /** [0,2] — see gpu/render.ts's own setBlur()/field.wgsl's own
   * blurDensity() comment. Only the "gradient" background mode's own
   * blur pass reads this — harmless to set regardless of which mode is
   * currently active, same as accent above. */
  setBlur(sigma: number): void {
    this.pendingBlur = sigma;
    this.renderer?.setBlur(sigma);
  }

  /** See gpu/render.ts's own setGradientExponent()/field.wgsl's own
   * colorizeGradient() comment. Only the "gradient" background mode's
   * own colorize pass reads this — harmless to set regardless of which
   * mode is currently active, same as accent/blur above. */
  setGradientExponent(exponent: number): void {
    this.pendingGradientExponent = exponent;
    this.renderer?.setGradientExponent(exponent);
  }

  setZoom(zoom: number): void {
    this.pendingZoom = Math.min(8, Math.max(1, zoom));
    this.renderer?.setZoom(this.pendingZoom);
  }

  setBloom(settings: BloomSettings): void {
    this.pendingBloom = settings;
    this.renderer?.setBloom(settings);
  }

  /** "Add Particle" tool — `(x, y)`: MpmCore's own [0,1]^2 domain
   * coords, already converted from screen space by the caller (render/
   * GridCanvas.tsx). Also tells Agents about the new, larger activeCount
   * (see agents.ts's own setActiveCount() — its own agentStep dispatch
   * is sized off this, independently of MpmCore's own particle
   * dispatches) so the newly-added particle is governed by the same
   * trained policy every other particle already is, starting next
   * step(). Silently does nothing before the first rebuild() (nothing to
   * add a particle to yet), same "ignore calls before ready" stance
   * every other GpuSimulation method already takes. */
  addParticleAt(x: number, y: number): void {
    if (!this.mpmCore || !this.agents) return;
    if (this.mpmCore.activeCount >= this.particleCap) return;
    if (this.mpmCore.addParticleAt(x, y)) {
      this.agents.setActiveCount(this.mpmCore.activeCount);
    }
  }

  /** "Move Particles" tool (gpu/interact.ts) — beginDrag() on pointerdown
   * grabs every particle within GRAB_RADIUS of `(x, y)`, not just the
   * nearest one (see interact.wgsl's own module docstring), dragTo()
   * every animation frame the pointer stays down (not just on
   * pointermove — see Interact.dragTo()'s own docstring for why), endDrag()
   * on pointerup/pointerleave. All three no-op before the first rebuild(). */
  beginDrag(x: number, y: number): void {
    this.interact?.beginGrab(x, y);
  }

  dragTo(x: number, y: number): void {
    this.interact?.dragTo(x, y);
  }

  endDrag(): void {
    this.interact?.endDrag();
  }

  /** "Deform" tool (gpu/deform.ts) — one-shot, called once per click (not
   * per frame the way dragTo() above is). Injects a radial push/pull
   * (`direction`, `strength`) at domain position (x,y), affecting every
   * particle within `radius` — see Deform.inject()'s own docstring for
   * the exact per-particle radial direction, the velocity-impulse vs
   * deformation-gradient-edit math, and the falloff. No-ops before the
   * first rebuild(), same stance every other GpuSimulation method here
   * already takes. */
  injectDeform(x: number, y: number, direction: DeformDirection, strength: number, radius: number, mode: DeformMode): void {
    this.deform?.inject(x, y, direction, strength, radius, mode);
  }

  /** Applies one frame of viewer-only coherent simplex displacement. */
  displaceWithNoise(strength: number, timeSeconds: number): void {
    this.noiseDisplacement?.apply(strength, timeSeconds);
  }

  /** Replaces the live update rule with a fresh random init (see
   * Agents.randomizeWeights()'s own docstring). Existing callers retain the
   * historical restart behavior by default; performance controls can opt out
   * to swap brains during the current rollout. Silently does nothing before
   * the first rebuild(), same stance every other tool method here takes. */
  randomizeWeights(restart = true): UpdateRuleWeights | null {
    if (!this.agents) return null;
    const weights = this.agents.randomizeWeights();
    if (restart) this.restartRollout();
    return weights;
  }

  private destroySimObjects(): void {
    this.mpmCore?.destroy();
    this.environment?.destroy();
    this.agents?.destroy();
    this.renderer?.destroy();
    this.interact?.destroy();
    this.deform?.destroy();
    this.noiseDisplacement?.destroy();
    this.mpmCore = null;
    this.environment = null;
    this.agents = null;
    this.renderer = null;
    this.interact = null;
    this.deform = null;
    this.noiseDisplacement = null;
  }

  destroy(): void {
    this.epoch++;
    this.destroySimObjects();
  }
}

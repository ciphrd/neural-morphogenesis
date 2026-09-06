import { VIEWER_DEFAULTS } from "../viewerConfig";
import { InitialCondition } from "./initialConditions";
import seedDensityModelConfig from "../../../core/config.json";
const seedDensityModel = seedDensityModelConfig.density;
import { cellMemoryFromConfig } from "./types";
import { StableMatchStop, targetMask, matchDomains } from "./shapeMatch";

import { Agents } from "./agents";
import type { BloomSettings } from "./bloom";
import { Deform, type DeformDirection, type DeformMode } from "./deform";
import { Environment } from "./environment";
import { Interact } from "./interact";
import { MAX_PARTICLES, MpmCore } from "./mpmCore";
import { MAX_ZOOM, Renderer, type FieldMode, type ParticleColorMode, type ParticleShape } from "./render";
import { seedBlob, seedRows } from "./rng";
import { chemicalCommunicationArchitectureFromConfig, physicsSettingsFromConfig, type PhysicsSettings, type SimulationConfig, type UpdateRuleWeights } from "./types";
import coreConstantsConfig from "../../../core/config.json";
const coreConstants = coreConstantsConfig.simulation;

export interface SimulationScenario {
  initialLayout:
    | { kind: "rows"; rows: number; columns: number }
    | { kind: "blob"; count: number };
  events: Array<{
    step: number;
    type: "grow";
    particleIndex: number;
    /** Number of contiguous particle slots participating in this event. */
    particleCount?: number;
    /** Fixed world axis. Omit to follow the local boundary tangent. */
    direction?: readonly [number, number];
  }>;
  suppressNaturalGrowth?: boolean;
  /** Lab-only analytic override applied at every MPM growth-grid node. */
  growthFieldOverride?: "radial-inward";
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

  private config: SimulationConfig | null = null;
  private resetKey: string | null = null;
  private pendingTargetPoints: Float32Array | null = null;
  // View-only display preferences — not simulation state, so they must
  // survive rebuild() destroying and recreating the Renderer (a new
  // generation with a different particle/channel/field/hidden-dim shape
  // gets a brand-new Renderer instance; the user's own render-option
  // choices shouldn't reset just because that happened).
  private pendingFieldMode: FieldMode = VIEWER_DEFAULTS.rendering.fieldMode;
  private pendingSubstrateChannelStart = VIEWER_DEFAULTS.rendering.substrateChannelStart;
  private pendingSubstrateZeroIsBlack = VIEWER_DEFAULTS.rendering.substrateZeroIsBlack;
  private pendingBoundaryGradientZeroIsBlack = VIEWER_DEFAULTS.rendering.boundaryGradientZeroIsBlack;
  private pendingParticleShape: ParticleShape = VIEWER_DEFAULTS.rendering.particleShape;
  private pendingParticleColorMode: ParticleColorMode = VIEWER_DEFAULTS.rendering.particleColorMode;
  private pendingParticleAlpha = VIEWER_DEFAULTS.rendering.particleAlpha;
  private pendingDirectionalLineVisible = VIEWER_DEFAULTS.rendering.directionalLineVisible;
  private pendingGrowthLineVisible = VIEWER_DEFAULTS.rendering.growthLineVisible;
  private pendingDomainVisible = VIEWER_DEFAULTS.rendering.domainVisible;
  private pendingGrowthMagnitudeBoost = VIEWER_DEFAULTS.rendering.growthMagnitudeBoost;
  private pendingInternalStateChannelStart = VIEWER_DEFAULTS.rendering.internalStateChannelStart;
  private pendingChemicalMemoryOpponentSubtraction = VIEWER_DEFAULTS.rendering.chemicalMemoryOpponentSubtraction;
  private pendingBoundaryGradientScale = VIEWER_DEFAULTS.rendering.boundaryGradientScale;
  private pendingPointRadiusPx: number | null = null;
  // 0 = identity — see gpu/render.ts's own setAccent()/field.wgsl's own
  // accent uniform comment. Same "view-only, survives rebuild()" reasoning
  // pendingFieldMode above already has.
  private pendingAccent = VIEWER_DEFAULTS.rendering.accent;
  private pendingMorphologyGradientVisible = VIEWER_DEFAULTS.rendering.morphologyGradientVisible;
  private pendingMorphologyDensityVisible = VIEWER_DEFAULTS.rendering.morphologyDensityVisible;
  // 0 = no blur — see gpu/render.ts's own setBlur()/field.wgsl's own
  // blurDensity() comment. Same "view-only, survives rebuild()" reasoning
  // pendingAccent above already has.
  private pendingBlur = VIEWER_DEFAULTS.rendering.blur;
  // 1 = identity — see gpu/render.ts's own setGradientExponent()/
  // field.wgsl's own colorizeGradient() comment. Same "view-only,
  // survives rebuild()" reasoning pendingAccent above already has.
  private pendingGradientExponent = VIEWER_DEFAULTS.rendering.gradientExponent;
  private pendingParticleCap: number | null = null;
  private pendingInitialParticleCount: number | null = null;
  private particleCap = 2;
  private pendingTargetVisible = VIEWER_DEFAULTS.rendering.targetVisible;
  private neuralUpdatesPerMacro = 1;
  private growthDuration = 0;
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
  private pendingZoom = VIEWER_DEFAULTS.rendering.zoom;
  private pendingBloom: BloomSettings = VIEWER_DEFAULTS.rendering.bloom;

  // Bumped by anything that invalidates in-flight GPU state (rebuild(),
  // restartRollout(), destroy()) — step() captures this at its own start
  // and checks it again after its own await (see that method's own
  // docstring for the exact race this guards against: growth's own
  // async readSampleCount() can resolve AFTER a user-triggered restart
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
  private shapeStop = new StableMatchStop({});
  private shapeMask: Float64Array | null = null;
  get shapeStatus() {
    return { complete: this.shapeStop.complete, settling: this.shapeStop.settlingSince !== null && !this.shapeStop.complete,
      match: this.shapeStop.match, ...this.samplingStatus };
  }
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
  get samplingStatus(): { atCapacity: boolean; capacityBlocked: boolean; unresolvedSamples: number } {
    return { atCapacity: this.particleCount >= this.particleCap,
      capacityBlocked: this.agents?.capacityBlocked ?? false,
      unresolvedSamples: this.agents?.unresolvedSamples ?? 0 };
  }

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
    return this.shapeStop.growthEnabled && (cutoff == null || this._currentStep < cutoff);
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
      JSON.stringify(config.chemicalChannelProfiles),
      config.hiddenDim,
      config.policyArchitecture,
      chemicalCommunicationArchitectureFromConfig(config),
      config.elasticStrainScale,
      config.elasticStrainInputsEnabled,
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
      depositRate: config.depositRate,
      normalizeDepositsByLocalDensity: config.normalizeDepositsByLocalDensity,
      advectionDt: config.substepsPerMacro * coreConstants.DT,
      chemicalCommunicationArchitecture: chemicalCommunicationArchitectureFromConfig(config),
      channelProfiles: config.chemicalChannelProfiles,
    }, mpmCore.gridVel);

    const agents = new Agents(this.device, mpmCore, environment, {
      channels: config.channels,
      hiddenDim: config.hiddenDim,
      policyArchitecture: config.policyArchitecture,
      chemicalCommunicationArchitecture: chemicalCommunicationArchitectureFromConfig(config),
      maxEnvWrite: config.maxEnvWrite,
      sampleSpacing: config.sampleSpacing,
      friction: config.friction,
      growthEnabled: 1.0,
      maxActiveParticles: this.particleCap,
      spawnX: config.spawnX,
      spawnY: config.spawnY,
      elasticStrainScale: config.elasticStrainScale,
      elasticStrainInputsEnabled: config.elasticStrainInputsEnabled,
      chemicalValueInputMultiplier: config.chemicalValueInputMultiplier,
      chemicalGradientInputScale: config.chemicalGradientInputScale,
      boundaryTangentMinGradient: config.boundaryTangentMinGradient,
    });
    agents.loadWeights(config.weights);

    const renderer = new Renderer(
      this.device,
      this.format,
      mpmCore,
      environment,
      agents.particleMetaState,
      mpmCore.growthField,
    );
    if (this.pendingCanvasSizePx) renderer.setCanvasSizePx(...this.pendingCanvasSizePx);
    if (this.pendingTargetPoints) renderer.setTargetPoints(this.pendingTargetPoints);
    renderer.setTargetVisible(this.pendingTargetVisible);
    renderer.setFieldMode(this.pendingFieldMode);
    renderer.setSubstrateChannelStart(this.pendingSubstrateChannelStart);
    renderer.setSubstrateZeroIsBlack(this.pendingSubstrateZeroIsBlack);
    renderer.setBoundaryGradientZeroIsBlack(this.pendingBoundaryGradientZeroIsBlack);
    renderer.setParticleShape(this.pendingParticleShape);
    renderer.setParticleColorMode(this.pendingParticleColorMode);
    renderer.setParticleAlpha(this.pendingParticleAlpha);
    renderer.setDirectionalLineVisible(this.pendingDirectionalLineVisible);
    renderer.setGrowthLineVisible(this.pendingGrowthLineVisible);
    renderer.setDomainVisible(this.pendingDomainVisible);
    renderer.setGrowthMagnitudeBoost(this.pendingGrowthMagnitudeBoost);
    renderer.setInternalStateChannelStart(this.pendingInternalStateChannelStart);
    renderer.setChemicalMemoryOpponentSubtraction(this.pendingChemicalMemoryOpponentSubtraction);
    renderer.setBoundaryGradientScale(this.pendingBoundaryGradientScale);
    if (this.pendingPointRadiusPx !== null) renderer.setPointRadiusPx(this.pendingPointRadiusPx);
    renderer.setAccent(this.pendingAccent);
    renderer.setMorphologyDisplay(this.pendingMorphologyGradientVisible, this.pendingMorphologyDensityVisible);
    renderer.setBlur(this.pendingBlur);
    renderer.setGradientExponent(this.pendingGradientExponent);
    renderer.setZoom(this.pendingZoom);
    renderer.setBloom(this.pendingBloom);

    const interact = new Interact(this.device, mpmCore);
    const deform = new Deform(this.device, mpmCore);

    this.mpmCore = mpmCore;
    this.environment = environment;
    this.agents = agents;
    this.renderer = renderer;
    this.interact = interact;
    this.deform = deform;
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
      Math.floor(this.particleCap / 2),
      Math.max(1, Math.floor(
        this.pendingInitialParticleCount
        ?? this.config.initialParticleCount
        ?? coreConstantsConfig.run.initialParticleCount
      ))
    );
    const scene = this.scenario?.initialLayout.kind === "rows"
      ? seedRows({
          rows: this.scenario.initialLayout.rows,
          columns: this.scenario.initialLayout.columns,
          centerX: this.config.spawnX,
          centerY: this.config.spawnY,
          spacing: this.config.sampleSpacing,
        })
      : seedBlob({
          count: this.scenario?.initialLayout.kind === "blob"
            ? this.scenario.initialLayout.count
            : initialCount,
          centerX: this.config.spawnX,
          centerY: this.config.spawnY,
          spacing: this.config.sampleSpacing,
          seed: this.config.seed,
        });
    if (scene.count > this.particleCap) {
      throw new Error(`Triangle seed needs ${scene.count} sample slots; capacity is ${this.particleCap}`);
    }
    const preset = this.config.initialCondition;
    if (preset === "internal-state" && cellMemoryFromConfig(this.config) !== "recurrent") {
      throw new Error("Internal-state initial condition requires recurrent cell memory");
    }
    const radius = Math.sqrt((scene.count/2)*(this.config.sampleSpacing*seedDensityModel.INITIAL_PACKING_SPACING_SCALE)**2*Math.sqrt(3)/(2*Math.PI));
    const perturbation = new InitialCondition(preset,
      this.config.initialConditionStrength,
      this.config.initialConditionChannel,
      this.config.seed, [this.config.spawnX, this.config.spawnY], radius);
    perturbation.deform(scene);
    this.mpmCore.resetGrowthBuffers(this.particleCap);
    this.mpmCore.loadScene(scene);
    // Every slot beyond the genuinely seeded particles is destined to become
    // a real particle via growth, at some unknown point in this rollout
    // — see MpmCore.resetGrowthBuffers()'s own docstring for why this
    // has to run every rollout (not just once, ever) despite seedBlob()
    // already giving genuinely-seeded particles these exact same fresh
    // defaults.
    this.environment.reset();
    this.agents.setSpawnCenter(this.config.spawnX, this.config.spawnY);
    this.agents.setMaxActiveParticles(this.particleCap);
    this.agents.setActiveCount(scene.count);
    // Clear rollout-scoped policy state. The first agent evaluation derives
    // alignment from chemical channel index 3's freshly sensed gradient.
    this.agents.resetState( perturbation.states(scene.positions, this.config.channels));
    if (this.environment.chemicalCommunicationArchitecture === "persistent-environment") {
      perturbation.seedEnvironment(this.device, this.environment);
    }
    this._currentStep = 0;
    this.shapeStop = new StableMatchStop(this.scenario ? {} : this.config);
    this.shapeMask = this.shapeStop.enabled && this.config.shapeTarget
      ? targetMask(this.config.shapeTarget, this.config.rasterResolution) : null;
  }

  /** Installs an isolated, deterministic lab scenario and restarts it. */
  setScenario(scenario: SimulationScenario | null): void {
    this.scenario = scenario;
    if (this.mpmCore) this.restartRollout();
  }

  private applyPhysics(physics: PhysicsSettings): void {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    this.mpmCore.setGravity(physics.gravity);
    this.neuralUpdatesPerMacro = Math.max(1, Math.round(physics.neuralUpdatesPerMacro));
    const communicationDt = Math.max(0, physics.communicationSpeed) / this.neuralUpdatesPerMacro;
    this.environment.setPhysics(
      physics.decay,
      physics.depositRate,
      this.neuralUpdatesPerMacro,
      physics.communicationSpeed,
      physics.normalizeDepositsByLocalDensity,
    );
    const growthCompressionStart = Math.max(0, physics.growthCompressionStart);
    const growthCompressionStop = Math.max(
      growthCompressionStart,
      physics.growthCompressionStop,
    );
    const growthCompressionFeedback = Math.max(
      0, Math.min(1, physics.growthCompressionFeedback),
    );
    const growthSpeedMultiplier = Math.max(0, physics.growthSpeedMultiplier);
    const effectiveGrowthDuration = growthSpeedMultiplier > 0
      ? physics.growthDuration / growthSpeedMultiplier
      : 0;
    this.mpmCore.setMaterial(
      physics.materialE,
      physics.materialNu,
      physics.materialHardening,
      physics.materialElasticity,
      // Controller ticks per uncompressed area doubling. MpmCore derives
      // the shader's internal per-substep rate from this and the run cadence.
      effectiveGrowthDuration,
      physics.growthAnisotropy,
      this.config.substepsPerMacro,
      physics.particleMass,
      physics.particleVolume,
      growthCompressionStart,
      growthCompressionStop,
      growthCompressionFeedback,
    );
    // Lab event admission and newborn fade must follow the same effective
    // timescale as the material-rate uniform, not the unscaled run setting.
    this.growthDuration = effectiveGrowthDuration;
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
      physics.repulsionMaxDelta,
    );
    this.agents.setCommunicationTimestep(communicationDt);
    this.agents.setInternalStateSpeed(physics.internalStateSpeed);

    this.agents.setChemicalValueInputMultiplier(physics.chemicalValueInputMultiplier);
    this.agents.setChemicalGradientInputScale(physics.chemicalGradientInputScale);

    this.agents.setBoundaryTangentMinGradient(physics.boundaryTangentMinGradient);

    this.agents.setMaterialAreaBudget(physics.materialAreaBudget);
    this.agents.setPhysics({
      maxEnvWrite: physics.maxEnvWrite,
      sampleSpacing: physics.sampleSpacing,
      friction: physics.friction,
      growthEnabled: this.growthIsEnabled() ? 1.0 : 0.0,
    });
    // ?? true — same pre-broadcast guard reasoning depositRate's own
    // ?? 1.0 guard above gives, for a train_server.py process still
    // running pre-mpmEnabled code. Not a GPU uniform write (unlike every
    // setting above) — a plain JS field step() reads to decide whether
    // to skip mpmCore.encodeSteps() at all (see that method's own
    // comment).
    this.mpmEnabled = physics.mpmEnabled;
    this.environment.setAdvectionTimestep(
      this.mpmEnabled ? this.config.substepsPerMacro * coreConstants.DT : 0,
    );
  }

  setPhysics(physics: PhysicsSettings): void {
    this.applyPhysics(physics);
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
   * deposit (+ the copy encodeReadSampleCount() adds) first, then —
   * *after* awaiting readSampleCount(), so the result is actually known —
   * mpmCore.encodeSteps()'s own physics substeps, sized off whatever
   * activeCount now is. Splitting into two submits like this costs
   * nothing extra beyond the readback itself already costs: WebGPU's
   * queue is a single in-order timeline, so the second submit correctly
   * sees the first one's positions/deposits/grown particles regardless
   * of how many submits that took.
   *
   * Captures `this.epoch` before the await and bails out (no activeCount
   * write, no physics submit, no currentStep bump) if it's changed by
   * the time readSampleCount() resolves — see that field's own comment
   * for the exact restart-vs-in-flight-step race this prevents. */
  async step(): Promise<void> {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    if (this.shapeStop.complete) return;
    const stepEpoch = this.epoch;
    const nextStep = this._currentStep + 1;
    const forcedGrowth = this.scenario?.events.find((event) => {
      const growthStartStep = Math.max(1, event.step - Math.ceil(this.growthDuration));
      return event.type === "grow" && nextStep >= growthStartStep && nextStep <= event.step;
    });
    const growthStartStep = forcedGrowth
      ? Math.max(1, forcedGrowth.step - Math.ceil(this.growthDuration))
      : -1;
    this.agents.setForcedGrowthControl(
      forcedGrowth?.particleIndex ?? null,
      forcedGrowth?.direction ?? null,
      nextStep === growthStartStep,
      forcedGrowth?.particleCount ?? 1,
    );
    this.agents.setForcedGrowthFieldOverride(
      this.scenario?.growthFieldOverride ?? null,
    );
    this.agents.setGrowthEnabled(
      !this.scenario?.suppressNaturalGrowth && this.growthIsEnabled(),
    );
    const encoder = this.device.createCommandEncoder();
    this.mpmCore.encodeMorphology(encoder);
    // Carry the persistent substrate through the preceding MPM motion before
    // this tick's first policy read. Divergent growth flow therefore expands
    // the substrate together with the material rather than leaving it behind.
    for (let communicationRound = 0; communicationRound < this.neuralUpdatesPerMacro; communicationRound++) {
      const finalRound = communicationRound === this.neuralUpdatesPerMacro - 1;
      this.environment.encodePreparePersistent(encoder, communicationRound === 0);
      this.environment.encodeClear(encoder);
      if (this.environment.chemicalCommunicationArchitecture === "cell-owned-projection") {
        this.agents.encodeSplatChemicalState(encoder);
      }
      this.environment.encodeSense(encoder);
      this.agents.encodeStep(
        encoder,
        this.environment.parity,
        finalRound
      );
      this.environment.encodeMergePersistent(encoder);
    }
    this.agents.encodeReadSampleCount(encoder);
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
    const grown = Math.min(await this.agents.readSampleCount(), this.particleCap);
    if (this.epoch !== stepEpoch) return;
    if (!this.mpmCore || !this.agents || !this.config) return;
    if (grown !== this.mpmCore.activeCount) {
      this.mpmCore.setActiveCount(grown);
      this.agents.setActiveCount(grown);
    }

    if (this.mpmEnabled) {
      const physicsEncoder = this.device.createCommandEncoder();
      this.mpmCore.encodeSteps(physicsEncoder, this.config.substepsPerMacro);
      this.device.queue.submit([physicsEncoder.finish()]);
    }
    this._currentStep += 1;
    if (this.shapeMask && this.config.shapeTarget && this.shapeStop.due(this._currentStep)) {
      const triangles = await this.mpmCore.readTriangles();
      if (this.epoch !== stepEpoch || !this.config?.shapeTarget) return;
      const match = matchDomains(triangles, this.config.shapeTarget, this.shapeMask);
      this.shapeStop.observe(this._currentStep, match,
        this.samplingStatus.atCapacity || this.agents.capacityBlocked || this.agents.unresolvedSamples > 0);
    }
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

  setSubstrateZeroIsBlack(enabled: boolean): void {
    this.pendingSubstrateZeroIsBlack = enabled;
    this.renderer?.setSubstrateZeroIsBlack(enabled);
  }

  setBoundaryGradientZeroIsBlack(enabled: boolean): void {
    this.pendingBoundaryGradientZeroIsBlack = enabled;
    this.renderer?.setBoundaryGradientZeroIsBlack(enabled);
  }

  setParticleShape(shape: ParticleShape): void {
    this.pendingParticleShape = shape;
    this.renderer?.setParticleShape(shape);
  }

  setParticleColorMode(mode: ParticleColorMode): void {
    this.pendingParticleColorMode = mode;
    this.renderer?.setParticleColorMode(mode);
  }

  setParticleAlpha(alpha: number): void {
    this.pendingParticleAlpha = alpha;
    this.renderer?.setParticleAlpha(alpha);
  }

  setDirectionalLineVisible(visible: boolean): void {
    this.pendingDirectionalLineVisible = visible;
    this.renderer?.setDirectionalLineVisible(visible);
  }

  setDomainVisible(visible: boolean): void {
    this.pendingDomainVisible = visible;
    this.renderer?.setDomainVisible(visible);
  }

  setGrowthLineVisible(visible: boolean): void {
    this.pendingGrowthLineVisible = visible;
    this.renderer?.setGrowthLineVisible(visible);
  }

  setGrowthMagnitudeBoost(boost: number): void {
    this.pendingGrowthMagnitudeBoost = boost;
    this.renderer?.setGrowthMagnitudeBoost(boost);
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
    this.pendingZoom = Math.min(MAX_ZOOM, Math.max(1, zoom));
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

  /** Replaces the live update rule with a fresh random init (see
   * Agents.randomizeWeights()'s own docstring) and restarts the rollout —
   * without the restart, whatever's already grown stays governed by the
   * OLD weights forever (a rollout only ever consults the update rule at
   * the moment a particle senses/acts, not retroactively), so the new
   * policy would only visibly affect brand-new growth from here, which
   * reads as broken rather than "randomized." Silently does nothing
   * before the first rebuild(), same stance every other tool method here
   * takes. */
  randomizeWeights(): UpdateRuleWeights | null {
    if (!this.agents) return null;
    const weights = this.agents.randomizeWeights();
    this.restartRollout();
    return weights;
  }

  private destroySimObjects(): void {
    this.mpmCore?.destroy();
    this.environment?.destroy();
    this.agents?.destroy();
    this.renderer?.destroy();
    this.interact?.destroy();
    this.deform?.destroy();
    this.mpmCore = null;
    this.environment = null;
    this.agents = null;
    this.renderer = null;
    this.interact = null;
    this.deform = null;
  }

  destroy(): void {
    this.epoch++;
    this.destroySimObjects();
  }
}

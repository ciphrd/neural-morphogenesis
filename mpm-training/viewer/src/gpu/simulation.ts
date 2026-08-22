// Ties MpmCore + Environment + Agents + Renderer into one autonomous
// macro step — the browser analogue of trainer/training_sim.py's own
// TrainingRollout.macro_step(). GPU-resident for every data-related
// buffer (positions, velocities, F/C/Jp, the chemical field, weights),
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
//   1. environment.encodeSense()      — clearScratch + computeGradient,
//      over whichever grid buffer is currently "current" (last step's
//      diffuse+decay output).
//   2. agents.encodeStep()            — NN forward pass: reads that same
//      grid + gradient, writes env_write into the deposit scratch, and
//      nudges MpmCore's own velocities buffer directly in place (strafe
//      — an acceleration, damped by friction, see agents.wgsl's own
//      module docstring; the separate `accel` output stays unused) —
//      may also grow activeCount (agents.wgsl's own agentStep()).
//   3. environment.encodeMergeAndDecay() — folds the deposit into the
//      current grid, blurs+decays into the other buffer, flips parity.
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
import { Environment } from "./environment";
import { Interact } from "./interact";
import { MpmCore } from "./mpmCore";
import { NnProbe, type NetworkProbe } from "./nnProbe";
import { Renderer, type FieldMode, type ParticleShape } from "./render";
import { seedBlob, spawnUniform01 } from "./rng";
import type { PhysicsSettings, SimulationConfig } from "./types";

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
  // Network visualization's own live forward-pass readback (gpu/nnProbe.ts)
  // — same "fresh instance per rebuild()" reasoning as interact above.
  private nnProbe: NnProbe | null = null;

  private config: SimulationConfig | null = null;
  private resetKey: string | null = null;
  private pendingTargetPoints: Float32Array | null = null;
  // View-only display preferences — not simulation state, so they must
  // survive rebuild() destroying and recreating the Renderer (a new
  // generation with a different particle/channel/field/hidden-dim shape
  // gets a brand-new Renderer instance; the user's own render-option
  // choices shouldn't reset just because that happened).
  private pendingFieldMode: FieldMode = "none";
  private pendingParticleShape: ParticleShape = "circle";
  private pendingPointRadiusPx: number | null = null;
  // 0 = identity — see gpu/render.ts's own setAccent()/field.wgsl's own
  // accent uniform comment. Same "view-only, survives rebuild()" reasoning
  // pendingFieldMode above already has.
  private pendingAccent = 0;

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
  // why velocities/F/C/Jp get that treatment but positions doesn't).
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
  get ready(): boolean {
    return this.mpmCore !== null;
  }

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  private resetKeyFor(config: SimulationConfig): string {
    return [config.particles, config.channels, config.fieldN, config.hiddenDim, config.chirality].join(":");
  }

  loadGeneration(config: SimulationConfig): void {
    const key = this.resetKeyFor(config);
    if (!this.mpmCore || key !== this.resetKey) {
      this.rebuild(config);
      this.resetKey = key;
    } else {
      this.agents!.loadWeights(config.weights);
      this.config = config;
      this.applyPhysics(config);
    }
    this.restartRollout();
  }

  private rebuild(config: SimulationConfig): void {
    this.epoch++;
    this.destroySimObjects();

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
    });

    const agents = new Agents(this.device, mpmCore, environment, {
      channels: config.channels,
      hiddenDim: config.hiddenDim,
      maxAccel: config.maxAccel,
      maxStrafe: config.maxStrafe,
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
      // config.particles is the growth CAP, not a starting count — see
      // types.ts's own SimulationConfig.particles docstring.
      maxActiveParticles: config.particles,
      spawnX: config.spawnX,
      spawnY: config.spawnY,
    });
    agents.loadWeights(config.weights);

    const renderer = new Renderer(this.device, this.format, mpmCore, environment, agents);
    if (this.pendingTargetPoints) renderer.setTargetPoints(this.pendingTargetPoints);
    renderer.setFieldMode(this.pendingFieldMode);
    renderer.setParticleShape(this.pendingParticleShape);
    if (this.pendingPointRadiusPx !== null) renderer.setPointRadiusPx(this.pendingPointRadiusPx);
    renderer.setAccent(this.pendingAccent);

    const interact = new Interact(this.device, mpmCore);

    const nnProbe = new NnProbe(this.device, mpmCore, environment, agents, {
      channels: config.channels,
      hiddenDim: config.hiddenDim,
      chirality: config.chirality,
    });

    this.mpmCore = mpmCore;
    this.environment = environment;
    this.agents = agents;
    this.renderer = renderer;
    this.interact = interact;
    this.nnProbe = nnProbe;
    this.config = config;
    this.applyPhysics(config);
  }

  /** Re-seeds particles from the current config's spawn params + winner
   * seed, zeroes the chemical field, and resets the step counter — a
   * fresh rollout of the *same* generation, same "restart" GridCanvas's
   * Playback controls and the RAF loop's own loop-at-trained-steps
   * behavior both call. */
  restartRollout(): void {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    this.epoch++;
    // HARDCODED experiment: start with 2 particles, back to back,
    // instead of the usual single starting particle (config.particles is
    // still the growth CAP, not a starting count — see types.ts's own
    // SimulationConfig.particles docstring) — particle 1 is placed
    // config.splitDisplacement behind particle 0 along a shared random
    // axis (same displacement/direction convention growth's own split
    // uses, agents.wgsl's own agentStep() `behindDir`), but with its
    // heading FLIPPED (theta + PI) rather than copied, so the two face
    // away from each other. Mirrors trainer/training_sim.py's own
    // TrainingRollout.__init__ — see that constructor's own comment.
    // Not a general N-agent or CLI-configurable start yet.
    const scene = seedBlob({
      count: 2,
      centerX: this.config.spawnX,
      centerY: this.config.spawnY,
      halfWidth: this.config.spawnHalfWidth,
      seed: this.config.seed,
    });
    // Index 4 — seedBlob({count:2,...}) above claims indices 0-3 for its
    // own 2 particles' x/y jitter (see that function's own docstring),
    // this is the next one over. Bit-exact with
    // trainer/training_sim.py's own theta draw.
    const theta = (spawnUniform01(this.config.seed, 4) * 2 - 1) * Math.PI;
    const behindDir = [-Math.cos(theta), -Math.sin(theta)];
    scene.positions[2] = ((scene.positions[0] + behindDir[0] * this.config.splitDisplacement) % 1 + 1) % 1;
    scene.positions[3] = ((scene.positions[1] + behindDir[1] * this.config.splitDisplacement) % 1 + 1) % 1;
    this.mpmCore.loadScene(scene);
    // Every slot beyond these 2 starting particles is destined to become
    // a real particle via growth, at some unknown point in this rollout
    // — see MpmCore.resetGrowthBuffers()'s own docstring for why this
    // has to run every rollout (not just once, ever) despite seedBlob()
    // already giving genuinely-seeded particles these exact same fresh
    // defaults.
    this.mpmCore.resetGrowthBuffers(this.config.particles);
    this.environment.reset();
    // Every rollout — same "run-constant in practice today, but a
    // rollout-scoped setter regardless" convention this method's own
    // seedBlob()/setActiveCount() calls already follow. See
    // Agents.setSpawnCenter()'s own docstring for what this drives (the
    // NN's own position input).
    this.agents.setSpawnCenter(this.config.spawnX, this.config.spawnY);
    this.agents.setActiveCount(2);
    // Heading's own per-slot fill and growth's own seed are both bit-
    // exact via rng.ts's own spawnUniform01()/growthSeed() respectively
    // (two DIFFERENT hash domains, see spawnUniform01()'s own comment
    // for why they're safe to derive from the identical raw seed
    // without correlating) — see Agents.resetHeading()'s own docstring.
    this.agents.resetHeading(this.config.seed);
    // resetHeading() above already randomized every slot's own heading
    // independently (including these 2) — overwrite just slots 0/1 with
    // the coordinated back-to-back pair computed above (theta/theta+PI),
    // mirroring trainer/agents_gpu.py's own set_headings() call site.
    this.agents.setHeadings(new Float32Array([theta, theta + Math.PI]));
    this._currentStep = 0;
  }

  /** Live-adjustable knobs only — see types.ts's own PhysicsSettings
   * docstring for why this is a strict subset of SimulationConfig. No
   * rebuild, just uniform writes — every field here has a live setter on
   * MpmCore/Environment/Agents, so this is safe to call on every
   * PhysicsPanel slider tick without disturbing the rollout in flight.
   * `physics` only needs to structurally match PhysicsSettings — a full
   * SimulationConfig (e.g. from rebuild()/loadGeneration()) works too,
   * extra fields simply ignored. damping's own substep count comes from
   * `this.config` (not `physics`), matching evolve.py's own rollout()
   * (which converts a run's damping loss-fraction using its own
   * --substeps-per-macro, not a fixed constant) — this.config must
   * already be set before this runs. */
  private applyPhysics(physics: PhysicsSettings): void {
    if (!this.mpmCore || !this.environment || !this.agents || !this.config) return;
    this.mpmCore.setGravity(physics.gravity);
    this.mpmCore.setMaterial(physics.materialE, physics.materialNu, physics.materialHardening, physics.materialElasticity);
    this.mpmCore.setDamping(physics.damping, this.config.substepsPerMacro);
    this.mpmCore.setSplatRadius(physics.splatRadius);
    this.mpmCore.setRepulsionStrength(physics.repulsionStrength);
    // ?? 1.0 (= unchanged) guards a call to this with a raw SimulationConfig
    // from a train_server.py process still running pre-depositRate code
    // (loadGeneration()/rebuild() both pass `config` straight through here,
    // bypassing types.ts's own physicsSettingsFromConfig() guard).
    this.environment.setPhysics({ decay: physics.decay, depositRate: physics.depositRate ?? 1.0 });
    this.agents.setPhysics({
      maxAccel: physics.maxAccel,
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

  /** `points`: flat [x0,y0,x1,y1,...] in MpmCore's own [0,1]^2 domain.
   * Cached (not just forwarded) since it can arrive before the first
   * rebuild() ever runs. */
  setTargetPoints(points: Float32Array): void {
    this.pendingTargetPoints = points;
    this.renderer?.setTargetPoints(points);
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
    const stepEpoch = this.epoch;
    const encoder = this.device.createCommandEncoder();
    this.environment.encodeSense(encoder);
    this.agents.encodeStep(encoder, this.environment.parity);
    this.environment.encodeMergeAndDecay(encoder);
    this.agents.encodeReadGrownCount(encoder);
    this.device.queue.submit([encoder.finish()]);

    // min(...) — growth's own atomic counter can overshoot config.particles
    // (the growth cap — see types.ts's own SimulationConfig.particles
    // docstring) slightly (several agents claiming a slot the same step,
    // right at the cap — see core/agents.wgsl's own agentStep() comment
    // for why that's left unguarded rather than compare-exchanged away);
    // clamping the *reported* count here is what actually enforces the
    // cap, since agents.wgsl itself already refuses to WRITE a claimed
    // slot past that either way. A plain != check below, not
    // unconditional writes, so a macro step where nothing actually split
    // costs one 4-byte readback and nothing else.
    const grown = Math.min(await this.agents.readGrownCount(), this.config.particles);
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

  /** Circle (default) or a triangle pointing along each particle's own
   * heading (Agents' own persistent per-particle heading state, not
   * velocity — see agents.wgsl's own module docstring) — see render.wgsl's
   * own triangleVertex. */
  setParticleShape(shape: ParticleShape): void {
    this.pendingParticleShape = shape;
    this.renderer?.setParticleShape(shape);
  }

  setPointRadiusPx(px: number): void {
    this.pendingPointRadiusPx = px;
    this.renderer?.setPointRadiusPx(px);
  }

  /** [0,2] — see gpu/render.ts's own setAccent()/field.wgsl's own
   * accent uniform comment for the exact exponential curve. Applies to
   * every background field mode at once. */
  setAccent(accent: number): void {
    this.pendingAccent = accent;
    this.renderer?.setAccent(accent);
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

  /** Replaces the live update rule with a fresh random init (see
   * Agents.randomizeWeights()'s own docstring) and restarts the rollout —
   * without the restart, whatever's already grown stays governed by the
   * OLD weights forever (a rollout only ever consults the update rule at
   * the moment a particle senses/acts, not retroactively), so the new
   * policy would only visibly affect brand-new growth from here, which
   * reads as broken rather than "randomized." Silently does nothing
   * before the first rebuild(), same stance every other tool method here
   * takes. */
  randomizeWeights(): void {
    if (!this.agents) return;
    this.agents.randomizeWeights();
    this.restartRollout();
  }

  /** One forward-pass snapshot for the Network panel's own brain
   * visualization — see gpu/nnProbe.ts's own module docstring. Callers
   * drive their own cadence (this is NOT called from step() — a once-in-
   * a-while diagnostic readback has no business adding a second async
   * GPU round-trip to the already-async main step() path); null before
   * the first rebuild(), or if a previous probe() call on this same
   * instance hasn't resolved yet (see NnProbe.probe()'s own docstring). */
  probeNetwork(): Promise<NetworkProbe | null> {
    if (!this.nnProbe || !this.environment) return Promise.resolve(null);
    return this.nnProbe.probe(this.environment.parity);
  }

  private destroySimObjects(): void {
    this.mpmCore?.destroy();
    this.environment?.destroy();
    this.agents?.destroy();
    this.renderer?.destroy();
    this.interact?.destroy();
    this.nnProbe?.destroy();
    this.mpmCore = null;
    this.environment = null;
    this.agents = null;
    this.renderer = null;
    this.interact = null;
    this.nnProbe = null;
  }

  destroy(): void {
    this.epoch++;
    this.destroySimObjects();
  }
}

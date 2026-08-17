import { GpuEnvironment } from "./environment";
import { GpuAgents } from "./agents";
import { DEFAULT_INTENSITY, GpuRender } from "./render";
import type { SpawnDistribution } from "./rng";
import type { BackgroundMode, SimulationConfig } from "./types";

function resetKeyFor(config: SimulationConfig): string {
  return [config.gridWidth, config.gridHeight, config.channels, config.agentCount, config.hiddenDim].join(":");
}

/** Orchestrator owning every GPU resource for one training-viewer
 * session: the chemical grid (GpuEnvironment), agents (GpuAgents), and
 * the render pipeline (GpuRender). Mirrors simulation.py's role — one
 * `step()` call runs exactly the pass sequence
 * Simulation.step()/environment.py/agent_state.py/update_rule.py
 * describe, in the same order. */
export class GpuSimulation {
  private readonly device: GPUDevice;
  private readonly canvasFormat: GPUTextureFormat;

  private environment: GpuEnvironment | null = null;
  private agents: GpuAgents | null = null;
  private renderer: GpuRender | null = null;

  private resetKey: string | null = null;
  // Which grid buffer (0=gridA, 1=gridB) holds the CURRENT state — see
  // GpuEnvironment's own docstring for the ping-pong convention.
  private parity = 0;
  private stepIndex = 0;
  private totalSteps = 0;
  private loopCount = 1;
  private targetPoints: readonly (readonly [number, number])[] | null = null;
  // Both survive a hard rebuild() (which constructs a fresh GpuRender) —
  // reapplied right after, so switching the target/agent count mid-
  // session doesn't silently reset the user's background choice.
  private backgroundMode: BackgroundMode = "substrate";
  private intensity = DEFAULT_INTENSITY;
  // Survives rebuild() same as the above — re-read by every restartRollout()
  // call, not just the next loadGeneration(). Viewer-only, see rng.ts.
  private spawnDistribution: SpawnDistribution = "default";
  // The most recently loaded generation's config — retained so
  // setSpawnDistribution() can restart the *current* rollout immediately
  // with the new distribution, instead of only taking effect on the next
  // loadGeneration()/loop.
  private currentConfig: SimulationConfig | null = null;

  constructor(device: GPUDevice, canvasFormat: GPUTextureFormat) {
    this.device = device;
    this.canvasFormat = canvasFormat;
  }

  get currentStep(): number {
    return this.stepIndex;
  }

  get steps(): number {
    return this.totalSteps;
  }

  get ready(): boolean {
    return this.environment !== null;
  }

  setTargetPoints(points: readonly (readonly [number, number])[]): void {
    this.targetPoints = points;
    this.renderer?.uploadTargetPoints(points);
  }

  setBackgroundMode(mode: BackgroundMode): void {
    this.backgroundMode = mode;
    this.renderer?.setBackgroundMode(mode);
  }

  setIntensity(intensity: number): void {
    this.intensity = intensity;
    this.renderer?.setIntensity(intensity);
  }

  /** Changes the initial-spread shape used the next time a rollout
   * (re)starts. Also restarts the *currently loaded* generation right
   * away (reusing loopCurrentGeneration()'s re-seed-and-restart logic) so
   * toggling this control gives immediate visual feedback instead of
   * waiting for the rollout to loop on its own. */
  setSpawnDistribution(distribution: SpawnDistribution): void {
    if (distribution === this.spawnDistribution) return;
    this.spawnDistribution = distribution;
    if (this.currentConfig) this.loopCurrentGeneration(this.currentConfig);
  }

  /** Loads a generation's config: a full GPU-resource rebuild if the
   * grid/agent shape changed since the last load (e.g. train_server.py
   * restarted with different CLI args mid-session — C/gridSize/N are
   * baked into WGSL as compile-time consts, so a shape change can't be
   * handled by just re-uploading weights), otherwise a cheap weight-swap.
   * Either way, starts a fresh rollout from step 0. */
  loadGeneration(config: SimulationConfig): void {
    const key = resetKeyFor(config);
    if (key !== this.resetKey) {
      this.rebuild(config);
      this.resetKey = key;
    } else {
      this.agents!.loadWeights(config.weights);
    }
    this.totalSteps = config.steps;
    this.currentConfig = config;
    this.restartRollout(config.gridWidth, config.gridHeight, config.seed);
  }

  /** Re-seeds a fresh rollout of the SAME generation once `steps` is
   * reached (see render/GridCanvas.tsx) — loops rather than freezing on
   * the last frame. Derives a new JS-side seed each loop (not the
   * original winner seed again) so consecutive loops don't look
   * identical. */
  loopCurrentGeneration(config: SimulationConfig): void {
    this.loopCount += 1;
    const seed = (config.seed ^ Math.imul(this.loopCount, 0x9e3779b9)) >>> 0;
    this.currentConfig = config;
    this.restartRollout(config.gridWidth, config.gridHeight, seed);
  }

  private rebuild(config: SimulationConfig): void {
    this.environment?.destroy();
    this.agents?.destroy();
    this.renderer?.destroy();

    this.environment = new GpuEnvironment(this.device, {
      width: config.gridWidth,
      height: config.gridHeight,
      channels: config.channels,
      decay: config.decay,
    });
    this.agents = new GpuAgents(this.device, this.environment, {
      width: config.gridWidth,
      height: config.gridHeight,
      channels: config.channels,
      hiddenDim: config.hiddenDim,
      agentCount: config.agentCount,
      spawnSpread: config.spawnSpread,
      maxSpeed: config.maxSpeed,
      maxAccel: config.maxAccel,
      maxStrafe: config.maxStrafe,
      edgeMargin: config.edgeMargin,
    });
    this.agents.loadWeights(config.weights);
    this.renderer = new GpuRender(this.device, this.canvasFormat, this.environment, this.agents, {
      width: config.gridWidth,
      height: config.gridHeight,
      channels: config.channels,
    });
    if (this.targetPoints) this.renderer.uploadTargetPoints(this.targetPoints);
    this.renderer.setBackgroundMode(this.backgroundMode);
    this.renderer.setIntensity(this.intensity);
  }

  private restartRollout(gridWidth: number, gridHeight: number, seed: number): void {
    this.environment!.reset();
    this.agents!.resetAgents(gridWidth, gridHeight, seed, this.spawnDistribution);
    this.parity = 0;
    this.stepIndex = 0;
  }

  /** One full simulation step — clearScratch, computeGradient,
   * agentStep, mergeDeposit, diffuseDecay. Each stage gets its own
   * compute pass (not one pass with 5 dispatches) — deliberately, not
   * for style: an earlier single-pass version let `depositScratch`'s
   * atomic writes leak across steps (agentStep's atomicAdd sometimes
   * running before clearScratch's atomicStore had taken visible effect,
   * on this browser/driver), corrupting roughly half the population
   * with unbounded deposit growth within the first few steps — a
   * cross-dispatch synchronization gap that doesn't exist across pass
   * boundaries. Pass creation itself is cheap; correctness comes first. */
  step(): void {
    if (!this.environment || !this.agents) return;
    const env = this.environment;
    const agents = this.agents;
    const parity = this.parity;

    const encoder = this.device.createCommandEncoder();

    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(env.clearScratchPipeline);
    clearPass.setBindGroup(0, env.clearScratchBindGroup);
    clearPass.dispatchWorkgroups(env.clearScratchGroups);
    clearPass.end();

    const gradientPass = encoder.beginComputePass();
    gradientPass.setPipeline(env.computeGradientPipeline);
    gradientPass.setBindGroup(0, env.computeGradientBindGroups[parity]);
    gradientPass.dispatchWorkgroups(env.grid2DGroups[0], env.grid2DGroups[1]);
    gradientPass.end();

    const agentPass = encoder.beginComputePass();
    agentPass.setPipeline(agents.pipeline);
    agentPass.setBindGroup(0, agents.bindGroups[parity]);
    agentPass.dispatchWorkgroups(agents.dispatchGroups);
    agentPass.end();

    const mergePass = encoder.beginComputePass();
    mergePass.setPipeline(env.mergeDepositPipeline);
    mergePass.setBindGroup(0, env.mergeDepositBindGroups[parity]);
    // mergeDeposit dispatches over the same PLANE_SIZE elements as
    // clearScratch, same workgroup size — reuses its group count.
    mergePass.dispatchWorkgroups(env.clearScratchGroups);
    mergePass.end();

    const diffusePass = encoder.beginComputePass();
    diffusePass.setPipeline(env.diffuseDecayPipeline);
    diffusePass.setBindGroup(0, env.diffuseDecayBindGroups[parity]);
    diffusePass.dispatchWorkgroups(env.grid2DGroups[0], env.grid2DGroups[1]);
    diffusePass.end();

    this.device.queue.submit([encoder.finish()]);

    // diffuseDecay just wrote the fresh state into the OTHER buffer —
    // that's "current" for next step's sensing and for rendering.
    this.parity = 1 - parity;
    this.stepIndex += 1;
  }

  render(context: GPUCanvasContext, canvasWidth: number, canvasHeight: number): void {
    if (!this.renderer || !this.agents) return;
    this.renderer.render(context, this.parity, this.agents.agentCount, canvasWidth, canvasHeight);
  }

  destroy(): void {
    this.environment?.destroy();
    this.agents?.destroy();
    this.renderer?.destroy();
  }
}

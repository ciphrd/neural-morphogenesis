import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { acquireGpuDevice, watchDeviceLoss } from "../gpu/device";
import { DEFAULT_INTENSITY } from "../gpu/render";
import type { SpawnDistribution } from "../gpu/rng";
import { GpuSimulation } from "../gpu/simulation";
import type { BackgroundMode, PhysicsSettings, SimulationConfig } from "../gpu/types";

interface GridCanvasProps {
  config: SimulationConfig | null;
  targetPoints: readonly (readonly [number, number])[] | null;
  backgroundMode: BackgroundMode;
  // Live decay/maxSpeed/maxAccel/maxStrafe for the "Physics" panel's
  // sliders — the caller (TrainingView) always resolves this to a
  // concrete value once a config is loaded (either config's own trained
  // values, or the user's in-progress override); null only means nothing
  // has loaded yet. Applied via a plain uniform-buffer write
  // (GpuSimulation.setPhysics()), never a rebuild, so dragging a slider
  // never disturbs the rollout in flight.
  physics: PhysicsSettings | null;
  // Contrast multiplier for the substrate colorize pass — see
  // gpu/render.ts's setIntensity() for what it actually does (shrinks
  // the EMA-tracked scale before upload, so pixels saturate to full
  // white/black sooner). Default matches GpuRender's own default.
  intensity?: number;
  // Viewer-only initial-spread shape (see gpu/rng.ts's SpawnDistribution
  // docstring) — doesn't affect training, just how a replayed rollout's
  // agents are scattered at step 0. Default matches training's own jitter.
  spawnDistribution?: SpawnDistribution;
  onStep?: (step: number) => void;
  // Default (true): restart with a fresh seed once currentStep reaches
  // config.steps — the rollout was only ever *trained* for that many
  // steps, so this is what keeps a long-idle viewer cycling through
  // varied looks rather than freezing on the last frame. false: keep
  // stepping straight past that count instead — nothing in the sim
  // itself enforces steps as a hard limit (it's just how long training
  // scored a candidate for), so this is purely "let me see what happens
  // if it kept going," e.g. to check whether a trained shape holds up or
  // degrades past its trained horizon.
  loopAtTrainedSteps?: boolean;
  // Freezes the simulation in place — sim.step() (and the loop-at-limit
  // check, which depends on stepping having happened) are skipped while
  // true, but rendering keeps running every frame regardless, so other
  // live controls (background mode, intensity) still take effect
  // immediately on a paused frame instead of only on the next step.
  paused?: boolean;
}

export interface GridCanvasHandle {
  /** Reloads the currently-active generation from step 0 — since
   * loadGeneration() always re-seeds with `config.seed` (the *original*
   * seed, not loopCurrentGeneration()'s derived-per-loop one), this is a
   * literal, deterministic replay of the exact same rollout from the
   * start, not a fresh random look. */
  restart(): void;
}

type Status = "loading" | "ready" | "unsupported" | "lost";

/** Owns the `<canvas>`, the WebGPU device/context, and the
 * requestAnimationFrame loop that drives GpuSimulation — the WebGPU
 * analogue of trainer/frontend's GraphRenderer.tsx, but presenting via a
 * WebGPU render pipeline instead of imperative canvas-2D draw calls (see
 * this project's design notes for why: the simulation itself is a
 * GPU-resident compute pipeline, and this component's canvas is just
 * where its render() pass's output lands). One sim step per rendered
 * frame — the faithful default, no ticksPerFrame control, since there's
 * no realtime/batch distinction here (unlike trainer/frontend's
 * Training tab). */
export const GridCanvas = forwardRef<GridCanvasHandle, GridCanvasProps>(function GridCanvas(
  {
    config,
    targetPoints,
    backgroundMode,
    physics,
    intensity = DEFAULT_INTENSITY,
    spawnDistribution = "default",
    onStep,
    loopAtTrainedSteps = true,
    paused = false,
  },
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simulationRef = useRef<GpuSimulation | null>(null);
  const contextRef = useRef<GPUCanvasContext | null>(null);
  const configRef = useRef<SimulationConfig | null>(null);
  const physicsRef = useRef(physics);
  const backgroundModeRef = useRef(backgroundMode);
  const intensityRef = useRef(intensity);
  const spawnDistributionRef = useRef(spawnDistribution);
  const onStepRef = useRef(onStep);
  const loopAtTrainedStepsRef = useRef(loopAtTrainedSteps);
  const pausedRef = useRef(paused);
  const [status, setStatus] = useState<Status>("loading");
  const [statusMessage, setStatusMessage] = useState<string>("");

  configRef.current = config;
  physicsRef.current = physics;
  backgroundModeRef.current = backgroundMode;
  intensityRef.current = intensity;
  spawnDistributionRef.current = spawnDistribution;
  onStepRef.current = onStep;
  loopAtTrainedStepsRef.current = loopAtTrainedSteps;
  pausedRef.current = paused;

  useImperativeHandle(ref, () => ({
    restart: () => {
      if (configRef.current) simulationRef.current?.loadGeneration(configRef.current);
    },
  }));

  // Acquire device + configure the canvas context once. StrictMode-safe:
  // if this effect is torn down (dev double-invoke, or a real unmount)
  // before or after the async acquisition resolves, whatever device got
  // acquired is released rather than leaked.
  useEffect(() => {
    let cancelled = false;
    let acquiredDevice: GPUDevice | null = null;

    (async () => {
      const result = await acquireGpuDevice();
      if (cancelled) {
        if (result.ok) result.device.destroy();
        return;
      }
      if (!result.ok) {
        setStatus("unsupported");
        setStatusMessage(result.reason);
        return;
      }
      acquiredDevice = result.device;
      const device = result.device;
      watchDeviceLoss(device, (message) => {
        setStatus("lost");
        setStatusMessage(message);
      });

      const canvas = canvasRef.current;
      const context = canvas?.getContext("webgpu");
      if (!canvas || !context) {
        setStatus("unsupported");
        setStatusMessage("canvas.getContext(\"webgpu\") returned null.");
        return;
      }
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });
      contextRef.current = context;

      const simulation = new GpuSimulation(device, format);
      if (targetPoints) simulation.setTargetPoints(targetPoints);
      simulation.setBackgroundMode(backgroundModeRef.current);
      simulation.setIntensity(intensityRef.current);
      simulation.setSpawnDistribution(spawnDistributionRef.current);
      if (configRef.current) simulation.loadGeneration(configRef.current);
      if (physicsRef.current) simulation.setPhysics(physicsRef.current);
      simulationRef.current = simulation;

      setStatus("ready");
    })();

    return () => {
      cancelled = true;
      simulationRef.current?.destroy();
      simulationRef.current = null;
      acquiredDevice?.destroy();
    };
    // Device/context acquisition happens exactly once — target points
    // and config are forwarded via the effects below, not by re-running
    // this one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resize the canvas's backing store to match its container, in device
  // pixels — presentation is NDC-based (see present.wgsl), so it adapts
  // to any canvas size with no knowledge of the simulation's own grid
  // resolution.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(entry.contentRect.width * dpr));
      const height = Math.max(1, Math.round(entry.contentRect.height * dpr));
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (targetPoints) simulationRef.current?.setTargetPoints(targetPoints);
  }, [targetPoints]);

  useEffect(() => {
    if (config) simulationRef.current?.loadGeneration(config);
  }, [config]);

  useEffect(() => {
    if (physics) simulationRef.current?.setPhysics(physics);
  }, [physics]);

  useEffect(() => {
    simulationRef.current?.setBackgroundMode(backgroundMode);
  }, [backgroundMode]);

  useEffect(() => {
    simulationRef.current?.setIntensity(intensity);
  }, [intensity]);

  useEffect(() => {
    simulationRef.current?.setSpawnDistribution(spawnDistribution);
  }, [spawnDistribution]);

  useEffect(() => {
    if (status !== "ready") return;
    let raf = 0;
    let cancelled = false;

    const frame = () => {
      const sim = simulationRef.current;
      const context = contextRef.current;
      const activeConfig = configRef.current;
      const canvas = canvasRef.current;
      if (sim?.ready && context && activeConfig && canvas) {
        if (!pausedRef.current) {
          sim.step();
          if (loopAtTrainedStepsRef.current && sim.currentStep >= sim.steps) {
            sim.loopCurrentGeneration(activeConfig);
          }
        }
        sim.render(context, canvas.width, canvas.height);
        onStepRef.current?.(sim.currentStep);
      }
      if (!cancelled) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [status]);

  return (
    <div className="grid-canvas-container">
      <canvas ref={canvasRef} className="gpu-canvas" />
      {status === "loading" && <div className="webgpu-banner hint">Acquiring WebGPU device…</div>}
      {status === "unsupported" && (
        <div className="webgpu-banner">
          <p>WebGPU isn't available in this browser.</p>
          <p className="hint">{statusMessage}</p>
          <p className="hint">Try Chrome or another WebGPU-capable browser.</p>
        </div>
      )}
      {status === "lost" && (
        <div className="webgpu-banner">
          <p>WebGPU device lost.</p>
          <p className="hint">{statusMessage}</p>
        </div>
      )}
    </div>
  );
});

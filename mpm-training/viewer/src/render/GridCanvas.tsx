import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { acquireGpuDevice, watchDeviceLoss, watchUncapturedErrors } from "../gpu/device";
import type { NetworkProbe } from "../gpu/nnProbe";
import type { FieldMode, ParticleShape } from "../gpu/render";
import { GpuSimulation } from "../gpu/simulation";
import type { PhysicsSettings, SimulationConfig } from "../gpu/types";
import { CanvasRecorder } from "./canvasRecorder";

// How often the Network panel's own probe (gpu/nnProbe.ts) re-samples the
// live forward pass — its own independent cadence, deliberately NOT once
// per macro step (that's what the RAF loop below runs at): a diagnostic
// readback has no business adding a second async GPU round-trip to
// step()'s own already-async path, and 8/sec is plenty for a human eye to
// read a "brain" visualization by, even at a much higher true step rate.
const PROBE_INTERVAL_MS = 125;

/** "none": no click/drag interaction (the default, passive-replay mode).
 * "add": click adds one particle at the clicked domain position (see
 * gpu/mpmCore.ts's own addParticleAt()). "move": press-drag picks up
 * whichever particle is nearest the pointer (within gpu/interact.wgsl's
 * own MAX_DIST) and pins it to the pointer until released (see
 * gpu/interact.ts). */
export type Tool = "none" | "add" | "move";

interface GridCanvasProps {
  config: SimulationConfig | null;
  /** Flat [x0,y0,x1,y1,...] in MpmCore's own [0,1]^2 domain. */
  targetPoints: Float32Array | null;
  // Live gravity/decay/maxAccel/maxStrafe/maxEnvWrite for the Physics
  // panel's sliders — the caller (TrainingView) always resolves this to
  // a concrete value once a config is loaded (either the config's own
  // trained values, or the user's in-progress override); null only means
  // nothing has loaded yet. Applied via a plain uniform-buffer write
  // (GpuSimulation.setPhysics()), never a rebuild.
  physics: PhysicsSettings | null;
  // View-only rendering options (gpu/render.ts) — none of these are
  // simulation state, so they're plain display props, not part of
  // PhysicsSettings/SimulationConfig.
  fieldMode?: FieldMode;
  particleShape?: ParticleShape;
  particleRadiusPx?: number;
  /** [0,2] — see gpu/render.ts's own setAccent()/field.wgsl's own accent
   * uniform comment. Default 0 (identity, every background mode renders
   * exactly as it did before this knob existed). */
  accent?: number;
  /** Which interaction tool (if any) is currently toggled on — see the
   * Tool type's own docstring. Default "none". */
  tool?: Tool;
  onStep?: (step: number) => void;
  // Default (true): restart with a fresh rollout (same seed) once
  // currentStep reaches config.macroSteps — the rollout was only ever
  // *trained* for that many macro steps, so this keeps a long-idle
  // viewer cycling rather than freezing on the last frame. false: keep
  // stepping straight past that count — nothing in the sim itself
  // enforces macroSteps as a hard limit, so this is purely "let me see
  // what happens if it kept going."
  loopAtTrainedSteps?: boolean;
  // Freezes the simulation in place — step() (and the loop-at-limit
  // check) are skipped while true, but rendering keeps running every
  // frame regardless.
  paused?: boolean;
  /** Fired roughly every PROBE_INTERVAL_MS with a fresh forward-pass
   * snapshot (see gpu/nnProbe.ts) for the Network panel's own brain
   * visualization — never fired with null (a probe that returned null
   * because a previous one was still resolving just skips that tick
   * rather than overwriting whatever's already displayed). */
  onProbe?: (probe: NetworkProbe) => void;
}

export interface GridCanvasHandle {
  /** Restarts the currently-active generation from macro step 0 — since
   * loadGeneration()/restartRollout() always re-seed with config.seed
   * (the winning rollout's own seed), this is a literal, deterministic
   * replay of the exact same rollout from the start. */
  restart(): void;
  /** Overwrites the update rule's own weights/biases with a fresh random
   * init and restarts the rollout under it — see
   * GpuSimulation.randomizeWeights()'s own docstring. A later "Restart"
   * click, or a new generation loading, both leave the randomized
   * weights in place until the next config change actually reloads
   * config.weights (loadGeneration() always does — see its own effect
   * below), same as any other in-place override this component exposes
   * (e.g. the Physics panel's own overrides). */
  randomizeWeights(): void;
  /** Starts capturing the canvas's own rendered output — see
   * canvasRecorder.ts's own CanvasRecorder.start() docstring. Throws if
   * this browser has no MediaRecorder at all — TrainingView checks
   * canvasRecorder.ts's own pickRecordingFormat() directly (a pure
   * browser feature-check, no canvas/mount needed) before ever calling
   * this, to disable the Record button instead. */
  startRecording(): void;
  /** Stops recording and triggers a browser download of the captured
   * video — see CanvasRecorder.stop()'s own docstring for exactly how.
   * Resolves once the download has been handed off. */
  stopRecording(): Promise<void>;
}

type Status = "loading" | "ready" | "unsupported" | "lost";

/** Sizes `canvas` to a SQUARE that fits inside its own parent element
 * (`.grid-canvas-container`, see this file's own JSX) — the simulation's
 * domain is always [0,1]^2 (MpmCore), so a non-square canvas would
 * visibly stretch it, the same reason mls-mpm's own resizeCanvas() (a
 * sibling project sharing this GPU pipeline's origins) keeps its canvas
 * square rather than just filling whatever rectangle its container
 * happens to be. Reads the PARENT's box (not the canvas's own — once
 * this has run once, the canvas's own box is the square, smaller answer
 * from last time, not "how much space is actually available now").
 * Sets both the CSS box (style.width/height, in CSS pixels) and the
 * backing store (canvas.width/height, in device pixels) — the container
 * itself centers that square via flex (see style.css's own
 * .grid-canvas-container). */
function applySquareSize(canvas: HTMLCanvasElement): void {
  const container = canvas.parentElement;
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const cssSize = Math.max(1, Math.min(rect.width, rect.height));
  canvas.style.width = `${cssSize}px`;
  canvas.style.height = `${cssSize}px`;
  const dpr = window.devicePixelRatio || 1;
  const pixelSize = Math.max(1, Math.round(cssSize * dpr));
  canvas.width = pixelSize;
  canvas.height = pixelSize;
}

/** Owns the <canvas>, the WebGPU device/context, and the
 * requestAnimationFrame loop that drives GpuSimulation. One macro step
 * per rendered frame — the faithful default, no batching control, since
 * unlike a fixed-timestep physics demo there's no realtime/offline
 * distinction to reconcile here. */
export const GridCanvas = forwardRef<GridCanvasHandle, GridCanvasProps>(function GridCanvas(
  { config, targetPoints, physics, fieldMode = "none", particleShape = "circle", particleRadiusPx, accent = 0, tool = "none", onStep, loopAtTrainedSteps = true, paused = false, onProbe },
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simulationRef = useRef<GpuSimulation | null>(null);
  const contextRef = useRef<GPUCanvasContext | null>(null);
  const configRef = useRef<SimulationConfig | null>(null);
  const physicsRef = useRef(physics);
  const fieldModeRef = useRef(fieldMode);
  const particleShapeRef = useRef(particleShape);
  const particleRadiusPxRef = useRef(particleRadiusPx);
  const accentRef = useRef(accent);
  const toolRef = useRef(tool);
  const onStepRef = useRef(onStep);
  const loopAtTrainedStepsRef = useRef(loopAtTrainedSteps);
  const pausedRef = useRef(paused);
  const onProbeRef = useRef(onProbe);
  // "Move" tool's own live drag state — a ref, not React state, since it
  // updates on every pointermove and is read every animation frame (see
  // the RAF loop below); re-rendering the component for either would be
  // pure waste.
  const draggingRef = useRef(false);
  const dragPosRef = useRef({ x: 0, y: 0 });
  // Constructed once, lazily, the first time anything actually needs it
  // (not on every render) — see canvasRecorder.ts's own CanvasRecorder
  // docstring for why one instance is reused across repeated
  // start()/stop() cycles rather than recreated each time.
  const recorderRef = useRef<CanvasRecorder | null>(null);
  function getRecorder(): CanvasRecorder {
    if (!recorderRef.current) recorderRef.current = new CanvasRecorder();
    return recorderRef.current;
  }
  const [status, setStatus] = useState<Status>("loading");
  const [statusMessage, setStatusMessage] = useState<string>("");

  configRef.current = config;
  physicsRef.current = physics;
  fieldModeRef.current = fieldMode;
  particleShapeRef.current = particleShape;
  particleRadiusPxRef.current = particleRadiusPx;
  accentRef.current = accent;
  toolRef.current = tool;
  onStepRef.current = onStep;
  loopAtTrainedStepsRef.current = loopAtTrainedSteps;
  pausedRef.current = paused;
  onProbeRef.current = onProbe;

  useImperativeHandle(ref, () => ({
    restart: () => {
      simulationRef.current?.restartRollout();
    },
    randomizeWeights: () => {
      simulationRef.current?.randomizeWeights();
    },
    startRecording: () => {
      if (canvasRef.current) getRecorder().start(canvasRef.current);
    },
    stopRecording: () => getRecorder().stop("mpm-training"),
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
      watchUncapturedErrors(device);

      const canvas = canvasRef.current;
      const context = canvas?.getContext("webgpu");
      if (!canvas || !context) {
        setStatus("unsupported");
        setStatusMessage('canvas.getContext("webgpu") returned null.');
        return;
      }
      // Size the backing store synchronously, from the container's own
      // current layout box, before configure() — otherwise it still
      // carries the HTML default (300x150) at configure time, and only
      // gets corrected once the ResizeObserver effect's first (async,
      // racy) callback fires. Presenting through a swap chain sized for
      // a moment-later-resized canvas is exactly the kind of race that
      // produces "invalid texture" cascades on some backends. The
      // ResizeObserver effect still owns every *subsequent* resize.
      applySquareSize(canvas);

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });
      contextRef.current = context;

      const simulation = new GpuSimulation(device, format);
      simulation.setCanvasSizePx(canvas.width, canvas.height);
      if (targetPoints) simulation.setTargetPoints(targetPoints);
      simulation.setFieldMode(fieldModeRef.current);
      simulation.setParticleShape(particleShapeRef.current);
      if (particleRadiusPxRef.current !== undefined) simulation.setPointRadiusPx(particleRadiusPxRef.current);
      simulation.setAccent(accentRef.current);
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
    // Device/context acquisition happens exactly once — target points/
    // config/physics are forwarded via the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resize the canvas's backing store to match its container, in device
  // pixels — point radius is a fixed device-pixel size converted to NDC
  // (see gpu/render.ts's setCanvasSizePx()), so the renderer needs to
  // know about a resize too, not just the <canvas> element itself.
  // Observes the CONTAINER (canvas.parentElement), not the canvas
  // itself: once applySquareSize() gives the canvas its own explicit
  // (square, generally smaller-than-container) box, observing the
  // canvas would only ever fire in response to a resize this same
  // callback just caused, never in response to the container growing on
  // its own — see applySquareSize's own comment for why the container's
  // box, not the canvas's, is "how much space is actually available."
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = canvas?.parentElement;
    if (!canvas || !container) return;
    const observer = new ResizeObserver(() => {
      applySquareSize(canvas);
      simulationRef.current?.setCanvasSizePx(canvas.width, canvas.height);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // "Add"/"Move" tools (see the Tool type's own docstring) — attached
  // once, like the ResizeObserver effect above; which tool is active,
  // and all the live drag state it reads/writes, come from refs so this
  // effect never needs to re-run. Uses Pointer Events (not mouse events)
  // specifically for setPointerCapture()/releasePointerCapture() below —
  // captured events keep arriving at this element even once the cursor
  // strays outside the canvas mid-drag, which plain mouse events don't
  // guarantee.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Screen pixel -> MpmCore's own [0,1]^2 domain coords. Y is flipped
    // (screen is top-down, the domain is bottom-up — gravity pulls
    // toward y=0, same convention render.wgsl's own particleVertex uses,
    // see that file's own comment) and both axes are clamped so a drag
    // that strays outside the canvas (legitimate, once captured) still
    // resolves to a position on the domain's own edge rather than
    // outside [0,1]^2.
    const domainPos = (ev: PointerEvent): { x: number; y: number } => {
      const rect = canvas.getBoundingClientRect();
      const nx = (ev.clientX - rect.left) / rect.width;
      const ny = (ev.clientY - rect.top) / rect.height;
      return { x: Math.min(1, Math.max(0, nx)), y: Math.min(1, Math.max(0, 1 - ny)) };
    };

    const handlePointerDown = (ev: PointerEvent) => {
      if (ev.button !== 0) return; // left/primary button only
      const sim = simulationRef.current;
      if (!sim) return;
      const pos = domainPos(ev);
      if (toolRef.current === "add") {
        sim.addParticleAt(pos.x, pos.y);
      } else if (toolRef.current === "move") {
        canvas.setPointerCapture(ev.pointerId);
        sim.beginDrag(pos.x, pos.y);
        dragPosRef.current = pos;
        draggingRef.current = true;
      }
    };

    const handlePointerMove = (ev: PointerEvent) => {
      if (!draggingRef.current) return;
      dragPosRef.current = domainPos(ev);
    };

    const handlePointerUp = (ev: PointerEvent) => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      simulationRef.current?.endDrag();
      if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
    };

    canvas.addEventListener("pointerdown", handlePointerDown);
    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("pointerup", handlePointerUp);
    canvas.addEventListener("pointercancel", handlePointerUp);
    return () => {
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("pointercancel", handlePointerUp);
    };
  }, []);

  // Cleanly ends an in-progress drag if the "Move" tool is switched away
  // from mid-gesture (e.g. the user clicks a different tool button while
  // still holding the mouse down) — without this, draggingRef would stay
  // stuck true and keep re-pinning a particle no pointerup ever arrives
  // for.
  useEffect(() => {
    if (tool !== "move" && draggingRef.current) {
      draggingRef.current = false;
      simulationRef.current?.endDrag();
    }
  }, [tool]);

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
    simulationRef.current?.setFieldMode(fieldMode);
  }, [fieldMode]);

  useEffect(() => {
    simulationRef.current?.setParticleShape(particleShape);
  }, [particleShape]);

  useEffect(() => {
    if (particleRadiusPx !== undefined) simulationRef.current?.setPointRadiusPx(particleRadiusPx);
  }, [particleRadiusPx]);

  useEffect(() => {
    simulationRef.current?.setAccent(accent);
  }, [accent]);

  // Stops any in-progress recording on unmount — a dangling
  // MediaRecorder/MediaStream left running past this component's own
  // lifetime would keep the canvas's own capture track alive for no
  // reason (and orphan the "stop" download this never gets to trigger).
  // Mount-only: recorderRef.current is only ever created lazily by
  // getRecorder() above, not on every render.
  useEffect(() => {
    return () => {
      if (recorderRef.current?.isRecording) recorderRef.current.stop("mpm-training");
    };
  }, []);

  useEffect(() => {
    if (status !== "ready") return;
    let raf = 0;
    let cancelled = false;

    // Async now — GpuSimulation.step() awaits a growth-count readback
    // every macro step (see that method's own docstring for why: WebGPU's
    // own buffer readback has no synchronous equivalent). requestAnimationFrame
    // itself doesn't care whether its callback is async (it just ignores
    // the returned promise), but awaiting inside means a frame's own
    // render() now happens after step()'s own GPU round-trip resolves,
    // not synchronously within the same tick — the real, accepted cost
    // of growth existing at all (see gpu/simulation.ts's own module
    // docstring). The try/catch below exists ONLY for the teardown race
    // this creates: if this component unmounts WHILE a frame() call is
    // suspended awaiting step(), sim.destroy() (a DIFFERENT effect's own
    // cleanup, not this one — see this component's own device-acquisition
    // effect) can destroy the very buffers readGrownCount()'s own
    // mapAsync() is waiting on, which WebGPU rejects rather than silently
    // ignores; `cancelled` (this effect's own flag) tells the two apart
    // from a real bug, which still surfaces via console.error rather than
    // vanishing silently.
    const frame = async () => {
      const sim = simulationRef.current;
      const context = contextRef.current;
      if (sim?.ready && context) {
        if (!pausedRef.current) {
          try {
            await sim.step();
          } catch (err) {
            if (!cancelled) console.error(err);
            return;
          }
          if (cancelled) return;
          if (loopAtTrainedStepsRef.current && sim.currentStep >= sim.steps) {
            sim.restartRollout();
          }
        }
        // Re-pins the dragged particle every frame, not just on
        // pointermove — see Interact.dragTo()'s own docstring for why —
        // and deliberately OUTSIDE the `!pausedRef.current` branch above:
        // a drag should still work while play is paused, same as any
        // other direct-manipulation tool would.
        if (draggingRef.current) {
          sim.dragTo(dragPosRef.current.x, dragPosRef.current.y);
        }
        sim.render(context);
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

  // Network panel's own probe (gpu/nnProbe.ts) — a separate, slower-
  // cadence timer, not tied to the RAF loop above, since it's a
  // diagnostic readback with its own async GPU round-trip (see
  // PROBE_INTERVAL_MS's own comment). Runs regardless of `paused` — the
  // sensed input only actually changes once step() runs again, but
  // re-probing the same, unchanged state while paused is harmless and
  // keeps the panel from just freezing on stale "still computing" state
  // the instant Play is toggled off.
  useEffect(() => {
    if (status !== "ready" || !onProbeRef.current) return;
    let cancelled = false;
    const interval = setInterval(() => {
      simulationRef.current?.probeNetwork().then((probe) => {
        if (!cancelled && probe) onProbeRef.current?.(probe);
      });
    }, PROBE_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [status]);

  return (
    <div className="grid-canvas-container">
      <canvas ref={canvasRef} className={`gpu-canvas${tool !== "none" ? ` gpu-canvas-tool-${tool}` : ""}`} />
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

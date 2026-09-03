import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { DeformDirection, DeformMode } from "../gpu/deform";
import type { BloomSettings } from "../gpu/bloom";
import { acquireGpuDevice, watchDeviceLoss, watchUncapturedErrors } from "../gpu/device";
import type { FieldMode, ParticleColorMode, ParticleShape } from "../gpu/render";
import { GpuSimulation, type SimulationScenario } from "../gpu/simulation";
import { policyWeightsShapeError } from "../gpu/policyEval";
import type { PhysicsSettings, SimulationConfig, UpdateRuleWeights } from "../gpu/types";
import { CanvasRecorder } from "./canvasRecorder";

/** "none": no click/drag interaction (the default, passive-replay mode).
 * "add": click adds one particle at the clicked domain position (see
 * gpu/mpmCore.ts's own addParticleAt()). "move": press-drag picks up
 * whichever particle is nearest the pointer (within gpu/interact.wgsl's
 * own MAX_DIST) and pins it to the pointer until released (see
 * gpu/interact.ts). "deform": hovering previews deformSettings.radius as
 * a circle around the pointer; press-hold injects a radial push/pull
 * centered there, once per rendered frame for as long as it's held (see
 * gpu/deform.ts — each particle's own push direction is computed
 * straight toward/away from the pointer, not a single uniform vector). */
export type Tool = "none" | "add" | "move" | "deform";

/** "Deform" tool's own live settings — TrainingView owns these (its own
 * small panel's inward/outward toggle + strength slider + radius slider
 * + mode checkbox), GridCanvas just reads them at click time and draws
 * the hover preview. `radius` is MpmCore's own [0,1]^2 domain units,
 * same as the preview circle drawn around the pointer. `strength` is a
 * raw scalar — see deform.wgsl's own VELOCITY_SCALE/DEFORMATION_SCALE
 * comment for the per-mode magnitude this actually gets scaled by. */
export interface DeformSettings {
  direction: DeformDirection;
  strength: number;
  radius: number;
  mode: DeformMode;
}

export interface AutoZoomSettings {
  enabled: boolean;
  sampleEveryFrames: number;
  maxSamples: number;
  fitFraction: number;
  padding: number;
  smoothing: number;
}

interface GridCanvasProps {
  config: SimulationConfig | null;
  /** Optional deterministic lab setup; null preserves training playback. */
  scenario?: SimulationScenario | null;
  /** Flat [x0,y0,x1,y1,...] in MpmCore's own [0,1]^2 domain. */
  targetPoints: Float32Array | null;
  /** Rendering-only visibility of the training-target overlay. */
  targetVisible?: boolean;
  // Live gravity/decay/maxAccel/maxStrafe/maxEnvWrite for the Physics
  // panel's sliders — the caller (TrainingView) always resolves this to
  // a concrete value once a config is loaded (either the config's own
  // trained values, or the user's in-progress override); null only means
  // nothing has loaded yet. Applied via a plain uniform-buffer write
  // (GpuSimulation.setPhysics()), never a rebuild.
  physics: PhysicsSettings | null;
  /** Playback-only growth/interaction cap; does not alter training. */
  particleCap?: number;
  /** Playback-only number of genuinely seeded agents. */
  initialParticleCount?: number;
  // View-only rendering options (gpu/render.ts) — none of these are
  // simulation state, so they're plain display props, not part of
  // PhysicsSettings/SimulationConfig.
  fieldMode?: FieldMode;
  /** First of three contiguous chemical channels mapped to substrate RGB. */
  substrateChannelStart?: number;
  substrateZeroIsBlack?: boolean;
  boundaryGradientZeroIsBlack?: boolean;
  particleShape?: ParticleShape;
  particleColorMode?: ParticleColorMode;
  particleAlpha?: number;
  directionalLineVisible?: boolean;
  particleRadiusPx?: number;
  /** Visualization-only multiplier for the signed mitosis drive. */
  mitosisSignalBoost?: number;
  /** Boundary diagnostic half-activation gradient g0. */
  boundaryGradientScale?: number;
  /** First of three contiguous private-state channels mapped to cell RGB. */
  internalStateChannelStart?: number;
  /** Amount of the wrapped next three private-state channels subtracted from particle RGB. */
  chemicalMemoryOpponentSubtraction?: number;
  /** [-2,2] — negative suppresses background-field contrast, 0 is
   * identity, positive accentuates faint values. */
  accent?: number;
  morphologyGradientVisible?: boolean;
  morphologyDensityVisible?: boolean;
  /** [0,2] — see gpu/render.ts's own setBlur()/field.wgsl's own
   * blurDensity() comment. Only the "gradient" background mode's own
   * blur pass reads this. Default 0 (no blur, unchanged from before this
   * knob existed). */
  blur?: number;
  /** [~0.25,4] — see gpu/render.ts's own setGradientExponent()/field.wgsl's
   * own colorizeGradient() comment. Only the "gradient" background mode's
   * own colorize pass reads this. Default 1 (identity, unchanged from
   * before this knob existed). */
  gradientExponent?: number;
  bloom?: BloomSettings;
  /** Geometry-stage camera zoom; particle rendering remains native-resolution. */
  zoom?: number;
  /** Periodically samples live agents and fits their bounds into the center. */
  autoZoom?: AutoZoomSettings;
  /** Reports the actual camera zoom while auto zoom is smoothing. */
  onEffectiveZoomChange?: (zoom: number) => void;
  /** Which interaction tool (if any) is currently toggled on — see the
   * Tool type's own docstring. Default "none". */
  tool?: Tool;
  /** "Deform" tool's own live settings — required whenever `tool ===
   * "deform"`, ignored otherwise (TrainingView still always passes
   * something, its own panel state, so this stays non-optional here —
   * simpler than threading an extra null-check through every reader
   * below for a case that can't actually arise in practice). */
  deformSettings: DeformSettings;
  /** Fired once per rendered frame with the rollout's own current macro
   * step AND its live particle count (which grows as growth splits —
   * see GpuSimulation's own particleCount getter). Both ride the same
   * callback rather than getting their own, since they're read from the
   * same sim at the same instant and always displayed together. */
  onStep?: (step: number, particleCount: number) => void;
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
  /** Returns the exact randomized weights loaded into the GPU so callers can
   * display the same policy instead of the selected generation's weights. */
  randomizeWeights(): UpdateRuleWeights | null;
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
  /** Runs deterministic rollouts under the already-loaded policy, capturing
   * the rendered final frame of each settings combination as a PNG. */
  collectSamples(
    samples: Array<{
      config: SimulationConfig;
      physics: PhysicsSettings;
      particleCap: number;
      initialParticleCount: number;
      particleDensityMultiplier: number;
      particleRadiusPx: number;
      filename: string;
    }>,
    steps: number,
    restorePhysics: PhysicsSettings,
    restoreParticleCap: number,
    restoreInitialParticleCount: number,
    restoreParticleRadiusPx: number,
    includeJson: boolean,
    onProgress: (completed: number) => void,
    signal: AbortSignal,
  ): Promise<Array<{ filename: string; blob: Blob }>>;
}

type Status = "loading" | "ready" | "unsupported" | "lost" | "incompatible";

function toroidalSpan(values: number[]): number {
  if (values.length <= 1) return 0;
  const sorted = values.map((v) => ((v % 1) + 1) % 1).sort((a, b) => a - b);
  let largestGap = sorted[0] + 1 - sorted[sorted.length - 1];
  for (let i = 1; i < sorted.length; i++) largestGap = Math.max(largestGap, sorted[i] - sorted[i - 1]);
  return 1 - largestGap;
}

function spatialMetrics(flatPositions: Float32Array) {
  const count = flatPositions.length / 2;
  if (count === 0) return { envelopeWidth: 0, envelopeHeight: 0, envelopeArea: 0, rmsRadius: 0, radiusP95: 0 };
  const xs: number[] = [];
  const ys: number[] = [];
  let cosX = 0, sinX = 0, cosY = 0, sinY = 0;
  for (let i = 0; i < count; i++) {
    const x = ((flatPositions[i * 2] % 1) + 1) % 1;
    const y = ((flatPositions[i * 2 + 1] % 1) + 1) % 1;
    xs.push(x); ys.push(y);
    cosX += Math.cos(2 * Math.PI * x); sinX += Math.sin(2 * Math.PI * x);
    cosY += Math.cos(2 * Math.PI * y); sinY += Math.sin(2 * Math.PI * y);
  }
  const centerX = ((Math.atan2(sinX, cosX) / (2 * Math.PI)) + 1) % 1;
  const centerY = ((Math.atan2(sinY, cosY) / (2 * Math.PI)) + 1) % 1;
  const radii = xs.map((x, i) => {
    const dx = ((x - centerX + 0.5) % 1 + 1) % 1 - 0.5;
    const dy = ((ys[i] - centerY + 0.5) % 1 + 1) % 1 - 0.5;
    return Math.hypot(dx, dy);
  }).sort((a, b) => a - b);
  const width = toroidalSpan(xs);
  const height = toroidalSpan(ys);
  return {
    envelopeWidth: width,
    envelopeHeight: height,
    envelopeArea: width * height,
    rmsRadius: Math.sqrt(radii.reduce((sum, r) => sum + r * r, 0) / count),
    radiusP95: radii[Math.min(radii.length - 1, Math.floor(0.95 * radii.length))],
  };
}

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
  // Floored to a WHOLE CSS pixel, not left fractional (e.g. 756.55px) —
  // otherwise the backing store below (necessarily an integer device-pixel
  // count) can't match the CSS box exactly, forcing the browser to scale
  // the bitmap down by a sub-pixel amount when painting it. That mismatch
  // is what produced the "1px gray border" seam on the right/bottom edge:
  // not a CSS border anywhere, but sub-pixel scaling blending the canvas's
  // own edge pixels against whatever's behind it. Flooring (not rounding)
  // keeps the square from ever exceeding the space actually available.
  const cssSize = Math.floor(Math.max(1, Math.min(rect.width, rect.height)));
  canvas.style.width = `${cssSize}px`;
  canvas.style.height = `${cssSize}px`;
  // Centered manually, in WHOLE CSS pixels, rather than via
  // .grid-canvas-container's own flexbox align-items/justify-content:
  // center — flex centering computes leftover space from the
  // container's own box, which is routinely fractional itself (e.g.
  // 806.578125px tall, from 100vh math), so even with cssSize now a
  // whole number, flex centering still lands the canvas at a
  // fractional left/top (observed: top: 0.28125px). That sub-pixel
  // POSITION — not the size, already fixed above — is what actually
  // produced the "1px gray border": the browser has to anti-alias a
  // sub-pixel-positioned element's edges when compositing it, and
  // that blending is what read as a thin seam, regardless of the
  // container's own background already color-matching the canvas's
  // clear color. Flooring the offset here pins both edges to a whole
  // physical pixel, eliminating the blend entirely. Requires
  // .gpu-canvas to be `position: absolute` (see style.css) — the
  // container's own flex centering rules are now dead weight, kept
  // only because .webgpu-banner still relies on the container being
  // `position: absolute` for ITS OWN inset:0 overlay.
  canvas.style.left = `${Math.floor((rect.width - cssSize) / 2)}px`;
  canvas.style.top = `${Math.floor((rect.height - cssSize) / 2)}px`;
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
  {
    config,
    scenario = null,
    targetPoints,
    targetVisible = true,
    physics,
    particleCap,
    initialParticleCount,
    fieldMode = "none",
    substrateChannelStart = 0,
    substrateZeroIsBlack = false,
    boundaryGradientZeroIsBlack = false,
    particleShape = "dot",
    particleColorMode = "white",
    particleAlpha = 1,
    directionalLineVisible = false,
    particleRadiusPx,
    mitosisSignalBoost = 1,
    boundaryGradientScale = 0.01,
    internalStateChannelStart = 0,
    chemicalMemoryOpponentSubtraction = 0,
    accent = 0,
    morphologyGradientVisible = true,
    morphologyDensityVisible = true,
    blur = 0,
    gradientExponent = 1,
    bloom,
    zoom = 1,
    autoZoom,
    onEffectiveZoomChange,
    tool = "none",
    deformSettings,
    onStep,
    loopAtTrainedSteps = true,
    paused = false,
  },
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simulationRef = useRef<GpuSimulation | null>(null);
  const contextRef = useRef<GPUCanvasContext | null>(null);
  const deviceRef = useRef<GPUDevice | null>(null);
  const configRef = useRef<SimulationConfig | null>(null);
  const scenarioRef = useRef(scenario);
  const physicsRef = useRef(physics);
  const particleCapRef = useRef(particleCap);
  const initialParticleCountRef = useRef(initialParticleCount);
  const fieldModeRef = useRef(fieldMode);
  const substrateChannelStartRef = useRef(substrateChannelStart);
  const substrateZeroIsBlackRef = useRef(substrateZeroIsBlack);
  const boundaryGradientZeroIsBlackRef = useRef(boundaryGradientZeroIsBlack);
  const particleShapeRef = useRef(particleShape);
  const particleColorModeRef = useRef(particleColorMode);
  const particleAlphaRef = useRef(particleAlpha);
  const directionalLineVisibleRef = useRef(directionalLineVisible);
  const particleRadiusPxRef = useRef(particleRadiusPx);
  const mitosisSignalBoostRef = useRef(mitosisSignalBoost);
  const boundaryGradientScaleRef = useRef(boundaryGradientScale);
  const internalStateChannelStartRef = useRef(internalStateChannelStart);
  const chemicalMemoryOpponentSubtractionRef = useRef(chemicalMemoryOpponentSubtraction);
  const accentRef = useRef(accent);
  const morphologyGradientVisibleRef = useRef(morphologyGradientVisible);
  const morphologyDensityVisibleRef = useRef(morphologyDensityVisible);
  const blurRef = useRef(blur);
  const gradientExponentRef = useRef(gradientExponent);
  const bloomRef = useRef(bloom);
  const zoomRef = useRef(zoom);
  const effectiveZoomRef = useRef(zoom);
  const autoZoomTargetRef = useRef(zoom);
  const autoZoomRef = useRef(autoZoom);
  const onEffectiveZoomChangeRef = useRef(onEffectiveZoomChange);
  const autoZoomFrameRef = useRef(0);
  const autoZoomReportFrameRef = useRef(0);
  const autoZoomHardResetRef = useRef(true);
  const toolRef = useRef(tool);
  const deformSettingsRef = useRef(deformSettings);
  const onStepRef = useRef(onStep);
  const loopAtTrainedStepsRef = useRef(loopAtTrainedSteps);
  const pausedRef = useRef(paused);
  // "Move" tool's own live drag state — a ref, not React state, since it
  // updates on every pointermove and is read every animation frame (see
  // the RAF loop below); re-rendering the component for either would be
  // pure waste.
  const draggingRef = useRef(false);
  const dragPosRef = useRef({ x: 0, y: 0 });
  // "Deform" tool's own live hover state — same "ref, not state" reasoning
  // as draggingRef/dragPosRef above (updates every pointermove). null
  // whenever the pointer isn't over the canvas (or the tool isn't
  // "deform") — the preview circle is hidden then, see
  // syncDeformPreview() below. Doubles as the injection POSITION while
  // deformingRef is true (below) — it already tracks the pointer
  // continuously regardless of press state, so there's no need for a
  // second, redundant "current drag position" ref the way move's own
  // dragPosRef is separate from nothing-equivalent.
  const deformHoverRef = useRef<{ x: number; y: number } | null>(null);
  // True from pointerdown to pointerup/leave while the "Deform" tool is
  // held — same "ref, read every RAF frame" reasoning as draggingRef
  // above. See the RAF loop below for why this now injects every frame,
  // not once per click.
  const deformingRef = useRef(false);
  // Plain DOM element, NOT drawn into the WebGPU canvas — a screen-space
  // overlay <div> is far simpler than threading a circle outline through
  // gpu/render.ts's own render pipeline for something this purely
  // cosmetic (never touches simulation state, doesn't need to composite
  // with the field/particle render at all). Positioned/sized directly
  // via style writes in syncDeformPreview() below, not React state/props
  // — this needs to update on every pointermove, which would be a lot of
  // wasted re-renders routed through React for a pure DOM style mutation.
  const deformPreviewRef = useRef<HTMLDivElement>(null);
  // Stable identity via useCallback (empty deps) — safe despite that,
  // since the body only ever reads through refs (canvasRef/
  // deformPreviewRef/deformHoverRef/deformSettingsRef), never closes over
  // a prop/state value directly, so a "stale" closure is never actually
  // stale. Called from the pointer handlers below AND from a dedicated
  // effect on deformSettings.radius (so dragging the radius slider while
  // already hovering resizes the preview immediately, not just on the
  // next pointermove).
  const syncDeformPreview = useCallback(() => {
    const canvas = canvasRef.current;
    const preview = deformPreviewRef.current;
    if (!canvas || !preview) return;
    const hover = deformHoverRef.current;
    if (!hover) {
      preview.style.display = "none";
      return;
    }
    const containerRect = canvas.parentElement!.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    // Y-flip matches domainPos()'s own convention (screen is top-down,
    // the domain is bottom-up) — see that function's own comment.
    const cameraZoom = effectiveZoomRef.current;
    const screenX = 0.5 + (hover.x - 0.5) * cameraZoom;
    const screenY = 0.5 + (hover.y - 0.5) * cameraZoom;
    const px = canvasRect.left - containerRect.left + screenX * canvasRect.width;
    const py = canvasRect.top - containerRect.top + (1 - screenY) * canvasRect.height;
    // Domain units -> CSS pixels — canvasRect.width/height are equal
    // (GridCanvas.tsx's own applySquareSize() guarantees a square
    // canvas), so either axis gives the same domain-to-pixel scale.
    const radiusPx = deformSettingsRef.current.radius * canvasRect.width;
    preview.style.display = "block";
    preview.style.left = `${px - radiusPx}px`;
    preview.style.top = `${py - radiusPx}px`;
    preview.style.width = `${radiusPx * 2}px`;
    preview.style.height = `${radiusPx * 2}px`;
  }, []);
  // Resizes the canvas's own backing store (applySquareSize()) AND
  // re-configure()s the WebGPU context against the new size — every
  // resize AFTER the very first one (the ResizeObserver effect below,
  // and the "next painted frame" re-validation in the mount effect)
  // needs both, not just the first. Skipping the reconfigure() is what
  // produced the "1px gray border on the right/bottom" artifact: a
  // canvas.width/height mutation alone doesn't reliably resize the
  // already-configured swap chain's own texture on every backend, so
  // getCurrentTexture() kept handing back a texture still sized for
  // whatever the LAST configure() call saw — one device pixel smaller
  // on the trailing edges than the canvas's own new (larger) backing
  // store, leaving that final row/column showing stale/uncleared
  // content instead of this frame's own clear color. Re-configuring on
  // every resize keeps the swap chain's own texture size and the
  // canvas's own backing store size exactly in lockstep. Stable
  // identity via useCallback (empty deps) — reads everything through
  // contextRef/deviceRef, so it's never stale despite that.
  const resizeCanvas = useCallback((canvas: HTMLCanvasElement) => {
    applySquareSize(canvas);
    const context = contextRef.current;
    const device = deviceRef.current;
    if (!context || !device) return;
    context.configure({
      device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
  }, []);
  // Constructed once, lazily, the first time anything actually needs it
  // (not on every render) — see canvasRecorder.ts's own CanvasRecorder
  // docstring for why one instance is reused across repeated
  // start()/stop() cycles rather than recreated each time.
  const recorderRef = useRef<CanvasRecorder | null>(null);
  const batchRunningRef = useRef(false);
  const activeStepRef = useRef<Promise<void> | null>(null);
  const configReloadingRef = useRef(false);
  const configRevisionRef = useRef(0);
  function getRecorder(): CanvasRecorder {
    if (!recorderRef.current) recorderRef.current = new CanvasRecorder();
    return recorderRef.current;
  }
  const [status, setStatus] = useState<Status>("loading");
  const [statusMessage, setStatusMessage] = useState<string>("");

  configRef.current = config;
  physicsRef.current = physics;
  particleCapRef.current = particleCap;
  initialParticleCountRef.current = initialParticleCount;
  fieldModeRef.current = fieldMode;
  substrateChannelStartRef.current = substrateChannelStart;
  substrateZeroIsBlackRef.current = substrateZeroIsBlack;
  boundaryGradientZeroIsBlackRef.current = boundaryGradientZeroIsBlack;
  particleShapeRef.current = particleShape;
  particleColorModeRef.current = particleColorMode;
  particleAlphaRef.current = particleAlpha;
  directionalLineVisibleRef.current = directionalLineVisible;
  particleRadiusPxRef.current = particleRadiusPx;
  mitosisSignalBoostRef.current = mitosisSignalBoost;
  boundaryGradientScaleRef.current = boundaryGradientScale;
  internalStateChannelStartRef.current = internalStateChannelStart;
  chemicalMemoryOpponentSubtractionRef.current = chemicalMemoryOpponentSubtraction;
  accentRef.current = accent;
  morphologyGradientVisibleRef.current = morphologyGradientVisible;
  morphologyDensityVisibleRef.current = morphologyDensityVisible;
  blurRef.current = blur;
  gradientExponentRef.current = gradientExponent;
  bloomRef.current = bloom;
  toolRef.current = tool;
  deformSettingsRef.current = deformSettings;
  onStepRef.current = onStep;
  loopAtTrainedStepsRef.current = loopAtTrainedSteps;
  pausedRef.current = paused;

  useImperativeHandle(ref, () => ({
    restart: () => {
      simulationRef.current?.restartRollout();
      autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
      autoZoomTargetRef.current = effectiveZoomRef.current;
      autoZoomHardResetRef.current = true;
    },
    randomizeWeights: () => {
      const weights = simulationRef.current?.randomizeWeights() ?? null;
      if (weights) {
        autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
        autoZoomTargetRef.current = effectiveZoomRef.current;
        autoZoomHardResetRef.current = true;
      }
      return weights;
    },
    startRecording: () => {
      if (canvasRef.current) getRecorder().start(canvasRef.current);
    },
    stopRecording: () => getRecorder().stop("mpm-training"),
    collectSamples: async (
      samples,
      steps,
      restorePhysics,
      restoreParticleCap,
      restoreInitialParticleCount,
      restoreParticleRadiusPx,
      includeJson,
      onProgress,
      signal,
    ) => {
      const sim = simulationRef.current;
      const context = contextRef.current;
      const canvas = canvasRef.current;
      const device = deviceRef.current;
      if (!sim?.ready || !context || !canvas || !device) {
        throw new Error("The simulation is not ready yet.");
      }
      const restoreConfig = configRef.current;
      if (!restoreConfig) throw new Error("No simulation configuration is loaded.");
      batchRunningRef.current = true;
      const captures: Array<{ filename: string; blob: Blob }> = [];
      try {
        // A normal animation frame may already be suspended in step()'s
        // particle-count readback. Let it finish before resetting state.
        await activeStepRef.current;
        for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
          const sample = samples[sampleIndex];
          if (signal.aborted) throw new DOMException("Sample collection cancelled", "AbortError");
          // Substrate resolution is baked into buffer sizes and shader
          // constants, so a resolution sweep must go through loadGeneration
          // rather than the live-uniform physics path.
          sim.loadGeneration(sample.config);
          sim.setPhysics(sample.physics);
          sim.setParticleCap(sample.particleCap);
          sim.setInitialParticleCount(sample.initialParticleCount);
          sim.setPointRadiusPx(sample.particleRadiusPx);
          sim.restartRollout();
          let firstCapStep: number | null = sim.particleCount >= sample.particleCap ? 0 : null;
          for (let step = 0; step < steps; step += 1) {
            if (signal.aborted) throw new DOMException("Sample collection cancelled", "AbortError");
            await sim.step();
            if (firstCapStep === null && sim.particleCount >= sample.particleCap) firstCapStep = step + 1;
          }
          sim.render(context);
          await device.queue.onSubmittedWorkDone();
          const blob = await new Promise<Blob>((resolve, reject) => {
            canvas.toBlob((result) => result ? resolve(result) : reject(new Error("Could not capture the simulation canvas.")), "image/png");
          });
          captures.push({ filename: sample.filename, blob });
          if (includeJson) {
            const positions = await sim.readPositions();
            const spatial = spatialMetrics(positions);
            const metadata = {
              particleDensityMultiplier: sample.particleDensityMultiplier,
              substrateResolution: sample.config.fieldN,
              particleCap: sample.particleCap,
              initialParticleCount: sample.initialParticleCount,
              finalParticleCount: sim.particleCount,
              representedInitialCount: sample.initialParticleCount / sample.particleDensityMultiplier,
              representedFinalCount: sim.particleCount / sample.particleDensityMultiplier,
              simulationSteps: steps,
              firstCapStep,
              ...spatial,
            };
            captures.push({
              filename: sample.filename.replace(/\.png$/i, ".json"),
              blob: new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" }),
            });
          }
          onProgress(sampleIndex + 1);
        }
        return captures;
      } finally {
        sim.loadGeneration(restoreConfig);
        sim.setPhysics(restorePhysics);
        sim.setParticleCap(restoreParticleCap);
        sim.setInitialParticleCount(restoreInitialParticleCount);
        sim.setPointRadiusPx(restoreParticleRadiusPx);
        sim.restartRollout();
        autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
        autoZoomHardResetRef.current = true;
        batchRunningRef.current = false;
      }
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
      deviceRef.current = device;

      const simulation = new GpuSimulation(device, format);
      simulation.setCanvasSizePx(canvas.width, canvas.height);
      if (targetPoints) simulation.setTargetPoints(targetPoints);
      simulation.setTargetVisible(targetVisible);
      simulation.setFieldMode(fieldModeRef.current);
      simulation.setSubstrateChannelStart(substrateChannelStartRef.current);
      simulation.setSubstrateZeroIsBlack(substrateZeroIsBlackRef.current);
      simulation.setBoundaryGradientZeroIsBlack(boundaryGradientZeroIsBlackRef.current);
      simulation.setParticleShape(particleShapeRef.current);
      simulation.setParticleColorMode(particleColorModeRef.current);
      simulation.setParticleAlpha(particleAlphaRef.current);
      simulation.setDirectionalLineVisible(directionalLineVisibleRef.current);
      simulation.setMitosisSignalBoost(mitosisSignalBoostRef.current);
      simulation.setBoundaryGradientScale(boundaryGradientScaleRef.current);
      simulation.setInternalStateChannelStart(internalStateChannelStartRef.current);
      simulation.setChemicalMemoryOpponentSubtraction(chemicalMemoryOpponentSubtractionRef.current);
      if (particleRadiusPxRef.current !== undefined) simulation.setPointRadiusPx(particleRadiusPxRef.current);
      simulation.setAccent(accentRef.current);
      simulation.setMorphologyDisplay(morphologyGradientVisibleRef.current, morphologyDensityVisibleRef.current);
      simulation.setBlur(blurRef.current);
      simulation.setGradientExponent(gradientExponentRef.current);
      if (bloomRef.current) simulation.setBloom(bloomRef.current);
      simulation.setZoom(zoomRef.current);
      effectiveZoomRef.current = zoomRef.current;
      autoZoomTargetRef.current = zoomRef.current;
      simulationRef.current = simulation;
      const initialConfig = configRef.current;
      const initialWeightsError = initialConfig
        ? policyWeightsShapeError(initialConfig.weights, initialConfig.channels, initialConfig.hiddenDim, initialConfig.policyArchitecture)
        : null;
      simulation.setScenario(scenarioRef.current);
      if (initialConfig && !initialWeightsError) simulation.loadGeneration(initialConfig);
      if (particleCapRef.current !== undefined) simulation.setParticleCap(particleCapRef.current);
      if (initialParticleCountRef.current !== undefined) {
        simulation.setInitialParticleCount(initialParticleCountRef.current);
      }
      if (physicsRef.current) simulation.setPhysics(physicsRef.current);
      if (initialWeightsError) {
        setStatus("incompatible");
        setStatusMessage(initialWeightsError);
      } else {
        setStatus("ready");
        setStatusMessage("");
      }

      // Re-validate the size ONE more time on the next painted frame.
      // The synchronous applySquareSize() above is necessary (see its own
      // call site comment — configuring the swap chain against the HTML
      // default 300x150 box causes texture-size cascades), but it's a
      // race against the surrounding page's OWN layout: acquireGpuDevice()
      // above is async, and when it resolves fast (warm adapter/device
      // cache), this whole callback can run before flex/sidebar layout has
      // reached its final box — container.getBoundingClientRect() then
      // returns a too-small rect, baking a too-small canvasMinDimPx into
      // the renderer (see render.ts's own setCanvasSizePx()), which makes
      // EVERY device-pixel-sized draw (particle radius, the target-point
      // overlay dots — both derive from canvasMinDimPx) render oversized.
      // This is exactly the "particles and dots load too big" bug: it
      // silently persists until the ResizeObserver effect below happens to
      // fire from an unrelated layout nudge (switching a run in history,
      // hitting Reset — anything that perturbs the container's own box by
      // even a pixel), which is why those actions "fix" it as a pure side
      // effect. rAF guarantees layout has settled by the time this runs,
      // so it re-derives the correct box unconditionally, every mount, not
      // just when something else happens to jog it loose.
      requestAnimationFrame(() => {
        if (cancelled) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        resizeCanvas(canvas);
        simulationRef.current?.setCanvasSizePx(canvas.width, canvas.height);
      });
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
      resizeCanvas(canvas);
      simulationRef.current?.setCanvasSizePx(canvas.width, canvas.height);
      // The canvas's own screen-pixel box just changed size — the
      // preview circle's own pixel geometry (computed from that box in
      // syncDeformPreview()) would otherwise go stale until the next
      // pointermove.
      syncDeformPreview();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [syncDeformPreview, resizeCanvas]);

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
      const ny = 1 - (ev.clientY - rect.top) / rect.height;
      const cameraZoom = effectiveZoomRef.current;
      const worldX = 0.5 + (nx - 0.5) / cameraZoom;
      const worldY = 0.5 + (ny - 0.5) / cameraZoom;
      return {
        x: Math.min(1, Math.max(0, worldX)),
        y: Math.min(1, Math.max(0, worldY)),
      };
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
      } else if (toolRef.current === "deform") {
        // Same capture reasoning as "move" above — keeps injecting even
        // if the pointer strays outside the canvas mid-hold. Sets
        // deformHoverRef immediately so the very first RAF frame already
        // has a fresh position, not whatever was last hovered (see the
        // RAF loop below for the actual per-frame injectDeform() call —
        // this handler no longer injects directly).
        canvas.setPointerCapture(ev.pointerId);
        deformHoverRef.current = pos;
        deformingRef.current = true;
      }
    };

    const handlePointerMove = (ev: PointerEvent) => {
      if (draggingRef.current) {
        dragPosRef.current = domainPos(ev);
      }
      // Independent of draggingRef/deformingRef — the deform preview
      // tracks the pointer on plain hover too, no press required (unlike
      // "move"'s own press-drag gesture); this same value doubles as the
      // injection position while deformingRef is true.
      deformHoverRef.current = toolRef.current === "deform" ? domainPos(ev) : null;
      syncDeformPreview();
    };

    const handlePointerUp = (ev: PointerEvent) => {
      if (deformingRef.current) {
        deformingRef.current = false;
        if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
      }
      if (!draggingRef.current) return;
      draggingRef.current = false;
      simulationRef.current?.endDrag();
      if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
    };

    // Clears the hover preview once the pointer actually leaves the
    // canvas — pointermove alone never fires again to naturally do this
    // (there's no "moved to nowhere" event), so without this the last
    // hovered position would keep drawing a stale preview circle.
    const handlePointerLeave = () => {
      deformHoverRef.current = null;
      syncDeformPreview();
    };

    canvas.addEventListener("pointerdown", handlePointerDown);
    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("pointerup", handlePointerUp);
    canvas.addEventListener("pointercancel", handlePointerUp);
    canvas.addEventListener("pointerleave", handlePointerLeave);
    return () => {
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("pointercancel", handlePointerUp);
      canvas.removeEventListener("pointerleave", handlePointerLeave);
    };
  }, []);

  // Cleanly ends an in-progress drag/hold if the "Move"/"Deform" tool is
  // switched away from mid-gesture (e.g. the user clicks a different
  // tool button while still holding the mouse down) — without this,
  // draggingRef/deformingRef would stay stuck true and keep re-pinning a
  // particle or injecting a force no pointerup ever arrives for.
  useEffect(() => {
    if (tool !== "move" && draggingRef.current) {
      draggingRef.current = false;
      simulationRef.current?.endDrag();
    }
    if (tool !== "deform") {
      deformingRef.current = false;
      deformHoverRef.current = null;
    }
    syncDeformPreview();
  }, [tool, syncDeformPreview]);

  // Resizes the preview circle immediately when deformSettings.radius
  // changes while already hovering (e.g. dragging the radius slider
  // without moving the mouse) — without this it would only catch up on
  // the next pointermove.
  useEffect(() => {
    syncDeformPreview();
  }, [deformSettings.radius, syncDeformPreview]);

  useEffect(() => {
    if (targetPoints) simulationRef.current?.setTargetPoints(targetPoints);
  }, [targetPoints]);

  useEffect(() => {
    simulationRef.current?.setTargetVisible(targetVisible);
  }, [targetVisible]);

  useEffect(() => {
    const simulation = simulationRef.current;
    if (!config || !simulation) return;
    const revision = ++configRevisionRef.current;
    const weightsError = policyWeightsShapeError(config.weights, config.channels, config.hiddenDim, config.policyArchitecture);
    if (weightsError) {
      configReloadingRef.current = false;
      setStatus("incompatible");
      setStatusMessage(weightsError);
      return;
    }
    configReloadingRef.current = true;
    void (async () => {
      // Architecture/width changes rebuild and destroy policy buffers. Wait
      // for the current growth-count mapAsync before doing so; otherwise a
      // rapid exploration switch aborts that in-flight readback.
      try {
        await activeStepRef.current;
      } catch {
        // The frame loop owns reporting genuine step failures.
      }
      if (revision !== configRevisionRef.current) return;
      simulation.loadGeneration(config);
      autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
      autoZoomTargetRef.current = effectiveZoomRef.current;
      autoZoomHardResetRef.current = true;
      // loadGeneration deliberately restores the run's recorded settings so
      // selecting another trained generation is reproducible. An exploration
      // brain reload uses that same path, but must then restore every current
      // playback override; React will not rerun the [physics] effect because
      // the override object itself did not change. This covers decay and the
      // entire PhysicsSettings/GrowthPanel surface, not a decay-only patch.
      if (physicsRef.current) simulation.setPhysics(physicsRef.current);
      configReloadingRef.current = false;
      setStatus("ready");
      setStatusMessage("");
    })();
  }, [config]);

  useEffect(() => {
    if (physics) simulationRef.current?.setPhysics(physics);
  }, [physics]);

  useEffect(() => {
    if (particleCap !== undefined) simulationRef.current?.setParticleCap(particleCap);
  }, [particleCap]);

  useEffect(() => {
    if (initialParticleCount !== undefined) {
      simulationRef.current?.setInitialParticleCount(initialParticleCount);
      autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
      autoZoomHardResetRef.current = true;
    }
  }, [initialParticleCount]);

  useEffect(() => {
    scenarioRef.current = scenario;
    simulationRef.current?.setScenario(scenario);
    autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
    autoZoomHardResetRef.current = true;
  }, [scenario]);

  useEffect(() => {
    simulationRef.current?.setFieldMode(fieldMode);
  }, [fieldMode]);

  useEffect(() => {
    simulationRef.current?.setSubstrateChannelStart(substrateChannelStart);
  }, [substrateChannelStart]);

  useEffect(() => {
    simulationRef.current?.setSubstrateZeroIsBlack(substrateZeroIsBlack);
  }, [substrateZeroIsBlack]);

  useEffect(() => {
    simulationRef.current?.setBoundaryGradientZeroIsBlack(boundaryGradientZeroIsBlack);
  }, [boundaryGradientZeroIsBlack]);

  useEffect(() => {
    simulationRef.current?.setParticleShape(particleShape);
  }, [particleShape]);

  useEffect(() => {
    simulationRef.current?.setParticleColorMode(particleColorMode);
  }, [particleColorMode]);

  useEffect(() => {
    simulationRef.current?.setParticleAlpha(particleAlpha);
  }, [particleAlpha]);

  useEffect(() => {
    simulationRef.current?.setDirectionalLineVisible(directionalLineVisible);
  }, [directionalLineVisible]);

  useEffect(() => {
    simulationRef.current?.setMitosisSignalBoost(mitosisSignalBoost);
  }, [mitosisSignalBoost]);

  useEffect(() => {
    simulationRef.current?.setBoundaryGradientScale(boundaryGradientScale);
  }, [boundaryGradientScale]);

  useEffect(() => {
    simulationRef.current?.setInternalStateChannelStart(internalStateChannelStart);
  }, [internalStateChannelStart]);

  useEffect(() => {
    simulationRef.current?.setChemicalMemoryOpponentSubtraction(chemicalMemoryOpponentSubtraction);
  }, [chemicalMemoryOpponentSubtraction]);

  useEffect(() => {
    if (particleRadiusPx !== undefined) simulationRef.current?.setPointRadiusPx(particleRadiusPx);
  }, [particleRadiusPx]);

  useEffect(() => {
    simulationRef.current?.setAccent(accent);
  }, [accent]);

  useEffect(() => {
    simulationRef.current?.setMorphologyDisplay(morphologyGradientVisible, morphologyDensityVisible);
  }, [morphologyGradientVisible, morphologyDensityVisible]);

  useEffect(() => {
    simulationRef.current?.setBlur(blur);
  }, [blur]);

  useEffect(() => {
    simulationRef.current?.setGradientExponent(gradientExponent);
  }, [gradientExponent]);

  useEffect(() => {
    if (bloom) simulationRef.current?.setBloom(bloom);
  }, [bloom]);

  useEffect(() => {
    zoomRef.current = zoom;
    if (!autoZoomRef.current?.enabled) {
      effectiveZoomRef.current = zoom;
      simulationRef.current?.setZoom(zoom);
      onEffectiveZoomChangeRef.current?.(zoom);
    }
    syncDeformPreview();
  }, [zoom, syncDeformPreview]);

  useEffect(() => {
    autoZoomRef.current = autoZoom;
    autoZoomFrameRef.current = 0;
    autoZoomReportFrameRef.current = 0;
    autoZoomTargetRef.current = effectiveZoomRef.current;
    autoZoomHardResetRef.current = Boolean(autoZoom?.enabled);
    if (autoZoom?.enabled) {
      autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
    }
    if (!autoZoom?.enabled) {
      effectiveZoomRef.current = zoomRef.current;
      simulationRef.current?.setZoom(zoomRef.current);
      onEffectiveZoomChangeRef.current?.(zoomRef.current);
    }
  }, [autoZoom]);

  useEffect(() => {
    onEffectiveZoomChangeRef.current = onEffectiveZoomChange;
  }, [onEffectiveZoomChange]);

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
      if (sim?.ready && context && !batchRunningRef.current && !configReloadingRef.current) {
        if (!pausedRef.current) {
          try {
            const stepPromise = sim.step();
            activeStepRef.current = stepPromise;
            await stepPromise;
            activeStepRef.current = null;
          } catch (err) {
            activeStepRef.current = null;
            if (!cancelled) console.error(err);
            return;
          }
          if (cancelled) return;
          if (loopAtTrainedStepsRef.current && sim.currentStep >= sim.steps) {
            sim.restartRollout();
            autoZoomFrameRef.current = Number.MAX_SAFE_INTEGER;
            autoZoomTargetRef.current = effectiveZoomRef.current;
            autoZoomHardResetRef.current = true;
          }
        }
        const auto = autoZoomRef.current;
        if (auto?.enabled) {
          autoZoomFrameRef.current += 1;
          const cadence = Math.max(1, Math.floor(auto.sampleEveryFrames));
          if (autoZoomFrameRef.current >= cadence) {
            autoZoomFrameRef.current = 0;
            try {
              const positions = await sim.readPositionSamples(auto.maxSamples);
              if (cancelled) return;
              if (positions.length >= 2) {
                let minX = positions[0];
                let maxX = positions[0];
                let minY = positions[1];
                let maxY = positions[1];
                for (let i = 2; i < positions.length; i += 2) {
                  minX = Math.min(minX, positions[i]);
                  maxX = Math.max(maxX, positions[i]);
                  minY = Math.min(minY, positions[i + 1]);
                  maxY = Math.max(maxY, positions[i + 1]);
                }
                // The existing camera is fixed on world center, so asymmetric
                // drift must count toward the centered fitting square too.
                const centeredExtent = 2 * Math.max(
                  Math.abs(minX - 0.5),
                  Math.abs(maxX - 0.5),
                  Math.abs(minY - 0.5),
                  Math.abs(maxY - 0.5),
                  1e-4,
                );
                const fitFraction = Math.min(1, Math.max(0.05, auto.fitFraction));
                const padding = Math.max(1, auto.padding);
                const target = Math.min(
                  8,
                  Math.max(1, fitFraction / (centeredExtent * padding)),
                );
                autoZoomTargetRef.current = target;
                if (autoZoomHardResetRef.current) {
                  autoZoomHardResetRef.current = false;
                  effectiveZoomRef.current = target;
                  sim.setZoom(target);
                  onEffectiveZoomChangeRef.current?.(target);
                  autoZoomReportFrameRef.current = 0;
                  syncDeformPreview();
                }
              }
            } catch (err) {
              if (!cancelled) {
                console.error("[auto-zoom] position sampling failed", err);
              }
            }
          }
          const current = effectiveZoomRef.current;
          const target = autoZoomTargetRef.current;
          const smoothing = Math.min(1, Math.max(0.001, auto.smoothing));
          const next = Math.abs(target - current) < 1e-4
            ? target
            : current + (target - current) * smoothing;
          const moved = next !== current;
          if (moved) {
            effectiveZoomRef.current = next;
            sim.setZoom(next);
            syncDeformPreview();
            // The camera moves every frame, but the numeric readout does not
            // need to force a parent React render at full animation cadence.
            autoZoomReportFrameRef.current += 1;
            if (autoZoomReportFrameRef.current >= 6 || next === target) {
              autoZoomReportFrameRef.current = 0;
              onEffectiveZoomChangeRef.current?.(next);
            }
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
        // Same "every frame, not once per click" treatment — Deform used
        // to fire a single injectDeform() on pointerdown; now it injects
        // continuously for as long as the tool's held, one dispatch per
        // rendered frame (same cadence dragTo() above already runs at),
        // at deformHoverRef's own live position (updated every
        // pointermove regardless of press state — see that ref's own
        // comment). Also outside `!pausedRef.current`, same reasoning.
        if (deformingRef.current && deformHoverRef.current) {
          const { direction, strength, radius, mode } = deformSettingsRef.current;
          sim.injectDeform(deformHoverRef.current.x, deformHoverRef.current.y, direction, strength, radius, mode);
        }
        sim.render(context);
        onStepRef.current?.(sim.currentStep, sim.particleCount);
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
      <canvas ref={canvasRef} className={`gpu-canvas${tool !== "none" ? ` gpu-canvas-tool-${tool}` : ""}`} />
      {/* "Deform" tool's own hover preview — see syncDeformPreview()'s own
       * comment for why this is a plain positioned <div>, not drawn into
       * the WebGPU canvas. Hidden by default (inline style, toggled by
       * syncDeformPreview() itself) rather than conditionally rendered —
       * this needs to update on every pointermove, which would be a lot
       * of wasted React re-renders for a pure style mutation. */}
      <div ref={deformPreviewRef} className="deform-preview" style={{ display: "none" }} />
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
      {status === "incompatible" && (
        <div className="webgpu-banner">
          <p>Policy architecture mismatch.</p>
          <p className="hint">{statusMessage}</p>
        </div>
      )}
    </div>
  );
});

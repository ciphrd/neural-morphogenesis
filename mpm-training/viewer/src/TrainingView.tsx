import { useEffect, useMemo, useRef, useState } from "react"
import coreConstants from "../../core/constants.json"
import { FitnessChart } from "./charts/FitnessChart"
import { randomWeights } from "./gpu/agents"
import { configAtDensity } from "./gpu/density"
import { MAX_PARTICLES } from "./gpu/mpmCore"
import type { FieldMode, ParticleRenderMode } from "./gpu/render"
import type {
  CellMemory,
  ChemicalCommunicationArchitecture,
  PhysicsSettings,
} from "./gpu/types"
import {
  cellMemoryFromConfig,
  chemicalCommunicationArchitectureFromConfig,
  hiddenLayersFromConfig,
  physicsSettingsFromConfig,
  policyArchitectureForCellMemory,
} from "./gpu/types"
import { generationImageUrl } from "./net/images"
import { fetchRunState } from "./net/runs"
import type { TrainingSocketState } from "./net/trainingSocket"
import { EMPTY_STATE, useTrainingSocket } from "./net/trainingSocket"
import { pickRecordingFormat } from "./render/canvasRecorder"
import { createSampleAtlas } from "./render/sampleAtlas"
import type {
  DeformSettings,
  GridCanvasHandle,
  Tool,
} from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"
import { createZip, downloadBlob } from "./render/zip"
import { ChannelWindowSlider } from "./ui/ChannelWindowSlider"
import { GrowthPanel } from "./ui/GrowthPanel"
import { NetworkPanel } from "./ui/NetworkPanel"
import { PhysicsPanel } from "./ui/PhysicsPanel"
import { RunPicker } from "./ui/RunPicker"
import type {
  SampleSweepRequest,
  SweepParameterKey,
} from "./ui/SampleSweepModal"
import { SampleSweepModal, sweepValues } from "./ui/SampleSweepModal"
import { Slider } from "./ui/Slider"
import { VIEWER_DEFAULTS } from "./viewerConfig"

const TRAIN_API_URL = "http://localhost:8003"
const TRAIN_WS_URL = "ws://localhost:8003/ws"

// A pure browser feature-check (no canvas/mount needed — see
// pickRecordingFormat()'s own docstring), so it's computed once here
// rather than round-tripped through GridCanvasHandle every render.
const RECORDING_FORMAT = pickRecordingFormat()

function explorationBrainSeed(seed: number, variant: number): number {
  if (variant === 0) return seed >>> 0
  let x = ((seed >>> 0) ^ Math.imul(variant, 0x9e3779b9)) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0
  return (x ^ (x >>> 16)) >>> 0
}

/** Passive training viewer — a live WebGPU replay of whichever
 * generation's weights are selected, plus a fitness-history timeline.
 * Same overall role as envnca/frontend/src/TrainingView.tsx. Initial particles
 * use the same compact hexagonal disk as training_sim.py's seed_blob(); the
 * "Rendering" section below otherwise
 * mirrors mls-mpm's own Field-mode/particle-size controls — see
 * gpu/render.ts's own module docstring for exactly which of mls-mpm's
 * field modes are portable without extending core/'s shared physics. */
export function TrainingView() {
  const liveState = useTrainingSocket(TRAIN_WS_URL, TRAIN_API_URL)
  // null = following the live/current run. Anything else is an archived
  // run's own id (net/runs.ts's RunSummary) — see the RunPicker below.
  const [viewingRunId, setViewingRunId] = useState<string | null>(null)
  const [archivedState, setArchivedState] =
    useState<TrainingSocketState>(EMPTY_STATE)
  useEffect(() => {
    if (viewingRunId === null) return
    let cancelled = false
    setArchivedState(EMPTY_STATE)
    fetchRunState(TRAIN_API_URL, viewingRunId)
      .then((state) => {
        if (!cancelled) setArchivedState(state)
      })
      .catch((err) =>
        console.error(
          "[mpm-training] failed to fetch archived run history",
          err
        )
      )
    return () => {
      cancelled = true
    }
  }, [viewingRunId])
  const { history, latest, configByGeneration } =
    viewingRunId === null ? liveState : archivedState

  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(
    null
  )
  // A generation number from one run means nothing in another's history
  // (or worse, coincidentally exists in both and would show mismatched
  // data) — always reset to "follow whatever's newest in this run" the
  // instant the viewed run itself changes.
  useEffect(() => {
    setSelectedGeneration(null)
  }, [viewingRunId])
  // Default true (loop): a candidate was only ever *trained* for
  // config.macroSteps, so that's what keeps a long-idle viewer cycling
  // through fresh rollouts instead of freezing on the last frame.
  // Unchecking lets a rollout run straight past that count instead —
  // there's no hard simulation limit at macroSteps, just the point
  // training stopped scoring it.
  const [loopAtTrainedSteps, setLoopAtTrainedSteps] = useState(
    VIEWER_DEFAULTS.playback.loopAtTrainedSteps
  )
  const [paused, setPaused] = useState(VIEWER_DEFAULTS.playback.paused)
  // Whether the Record button currently reads "● REC" — GridCanvas's own
  // CanvasRecorder (render/canvasRecorder.ts) owns the actual
  // MediaRecorder/download lifecycle, this is just the button's own
  // display state; toggled on click (see handleRecordClick below), not
  // by GridCanvas ever calling back in (recording never stops on its
  // own mid-session the way, say, a rollout-length limit might).
  const [recording, setRecording] = useState(false)
  const gridCanvasRef = useRef<GridCanvasHandle>(null)
  const [sampleModalOpen, setSampleModalOpen] = useState(false)
  const [sampleRunning, setSampleRunning] = useState(false)
  const [sampleCompleted, setSampleCompleted] = useState(0)
  const [sampleTotal, setSampleTotal] = useState(0)
  const [sampleError, setSampleError] = useState<string | null>(null)
  const sampleAbortRef = useRef<AbortController | null>(null)

  // View-only rendering options (gpu/render.ts) — not simulation state,
  // so plain component state, never reset by a run/generation change.
  const [fieldMode, setFieldMode] = useState<FieldMode>(
    VIEWER_DEFAULTS.rendering.fieldMode
  )
  // First channel in the contiguous substrate RGB window. The renderer maps
  // start/start+1/start+2 to red/green/blue respectively.
  const [substrateChannelStart, setSubstrateChannelStart] = useState(
    VIEWER_DEFAULTS.rendering.substrateChannelStart
  )
  const [morphologyGradientVisible, setMorphologyGradientVisible] =
    useState(VIEWER_DEFAULTS.rendering.morphologyGradientVisible)
  const [morphologyDensityVisible, setMorphologyDensityVisible] = useState(
    VIEWER_DEFAULTS.rendering.morphologyDensityVisible
  )
  // [-2,2] exponential background contrast. Negative suppresses submaximal
  // field values, 0 is identity, positive accentuates faint values.
  const [accent, setAccent] = useState(VIEWER_DEFAULTS.rendering.accent)
  // [0,2] — Gaussian sigma, in repulsion-field texels, for the "gradient"
  // mode's own blur pass (see gpu/render.ts's own setBlur()/field.wgsl's
  // own blurDensity() comment for why: raw per-particle density is too
  // grainy for a clean shape-boundary gradient). 0 = no blur, unchanged
  // from before this knob existed; only read by that one background mode.
  const [blur, setBlur] = useState(VIEWER_DEFAULTS.rendering.blur)
  // [~0.25,4] — power curve on the "gradient" mode's own gradient
  // MAGNITUDE (direction preserved — see gpu/render.ts's own
  // setGradientExponent()/field.wgsl's own colorizeGradient() comment).
  // 1 = identity, unchanged from before this knob existed; only read by
  // that one background mode.
  const [gradientExponent, setGradientExponent] = useState(
    VIEWER_DEFAULTS.rendering.gradientExponent
  )
  const [particleRenderMode, setParticleRenderMode] =
    useState<ParticleRenderMode>(VIEWER_DEFAULTS.rendering.particleRenderMode)
  const [boundaryGradientScale, setBoundaryGradientScale] = useState(
    VIEWER_DEFAULTS.rendering.boundaryGradientScale
  )
  const [zoom, setZoom] = useState(VIEWER_DEFAULTS.rendering.zoom)
  const [autoZoomEnabled, setAutoZoomEnabled] = useState(
    VIEWER_DEFAULTS.rendering.autoZoom.enabled
  )
  const [effectiveZoom, setEffectiveZoom] = useState(
    VIEWER_DEFAULTS.rendering.zoom
  )
  const autoZoomSettings = useMemo(
    () => ({
      ...VIEWER_DEFAULTS.rendering.autoZoom,
      enabled: autoZoomEnabled,
    }),
    [autoZoomEnabled]
  )
  const [bloom, setBloom] = useState(() => ({
    ...VIEWER_DEFAULTS.rendering.bloom,
  }))
  const [particleRadiusPx, setParticleRadiusPx] = useState(
    VIEWER_DEFAULTS.rendering.particleRadiusPx
  )
  const [frontendParticleCap, setFrontendParticleCap] = useState(
    VIEWER_DEFAULTS.playback.particleCap ?? 2
  )
  const [frontendInitialParticleCount, setFrontendInitialParticleCount] =
    useState(VIEWER_DEFAULTS.playback.initialParticleCount ?? 1)
  const [
    frontendInitialParticleCountInput,
    setFrontendInitialParticleCountInput,
  ] = useState(String(VIEWER_DEFAULTS.playback.initialParticleCount ?? 1))
  const particleCapRunRef = useRef<string | null>(null)
  const [targetVisible, setTargetVisible] = useState(
    VIEWER_DEFAULTS.rendering.targetVisible
  )
  const [whiteDotsAlpha, setWhiteDotsAlpha] = useState(
    VIEWER_DEFAULTS.rendering.whiteDotsAlpha
  )
  const [activationAlpha, setActivationAlpha] = useState(
    VIEWER_DEFAULTS.rendering.activationAlpha
  )
  const [neuralColorAlpha, setNeuralColorAlpha] = useState(
    VIEWER_DEFAULTS.rendering.neuralColorAlpha
  )
  const [internalStateAlpha, setInternalStateAlpha] = useState(
    VIEWER_DEFAULTS.rendering.internalStateAlpha
  )
  const [internalStateChannelStart, setInternalStateChannelStart] = useState(
    VIEWER_DEFAULTS.rendering.internalStateChannelStart
  )
  const [chemicalMemoryOpponentSubtraction, setChemicalMemoryOpponentSubtraction] =
    useState(VIEWER_DEFAULTS.rendering.chemicalMemoryOpponentSubtraction)
  const [growthAxisLengthPx, setGrowthAxisLengthPx] = useState(
    VIEWER_DEFAULTS.rendering.growthAxisLengthPx
  )
  // "Add"/"Move"/"Deform" interaction tools (render/GridCanvas.tsx's own
  // Tool type) — toggled on/off by clicking their own icon button again
  // (see the Tools section below), not reset by a run/generation change
  // either, same reasoning as the rendering options above.
  const [tool, setTool] = useState<Tool>(VIEWER_DEFAULTS.tools.selected)
  // "Deform" tool's own live settings (direction/strength/radius/mode) —
  // owned here (this component's own small panel below), read by
  // GridCanvas at click/hover time (see that component's own
  // DeformSettings docstring). direction/strength/radius/mode defaults
  // mirror gpu/deform.wgsl's own starting-guess scale comments.
  const [deformSettings, setDeformSettings] = useState<DeformSettings>(() => ({
    ...VIEWER_DEFAULTS.tools.deform,
  }))

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. configByGeneration and history are
  // evicted in lockstep (see net/trainingSocket.ts), so any generation
  // number that still appears on the chart is guaranteed to resolve here.
  const activeConfig =
    selectedGeneration !== null
      ? (configByGeneration.get(selectedGeneration) ?? latest)
      : latest
  const [chemicalArchitectureOverride, setChemicalArchitectureOverride] =
    useState<ChemicalCommunicationArchitecture | null>(null)
  const [chiralityOverride, setChiralityOverride] = useState<boolean | null>(
    null
  )
  const [particleDensityOverride, setParticleDensityOverride] = useState<
    number | null
  >(null)
  const [substrateResolutionOverride, setSubstrateResolutionOverride] = useState<
    number | null
  >(null)
  useEffect(() => {
    setChemicalArchitectureOverride(null)
    setChiralityOverride(null)
    setParticleDensityOverride(null)
    setSubstrateResolutionOverride(null)
  }, [viewingRunId, activeConfig?.generation])
  const defaultParticleDensity =
    VIEWER_DEFAULTS.playback.particleDensityMultiplier ??
    activeConfig?.particleDensityMultiplier ??
    1
  const effectiveParticleDensity =
    particleDensityOverride ?? defaultParticleDensity
  const defaultSubstrateResolution =
    VIEWER_DEFAULTS.playback.substrateResolution ?? activeConfig?.fieldN ?? 256
  const effectiveSubstrateResolution =
    substrateResolutionOverride ?? defaultSubstrateResolution
  const effectiveChirality =
    chiralityOverride ?? activeConfig?.chirality ?? true
  const playbackConfig = useMemo(
    () => {
      if (!activeConfig) return null
      const densityResolved = configAtDensity(
        {
          ...activeConfig,
          fieldN: effectiveSubstrateResolution,
          chemicalCommunicationArchitecture:
            chemicalArchitectureOverride ??
            chemicalCommunicationArchitectureFromConfig(activeConfig),
          chirality: effectiveChirality,
        },
        effectiveParticleDensity
      )
      // Playback population is an explicit viewer control. Density currently
      // changes numerical physics/chemistry only, never count or seed count.
      return {
        ...densityResolved,
        particles: activeConfig.particles,
        initialParticleCount: activeConfig.initialParticleCount,
      }
    },
    [
      activeConfig,
      chemicalArchitectureOverride,
      effectiveChirality,
      effectiveParticleDensity,
      effectiveSubstrateResolution,
    ]
  )
  useEffect(() => {
    const maxStart = Math.max(0, (activeConfig?.channels ?? 3) - 3)
    setSubstrateChannelStart((start) => Math.min(start, maxStart))
  }, [activeConfig?.channels])
  // Shape-changing exploration never reinterprets checkpoint weights. It
  // creates a deterministic fresh policy from this rollout's own seed.
  const [policyExploration, setPolicyExploration] = useState<{
    cellMemory: CellMemory
    hiddenWidth: number
    variant: number
  } | null>(null)
  useEffect(() => {
    setPolicyExploration(null)
  }, [viewingRunId, activeConfig?.generation])
  const previewConfig = useMemo(() => {
    if (!playbackConfig || !policyExploration) return playbackConfig
    const policyArchitecture = policyArchitectureForCellMemory(
      policyExploration.cellMemory
    )
    return {
      ...playbackConfig,
      cellMemory: policyExploration.cellMemory,
      hiddenDim: policyExploration.hiddenWidth,
      hiddenLayers: [policyExploration.hiddenWidth],
      policyArchitecture,
      weights: randomWeights(
        playbackConfig.channels,
        policyExploration.hiddenWidth,
        policyArchitecture,
        explorationBrainSeed(playbackConfig.seed, policyExploration.variant)
      ),
    }
  }, [playbackConfig, policyExploration])
  const displayedCellMemory =
    policyExploration?.cellMemory ??
    (activeConfig ? cellMemoryFromConfig(activeConfig) : "recurrent")
  const displayedHiddenWidth =
    policyExploration?.hiddenWidth ??
    (activeConfig ? hiddenLayersFromConfig(activeConfig)[0] : 128)
  const [replayStep, setReplayStep] = useState(0)
  // Live particle count — grows as growth splits (see GpuSimulation's
  // own particleCount getter), reported alongside the step by
  // GridCanvas's own onStep every rendered frame.
  const [cellCount, setCellCount] = useState(0)
  useEffect(() => {
    setReplayStep(0)
    setCellCount(0)
  }, [activeConfig?.generation])
  useEffect(() => {
    if (!activeConfig) return
    const runKey = viewingRunId ?? "current"
    if (particleCapRunRef.current === runKey) return
    particleCapRunRef.current = runKey
    const configuredCap = VIEWER_DEFAULTS.playback.particleCap
    const particleCap = Math.min(
      MAX_PARTICLES,
      Math.max(2, Math.floor(configuredCap ?? activeConfig.particles))
    )
    setFrontendParticleCap(particleCap)
    const initialCount = Math.min(
      particleCap,
      Math.max(
        1,
        Math.floor(
          VIEWER_DEFAULTS.playback.initialParticleCount ??
            activeConfig.initialParticleCount ??
            coreConstants.INITIAL_PARTICLE_COUNT
        )
      )
    )
    setFrontendInitialParticleCount(initialCount)
    setFrontendInitialParticleCountInput(String(initialCount))
  }, [viewingRunId, activeConfig?.particles])
  // null = following this generation's own trained physics/growth values;
  // non-null once either live-control panel's sliders have been touched.
  // Reset whenever the run or the generation
  // being viewed changes — an override dialed in for one generation's
  // behavior means nothing once a different one is loaded.
  const [physicsOverride, setPhysicsOverride] =
    useState<PhysicsSettings | null>(null)
  useEffect(() => {
    setPhysicsOverride(null)
  }, [viewingRunId, activeConfig?.generation])
  const trainedPhysics = playbackConfig
    ? physicsSettingsFromConfig(playbackConfig)
    : null
  const physicsValues = physicsOverride ?? trainedPhysics
  const activeStat =
    selectedGeneration !== null
      ? (history.find((h) => h.generation === selectedGeneration) ?? null)
      : history.length > 0
        ? history[history.length - 1]
        : null
  // train_server.py's /runs/{run_id}/... routes use "current" as the
  // live run's own id — viewingRunId is null for that case.
  const activeRunId = viewingRunId ?? "current"

  // Unlike the live server's own --target (fixed for its whole process
  // lifetime), an *archived* run being browsed may have been trained
  // against a completely different target — GET /targets/{name}/points
  // (not the fixed /target/points) loads any target's points by name, so
  // this re-fetches whenever the run/generation actually being viewed
  // changes rather than once on mount. Points already arrive in
  // MpmCore's own [0,1]^2 domain (targets.py's own TargetShape) — no
  // grid_size/rescaling step needed, unlike envnca's pixel-space targets.
  const [targetPoints, setTargetPoints] = useState<Float32Array | null>(null)
  useEffect(() => {
    if (!activeConfig) return
    let cancelled = false
    fetch(
      `${TRAIN_API_URL}/targets/${encodeURIComponent(activeConfig.target)}/points`
    )
      .then((res) => res.json())
      .then((data: { points: [number, number][] }) => {
        if (cancelled) return
        setTargetPoints(Float32Array.from(data.points.flat()))
      })
      .catch((err) =>
        console.error("[mpm-training] failed to fetch target points", err)
      )
    return () => {
      cancelled = true
    }
  }, [activeConfig?.target])

  // Toggles GridCanvas's own CanvasRecorder — click while idle starts
  // capturing (button reads "● REC"), click again stops and triggers
  // the video download (see canvasRecorder.ts's own CanvasRecorder.stop()
  // for exactly how). Awaits stopRecording() before flipping the button
  // back, so "● REC" stays showing through the brief encode/download
  // handoff rather than reading "Record" again before the file's
  // actually been saved.
  const handleRecordClick = async () => {
    if (recording) {
      await gridCanvasRef.current?.stopRecording()
      setRecording(false)
    } else {
      gridCanvasRef.current?.startRecording()
      setRecording(true)
    }
  }

  const handleSampleSweep = async (request: SampleSweepRequest) => {
    const canvas = gridCanvasRef.current
    if (!canvas || !physicsValues || !activeConfig || !previewConfig) return
    const combinations: Array<Partial<Record<SweepParameterKey, number>>> = []
    const visit = (
      axisIndex: number,
      values: Partial<Record<SweepParameterKey, number>>
    ) => {
      if (axisIndex === request.axes.length) {
        combinations.push({ ...values })
        return
      }
      const axis = request.axes[axisIndex]
      for (const value of sweepValues(axis)) {
        values[axis.key] = value
        visit(axisIndex + 1, values)
      }
    }
    visit(0, {})

    const controller = new AbortController()
    sampleAbortRef.current = controller
    setSampleRunning(true)
    setSampleCompleted(0)
    setSampleTotal(combinations.length)
    setSampleError(null)
    const basePhysics = { ...physicsValues }
    const keyLabel = (key: string) =>
      key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`)
    const valueLabel = (value: number) =>
      Number(value.toPrecision(12)).toString()

    try {
      const densityPhysicsKeys = [
        "particleMass",
        "particleVolume",
        "chemicalGradientInputScale",
        "chemicalProjectionWeight",
        "depositSigma",
        "splitDisplacement",
        "splatRadius",
        "repulsionStrength",
        "repulsionMaxDelta",
      ] as const satisfies readonly (keyof PhysicsSettings)[]
      const samples = combinations.map((combination) => {
        const density = combination.particleDensityMultiplier
          ?? effectiveParticleDensity
        const substrateResolution = combination.substrateResolution
          ?? effectiveSubstrateResolution
        const resolvedConfig = configAtDensity(
          { ...previewConfig, fieldN: substrateResolution },
          density,
        )
        const physics = { ...basePhysics }
        if (combination.particleDensityMultiplier !== undefined) {
          const resolvedPhysics = physicsSettingsFromConfig(resolvedConfig)
          for (const key of densityPhysicsKeys)
            physics[key] = resolvedPhysics[key]
        }
        for (const axis of request.axes) {
          if (
            axis.key !== "particleDensityMultiplier"
            && axis.key !== "substrateResolution"
          ) {
            physics[axis.key] = combination[axis.key]!
          }
        }
        return {
          config: resolvedConfig,
          physics,
          particleCap: frontendParticleCap,
          initialParticleCount: frontendInitialParticleCount,
          particleDensityMultiplier: density,
          particleRadiusPx:
            particleRadiusPx / Math.sqrt(density),
          filename: `${request.axes.map((axis) => `${keyLabel(axis.key)}=${valueLabel(combination[axis.key]!)}`).join(",")}.png`,
        }
      })
      const captures = await canvas.collectSamples(
        samples,
        request.steps,
        basePhysics,
        frontendParticleCap,
        frontendInitialParticleCount,
        particleRadiusPx,
        request.includeJson,
        setSampleCompleted,
        controller.signal
      )
      if (request.axes.length === 1 || request.axes.length === 2) {
        const [firstAxis, secondAxis] = request.axes
        captures.push({
          filename: "atlas.png",
          blob: await createSampleAtlas({
            images: captures.filter((capture) => capture.filename.endsWith(".png")),
            rows: secondAxis
              ? {
                  label: firstAxis.label,
                  values: sweepValues(firstAxis),
                }
              : undefined,
            columns: {
              label: (secondAxis ?? firstAxis).label,
              values: sweepValues(secondAxis ?? firstAxis),
            },
            signal: controller.signal,
          }),
        })
      }
      const zip = await createZip(captures)
      const generation = activeConfig?.generation ?? "unknown"
      downloadBlob(zip, `mpm-samples-generation-${generation}.zip`)
      setSampleModalOpen(false)
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setSampleError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      sampleAbortRef.current = null
      setSampleRunning(false)
    }
  }

  return (
    <div className="training-layout">
      <div className="controls">
        <h1>mpm-training viewer</h1>

        <section>
          <RunPicker
            apiUrl={TRAIN_API_URL}
            activeRunId={viewingRunId}
            onSelectRun={setViewingRunId}
          />
        </section>

        <section>
          <h2>Rollout</h2>
          <div className="stat-row">
            <span>Training particle cap</span>
            <span>{activeConfig ? activeConfig.particles : "—"}</span>
          </div>
          <div className="stat-row">
            <span>Chemical field</span>
            <span>
              {activeConfig
                ? `${activeConfig.fieldN}×${activeConfig.fieldN}`
                : "—"}
            </span>
          </div>
          <div className="stat-row">
            <span>Channels</span>
            <span>{activeConfig ? activeConfig.channels : "—"}</span>
          </div>
        </section>

        <section>
          <h2>Simulation</h2>
          <div className="stat-row">
            <span>Cell memory</span>
            <select
              className="select"
              aria-label="Cell memory"
              value={displayedCellMemory}
              disabled={!activeConfig}
              onChange={(event) =>
                setPolicyExploration({
                  cellMemory: event.target.value as CellMemory,
                  hiddenWidth: displayedHiddenWidth,
                  variant: 0,
                })
              }
            >
              <option value="none">None</option>
              <option value="recurrent">Recurrent</option>
            </select>
          </div>
          <div className="stat-row">
            <span>Hidden layer</span>
            <select
              className="select"
              aria-label="Hidden layer width"
              value={displayedHiddenWidth}
              disabled={!activeConfig}
              onChange={(event) =>
                setPolicyExploration({
                  cellMemory: displayedCellMemory,
                  hiddenWidth: Number(event.target.value),
                  variant: 0,
                })
              }
            >
              {[16, 32, 64, 128, 256].map((width) => (
                <option key={width} value={width}>
                  {width}
                </option>
              ))}
            </select>
          </div>
          <div className="stat-row">
            <span>Brain source</span>
            <button
              className="select simulation-control-button"
              disabled={!policyExploration}
              onClick={() => setPolicyExploration(null)}
              title="Restore the selected generation's trained brain"
              aria-label="Restore trained brain"
            >
              {policyExploration
                ? `Seeded random #${policyExploration.variant + 1} ↺`
                : "Trained"}
            </button>
          </div>
          <div className="stat-row">
            <span>Chemical architecture</span>
            <select
              className="select"
              aria-label="Chemical architecture"
              value={
                playbackConfig?.chemicalCommunicationArchitecture ??
                "cell-owned-projection"
              }
              disabled={!activeConfig}
              onChange={(event) => {
                const selected = event.target
                  .value as ChemicalCommunicationArchitecture
                const trained = activeConfig
                  ? chemicalCommunicationArchitectureFromConfig(activeConfig)
                  : "cell-owned-projection"
                setChemicalArchitectureOverride(
                  selected === trained ? null : selected
                )
              }}
            >
              <option value="cell-owned-projection">
                Cell-owned projection
              </option>
              <option value="persistent-environment">
                Persistent environment
              </option>
            </select>
          </div>
          <div className="stat-row">
            <span>Particle density</span>
            <select
              className="select"
              aria-label="Particle density"
              value={effectiveParticleDensity}
              disabled={!activeConfig}
              onChange={(event) => {
                const selected = Number(event.target.value)
                setParticleDensityOverride(
                  selected === defaultParticleDensity ? null : selected
                )
                setPhysicsOverride(null)
              }}
            >
              {Array.from(new Set([0.5, 1, 2, 4, effectiveParticleDensity]))
                .sort((a, b) => a - b)
                .map((density) => (
                  <option key={density} value={density}>
                    {density}×
                  </option>
                ))}
            </select>
          </div>
          <div className="stat-row">
            <span>Substrate resolution</span>
            <select
              className="select"
              aria-label="Substrate resolution"
              value={effectiveSubstrateResolution}
              disabled={!activeConfig}
              title="Changing substrate resolution rebuilds and restarts playback"
              onChange={(event) => {
                const selected = Number(event.target.value)
                setSubstrateResolutionOverride(
                  selected === defaultSubstrateResolution ? null : selected
                )
              }}
            >
              {Array.from(new Set([64, 128, 256, 512, 1024, 2048, effectiveSubstrateResolution]))
                .sort((a, b) => a - b)
                .map((resolution) => (
                  <option key={resolution} value={resolution}>
                    {resolution}×{resolution}
                  </option>
                ))}
            </select>
          </div>
          <label
            className="checkbox-row"
            title="Changing chirality rebuilds and restarts playback"
          >
            <input
              type="checkbox"
              checked={effectiveChirality}
              disabled={!activeConfig}
              onChange={(event) => {
                const trained = activeConfig?.chirality ?? true
                const selected = event.target.checked
                setChiralityOverride(selected === trained ? null : selected)
              }}
            />
            Chirality
          </label>
          <label className="slider-row">
            <span>Playback particle cap</span>
            <Slider
              min={2}
              max={MAX_PARTICLES}
              step={1}
              value={frontendParticleCap}
              disabled={!activeConfig}
              onChange={(value) => {
                const cap = Math.floor(value)
                setFrontendParticleCap(cap)
                if (frontendInitialParticleCount > cap) {
                  setFrontendInitialParticleCount(cap)
                  setFrontendInitialParticleCountInput(String(cap))
                }
              }}
            />
            <span className="slider-value playback-cap-value">
              {frontendParticleCap.toLocaleString()}
            </span>
          </label>
          <label className="slider-row">
            <span>Initial agents</span>
            <input
              className="number-input"
              type="number"
              min={1}
              max={frontendParticleCap}
              step={1}
              value={frontendInitialParticleCountInput}
              onChange={(e) => {
                setFrontendInitialParticleCountInput(e.currentTarget.value)
                const value = e.currentTarget.valueAsNumber
                if (
                  Number.isFinite(value) &&
                  value >= 1 &&
                  value <= frontendParticleCap
                ) {
                  setFrontendInitialParticleCount(Math.floor(value))
                }
              }}
              onBlur={(e) => {
                const value = e.currentTarget.valueAsNumber
                const count = Math.min(
                  frontendParticleCap,
                  Math.max(
                    1,
                    Number.isFinite(value)
                      ? Math.floor(value)
                      : frontendInitialParticleCount
                  )
                )
                setFrontendInitialParticleCountInput(String(count))
                setFrontendInitialParticleCount(count)
              }}
            />
          </label>
        </section>

        <section>
          <h2>Rendering</h2>
          <div className="slider-row">
            <span>Zoom</span>
            <label className="auto-zoom-toggle">
              <input
                type="checkbox"
                checked={autoZoomEnabled}
                onChange={(event) => setAutoZoomEnabled(event.target.checked)}
              />
              Auto
            </label>
            <Slider
              min={1}
              max={8}
              step={0.05}
              value={zoom}
              disabled={autoZoomEnabled}
              onChange={(value) => {
                setZoom(value)
                setEffectiveZoom(value)
              }}
            />
            <span className="slider-value">
              {(autoZoomEnabled ? effectiveZoom : zoom).toFixed(2)}×
            </span>
          </div>
          <details className="settings-category">
            <summary>Post-processing</summary>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={bloom.enabled}
                onChange={(event) =>
                  setBloom((value) => ({ ...value, enabled: event.target.checked }))
                }
              />
              Bloom
            </label>
            {bloom.enabled && (
              <>
                <label className="slider-row">
                  <span>Bloom intensity</span>
                  <Slider
                    min={0}
                    max={3}
                    step={0.05}
                    value={bloom.intensity}
                    onChange={(intensity) => setBloom((value) => ({ ...value, intensity }))}
                  />
                  <span className="slider-value">{bloom.intensity.toFixed(2)}</span>
                </label>
                <label className="slider-row">
                  <span>Bloom threshold</span>
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={bloom.threshold}
                    onChange={(threshold) => setBloom((value) => ({ ...value, threshold }))}
                  />
                  <span className="slider-value">{bloom.threshold.toFixed(2)}</span>
                </label>
                <label className="slider-row">
                  <span>Bloom radius</span>
                  <Slider
                    min={0.25}
                    max={8}
                    step={0.25}
                    value={bloom.radiusPx}
                    onChange={(radiusPx) => setBloom((value) => ({ ...value, radiusPx }))}
                  />
                  <span className="slider-value">{bloom.radiusPx.toFixed(2)}px</span>
                </label>
                <label className="slider-row">
                  <span>Bloom levels</span>
                  <Slider
                    min={2}
                    max={10}
                    step={1}
                    value={bloom.levels}
                    onChange={(levels) => setBloom((value) => ({ ...value, levels }))}
                  />
                  <span className="slider-value">{bloom.levels}</span>
                </label>
                <label className="slider-row">
                  <span>Bloom scatter</span>
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={bloom.scatter}
                    onChange={(scatter) => setBloom((value) => ({ ...value, scatter }))}
                  />
                  <span className="slider-value">{bloom.scatter.toFixed(2)}</span>
                </label>
              </>
            )}
          </details>
          <label className="slider-row">
            <span>Particle size</span>
            <Slider
              min={1}
              max={16}
              step={1}
              value={particleRadiusPx}
              onChange={setParticleRadiusPx}
            />
            <span className="slider-value">{particleRadiusPx}px</span>
          </label>
          <label className="slider-row">
            <span>Particles</span>
            <select
              className="select"
              value={particleRenderMode}
              onChange={(e) =>
                setParticleRenderMode(e.target.value as ParticleRenderMode)
              }
            >
              <option value="dots-white">Dots (white)</option>
              <option value="dots-neural-color">Dots (neural RGB)</option>
              <option value="dots-internal-state">Chemical memory</option>
              <option value="dots-chemical-levels">Chemical levels</option>
              <option value="dots-boundary-value">Boundary value</option>
              <option value="dots-activation">Dots (neurons)</option>
              <option value="dots-activation-translucent">
                Dots (translucent neurons)
              </option>
              <option value="directional-arrows">Directional triangles</option>
            </select>
          </label>
          {particleRenderMode === "dots-white" && (
            <label className="slider-row">
              <span>Dots alpha</span>
              <Slider
                min={0}
                max={1}
                step={0.01}
                value={whiteDotsAlpha}
                onChange={setWhiteDotsAlpha}
              />
              <span className="slider-value">{whiteDotsAlpha.toFixed(2)}</span>
            </label>
          )}
          {particleRenderMode === "dots-neural-color" && (
            <label className="slider-row">
              <span>Neural RGB alpha</span>
              <Slider
                min={0}
                max={1}
                step={0.01}
                value={neuralColorAlpha}
                onChange={setNeuralColorAlpha}
              />
              <span className="slider-value">
                {neuralColorAlpha.toFixed(2)}
              </span>
            </label>
          )}
          {(particleRenderMode === "dots-internal-state" ||
            particleRenderMode === "dots-chemical-levels") && (
            <>
              <label className="slider-row">
                <span>
                  {particleRenderMode === "dots-internal-state"
                    ? "Chemical memory alpha"
                    : "Chemical levels alpha"}
                </span>
                <Slider
                  min={0}
                  max={1}
                  step={0.01}
                  value={internalStateAlpha}
                  onChange={setInternalStateAlpha}
                />
                <span className="slider-value">
                  {internalStateAlpha.toFixed(2)}
                </span>
              </label>
              <div className="channel-window-control">
                <div className="channel-window-label">
                  <span>Channels</span>
                  <span>
                    {internalStateChannelStart}–{internalStateChannelStart + 2}
                  </span>
                </div>
                <ChannelWindowSlider
                  channels={
                    particleRenderMode === "dots-internal-state"
                      ? 8
                      : (activeConfig?.channels ?? 8)
                  }
                  value={internalStateChannelStart}
                  onChange={setInternalStateChannelStart}
                  channelKind={
                    particleRenderMode === "dots-internal-state"
                      ? "chemical memory"
                      : "chemical levels"
                  }
                />
              </div>
              {particleRenderMode === "dots-internal-state" && (
                <label className="slider-row">
                  <span>Opponent subtraction</span>
                  <Slider
                    min={0}
                    max={1}
                    step={0.01}
                    value={chemicalMemoryOpponentSubtraction}
                    onChange={setChemicalMemoryOpponentSubtraction}
                  />
                  <span className="slider-value">
                    {chemicalMemoryOpponentSubtraction.toFixed(2)}
                  </span>
                </label>
              )}
              {particleRenderMode === "dots-internal-state" &&
                previewConfig &&
                cellMemoryFromConfig(previewConfig) !== "recurrent" && (
                  <p className="hint">
                    Cell memory is disabled, so these channels remain zero.
                  </p>
                )}
              {particleRenderMode === "dots-chemical-levels" &&
                activeConfig &&
                chemicalCommunicationArchitectureFromConfig(
                  previewConfig ?? activeConfig
                ) !== "cell-owned-projection" && (
                  <p className="hint">
                    Chemical levels are inactive in persistent-environment mode.
                  </p>
                )}
            </>
          )}
          {particleRenderMode === "dots-boundary-value" && (
            <>
              <label className="slider-row">
                <span>Boundary g0</span>
                <Slider
                  min={0.001}
                  max={0.1}
                  step={0.001}
                  value={boundaryGradientScale}
                  onChange={setBoundaryGradientScale}
                />
                <span className="slider-value">
                  {boundaryGradientScale.toFixed(3)}
                </span>
              </label>
              <p className="hint">
                Cell color shows |∇ρ|/(|∇ρ|+g0): dark blue is interior, teal is
                0.5, and yellow is a strong boundary.
              </p>
            </>
          )}
          {particleRenderMode === "dots-activation-translucent" && (
            <label className="slider-row">
              <span>Activation alpha</span>
              <Slider
                min={0}
                max={1}
                step={0.01}
                value={activationAlpha}
                onChange={setActivationAlpha}
              />
              <span className="slider-value">{activationAlpha.toFixed(2)}</span>
            </label>
          )}
          {particleRenderMode === "directional-arrows" && (
            <>
              <label className="slider-row">
                <span>Triangle size</span>
                <Slider
                  min={1}
                  max={20}
                  step={1}
                  value={growthAxisLengthPx}
                  onChange={setGrowthAxisLengthPx}
                />
                <span className="slider-value">{growthAxisLengthPx}px</span>
              </label>
              <p className="hint">
                Compact white triangles point toward +n division polarity;
                size and opacity show signal strength.
              </p>
            </>
          )}
          <label className="slider-row">
            <span>Background</span>
            <select
              className="select"
              value={fieldMode}
              onChange={(e) => setFieldMode(e.target.value as FieldMode)}
            >
              <option value="none">None</option>
              <option value="density">Density</option>
              <option value="speed">Speed</option>
              <option value="deformation">Deformation</option>
              <option value="pressure">Pressure</option>
              <option value="shear">Shear</option>
              <option value="repulsion">Repulsion field</option>
              <option value="morphology">
                Policy morphology (gradient + density)
              </option>
              <option value="substrate">Substrate</option>
              <option value="growth">Growth (cividis)</option>
              <option value="gradient">Boundary gradient</option>
            </select>
          </label>
          {fieldMode === "morphology" && (
            <>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={morphologyGradientVisible}
                  onChange={(e) =>
                    setMorphologyGradientVisible(e.target.checked)
                  }
                />
                Show morphology gradient (R/G)
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={morphologyDensityVisible}
                  onChange={(e) =>
                    setMorphologyDensityVisible(e.target.checked)
                  }
                />
                Show morphology density (B)
              </label>
            </>
          )}
          {fieldMode === "substrate" && activeConfig && (
            <div className="channel-window-control">
              <div className="channel-window-label">
                <span>RGB channels</span>
                <span>
                  {substrateChannelStart}–
                  {Math.min(
                    activeConfig.channels - 1,
                    substrateChannelStart + 2
                  )}
                </span>
              </div>
              <ChannelWindowSlider
                channels={activeConfig.channels}
                value={substrateChannelStart}
                onChange={setSubstrateChannelStart}
              />
            </div>
          )}
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={targetVisible}
              onChange={(e) => setTargetVisible(e.target.checked)}
            />
            Show training target
          </label>
          <label className="slider-row">
            <span>Accent</span>
            <Slider
              min={-2}
              max={2}
              step={0.01}
              value={accent}
              onChange={setAccent}
            />
            <span className="slider-value">{accent.toFixed(2)}</span>
          </label>
          {/* Blur/Gradient exponent only drive the "gradient" (Boundary
              gradient) background mode's own blur+colorize passes — see
              gpu/render.ts's own setBlur()/setGradientExponent()
              comments — so they'd do nothing under any other mode;
              hidden rather than shown-but-inert. */}
          {fieldMode === "gradient" && (
            <>
              <label className="slider-row">
                <span>Blur</span>
                <Slider
                  min={0}
                  max={2}
                  step={0.01}
                  value={blur}
                  onChange={setBlur}
                />
                <span className="slider-value">{blur.toFixed(2)}</span>
              </label>
              <label className="slider-row">
                <span>Gradient exponent</span>
                <Slider
                  min={0.25}
                  max={4}
                  step={0.05}
                  value={gradientExponent}
                  onChange={setGradientExponent}
                />
                <span className="slider-value">
                  {gradientExponent.toFixed(2)}
                </span>
              </label>
            </>
          )}
        </section>

        {trainedPhysics && physicsValues && (
          <>
            <PhysicsPanel
              trained={trainedPhysics}
              value={physicsValues}
              onChange={setPhysicsOverride}
              isOverridden={physicsOverride !== null}
              onReset={() => setPhysicsOverride(null)}
            />
            <GrowthPanel
              trained={trainedPhysics}
              value={physicsValues}
              onChange={setPhysicsOverride}
              isOverridden={physicsOverride !== null}
              onReset={() => setPhysicsOverride(null)}
            />
          </>
        )}
      </div>
      <div className="center-column">
        <div className="viewport">
          <GridCanvas
            ref={gridCanvasRef}
            config={previewConfig}
            targetPoints={targetPoints}
            targetVisible={targetVisible}
            physics={physicsValues}
            particleCap={frontendParticleCap}
            initialParticleCount={frontendInitialParticleCount}
            fieldMode={fieldMode}
            substrateChannelStart={substrateChannelStart}
            accent={accent}
            morphologyGradientVisible={morphologyGradientVisible}
            morphologyDensityVisible={morphologyDensityVisible}
            blur={blur}
            gradientExponent={gradientExponent}
            particleRenderMode={particleRenderMode}
            zoom={zoom}
            autoZoom={autoZoomSettings}
            onEffectiveZoomChange={setEffectiveZoom}
            bloom={bloom}
            particleRadiusPx={particleRadiusPx}
            whiteDotsAlpha={whiteDotsAlpha}
            activationAlpha={activationAlpha}
            neuralColorAlpha={neuralColorAlpha}
            internalStateAlpha={internalStateAlpha}
            boundaryGradientScale={boundaryGradientScale}
            internalStateChannelStart={internalStateChannelStart}
            chemicalMemoryOpponentSubtraction={chemicalMemoryOpponentSubtraction}
            growthAxisLengthPx={growthAxisLengthPx}
            tool={tool}
            deformSettings={deformSettings}
            onStep={(step, particles) => {
              setReplayStep(step)
              setCellCount(particles)
            }}
            loopAtTrainedSteps={loopAtTrainedSteps}
            paused={paused}
          />
          <div className="viewport-telemetry" aria-label="Rollout status">
            <span>
              {activeConfig
                ? `${replayStep} / ${activeConfig.macroSteps} steps`
                : "— steps"}
            </span>
            {/* Live count, not the cap — grows as growth splits. */}
            <span>
              {activeConfig
                ? `${cellCount} / ${frontendParticleCap} cells`
                : "— cells"}
            </span>
          </div>
        </div>
        <div className="toolbar">
          <div className="tool-buttons-wrap">
            {/* The active tool's own contextual settings — pops up
                directly above the tool-selector buttons instead of
                living in the left sidebar, so it stays visually
                attached to the tool it belongs to. Renders nothing
                while no tool is active. */}
            {tool !== "none" && (
              <div className="tool-settings-panel">
                <h3>
                  {tool === "add" ? "Add" : tool === "move" ? "Move" : "Deform"}
                </h3>
                {tool !== "deform" && (
                  <p className="hint">
                    {tool === "add"
                      ? "Click the sim to add a particle."
                      : "Drag a particle to move it."}
                  </p>
                )}
                {tool === "deform" && (
                  <>
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={deformSettings.direction === "outward"}
                        onChange={(e) =>
                          setDeformSettings((s) => ({
                            ...s,
                            direction: e.target.checked ? "outward" : "inward",
                          }))
                        }
                      />
                      {deformSettings.direction === "outward"
                        ? "Push outward (explode)"
                        : "Pull inward (implode)"}
                    </label>
                    <label className="slider-row">
                      <span>Strength</span>
                      <Slider
                        min={0}
                        max={2}
                        step={0.01}
                        value={deformSettings.strength}
                        onChange={(v) =>
                          setDeformSettings((s) => ({ ...s, strength: v }))
                        }
                      />
                      <span className="slider-value">
                        {deformSettings.strength.toFixed(2)}
                      </span>
                    </label>
                    <label className="slider-row">
                      <span>Radius</span>
                      <Slider
                        min={0.01}
                        max={0.5}
                        step={0.01}
                        value={deformSettings.radius}
                        onChange={(v) =>
                          setDeformSettings((s) => ({ ...s, radius: v }))
                        }
                      />
                      <span className="slider-value">
                        {deformSettings.radius.toFixed(2)}
                      </span>
                    </label>
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={deformSettings.mode === "deformation"}
                        onChange={(e) =>
                          setDeformSettings((s) => ({
                            ...s,
                            mode: e.target.checked ? "deformation" : "velocity",
                          }))
                        }
                      />
                      Direct deformation (F) edit
                    </label>
                  </>
                )}
              </div>
            )}
            <div className="tool-buttons">
              <button
                className={`icon-button${tool === "add" ? " is-active" : ""}`}
                onClick={() => setTool((t) => (t === "add" ? "none" : "add"))}
                title="Add particle — click the sim to place one"
                aria-label="Add particle"
                aria-pressed={tool === "add"}
              >
                ＋
              </button>
              <button
                className={`icon-button${tool === "move" ? " is-active" : ""}`}
                onClick={() => setTool((t) => (t === "move" ? "none" : "move"))}
                title="Move particles — drag one in the sim"
                aria-label="Move particles"
                aria-pressed={tool === "move"}
              >
                ✥
              </button>
              <button
                className={`icon-button${tool === "deform" ? " is-active" : ""}`}
                onClick={() =>
                  setTool((t) => (t === "deform" ? "none" : "deform"))
                }
                title="Deform — click the sim to inject a directional deformation"
                aria-label="Deform"
                aria-pressed={tool === "deform"}
              >
                ⤢
              </button>
            </div>
          </div>
          <div className="toolbar-actions">
            <label
              className="toolbar-checkbox"
              title={
                loopAtTrainedSteps
                  ? "Restart with a fresh rollout once it reaches the step count it was trained/scored at"
                  : `Running past step ${activeConfig?.macroSteps ?? "—"} — the horizon it was scored at`
              }
            >
              <input
                type="checkbox"
                checked={loopAtTrainedSteps}
                onChange={(e) => setLoopAtTrainedSteps(e.target.checked)}
              />
              Loop
            </label>
            <button
              className="icon-button"
              onClick={() => {
                setSampleError(null)
                setSampleModalOpen(true)
              }}
              disabled={
                !activeConfig || !physicsValues || sampleRunning || recording
              }
              title="Collect a matrix of parameter-sweep screenshots"
              aria-label="Collect parameter samples"
            >
              ◫
            </button>
            <button
              className="icon-button"
              onClick={() => {
                if (!activeConfig) return
                setPolicyExploration((current) => ({
                  cellMemory: displayedCellMemory,
                  hiddenWidth: displayedHiddenWidth,
                  variant:
                    current &&
                    current.cellMemory === displayedCellMemory &&
                    current.hiddenWidth === displayedHiddenWidth
                      ? current.variant + 1
                      : 0,
                }))
              }}
              disabled={!activeConfig}
              title="Advance to the next deterministic random brain derived from the active rollout seed"
              aria-label="Load seeded random brain"
            >
              🎲
            </button>
            <button
              className="icon-button"
              onClick={() => setPaused((p) => !p)}
              title={paused ? "Play" : "Pause"}
              aria-label={paused ? "Play" : "Pause"}
            >
              {paused ? "▶" : "⏸"}
            </button>
            <button
              className="icon-button"
              onClick={() => gridCanvasRef.current?.restart()}
              title="Restart"
              aria-label="Restart"
            >
              ↺
            </button>
            <button
              className={`icon-button${recording ? " is-recording" : ""}`}
              onClick={handleRecordClick}
              disabled={!RECORDING_FORMAT}
              title={
                !RECORDING_FORMAT
                  ? "This browser can't record video (MediaRecorder unsupported)"
                  : recording
                    ? "Stop recording"
                    : `Record — saves as .${RECORDING_FORMAT.ext}`
              }
              aria-label={recording ? "Stop recording" : "Record"}
            >
              {recording ? "●" : "⏺"}
            </button>
          </div>
        </div>
        <div className="training-timeline">
          <FitnessChart
            history={history}
            selectedGeneration={selectedGeneration}
            onSelectGeneration={setSelectedGeneration}
            getPreviewImageUrl={(generation) =>
              generationImageUrl(
                TRAIN_API_URL,
                activeRunId,
                generation,
                "agents"
              )
            }
          />
        </div>
      </div>
      <div className="controls-right">
        <section>
          <h2>Stats</h2>
          <div className="stat-row">
            <span>Generation</span>
            <span>{activeStat ? activeStat.generation : "—"}</span>
          </div>
          <div className="stat-row">
            <span>Best (this gen)</span>
            <span>{activeStat ? activeStat.best.toFixed(3) : "—"}</span>
          </div>
          <div className="stat-row">
            <span>Mean (this gen)</span>
            <span>{activeStat ? activeStat.mean.toFixed(3) : "—"}</span>
          </div>
          <div className="stat-row">
            <span>Worst (this gen)</span>
            <span>{activeStat ? activeStat.worst.toFixed(3) : "—"}</span>
          </div>
        </section>

        <section>
          <h2>Snapshot</h2>
          {activeStat ? (
            <div className="snapshot-grid">
              {/* Target/agents come first, side by side — the pair
                  meant to be checked directly against each other (see
                  debug_images.py's own module docstring: "agents" is
                  rasterized at literally the same pose
                  raster.training_raster_distance() scored it under, so
                  it lines up pixel-for-pixel with "target"). Grown
                  (raw, un-aligned positions) is context below, spanning
                  the full row — see .snapshot-item-wide. */}
              <div className="snapshot-item">
                <img
                  className="snapshot-image"
                  src={generationImageUrl(
                    TRAIN_API_URL,
                    activeRunId,
                    activeStat.generation,
                    "target"
                  )}
                  alt="Target raster this run is training against"
                />
                <span className="snapshot-label">Target</span>
              </div>
              <div className="snapshot-item">
                <img
                  className="snapshot-image"
                  src={generationImageUrl(
                    TRAIN_API_URL,
                    activeRunId,
                    activeStat.generation,
                    "agents"
                  )}
                  alt="Winning rollout, rasterized at its best-scoring pose"
                />
                <span className="snapshot-label">Agents (aligned)</span>
              </div>
              <div className="snapshot-item snapshot-item-wide">
                <img
                  className="snapshot-image"
                  src={generationImageUrl(
                    TRAIN_API_URL,
                    activeRunId,
                    activeStat.generation,
                    "grown"
                  )}
                  alt="Winning rollout, raw final positions"
                />
                <span className="snapshot-label">Grown (raw)</span>
              </div>
            </div>
          ) : (
            <p className="hint">No generation selected yet.</p>
          )}
        </section>

        <NetworkPanel config={previewConfig} physics={physicsValues} />
      </div>
      {sampleModalOpen && physicsValues && (
        <SampleSweepModal
          current={physicsValues}
          currentDensity={effectiveParticleDensity}
          currentSubstrateResolution={effectiveSubstrateResolution}
          defaultSteps={activeConfig?.macroSteps ?? 1}
          running={sampleRunning}
          completed={sampleCompleted}
          total={sampleTotal}
          error={sampleError}
          onClose={() => setSampleModalOpen(false)}
          onRun={handleSampleSweep}
          onCancel={() => sampleAbortRef.current?.abort()}
        />
      )}
    </div>
  )
}

import { useEffect, useRef, useState } from "react"
import { FitnessChart } from "./charts/FitnessChart"
import { MAX_PARTICLES } from "./gpu/mpmCore"
import type { FieldMode, ParticleRenderMode } from "./gpu/render"
import type { PhysicsSettings } from "./gpu/types"
import { physicsSettingsFromConfig } from "./gpu/types"
import { generationImageUrl } from "./net/images"
import { fetchRunState } from "./net/runs"
import type { TrainingSocketState } from "./net/trainingSocket"
import { EMPTY_STATE, useTrainingSocket } from "./net/trainingSocket"
import { pickRecordingFormat } from "./render/canvasRecorder"
import type {
  DeformSettings,
  GridCanvasHandle,
  Tool,
} from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"
import { GrowthPanel } from "./ui/GrowthPanel"
import { NetworkPanel } from "./ui/NetworkPanel"
import { PhysicsPanel } from "./ui/PhysicsPanel"
import { RunPicker } from "./ui/RunPicker"
import { Slider } from "./ui/Slider"

const TRAIN_API_URL = "http://localhost:8003"
const TRAIN_WS_URL = "ws://localhost:8003/ws"

// A pure browser feature-check (no canvas/mount needed — see
// pickRecordingFormat()'s own docstring), so it's computed once here
// rather than round-tripped through GridCanvasHandle every render.
const RECORDING_FORMAT = pickRecordingFormat()

/** Passive training viewer — a live WebGPU replay of whichever
 * generation's weights are selected, plus a fitness-history timeline.
 * Same overall role as envnca/frontend/src/TrainingView.tsx. No spawn-
 * distribution toggle (particles always seed as a single jittered blob,
 * matching training_sim.py's own seed_blob() exactly — there's nothing
 * else to toggle between); the "Rendering" section below otherwise
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
  const [loopAtTrainedSteps, setLoopAtTrainedSteps] = useState(true)
  const [paused, setPaused] = useState(false)
  // Whether the Record button currently reads "● REC" — GridCanvas's own
  // CanvasRecorder (render/canvasRecorder.ts) owns the actual
  // MediaRecorder/download lifecycle, this is just the button's own
  // display state; toggled on click (see handleRecordClick below), not
  // by GridCanvas ever calling back in (recording never stops on its
  // own mid-session the way, say, a rollout-length limit might).
  const [recording, setRecording] = useState(false)
  const gridCanvasRef = useRef<GridCanvasHandle>(null)

  // View-only rendering options (gpu/render.ts) — not simulation state,
  // so plain component state, never reset by a run/generation change.
  const [fieldMode, setFieldMode] = useState<FieldMode>("none")
  const [morphologyGradientVisible, setMorphologyGradientVisible] = useState(true)
  const [morphologyDensityVisible, setMorphologyDensityVisible] = useState(true)
  // [-2,2] exponential background contrast. Negative suppresses submaximal
  // field values, 0 is identity, positive accentuates faint values.
  const [accent, setAccent] = useState(0)
  // [0,2] — Gaussian sigma, in repulsion-field texels, for the "gradient"
  // mode's own blur pass (see gpu/render.ts's own setBlur()/field.wgsl's
  // own blurDensity() comment for why: raw per-particle density is too
  // grainy for a clean shape-boundary gradient). 0 = no blur, unchanged
  // from before this knob existed; only read by that one background mode.
  const [blur, setBlur] = useState(0)
  // [~0.25,4] — power curve on the "gradient" mode's own gradient
  // MAGNITUDE (direction preserved — see gpu/render.ts's own
  // setGradientExponent()/field.wgsl's own colorizeGradient() comment).
  // 1 = identity, unchanged from before this knob existed; only read by
  // that one background mode.
  const [gradientExponent, setGradientExponent] = useState(1)
  const [particleRenderMode, setParticleRenderMode] =
    useState<ParticleRenderMode>("dots-white")
  const [particleRadiusPx, setParticleRadiusPx] = useState(4)
  const [frontendParticleCap, setFrontendParticleCap] = useState(2)
  const [frontendParticleCapInput, setFrontendParticleCapInput] = useState("2")
  const particleCapRunRef = useRef<string | null>(null)
  const [targetVisible, setTargetVisible] = useState(true)
  const [whiteDotsAlpha, setWhiteDotsAlpha] = useState(1)
  const [activationAlpha, setActivationAlpha] = useState(0.2)
  const [neuralColorAlpha, setNeuralColorAlpha] = useState(1)
  const [growthAxisLengthPx, setGrowthAxisLengthPx] = useState(28)
  // "Add"/"Move"/"Deform" interaction tools (render/GridCanvas.tsx's own
  // Tool type) — toggled on/off by clicking their own icon button again
  // (see the Tools section below), not reset by a run/generation change
  // either, same reasoning as the rendering options above.
  const [tool, setTool] = useState<Tool>("none")
  // "Deform" tool's own live settings (direction/strength/radius/mode) —
  // owned here (this component's own small panel below), read by
  // GridCanvas at click/hover time (see that component's own
  // DeformSettings docstring). direction/strength/radius/mode defaults
  // mirror gpu/deform.wgsl's own starting-guess scale comments.
  const [deformSettings, setDeformSettings] = useState<DeformSettings>({
    direction: "outward",
    strength: 1,
    radius: 0.08,
    mode: "velocity",
  })

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. configByGeneration and history are
  // evicted in lockstep (see net/trainingSocket.ts), so any generation
  // number that still appears on the chart is guaranteed to resolve here.
  const activeConfig =
    selectedGeneration !== null
      ? (configByGeneration.get(selectedGeneration) ?? latest)
      : latest
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
    setFrontendParticleCap(activeConfig.particles)
    setFrontendParticleCapInput(String(activeConfig.particles))
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
  const trainedPhysics = activeConfig
    ? physicsSettingsFromConfig(activeConfig)
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
            <span>Step (replay)</span>
            <span>
              {activeConfig
                ? `${replayStep} / ${activeConfig.macroSteps}`
                : "—"}
            </span>
          </div>
          {/* Live count, not the cap — grows as growth splits. */}
          <div className="stat-row">
            <span>Cells</span>
            <span>
              {activeConfig ? `${cellCount} / ${frontendParticleCap}` : "—"}
            </span>
          </div>
          <div className="stat-row">
            <span>All-time best</span>
            <span>{activeStat ? activeStat.allTimeBest.toFixed(3) : "—"}</span>
          </div>
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
          <h2>Rendering</h2>
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
            <span>Playback particle cap</span>
            <input
              className="number-input"
              type="number"
              min={2}
              max={MAX_PARTICLES}
              step={1}
              value={frontendParticleCapInput}
              onChange={(e) => {
                setFrontendParticleCapInput(e.currentTarget.value)
                const value = e.currentTarget.valueAsNumber
                if (Number.isFinite(value) && value >= 2 && value <= MAX_PARTICLES) {
                  setFrontendParticleCap(Math.floor(value))
                }
              }}
              onBlur={(e) => {
                const value = e.currentTarget.valueAsNumber
                const cap = Math.min(
                  MAX_PARTICLES,
                  Math.max(2, Number.isFinite(value) ? Math.floor(value) : frontendParticleCap)
                )
                setFrontendParticleCapInput(String(cap))
                setFrontendParticleCap(cap)
              }}
            />
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
              <span className="slider-value">{neuralColorAlpha.toFixed(2)}</span>
            </label>
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
                  min={8}
                  max={80}
                  step={1}
                  value={growthAxisLengthPx}
                  onChange={setGrowthAxisLengthPx}
                />
                <span className="slider-value">{growthAxisLengthPx}px</span>
              </label>
              <p className="hint">
                X-squashed cyan triangles point toward +n division polarity;
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
              <option value="morphology">Policy morphology (gradient + density)</option>
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
                  onChange={(e) => setMorphologyGradientVisible(e.target.checked)}
                />
                Show morphology gradient (R/G)
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={morphologyDensityVisible}
                  onChange={(e) => setMorphologyDensityVisible(e.target.checked)}
                />
                Show morphology density (B)
              </label>
              <p className="hint">
                R/G encode the signed world-space density gradient (0.5 is
                zero); B is the blurred, normalized density. These are the
                exact quantities sampled by the policy before heading rotation.
              </p>
            </>
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
            config={activeConfig}
            targetPoints={targetPoints}
            targetVisible={targetVisible}
            physics={physicsValues}
            particleCap={frontendParticleCap}
            fieldMode={fieldMode}
            accent={accent}
            morphologyGradientVisible={morphologyGradientVisible}
            morphologyDensityVisible={morphologyDensityVisible}
            blur={blur}
            gradientExponent={gradientExponent}
            particleRenderMode={particleRenderMode}
            particleRadiusPx={particleRadiusPx}
            whiteDotsAlpha={whiteDotsAlpha}
            activationAlpha={activationAlpha}
            neuralColorAlpha={neuralColorAlpha}
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
              onClick={() => gridCanvasRef.current?.randomizeWeights()}
              disabled={!activeConfig}
              title="Randomize weights — replace the update rule's weights/biases with a fresh random init and restart the rollout under it, until a new one loads or you switch generations"
              aria-label="Randomize weights"
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

        <NetworkPanel config={activeConfig} physics={physicsValues} />
      </div>
    </div>
  )
}

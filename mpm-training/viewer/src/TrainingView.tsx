import { useEffect, useRef, useState } from "react"
import { FitnessChart } from "./charts/FitnessChart"
import type { NetworkProbe } from "./gpu/nnProbe"
import type { FieldMode, ParticleShape } from "./gpu/render"
import type { PhysicsSettings } from "./gpu/types"
import { physicsSettingsFromConfig } from "./gpu/types"
import { generationImageUrl } from "./net/images"
import { fetchRunState } from "./net/runs"
import type { TrainingSocketState } from "./net/trainingSocket"
import { EMPTY_STATE, useTrainingSocket } from "./net/trainingSocket"
import { pickRecordingFormat } from "./render/canvasRecorder"
import type { GridCanvasHandle, Tool } from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"
import { NetworkPanel } from "./ui/NetworkPanel"
import { PhysicsPanel } from "./ui/PhysicsPanel"
import { RunPicker } from "./ui/RunPicker"

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
  // [0,2] — exponentially accentuates whichever background mode is
  // active (see gpu/render.ts's own setAccent()/field.wgsl's own accent
  // uniform comment for the exact curve). 0 = identity, every mode
  // renders exactly as it did before this knob existed.
  const [accent, setAccent] = useState(0)
  const [particleShape, setParticleShape] = useState<ParticleShape>("triangle")
  const [particleRadiusPx, setParticleRadiusPx] = useState(4)
  // "Add"/"Move" interaction tools (render/GridCanvas.tsx's own Tool
  // type) — toggled on/off by clicking their own icon button again (see
  // the Tools section below), not reset by a run/generation change
  // either, same reasoning as the rendering options above.
  const [tool, setTool] = useState<Tool>("none")

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. configByGeneration and history are
  // evicted in lockstep (see net/trainingSocket.ts), so any generation
  // number that still appears on the chart is guaranteed to resolve here.
  const activeConfig =
    selectedGeneration !== null
      ? (configByGeneration.get(selectedGeneration) ?? latest)
      : latest
  const [replayStep, setReplayStep] = useState(0)
  useEffect(() => {
    setReplayStep(0)
  }, [activeConfig?.generation])
  // null = following this generation's own trained gravity/decay/
  // maxAccel/maxStrafe/maxEnvWrite; non-null once the "Physics" panel's
  // sliders have been touched. Reset whenever the run or the generation
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
  // Network panel's own live forward-pass snapshot (gpu/nnProbe.ts,
  // refreshed on GridCanvas's own timer — see PROBE_INTERVAL_MS there).
  // Reset alongside physicsOverride/targetPoints whenever the run/
  // generation being viewed changes — a stale probe from a DIFFERENT
  // config could carry the wrong channels/hiddenDim shape, and even a
  // same-shape one is just a snapshot of some other rollout's own state.
  const [probe, setProbe] = useState<NetworkProbe | null>(null)
  useEffect(() => {
    setProbe(null)
  }, [viewingRunId, activeConfig?.generation])
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
      <div className="training-main">
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
            <div className="stat-row">
              <span>All-time best</span>
              <span>
                {activeStat ? activeStat.allTimeBest.toFixed(3) : "—"}
              </span>
            </div>
            <div className="stat-row">
              <span>Max particles</span>
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
            <h2>Tools</h2>
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
            </div>
            {tool !== "none" && (
              <p className="hint">
                {tool === "add"
                  ? "Click the sim to add a particle."
                  : "Drag a particle to move it."}
              </p>
            )}
          </section>

          {trainedPhysics && physicsValues && (
            <PhysicsPanel
              trained={trainedPhysics}
              value={physicsValues}
              onChange={setPhysicsOverride}
              isOverridden={physicsOverride !== null}
              onReset={() => setPhysicsOverride(null)}
            />
          )}

          <section>
            <h2>Rendering</h2>
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
                <option value="substrate">Substrate</option>
                <option value="growth">Growth (cividis)</option>
              </select>
            </label>
            <label className="slider-row">
              <span>Accent</span>
              <input
                type="range"
                min={0}
                max={2}
                step={0.01}
                value={accent}
                onChange={(e) => setAccent(Number(e.target.value))}
              />
              <span className="slider-value">{accent.toFixed(2)}</span>
            </label>
            <label className="slider-row">
              <span>Particle size</span>
              <input
                type="range"
                min={1}
                max={16}
                step={1}
                value={particleRadiusPx}
                onChange={(e) => setParticleRadiusPx(Number(e.target.value))}
              />
              <span className="slider-value">{particleRadiusPx}px</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={particleShape === "triangle"}
                onChange={(e) =>
                  setParticleShape(e.target.checked ? "triangle" : "circle")
                }
              />
              Point particles toward heading
            </label>
          </section>

          <section>
            <h2>Playback</h2>
            <div className="playback-buttons">
              <button
                className="playback-button"
                onClick={() => setPaused((p) => !p)}
              >
                {paused ? "▶ Play" : "⏸ Pause"}
              </button>
              <button
                className="playback-button"
                onClick={() => gridCanvasRef.current?.restart()}
              >
                ↺ Restart
              </button>
              <button
                className={
                  "playback-button" + (recording ? " is-recording" : "")
                }
                onClick={handleRecordClick}
                disabled={!RECORDING_FORMAT}
                title={
                  RECORDING_FORMAT
                    ? undefined
                    : "This browser can't record video (MediaRecorder unsupported)"
                }
              >
                {recording ? "● REC" : "⏺ Record"}
              </button>
            </div>
            {RECORDING_FORMAT && RECORDING_FORMAT.ext !== "mp4" && (
              <p className="hint">
                Recording saves as .{RECORDING_FORMAT.ext} — this browser can't
                encode MP4 directly.
              </p>
            )}
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={loopAtTrainedSteps}
                onChange={(e) => setLoopAtTrainedSteps(e.target.checked)}
              />
              Loop at trained step count
            </label>
            {!loopAtTrainedSteps && (
              <p className="hint">
                Running past step {activeConfig?.macroSteps ?? "—"} — the
                horizon it was scored at.
              </p>
            )}
          </section>

          <section>
            <h2>Update rule</h2>
            <button
              className="playback-button"
              onClick={() => gridCanvasRef.current?.randomizeWeights()}
              disabled={!activeConfig}
              title="Replace the update rule's weights/biases with a fresh random init and restart the rollout under it"
            >
              🎲 Randomize weights
            </button>
            <p className="hint">
              Overrides this generation's trained weights until a new one loads
              or you switch generations.
            </p>
          </section>
        </div>
        <div className="viewport">
          <GridCanvas
            ref={gridCanvasRef}
            config={activeConfig}
            targetPoints={targetPoints}
            physics={physicsValues}
            fieldMode={fieldMode}
            accent={accent}
            particleShape={particleShape}
            particleRadiusPx={particleRadiusPx}
            tool={tool}
            onStep={setReplayStep}
            onProbe={setProbe}
            loopAtTrainedSteps={loopAtTrainedSteps}
            paused={paused}
          />
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

          <NetworkPanel probe={probe} physics={physicsValues} />
        </div>
      </div>
      <div className="training-timeline">
        <FitnessChart
          history={history}
          selectedGeneration={selectedGeneration}
          onSelectGeneration={setSelectedGeneration}
          getPreviewImageUrl={(generation) =>
            generationImageUrl(TRAIN_API_URL, activeRunId, generation, "agents")
          }
        />
      </div>
    </div>
  )
}

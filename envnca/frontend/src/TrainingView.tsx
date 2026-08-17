import { useEffect, useRef, useState } from "react"
import { FitnessChart } from "./charts/FitnessChart"
import { DEFAULT_INTENSITY } from "./gpu/render"
import type { SpawnDistribution } from "./gpu/rng"
import type { BackgroundMode, PhysicsSettings } from "./gpu/types"
import { physicsSettingsFromConfig } from "./gpu/types"
import { generationImageUrl } from "./net/images"
import { fetchRunState } from "./net/runs"
import type { TrainingSocketState } from "./net/trainingSocket"
import { EMPTY_STATE, useTrainingSocket } from "./net/trainingSocket"
import type { GridCanvasHandle } from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"
import { PhysicsPanel } from "./ui/PhysicsPanel"
import { RunPicker } from "./ui/RunPicker"
import { SpawnDistributionToggle } from "./ui/SpawnDistributionToggle"

const TRAIN_API_URL = "http://localhost:8002"
const TRAIN_WS_URL = "ws://localhost:8002/ws"

/** Passive training viewer — no interactive editing tool (no select/move
 * tool, no color/field-mode dropdowns), just a fitness-history timeline
 * and a live WebGPU replay of whichever generation's weights are
 * selected. Same overall role as trainer/frontend's TrainingView.tsx,
 * scoped down per this project's own requirements. */
export function TrainingView() {
  const liveState = useTrainingSocket(TRAIN_WS_URL, TRAIN_API_URL)
  // null = following the live/current run. Anything else is an archived
  // run's own id (net/runs.ts's RunSummary) — see the RunPicker below,
  // and archivedState's own effect just under this, which fetches that
  // run's full history exactly once (an archived run is static, nothing
  // to subscribe to) whenever this changes.
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
        console.error("[envnca] failed to fetch archived run history", err)
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
  const [backgroundMode, setBackgroundMode] =
    useState<BackgroundMode>("substrate")
  const [intensity, setIntensity] = useState(DEFAULT_INTENSITY)
  const [spawnDistribution, setSpawnDistribution] =
    useState<SpawnDistribution>("default")
  // Default true (loop): a candidate was only ever *trained* for
  // config.steps, so that's what keeps a long-idle viewer cycling
  // through fresh-seeded looks instead of freezing on the last frame.
  // Unchecking lets a rollout run straight past that count instead —
  // there's no hard simulation limit at config.steps, just the point
  // training stopped scoring it, so this is for watching whether a
  // trained shape holds up (or degrades) beyond its trained horizon.
  const [loopAtTrainedSteps, setLoopAtTrainedSteps] = useState(true)
  const [paused, setPaused] = useState(false)
  const gridCanvasRef = useRef<GridCanvasHandle>(null)

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. configByGeneration and history are
  // evicted in lockstep (see trainingSocket.ts), so any generation number
  // that still appears on the chart is guaranteed to resolve here.
  const activeConfig =
    selectedGeneration !== null
      ? (configByGeneration.get(selectedGeneration) ?? latest)
      : latest
  const [replayStep, setReplayStep] = useState(0)
  // Reset the visible step counter whenever the replayed generation
  // actually changes (a new set of weights, not just the same config
  // object re-arriving) — GridCanvas restarts its own rollout the same
  // way (see gpu/simulation.ts's loadGeneration()).
  useEffect(() => {
    setReplayStep(0)
  }, [activeConfig?.generation])
  // null = following this generation's own trained decay/maxSpeed/
  // maxAccel/maxStrafe; non-null once the "Physics" panel's sliders have
  // been touched. Reset (falls back to trained values again) whenever
  // the run or the generation being viewed changes — same reasoning as
  // replayStep just above: an override dialed in for one generation's
  // behavior means nothing once a different one is loaded, and a stale
  // one silently carrying over would be confusing (not a "same shape"
  // resetKeyFor(config) shape change, so gpu/simulation.ts wouldn't
  // reset it on its own — see GridCanvas's own physics effect).
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
  // live run's own id — viewingRunId is null for that case (see its own
  // declaration above), so this is the one place that translates between
  // the two conventions.
  const activeRunId = viewingRunId ?? "current"

  // Unlike the live server's own --target (fixed for its whole process
  // lifetime), an *archived* run being browsed may have been trained
  // against a completely different target — GET /targets/{name}/points
  // (not the older, server-fixed /target/points) can load any target's
  // points at any grid size, so this re-fetches whenever the run/
  // generation actually being viewed changes rather than once on mount.
  const [targetPoints, setTargetPoints] = useState<[number, number][] | null>(
    null
  )
  useEffect(() => {
    if (!activeConfig) return
    let cancelled = false
    const params = new URLSearchParams({
      grid_size: String(activeConfig.gridWidth),
    })
    fetch(
      `${TRAIN_API_URL}/targets/${encodeURIComponent(activeConfig.target)}/points?${params}`
    )
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) setTargetPoints(data.points)
      })
      .catch((err) =>
        console.error("[envnca] failed to fetch target points", err)
      )
    return () => {
      cancelled = true
    }
  }, [activeConfig?.target, activeConfig?.gridWidth])

  return (
    <div className="training-layout">
      <div className="training-main">
        <div className="controls">
          <h1>training viewer</h1>

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
                {activeConfig ? `${replayStep} / ${activeConfig.steps}` : "—"}
              </span>
            </div>
            <div className="stat-row">
              <span>All-time best</span>
              <span>
                {activeStat ? activeStat.allTimeBest.toFixed(3) : "—"}
              </span>
            </div>
            <div className="stat-row">
              <span>Agents</span>
              <span>{activeConfig ? activeConfig.agentCount : "—"}</span>
            </div>
            <div className="stat-row">
              <span>Grid</span>
              <span>
                {activeConfig
                  ? `${activeConfig.gridWidth}×${activeConfig.gridHeight}`
                  : "—"}
              </span>
            </div>
            <div className="stat-row">
              <span>Channels</span>
              <span>{activeConfig ? activeConfig.channels : "—"}</span>
            </div>
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
            <h2>Background</h2>
            <select
              className="select"
              value={backgroundMode}
              onChange={(e) =>
                setBackgroundMode(e.target.value as BackgroundMode)
              }
            >
              <option value="substrate">Chemical substrate</option>
              <option value="gray">Gray</option>
              <option value="black">Black</option>
            </select>
            <label className="slider-row">
              <span>Intensity</span>
              <input
                type="range"
                min={0.5}
                max={8}
                step={0.1}
                value={intensity}
                onChange={(e) => setIntensity(Number(e.target.value))}
              />
              <span className="slider-value">{intensity.toFixed(1)}×</span>
            </label>
          </section>

          <section>
            <h2>Spawn</h2>
            <SpawnDistributionToggle
              value={spawnDistribution}
              onChange={setSpawnDistribution}
            />
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
            </div>
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
                Running past step {activeConfig?.steps ?? "—"} — the horizon it
                was scored at.
              </p>
            )}
          </section>
        </div>
        <div className="viewport">
          <GridCanvas
            ref={gridCanvasRef}
            config={activeConfig}
            targetPoints={targetPoints}
            backgroundMode={backgroundMode}
            physics={physicsValues}
            intensity={intensity}
            spawnDistribution={spawnDistribution}
            onStep={setReplayStep}
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
                <div className="snapshot-item">
                  <img
                    className="snapshot-image"
                    src={generationImageUrl(
                      TRAIN_API_URL,
                      activeRunId,
                      activeStat.generation,
                      "target"
                    )}
                    alt="Target, rasterized"
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
                      "raster"
                    )}
                    alt="Winning agents, rasterized and pose-aligned"
                  />
                  <span className="snapshot-label">Agents (raster)</span>
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
                    alt="Winning agents, raw replay positions"
                  />
                  <span className="snapshot-label">Agents (raw)</span>
                </div>
              </div>
            ) : (
              <p className="hint">No generation selected yet.</p>
            )}
          </section>
        </div>
      </div>
      <div className="training-timeline">
        <FitnessChart
          history={history}
          selectedGeneration={selectedGeneration}
          onSelectGeneration={setSelectedGeneration}
          getRasterImageUrl={(generation) =>
            generationImageUrl(TRAIN_API_URL, activeRunId, generation, "raster")
          }
        />
      </div>
    </div>
  )
}

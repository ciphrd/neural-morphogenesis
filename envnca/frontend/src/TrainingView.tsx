import { useEffect, useState } from "react";
import { FitnessChart } from "./charts/FitnessChart";
import { DEFAULT_INTENSITY } from "./gpu/render";
import type { SpawnDistribution } from "./gpu/rng";
import { generationImageUrl } from "./net/images";
import { useTrainingSocket } from "./net/trainingSocket";
import { GridCanvas } from "./render/GridCanvas";
import { SpawnDistributionToggle } from "./ui/SpawnDistributionToggle";
import type { BackgroundMode } from "./gpu/types";

const TRAIN_API_URL = "http://localhost:8002";
const TRAIN_WS_URL = "ws://localhost:8002/ws";

/** Passive training viewer — no interactive editing tool (no select/move
 * tool, no color/field-mode dropdowns), just a fitness-history timeline
 * and a live WebGPU replay of whichever generation's weights are
 * selected. Same overall role as trainer/frontend's TrainingView.tsx,
 * scoped down per this project's own requirements. */
export function TrainingView() {
  const { history, latest, configByGeneration } = useTrainingSocket(TRAIN_WS_URL, TRAIN_API_URL);
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);
  const [backgroundMode, setBackgroundMode] = useState<BackgroundMode>("substrate");
  const [intensity, setIntensity] = useState(DEFAULT_INTENSITY);
  const [spawnDistribution, setSpawnDistribution] = useState<SpawnDistribution>("default");
  // Default true (loop): a candidate was only ever *trained* for
  // config.steps, so that's what keeps a long-idle viewer cycling
  // through fresh-seeded looks instead of freezing on the last frame.
  // Unchecking lets a rollout run straight past that count instead —
  // there's no hard simulation limit at config.steps, just the point
  // training stopped scoring it, so this is for watching whether a
  // trained shape holds up (or degrades) beyond its trained horizon.
  const [loopAtTrainedSteps, setLoopAtTrainedSteps] = useState(true);

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. configByGeneration and history are
  // evicted in lockstep (see trainingSocket.ts), so any generation number
  // that still appears on the chart is guaranteed to resolve here.
  const activeConfig = selectedGeneration !== null ? (configByGeneration.get(selectedGeneration) ?? latest) : latest;
  const [replayStep, setReplayStep] = useState(0);
  // Reset the visible step counter whenever the replayed generation
  // actually changes (a new set of weights, not just the same config
  // object re-arriving) — GridCanvas restarts its own rollout the same
  // way (see gpu/simulation.ts's loadGeneration()).
  useEffect(() => {
    setReplayStep(0);
  }, [activeConfig?.generation]);
  const activeStat =
    selectedGeneration !== null
      ? (history.find((h) => h.generation === selectedGeneration) ?? null)
      : history.length > 0
        ? history[history.length - 1]
        : null;

  // The training server's target is fixed for its whole run (set once at
  // launch via --target) — no switching UI needed, just a one-shot fetch.
  const [targetPoints, setTargetPoints] = useState<[number, number][] | null>(null);
  useEffect(() => {
    fetch(`${TRAIN_API_URL}/target/points`)
      .then((res) => res.json())
      .then((data) => setTargetPoints(data.points))
      .catch((err) => console.error("[envnca] failed to fetch target points", err));
  }, []);

  return (
    <div className="training-layout">
      <div className="training-main">
        <div className="controls">
          <h1>envnca training</h1>
          <p className="subtitle">Live view of a random-evolution run (train_server.py), replayed on WebGPU</p>

          <section>
            <h2>Stats</h2>
            <div className="stat-row">
              <span>Generation</span>
              <span>{activeStat ? activeStat.generation : "—"}</span>
            </div>
            <div className="stat-row">
              <span>Step (replay)</span>
              <span>{activeConfig ? `${replayStep} / ${activeConfig.steps}` : "—"}</span>
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
            <div className="stat-row">
              <span>All-time best</span>
              <span>{activeStat ? activeStat.allTimeBest.toFixed(3) : "—"}</span>
            </div>
          </section>

          <section>
            <h2>Snapshot</h2>
            {activeStat ? (
              <div className="snapshot-grid">
                <div className="snapshot-item">
                  <img
                    className="snapshot-image"
                    src={generationImageUrl(TRAIN_API_URL, activeStat.generation, "target")}
                    alt="Target, rasterized"
                  />
                  <span className="snapshot-label">Target</span>
                </div>
                <div className="snapshot-item">
                  <img
                    className="snapshot-image"
                    src={generationImageUrl(TRAIN_API_URL, activeStat.generation, "raster")}
                    alt="Winning agents, rasterized and pose-aligned"
                  />
                  <span className="snapshot-label">Agents (raster)</span>
                </div>
                <div className="snapshot-item">
                  <img
                    className="snapshot-image"
                    src={generationImageUrl(TRAIN_API_URL, activeStat.generation, "agents")}
                    alt="Winning agents, raw replay positions"
                  />
                  <span className="snapshot-label">Agents (raw)</span>
                </div>
              </div>
            ) : (
              <p className="hint">No generation selected yet.</p>
            )}
          </section>

          <section>
            <h2>Rollout</h2>
            <div className="stat-row">
              <span>Agents</span>
              <span>{activeConfig ? activeConfig.agentCount : "—"}</span>
            </div>
            <div className="stat-row">
              <span>Grid</span>
              <span>{activeConfig ? `${activeConfig.gridWidth}×${activeConfig.gridHeight}` : "—"}</span>
            </div>
            <div className="stat-row">
              <span>Channels</span>
              <span>{activeConfig ? activeConfig.channels : "—"}</span>
            </div>
          </section>

          <section>
            <h2>Background</h2>
            <select
              className="select"
              value={backgroundMode}
              onChange={(e) => setBackgroundMode(e.target.value as BackgroundMode)}
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
            <SpawnDistributionToggle value={spawnDistribution} onChange={setSpawnDistribution} />
            <p className="hint">
              Viewer-only — changes how a replayed rollout scatters agents at step 0, not how training itself spawns
              them.
            </p>
          </section>

          <section>
            <h2>Playback</h2>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={loopAtTrainedSteps}
                onChange={(e) => setLoopAtTrainedSteps(e.target.checked)}
              />
              Loop at trained step count
            </label>
            {!loopAtTrainedSteps && (
              <p className="hint">Running past step {activeConfig?.steps ?? "—"} — the horizon it was scored at.</p>
            )}
          </section>
        </div>
        <div className="viewport">
          <GridCanvas
            config={activeConfig}
            targetPoints={targetPoints}
            backgroundMode={backgroundMode}
            intensity={intensity}
            spawnDistribution={spawnDistribution}
            onStep={setReplayStep}
            loopAtTrainedSteps={loopAtTrainedSteps}
          />
        </div>
      </div>
      <div className="training-timeline">
        <FitnessChart
          history={history}
          selectedGeneration={selectedGeneration}
          onSelectGeneration={setSelectedGeneration}
          getRasterImageUrl={(generation) => generationImageUrl(TRAIN_API_URL, generation, "raster")}
        />
      </div>
    </div>
  );
}

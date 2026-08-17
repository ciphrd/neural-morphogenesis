import { useEffect, useState } from "react"
import { FitnessChart } from "./charts/FitnessChart"
import { useTrainingSocket } from "./net/trainingSocket"
import {
  type ColorMode,
  type FieldMode,
  GraphRenderer,
} from "./render/GraphRenderer"
import { useLocalSimulation } from "./sim/useLocalSimulation"
import { useRealtimeSimulation } from "./sim/useRealtimeSimulation"

const TRAIN_API_URL = "http://localhost:8001"
const TRAIN_WS_URL = "ws://localhost:8001/ws"

// The replay drives itself — nothing in this view ever triggers a split or
// a selection, so these are true no-ops, not stubs waiting to be filled in.
const noop = () => {}

type TrainingTool = "select" | "move"

export function TrainingView() {
  const { history, latest, weightsByGeneration } = useTrainingSocket(
    TRAIN_WS_URL,
    TRAIN_API_URL
  )
  const [colorMode, setColorMode] = useState<ColorMode>("solid")
  const [fieldMode, setFieldMode] = useState<FieldMode>("none")
  const [tool, setTool] = useState<TrainingTool>("select")
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(
    null
  )
  // "Realtime growth": swaps the batch replay (sense/decide/act ->
  // physics settled to convergence -> repeat, animated but exact) for a
  // loose, always-ticking visualization (physics.ts's relaxTick(), never
  // run to convergence — see useRealtimeSimulation.ts) that doesn't try
  // to be accurate, just alive-looking. Both hooks below are always
  // mounted; only the active one's loop actually runs (see each hook's
  // own `enabled`/`running` gating) so flipping this doesn't leave a
  // second animation loop running in the background for no reason.
  const [realtime, setRealtime] = useState(false)
  const [ticksPerFrame, setTicksPerFrame] = useState(1)

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. weightsByGeneration and history are
  // evicted in lockstep (see trainingSocket.ts), so any generation number
  // that still appears on the chart is guaranteed to resolve here.
  const activeGeneration =
    selectedGeneration !== null
      ? (weightsByGeneration.get(selectedGeneration) ?? latest)
      : latest
  const activeStat =
    selectedGeneration !== null
      ? (history.find((h) => h.generation === selectedGeneration) ?? null)
      : history.length > 0
        ? history[history.length - 1]
        : null

  const batchSim = useLocalSimulation(activeGeneration, !realtime)
  const realtimeSim = useRealtimeSimulation(
    activeGeneration,
    realtime,
    ticksPerFrame
  )
  const { nodes, radius, dragNode, releaseNode } = realtime
    ? realtimeSim
    : batchSim

  // The training server's target is fixed for its whole run (set once at
  // launch via --target) — no switching UI needed, just a one-shot fetch.
  const [targetPoints, setTargetPoints] = useState<[number, number][] | null>(
    null
  )
  useEffect(() => {
    fetch(`${TRAIN_API_URL}/target/points`)
      .then((res) => res.json())
      .then((data) => setTargetPoints(data.points))
      .catch((err) =>
        console.error("[trainer] failed to fetch training target points", err)
      )
  }, [])

  return (
    <div className="training-layout">
      <div className="training-main">
        <div className="controls">
          <h1>Training</h1>
          <p className="subtitle">
            Live view of a random-evolution run (train_server.py)
          </p>

          <section>
            <h2>Stats</h2>
            <div className="stat-row">
              <span>Generation</span>
              <span>{activeStat ? activeStat.generation : "—"}</span>
            </div>
            <div className="stat-row">
              <span>{realtime ? "Ticks" : "Step (replay)"}</span>
              <span>
                {realtime
                  ? realtimeSim.tick
                  : activeGeneration
                    ? `${batchSim.step} / ${activeGeneration.steps}`
                    : "—"}
              </span>
            </div>
            <div className="stat-row">
              <span>Nodes</span>
              <span>{nodes.length}</span>
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
              <span>
                {activeStat ? activeStat.allTimeBest.toFixed(3) : "—"}
              </span>
            </div>
          </section>

          <section>
            <h2>Simulation</h2>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={realtime}
                onChange={(e) => setRealtime(e.target.checked)}
              />
              Realtime growth
            </label>
            {realtime && (
              <div className="slider-row">
                <label>Ticks/frame</label>
                <input
                  type="range"
                  min={1}
                  max={5}
                  step={1}
                  value={ticksPerFrame}
                  onChange={(e) =>
                    setTicksPerFrame(parseInt(e.target.value, 10))
                  }
                />
                <span className="slider-value">{ticksPerFrame}</span>
              </div>
            )}
          </section>

          <section>
            <h2>Tool</h2>
            <div className="tabs">
              <button
                className={tool === "select" ? "active" : ""}
                onClick={() => setTool("select")}
              >
                Select
              </button>
              <button
                className={tool === "move" ? "active" : ""}
                onClick={() => setTool("move")}
              >
                Move
              </button>
            </div>
            {tool === "move" && (
              <p className="hint">
                Drag any node to reposition it — physics reacts live (tension
                pulls neighbors along, collision pushes them clear) for as long
                as you hold it, then lets go once released.
              </p>
            )}
          </section>

          <section>
            <h2>Node color</h2>
            <select
              className="select"
              value={colorMode}
              onChange={(e) => setColorMode(e.target.value as ColorMode)}
            >
              <option value="solid">Solid</option>
              <option value="channels">Channels (first 3)</option>
              <option value="id">ID</option>
              <option value="direction">Spawn direction</option>
            </select>
            {colorMode !== "solid" && colorMode !== "direction" && (
              <p className="hint">
                Each node's{" "}
                {colorMode === "channels"
                  ? "first 3 chemical channels"
                  : "3 ID channels"}{" "}
                mapped to R/G/B (
                {colorMode === "channels" ? "[-10, 10]" : "[-1, 1]"} → [0, 255],
                raw — a fully saturated channel means it's genuinely sitting at{" "}
                {colorMode === "channels" ? "its clip" : "±1"}, not a
                color-mapping artifact). White ring marks the most recently
                added node.
              </p>
            )}
            {colorMode === "direction" && (
              <p className="hint">
                Fill = the network's own raw split probability (red 0 → green
                1), before the energy gate scales it down — how much this node
                wants to split right now, regardless of whether it's actually
                allowed to. A white tick points the learned spawn-direction
                output (which way it would place a child); no tick means the
                network hasn't expressed a direction preference yet (near-zero
                output). White ring marks the most recently added node.
              </p>
            )}
          </section>

          <section>
            <h2>Background</h2>
            <select
              className="select"
              value={fieldMode}
              onChange={(e) => setFieldMode(e.target.value as FieldMode)}
            >
              <option value="none">None</option>
              <option value="field">Chemical field</option>
              <option value="gradientX">Gradient ∂x</option>
              <option value="gradientY">Gradient ∂y</option>
            </select>
          </section>
        </div>
        <div className="viewport">
          <GraphRenderer
            nodes={nodes}
            radius={radius}
            targetPoints={targetPoints}
            tool={tool}
            selectedNodeId={null}
            onSplitNode={noop}
            onSelectNode={noop}
            colorMode={colorMode}
            onDragNode={dragNode}
            onDragEnd={releaseNode}
            fieldMode={fieldMode}
            fieldSigma={activeGeneration?.sensingSigma}
            maxEnergy={activeGeneration?.maxEnergy}
          />
        </div>
      </div>
      <div className="training-timeline">
        <FitnessChart
          history={history}
          selectedGeneration={selectedGeneration}
          onSelectGeneration={setSelectedGeneration}
        />
      </div>
    </div>
  )
}

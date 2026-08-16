import { useEffect, useState } from "react";
import { GraphRenderer, type ColorMode, type FieldMode } from "./render/GraphRenderer";
import { FitnessChart } from "./charts/FitnessChart";
import { useTrainingSocket } from "./net/trainingSocket";
import { useLocalSimulation } from "./sim/useLocalSimulation";

const TRAIN_API_URL = "http://localhost:8001";
const TRAIN_WS_URL = "ws://localhost:8001/ws";

// The replay drives itself — nothing in this view ever triggers a split or
// a selection, so these are true no-ops, not stubs waiting to be filled in.
const noop = () => {};

type TrainingTool = "select" | "move";

export function TrainingView() {
  const { history, latest, weightsByGeneration } = useTrainingSocket(TRAIN_WS_URL, TRAIN_API_URL);
  const [colorMode, setColorMode] = useState<ColorMode>("solid");
  const [fieldMode, setFieldMode] = useState<FieldMode>("none");
  const [tool, setTool] = useState<TrainingTool>("select");
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);

  // null selection = follow whatever's newest; otherwise replay whichever
  // past generation was scrubbed to. weightsByGeneration and history are
  // evicted in lockstep (see trainingSocket.ts), so any generation number
  // that still appears on the chart is guaranteed to resolve here.
  const activeGeneration = selectedGeneration !== null ? weightsByGeneration.get(selectedGeneration) ?? latest : latest;
  const activeStat =
    selectedGeneration !== null
      ? history.find((h) => h.generation === selectedGeneration) ?? null
      : history.length > 0
        ? history[history.length - 1]
        : null;

  const { nodes, radius, dragNode, releaseNode, step } = useLocalSimulation(activeGeneration);

  // The training server's target is fixed for its whole run (set once at
  // launch via --target) — no switching UI needed, just a one-shot fetch.
  const [targetPoints, setTargetPoints] = useState<[number, number][] | null>(null);
  useEffect(() => {
    fetch(`${TRAIN_API_URL}/target/points`)
      .then((res) => res.json())
      .then((data) => setTargetPoints(data.points))
      .catch((err) => console.error("[trainer] failed to fetch training target points", err));
  }, []);

  return (
    <div className="training-layout">
      <div className="training-main">
        <div className="controls">
          <h1>Training</h1>
          <p className="subtitle">Live view of a random-evolution run (train_server.py)</p>

          <section>
            <h2>Stats</h2>
            <div className="stat-row">
              <span>Generation</span>
              <span>{activeStat ? activeStat.generation : "—"}</span>
            </div>
            <div className="stat-row">
              <span>Step (replay)</span>
              <span>{activeGeneration ? `${step} / ${activeGeneration.steps}` : "—"}</span>
            </div>
            <div className="stat-row">
              <span>Nodes (replay)</span>
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
              <span>{activeStat ? activeStat.allTimeBest.toFixed(3) : "—"}</span>
            </div>
          </section>

          <section>
            <h2>Tool</h2>
            <div className="tabs">
              <button className={tool === "select" ? "active" : ""} onClick={() => setTool("select")}>
                Select
              </button>
              <button className={tool === "move" ? "active" : ""} onClick={() => setTool("move")}>
                Move
              </button>
            </div>
            {tool === "move" && (
              <p className="hint">
                Drag any node to reposition it — physics reacts live (tension pulls
                neighbors along, collision pushes them clear) for as long as you hold it,
                then lets go once released.
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
                Each node's {colorMode === "channels" ? "first 3 chemical channels" : "3 ID channels"} mapped
                to R/G/B ({colorMode === "channels" ? "[-10, 10]" : "[-1, 1]"} → [0, 255], raw — a fully
                saturated channel means it's genuinely sitting at {colorMode === "channels" ? "its clip" : "±1"}, not
                a color-mapping artifact). White ring marks the most recently added node.
              </p>
            )}
            {colorMode === "direction" && (
              <p className="hint">
                Fill = the network's own raw split probability (red 0 → green 1), before the
                energy gate scales it down — how much this node wants to split right now,
                regardless of whether it's actually allowed to. A white tick points the
                learned spawn-direction output (which way it would place a child); no tick
                means the network hasn't expressed a direction preference yet (near-zero
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
            {fieldMode === "field" && (
              <p className="hint">
                The field every node's own gradient-sensing step actually reads — same
                Gaussian-summed first-3-channels substrate (substrate.weighted_field_and_gradient),
                just its raw value instead of its gradient. Gray means no chemical influence
                reaches that point; color means it does, at the same [-10, 10] scale as
                "Channels" node color — several overlapping saturated nodes will push it further
                into saturation than any one node alone, which is real (their influence is
                genuinely stacking), not a rendering cap.
              </p>
            )}
            {(fieldMode === "gradientX" || fieldMode === "gradientY") && (
              <p className="hint">
                That same field's signed {fieldMode === "gradientX" ? "∂/∂x" : "∂/∂y"} component
                per channel instead of its raw value — one axis of the literal derivative a
                node's own sensing step differentiates
                (substrate.weighted_field_and_gradient) to decide which way to grow. Gray is
                zero (locally flat along this axis); it darkens toward black as the field
                decreases along {fieldMode === "gradientX" ? "x" : "y"} and brightens toward
                white as it increases — a single axis, not the vector's magnitude, since
                magnitude (length) can never be negative and so can't show a black/white split
                the way a signed axis can. Direction as a full 2D vector isn't shown here — a
                single node's own spawn-direction arrow is the "Spawn direction" node color
                mode / node inspector, not this per-pixel background.
              </p>
            )}
          </section>

          <section>
            <h2>About</h2>
            <p className="hint">
              The viewport runs the selected generation's winning weights entirely in your
              browser — sensing, the update rule, and physics all replayed client-side (see
              src/sim/) — so growth animates smoothly with no per-frame network round-trip.
              Scrub the timeline below to revisit any past generation; jump back to live
              anytime. Hyperparameters (target, population, generations, ...) are fixed for
              this run's lifetime, set at launch — see the training server's own terminal
              output for the exact configuration. Every node also carries a blue ring
              just outside its edge, in every color mode — a clock face showing its current
              energy as a fraction of the per-run max (full circle = full energy, gated split
              threshold — see MIN_SPLIT_ENERGY — is not marked on it).
            </p>
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
  );
}

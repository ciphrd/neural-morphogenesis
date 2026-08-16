import { useState } from "react";
import { GraphRenderer, type ColorMode, type Tool } from "./render/GraphRenderer";
import { TargetPreview } from "./render/TargetPreview";
import { NodeInspector } from "./ui/NodeInspector";
import { useGraphSocket } from "./net/socket";
import { useTargets } from "./net/targets";
import { useTargetDistance } from "./net/distance";
import { useNodeInspector } from "./net/nodes";

const WS_URL = "ws://localhost:8000/ws";

export function InteractiveView() {
  const { state, splitNode, step, playing, togglePlay } = useGraphSocket(WS_URL);
  const { targets, selected, targetPoints, select } = useTargets();
  const distance = useTargetDistance(selected, state.nodes);
  const [tool, setTool] = useState<Tool>("add");
  const [colorMode, setColorMode] = useState<ColorMode>("solid");
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const inspectedNode = useNodeInspector(selectedNodeId, state.nodes);

  return (
    <>
      <div className="controls">
        <h1>Trainer 2D</h1>
        <p className="subtitle">Particle + surface-tension scaffold</p>

        <section>
          <h2>Stats</h2>
          <div className="stat-row">
            <span>Nodes</span>
            <span>{state.nodes.length}</span>
          </div>
        </section>

        <section>
          <h2>Simulation</h2>
          <div className="tabs">
            <button onClick={step} disabled={playing}>
              Step
            </button>
            <button className={playing ? "active" : ""} onClick={togglePlay}>
              {playing ? "Pause" : "Play"}
            </button>
          </div>
          <p className="hint">
            Every node senses, decides, and acts via the (untrained, randomly-initialized)
            learned update rule. Expect undirected/chaotic growth — nothing has taught it
            to do anything useful yet.
          </p>
        </section>

        <section>
          <h2>Tool</h2>
          <div className="tabs">
            <button className={tool === "add" ? "active" : ""} onClick={() => setTool("add")}>
              Add node
            </button>
            <button
              className={tool === "select" ? "active" : ""}
              onClick={() => setTool("select")}
            >
              Select node
            </button>
          </div>
        </section>

        <section>
          <h2>Node color</h2>
          <select
            className="select"
            value={colorMode}
            onChange={(e) => setColorMode(e.target.value as ColorMode)}
          >
            <option value="solid">Solid</option>
            <option value="direction">Spawn direction</option>
          </select>
          {colorMode === "direction" && (
            <p className="hint">
              Fill = the network's own raw split probability (red 0 → green 1), before the
              energy gate scales it down. A white tick points the learned spawn-direction
              output (which way it would place a child); no tick means the network hasn't
              expressed a direction preference yet. (Only "Solid"/"Spawn direction" are
              offered here — "Channels"/"ID" need per-node chemicals/id data this view
              doesn't broadcast; see the Training tab.) Every node also carries a blue ring
              just outside its edge, in every color mode, showing its current energy as
              a fraction of the per-run max.
            </p>
          )}
        </section>

        <section>
          <h2>Targets</h2>
          <div className="target-list">
            {targets.map((t) => (
              <button
                key={t.name}
                className={"target-row" + (selected === t.name ? " selected" : "")}
                onClick={() => select(t.name)}
              >
                <TargetPreview points={t.preview} />
                <div className="target-info">
                  <span className="target-name">{t.name}</span>
                  <span className="target-meta">{t.points} pixels</span>
                </div>
              </button>
            ))}
            {targets.length === 0 && (
              <p className="hint">No targets found in trainer/backend/targets.</p>
            )}
          </div>
        </section>

        {selected && (
          <section>
            <h2>Distance to {selected}</h2>
            {distance ? (
              <>
                <div className="stat-row">
                  <span>Chamfer</span>
                  <span>{distance.chamfer.toFixed(3)}</span>
                </div>
                <div className="stat-row">
                  <span>Structure → target</span>
                  <span>{distance.points_to_target.toFixed(3)}</span>
                </div>
                <div className="stat-row">
                  <span>Target → structure</span>
                  <span>{distance.target_to_points.toFixed(3)}</span>
                </div>
                <div className="stat-row">
                  <span>Best-fit rotation</span>
                  <span>{((distance.rotation * 180) / Math.PI).toFixed(1)}°</span>
                </div>
                <p className="hint">
                  Measured at the best-found alignment, not the structure's actual
                  orientation.
                </p>
              </>
            ) : (
              <p className="hint">Computing…</p>
            )}
          </section>
        )}

        <section>
          <h2>Controls</h2>
          <p className="hint">
            {tool === "add"
              ? "Click a node to split it — a new node spawns touching it at a random angle. Physics settles the whole cluster: nodes never overlap (hard collision), and nearby nodes pull gently toward contact (breakable surface tension)."
              : "Click a node to inspect its internal state in the panel on the right. Click it again, or click empty space, to deselect."}{" "}
            Scroll to zoom, right-drag to pan.
          </p>
        </section>
      </div>
      <div className="viewport">
        <GraphRenderer
          nodes={state.nodes}
          radius={state.radius}
          targetPoints={targetPoints}
          tool={tool}
          selectedNodeId={selectedNodeId}
          onSplitNode={splitNode}
          onSelectNode={setSelectedNodeId}
          colorMode={colorMode}
          maxEnergy={state.maxEnergy}
        />
      </div>
      <NodeInspector node={inspectedNode} />
    </>
  );
}

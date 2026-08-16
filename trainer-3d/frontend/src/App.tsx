import { GraphRenderer } from "./render/GraphRenderer";
import { TargetPreview } from "./render/TargetPreview";
import { useGraphSocket } from "./net/socket";
import { useTargets } from "./net/targets";
import { useTargetDistance } from "./net/distance";

const WS_URL = "ws://localhost:8000/ws";

export default function App() {
  const { state, growTriangle } = useGraphSocket(WS_URL);
  const { targets, selected, targetPoints, select } = useTargets();
  const distance = useTargetDistance(selected, state.nodes);
  const growable = state.triangles.filter((t) => !t.grown).length;

  return (
    <div className="app">
      <div className="controls">
        <h1>Trainer</h1>
        <p className="subtitle">Graph + physics scaffold</p>

        <section>
          <h2>Stats</h2>
          <div className="stat-row">
            <span>Nodes</span>
            <span>{state.nodes.length}</span>
          </div>
          <div className="stat-row">
            <span>Triangles</span>
            <span>{state.triangles.length}</span>
          </div>
          <div className="stat-row">
            <span>Growable</span>
            <span>{growable}</span>
          </div>
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
                  <span className="target-meta">{t.points} voxels</span>
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
              </>
            ) : (
              <p className="hint">Computing…</p>
            )}
          </section>
        )}

        <section>
          <h2>Controls</h2>
          <p className="hint">
            Hover a growable (brighter) triangle to preview where it would extrude a new
            node. Click to commit — the apex connects to the triangle's 3 vertices,
            exposing 3 new growable faces, and physics relaxes the whole graph. Each
            triangle can only grow once.
          </p>
        </section>
      </div>
      <div className="viewport">
        <GraphRenderer
          nodes={state.nodes}
          edges={state.edges}
          triangles={state.triangles}
          targetPoints={targetPoints}
          onGrowTriangle={growTriangle}
        />
      </div>
    </div>
  );
}

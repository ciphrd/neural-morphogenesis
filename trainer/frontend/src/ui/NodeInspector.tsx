import type { NodeState } from "../net/nodes";

interface NodeInspectorProps {
  node: NodeState | null;
}

function ChannelBars({ values, label }: { values: number[]; label: string }) {
  return (
    <div className="channel-bars">
      {values.map((v, i) => (
        <div key={i} className="channel-row">
          <span className="channel-label">
            {label} {i}
          </span>
          <div className="channel-track">
            <div
              className="channel-fill"
              style={{
                left: v < 0 ? `${50 + v * 50}%` : "50%",
                width: `${Math.abs(v) * 50}%`,
              }}
            />
          </div>
          <span className="channel-value">{v.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

// Only used when the vector happens to be 2D (falls back to bars
// otherwise) — a quick visual read of direction/magnitude.
function Vector2Arrow({ vector }: { vector: [number, number] }) {
  const size = 80;
  const center = size / 2;
  const scale = center * 0.85;
  const x = center + vector[0] * scale;
  const y = center - vector[1] * scale;

  return (
    <svg width={size} height={size} className="bindability-arrow">
      <circle cx={center} cy={center} r={center - 2} fill="none" stroke="#2a2d35" />
      <line x1={center} y1={center} x2={x} y2={y} stroke="#4ade80" strokeWidth={2} />
      <circle cx={x} cy={y} r={3} fill="#4ade80" />
    </svg>
  );
}

export function NodeInspector({ node }: NodeInspectorProps) {
  return (
    <div className="controls inspector">
      <h1>Inspector</h1>

      {!node ? (
        <p className="hint">Select a node (with the Select tool) to inspect its internal state.</p>
      ) : (
        <>
          <section>
            <h2>Node {node.id}</h2>
            <div className="stat-row">
              <span>Position</span>
              <span>
                {node.position[0].toFixed(2)}, {node.position[1].toFixed(2)}
              </span>
            </div>
            <div className="stat-row">
              <span>Energy</span>
              <span>{node.energy.toFixed(1)}</span>
            </div>
            <div className="stat-row">
              <span>Speed</span>
              <span>{node.speed.toFixed(4)}</span>
            </div>
          </section>

          <section>
            <h2>Heading</h2>
            <Vector2Arrow vector={[Math.cos(node.heading), Math.sin(node.heading)]} />
            <p className="hint">
              Which way this node is currently facing — derived from its own velocity
              (atan2(vy, vx)), not stored separately, and the frame its own chemical-gradient
              sensing is expressed in (forward/lateral), not world x/y. Arrow length is always
              1 regardless of speed; see the "Speed" stat above (velocity's magnitude) for how
              fast it's actually moving in this direction. The network accelerates velocity
              directly (in this same local frame), so heading only changes once the node is
              actually moving somewhere.
            </p>
          </section>

          <section>
            <h2>Identity (id)</h2>
            {node.idVector.length === 2 ? (
              <Vector2Arrow vector={node.idVector as [number, number]} />
            ) : (
              <ChannelBars values={node.idVector} label="Id" />
            )}
          </section>

          <section>
            <h2>Chemical channels</h2>
            <ChannelBars values={node.chemicals} label="Ch" />
          </section>

          <section>
            <h2>Spawn direction</h2>
            <Vector2Arrow vector={node.spawnDirection as [number, number]} />
            <p className="hint">
              Which way this node would place a child if it split right now — computed every
              step regardless of whether it actually does (see the "Spawn direction" node color
              mode for this reading across every node at once). A dot at the center means the
              network hasn't expressed a preference yet.
            </p>
          </section>
        </>
      )}
    </div>
  );
}

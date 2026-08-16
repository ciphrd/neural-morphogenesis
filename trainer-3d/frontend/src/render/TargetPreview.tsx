// Fixed isometric projection — cheap, and gives a recognizable 3D-ish
// silhouette from a static SVG without spinning up a WebGL context per
// list row (a real risk with a dozen+ targets, given browsers cap
// simultaneous WebGL contexts).
function isoProject([x, y, z]: [number, number, number]): [number, number] {
  return [(x - z) * 0.866, (x + z) * 0.5 - y];
}

interface TargetPreviewProps {
  points: [number, number, number][];
}

export function TargetPreview({ points }: TargetPreviewProps) {
  if (points.length === 0) {
    return <div className="target-preview target-preview-empty" />;
  }

  const projected = points.map(isoProject);
  const xs = projected.map((p) => p[0]);
  const ys = projected.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const size = Math.max(maxX - minX, maxY - minY, 1e-6);
  const pad = size * 0.15;

  return (
    <svg
      viewBox={`${minX - pad} ${minY - pad} ${size + pad * 2} ${size + pad * 2}`}
      className="target-preview"
    >
      {projected.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={size * 0.015} fill="#4f8cff" />
      ))}
    </svg>
  );
}

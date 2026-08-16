interface TargetPreviewProps {
  points: [number, number][];
}

// Simpler than trainer-3d's version: 2D points need no projection, just a
// direct plot into an SVG viewBox sized to their bounding box.
export function TargetPreview({ points }: TargetPreviewProps) {
  if (points.length === 0) {
    return <div className="target-preview target-preview-empty" />;
  }

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
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
      {points.map(([x, y], i) => (
        <circle key={i} cx={x} cy={-y} r={size * 0.02} fill="#4f8cff" />
      ))}
    </svg>
  );
}

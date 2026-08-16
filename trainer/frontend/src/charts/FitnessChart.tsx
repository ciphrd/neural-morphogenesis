import { useRef, useState } from "react";
import type { PointerEvent } from "react";

export interface GenerationStat {
  generation: number;
  best: number;
  mean: number;
  worst: number;
  allTimeBest: number;
}

interface FitnessChartProps {
  history: GenerationStat[];
  /** null = following the live/latest generation. */
  selectedGeneration: number | null;
  onSelectGeneration: (generation: number | null) => void;
}

// Wide and short — this now lives as a full-width strip at the bottom of
// the page, not a small chart in the sidebar, so the aspect ratio needs
// to actually look like a timeline rather than a square-ish plot.
const WIDTH = 1400;
const HEIGHT = 170;
const MARGIN = { top: 16, right: 20, bottom: 28, left: 48 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

// Validated categorical palette, dark-surface steps (see the dataviz skill's
// palette.md) — slots 1-3, the only three guaranteed CVD-safe against every
// pairing at once, not just neighbors.
const SERIES = [
  { key: "best", label: "Best", color: "#3987e5" },
  { key: "mean", label: "Mean", color: "#d95926" },
  { key: "worst", label: "Worst", color: "#199e70" },
] as const;

export function FitnessChart({ history, selectedGeneration, onSelectGeneration }: FitnessChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const draggingRef = useRef(false);

  if (history.length === 0) {
    return (
      <div className="fitness-chart-empty">
        <p className="hint">No generations completed yet.</p>
      </div>
    );
  }

  const maxValue = Math.max(...history.map((h) => h.worst), 1e-6) * 1.08;
  const xForIndex = (i: number) => (history.length > 1 ? (i / (history.length - 1)) * PLOT_WIDTH : PLOT_WIDTH / 2);
  const yForValue = (v: number) => PLOT_HEIGHT - (v / maxValue) * PLOT_HEIGHT;

  const linePath = (key: "best" | "mean" | "worst") =>
    history.map((h, i) => `${i === 0 ? "M" : "L"} ${xForIndex(i)} ${yForValue(h[key])}`).join(" ");

  // Shared by hover-preview and click/drag-select: snap to the nearest
  // generation rather than requiring the pointer to land exactly on a line.
  const indexFromEvent = (ev: PointerEvent<SVGSVGElement>): number => {
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const localX = (ev.clientX - rect.left) * (WIDTH / rect.width) - MARGIN.left;
    const frac = history.length > 1 ? localX / PLOT_WIDTH : 0;
    return Math.min(history.length - 1, Math.max(0, Math.round(frac * (history.length - 1))));
  };

  const handleMove = (ev: PointerEvent<SVGSVGElement>) => {
    const index = indexFromEvent(ev);
    setHoverIndex(index);
    if (draggingRef.current) onSelectGeneration(history[index].generation);
  };

  const handleDown = (ev: PointerEvent<SVGSVGElement>) => {
    draggingRef.current = true;
    onSelectGeneration(history[indexFromEvent(ev)].generation);
  };

  const handleUp = () => {
    draggingRef.current = false;
  };

  const yTicks = [0, maxValue / 2, maxValue];
  const xTickIndices = Array.from(
    new Set([0, Math.floor((history.length - 1) / 2), history.length - 1])
  );
  const hovered = hoverIndex !== null ? history[hoverIndex] : null;

  // The persistent pointer always shows what the viewport is actually
  // replaying: the selected generation if scrubbed back, otherwise
  // whatever's newest — there's always something to point at.
  const selectedIndex =
    selectedGeneration !== null
      ? history.findIndex((h) => h.generation === selectedGeneration)
      : history.length - 1;
  const isLive = selectedGeneration === null;

  return (
    <div className="fitness-chart">
      <div className="fitness-chart-legend">
        {SERIES.map((s) => (
          <span key={s.key} className="fitness-chart-legend-item">
            <span className="fitness-chart-legend-swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        <span className="fitness-chart-legend-spacer" />
        {!isLive && (
          <button className="fitness-chart-live-button" onClick={() => onSelectGeneration(null)}>
            Jump to live
          </button>
        )}
        <span className={"fitness-chart-live-indicator" + (isLive ? " is-live" : "")}>
          {isLive ? "● Live" : `Viewing gen ${selectedGeneration}`}
        </span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="fitness-chart-svg"
        onPointerMove={handleMove}
        onPointerDown={handleDown}
        onPointerUp={handleUp}
        onPointerLeave={() => {
          setHoverIndex(null);
          draggingRef.current = false;
        }}
      >
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {yTicks.map((tick, i) => (
            <g key={i}>
              <line x1={0} x2={PLOT_WIDTH} y1={yForValue(tick)} y2={yForValue(tick)} className="fitness-chart-grid" />
              <text x={-8} y={yForValue(tick)} textAnchor="end" dominantBaseline="middle" className="fitness-chart-tick">
                {tick.toFixed(1)}
              </text>
            </g>
          ))}
          <line x1={0} x2={0} y1={0} y2={PLOT_HEIGHT} className="fitness-chart-axis" />
          <line x1={0} x2={PLOT_WIDTH} y1={PLOT_HEIGHT} y2={PLOT_HEIGHT} className="fitness-chart-axis" />

          {xTickIndices.map((i) => (
            <text key={i} x={xForIndex(i)} y={PLOT_HEIGHT + 18} textAnchor="middle" className="fitness-chart-tick">
              {history[i].generation}
            </text>
          ))}

          {SERIES.map((s) => (
            <path
              key={s.key}
              d={linePath(s.key)}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}

          {/* Persistent selection pointer — the "vertical pointer indicating
              the timeline": always visible, marks what's currently replaying. */}
          {selectedIndex >= 0 && (
            <g>
              <line
                x1={xForIndex(selectedIndex)}
                x2={xForIndex(selectedIndex)}
                y1={0}
                y2={PLOT_HEIGHT}
                className={"fitness-chart-pointer" + (isLive ? " is-live" : "")}
              />
              <path
                d={`M ${xForIndex(selectedIndex) - 5} -4 L ${xForIndex(selectedIndex) + 5} -4 L ${xForIndex(selectedIndex)} 4 Z`}
                className={"fitness-chart-pointer-head" + (isLive ? " is-live" : "")}
              />
            </g>
          )}

          {/* Transient hover crosshair — lighter, for reading exact values
              without disturbing the current selection. */}
          {hoverIndex !== null && hoverIndex !== selectedIndex && (
            <line
              x1={xForIndex(hoverIndex)}
              x2={xForIndex(hoverIndex)}
              y1={0}
              y2={PLOT_HEIGHT}
              className="fitness-chart-crosshair"
            />
          )}

          {hoverIndex !== null &&
            SERIES.map((s) => (
              <circle
                key={s.key}
                cx={xForIndex(hoverIndex)}
                cy={yForValue(history[hoverIndex][s.key])}
                r={4}
                fill={s.color}
                className="fitness-chart-dot"
              />
            ))}
        </g>
      </svg>

      {hovered && (
        <div className="fitness-chart-tooltip">
          <span className="fitness-chart-tooltip-title">Gen {hovered.generation}</span>
          {SERIES.map((s) => (
            <span key={s.key} className="fitness-chart-tooltip-row">
              <span className="fitness-chart-tooltip-key" style={{ background: s.color }} />
              <span className="fitness-chart-tooltip-value">{hovered[s.key].toFixed(3)}</span>
              <span className="fitness-chart-tooltip-label">{s.label}</span>
            </span>
          ))}
          <span className="fitness-chart-tooltip-hint">Click to view this generation's replay</span>
        </div>
      )}
    </div>
  );
}

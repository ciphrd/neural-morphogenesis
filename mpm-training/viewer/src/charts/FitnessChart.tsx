import { useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { GenerationStat } from "../net/trainingSocket";

interface FitnessChartProps {
  history: GenerationStat[];
  /** null = following the live/latest generation. */
  selectedGeneration: number | null;
  onSelectGeneration: (generation: number | null) => void;
  /** Builds the URL for a generation's aligned-agents-raster debug PNG
   * (see net/images.ts) — a callback, not a raw API base URL, so this
   * component doesn't need to know anything about that module. Omit to
   * skip the hover preview entirely. */
  getPreviewImageUrl?: (generation: number) => string;
}

// Wide and short — lives as a full-width strip at the bottom of the
// page, not a small chart in a sidebar.
const WIDTH = 1400;
const HEIGHT = 170;
const MARGIN = { top: 16, right: 20, bottom: 28, left: 48 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

// Validated categorical palette, dark-surface steps — slots 1-3, the
// only three guaranteed CVD-safe against every pairing at once, not just
// neighbors. Same values envnca/frontend's own FitnessChart.tsx uses.
const SERIES = [
  { key: "best", label: "Best", color: "#3987e5" },
  { key: "mean", label: "Mean", color: "#d95926" },
  { key: "worst", label: "Worst", color: "#199e70" },
] as const;

type SeriesKey = (typeof SERIES)[number]["key"];

// Best only, by default — mean/worst are legitimately useful (spread
// across the population) but Worst in particular can be +Infinity (a
// diverged candidate — see evolve.py's own _score_fitness() docstring),
// which would otherwise dominate the log-scale y-axis and squash Best's
// own much more interesting curve toward the bottom every time. Click a
// legend label to toggle its own curve.
const DEFAULT_VISIBLE: ReadonlySet<SeriesKey> = new Set(["best"]);

export function FitnessChart({ history, selectedGeneration, onSelectGeneration, getPreviewImageUrl }: FitnessChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const draggingRef = useRef(false);
  const [visibleKeys, setVisibleKeys] = useState<ReadonlySet<SeriesKey>>(DEFAULT_VISIBLE);
  const toggleSeries = (key: SeriesKey) => {
    setVisibleKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const visibleSeries = SERIES.filter((s) => visibleKeys.has(s.key));

  if (history.length === 0) {
    return (
      <div className="fitness-chart-empty">
        <p className="hint">No generations completed yet.</p>
      </div>
    );
  }

  // Log scale: fitness (alignment/Chamfer distance) typically spans
  // orders of magnitude over a run — log scale gives each order of
  // magnitude the same amount of vertical space, so late-run incremental
  // progress stays legible instead of compressing into a flat line near
  // the bottom of a linear axis. Only finite, strictly-positive values
  // are loggable — `worst`/`mean` can legitimately be Infinity
  // (evolve.py scores a diverged candidate that way rather than
  // crashing the generation); non-finite values pin to the plot's top, a
  // value of exactly 0 pins to the bottom.
  //
  // Scaled to whichever series are actually VISIBLE, not all three
  // always — Worst's own occasional +Infinity would otherwise dominate
  // the range and squash Best's curve toward the bottom even while
  // Worst itself sits hidden (see DEFAULT_VISIBLE's own comment).
  const plottableValues = history.flatMap((h) => visibleSeries.map((s) => h[s.key])).filter((v) => Number.isFinite(v) && v > 0);
  const minValue = plottableValues.length > 0 ? Math.min(...plottableValues) : 0.01;
  const maxValue = (plottableValues.length > 0 ? Math.max(...plottableValues) : 1) * 1.08 || 1;
  const logMin = Math.log(minValue);
  const logMax = Math.log(Math.max(maxValue, minValue * 1.01));
  const logRange = Math.max(logMax - logMin, 1e-6);

  const xForIndex = (i: number) => (history.length > 1 ? (i / (history.length - 1)) * PLOT_WIDTH : PLOT_WIDTH / 2);
  const yForValue = (v: number) => {
    if (!Number.isFinite(v)) return 0; // pinned to the top
    const clamped = Math.min(maxValue, Math.max(minValue, v > 0 ? v : minValue));
    const t = (Math.log(clamped) - logMin) / logRange;
    return PLOT_HEIGHT - t * PLOT_HEIGHT;
  };

  const formatFitness = (v: number): string => {
    if (!Number.isFinite(v)) return "∞";
    if (v === 0) return "0";
    if (v < 0.01) return v.toExponential(1);
    if (v < 10) return v.toFixed(2);
    if (v < 100) return v.toFixed(1);
    return v.toFixed(0);
  };

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

  // Geometric-mean midpoint, not arithmetic — evenly spaced in log space
  // (matches yForValue's own log interpolation), so the middle tick
  // actually lands halfway up the plot.
  const yTicks = [minValue, Math.sqrt(minValue * maxValue), maxValue];
  const xTickIndices = Array.from(new Set([0, Math.floor((history.length - 1) / 2), history.length - 1]));
  const hovered = hoverIndex !== null ? history[hoverIndex] : null;

  // The persistent pointer always shows what the viewport is actually
  // replaying: the selected generation if scrubbed back, otherwise
  // whatever's newest.
  const selectedIndex = selectedGeneration !== null ? history.findIndex((h) => h.generation === selectedGeneration) : history.length - 1;
  const isLive = selectedGeneration === null;

  return (
    <div className="fitness-chart">
      <div className="fitness-chart-legend">
        {SERIES.map((s) => {
          const isVisible = visibleKeys.has(s.key);
          return (
            <button
              key={s.key}
              type="button"
              className={"fitness-chart-legend-item" + (isVisible ? "" : " is-hidden")}
              onClick={() => toggleSeries(s.key)}
              aria-pressed={isVisible}
              title={isVisible ? `Hide ${s.label}` : `Show ${s.label}`}
            >
              <span className="fitness-chart-legend-swatch" style={{ background: s.color }} />
              {s.label}
            </button>
          );
        })}
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
                {formatFitness(tick)}
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

          {visibleSeries.map((s) => (
            <path key={s.key} d={linePath(s.key)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}

          {/* Persistent selection pointer — always visible, marks what's
              currently replaying. */}
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

          {/* Transient hover crosshair — lighter, for reading exact
              values without disturbing the current selection. */}
          {hoverIndex !== null && hoverIndex !== selectedIndex && (
            <line x1={xForIndex(hoverIndex)} x2={xForIndex(hoverIndex)} y1={0} y2={PLOT_HEIGHT} className="fitness-chart-crosshair" />
          )}

          {hoverIndex !== null &&
            visibleSeries.map((s) => (
              <circle key={s.key} cx={xForIndex(hoverIndex)} cy={yForValue(history[hoverIndex][s.key])} r={4} fill={s.color} className="fitness-chart-dot" />
            ))}

          {/* Drawn inside the SVG, not as a sibling DOM element — an
              external tooltip div appearing/disappearing on hover would
              shift the page's layout height every time; this overlays
              the fixed-size chart instead. Flips to the left of the
              cursor when it would overflow the plot's right edge. */}
          {hovered && (
            <g pointerEvents="none">
              {(() => {
                const rowHeight = 15;
                const textWidth = 128;
                const imageSize = 72;
                const imageMargin = 10;
                const imageUrl = getPreviewImageUrl?.(hovered.generation);
                const panelWidth = imageUrl ? textWidth + imageMargin + imageSize + imageMargin : textWidth;
                const textPanelHeight = 14 + rowHeight * (visibleSeries.length + 1) + 6;
                const panelHeight = imageUrl ? Math.max(textPanelHeight, imageSize + imageMargin * 2) : textPanelHeight;
                const hoverX = xForIndex(hoverIndex!);
                const overflowsRight = hoverX + 12 + panelWidth > PLOT_WIDTH;
                const panelX = overflowsRight ? hoverX - 12 - panelWidth : hoverX + 12;
                const panelY = 6;
                return (
                  <g transform={`translate(${panelX}, ${panelY})`}>
                    <rect width={panelWidth} height={panelHeight} rx={6} className="fitness-chart-tooltip-bg" />
                    <text x={10} y={16} className="fitness-chart-tooltip-title">
                      Gen {hovered.generation}
                    </text>
                    {visibleSeries.map((s, i) => (
                      <g key={s.key} transform={`translate(0, ${16 + rowHeight * (i + 1)})`}>
                        <rect x={10} y={-8} width={8} height={8} rx={2} fill={s.color} />
                        <text x={24} className="fitness-chart-tooltip-label">
                          {s.label}
                        </text>
                        <text x={textWidth - 10} textAnchor="end" className="fitness-chart-tooltip-value">
                          {formatFitness(hovered[s.key])}
                        </text>
                      </g>
                    ))}
                    <text x={10} y={textPanelHeight - 6} className="fitness-chart-tooltip-hint">
                      Click to view this generation
                    </text>
                    {imageUrl && (
                      <g transform={`translate(${textWidth + imageMargin}, ${imageMargin})`}>
                        <rect width={imageSize} height={imageSize} rx={3} className="fitness-chart-tooltip-image-bg" />
                        <image href={imageUrl} width={imageSize} height={imageSize} style={{ imageRendering: "pixelated" }} preserveAspectRatio="xMidYMid meet" />
                      </g>
                    )}
                  </g>
                );
              })()}
            </g>
          )}
        </g>
      </svg>
    </div>
  );
}

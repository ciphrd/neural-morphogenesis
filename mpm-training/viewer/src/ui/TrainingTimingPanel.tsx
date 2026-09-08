import { averageTimings, timingReport } from "./timingAverages";
import { useEffect, useMemo, useRef, useState } from "react";
import type { TimingEntry, TimingStage } from "../gpu/types";

const labels: Record<string, string> = {
  gpuMorphologyDepositQuads: "Morphology · quad deposition",
  gpuMorphologyClear: "Morphology · clear density",
  gpuMorphologyDepositCompute: "Morphology · compute deposition",
  gpuMorphologyTexture: "Morphology · density texture",
  gpuMorphologyBlurHorizontal: "Morphology · horizontal blur",
  gpuMorphologyBlurVertical: "Morphology · vertical blur + normalize",
  "gpuPhysicsRepulsionClear": "Repulsion · clear density",
  "gpuPhysicsRepulsionSplat": "Repulsion · deposit density",
  "gpuPhysicsRepulsionTexture": "Repulsion · density texture",
  "gpuPhysicsRepulsionApply": "Repulsion · apply forces",
  "gpuPhysicsGridClear": "Physics · clear grid",
  "gpuPhysicsP2G": "Physics · particles → grid",
  "gpuPhysicsGridUpdate": "Physics · update grid",
  "gpuPhysicsG2P": "Physics · grid → particles",
  "gpuChemistryClear": "Chemistry · clear deposits",
  "gpuChemistryMaterialize": "Chemistry · materialize deposits",
  "gpuChemistryGradient": "Chemistry · compute gradients",
  "gpuChemistryTransport": "Chemistry · transport + diffuse + decay",
  "gpuChemistryDiffuse": "Chemistry · diffuse + decay",
  "gpuChemistryMerge": "Chemistry · merge deposits",
  "gpuChemistrySplat": "Chemistry · deposit cell state",
  "gpuGrowth:clearGrowthField": "Growth · clear field",
  "gpuGrowth:clearRefinement": "Growth · clear refinement",
  "gpuGrowth:indexRefinementEdges": "Growth · index edges",
  "gpuGrowth:scatterGrowthIntent": "Growth · deposit intent",
  "gpuGrowth:scatterGrowthBoundary": "Growth · deposit boundary",
  "gpuGrowth:enforceGrowthField": "Growth · enforce field",
  "gpuGrowth:linkRefinementEdges": "Growth · link edges",
  "gpuGrowth:propagateRefinement": "Growth · propagate refinement",
  "gpuGrowth:requestRefinement": "Growth · request refinement",
  "gpuGrowth:reserveRefinement": "Growth · reserve refinement",
  "gpuGrowth:commitResample": "Growth · commit resampling",
  "gpuGrowth:classifyPruning": "Growth · pruning checks",
  "gpuGrowth:pruneMaterial": "Growth · prune material",
  "gpuGrowth:stopGrowthAtCapacity": "Growth · stop at capacity",

  setup: "Rollout setup",
  neuralCommands: "Neural + growth commands",
  growthSync: "GPU completion / status readback",
  growthStatusUpdate: "CPU sample-count update",
  gpuProfilingReadback: "GPU profiling readback",
  gpuPhysics: "GPU physics",
  gpuNeural: "GPU neural inference",
  gpuChemistry: "GPU chemical fields",
  gpuMorphology: "GPU morphology",
  gpuGrowth: "GPU growth / refinement",
  physics: "Physics + GPU wait",
  geometryReadback: "Geometry readback",
  colorReadback: "Color readback",
  fitness: "Raster + fitness scoring",
  positionsReadback: "Preview position readback",
};
const duration = (s: number) => s < 1 ? `${(s * 1000).toFixed(1)} ms` : `${s.toFixed(2)} s`;

const stageColors: Record<string, string> = {
  gpuMorphologyDepositQuads: "#e5ce70", gpuMorphologyClear: "#f5e6a4",
  gpuMorphologyDepositCompute: "#e5ce70", gpuMorphologyTexture: "#d4b958",
  gpuMorphologyBlurHorizontal: "#c3a348", gpuMorphologyBlurVertical: "#a68932",
  "gpuPhysicsRepulsionClear": "#a9c9ff",
  "gpuPhysicsRepulsionSplat": "#89b4fa",
  "gpuPhysicsRepulsionTexture": "#6499eb",
  "gpuPhysicsRepulsionApply": "#447bd2",
  "gpuPhysicsGridClear": "#bedfff",
  "gpuPhysicsP2G": "#78a9ff",
  "gpuPhysicsGridUpdate": "#467bd8",
  "gpuPhysicsG2P": "#315baa",
  "gpuChemistryClear": "#e2c6ff",
  "gpuChemistryMaterialize": "#ceabed",
  "gpuChemistryGradient": "#bd93f9",
  "gpuChemistryTransport": "#a677df",
  "gpuChemistryDiffuse": "#8956bf",
  "gpuChemistryMerge": "#70439f",
  "gpuChemistrySplat": "#efb9ee",
  "gpuGrowth:clearGrowthField": "#ffe0b2",
  "gpuGrowth:clearRefinement": "#ffcc80",
  "gpuGrowth:indexRefinementEdges": "#ffb74d",
  "gpuGrowth:scatterGrowthIntent": "#f3ac66",
  "gpuGrowth:scatterGrowthBoundary": "#ff9800",
  "gpuGrowth:enforceGrowthField": "#ef8c42",
  "gpuGrowth:linkRefinementEdges": "#dd7937",
  "gpuGrowth:propagateRefinement": "#d56b27",
  "gpuGrowth:requestRefinement": "#c66039",
  "gpuGrowth:reserveRefinement": "#ffab91",
  "gpuGrowth:commitResample": "#ff8a65",
  "gpuGrowth:classifyPruning": "#ee9292",
  "gpuGrowth:pruneMaterial": "#e57373",
  "gpuGrowth:stopGrowthAtCapacity": "#bc6955",

  gpuPhysics: "#78a9ff", gpuNeural: "#64d8bd", gpuChemistry: "#bd93f9",
  gpuMorphology: "#e5ce70", gpuGrowth: "#f3ac66", gpuProfilingReadback: "#ec82aa", growthStatusUpdate: "#adb5c5",
  physics: "#78a9ff", growthSync: "#f3ac66", fitness: "#bd93f9",
  neuralCommands: "#64d8bd", setup: "#ec82aa", geometryReadback: "#e5ce70",
  colorReadback: "#68c7ed", positionsReadback: "#adb5c5",
};

function TimingPie({ stages, total, gpu = false }: { stages: [string, TimingStage][]; total: number; gpu?: boolean }) {
  let offset = 0;
  const slices = stages.filter(([, stage]) => stage.seconds > 0).map(([name, stage]) => {
    const percent = total > 0 ? stage.seconds / total * 100 : 0;
    const slice = { name, stage, percent, offset, color: stageColors[name] ?? "#999" };
    offset += percent;
    return slice;
  });
  return <div className="timing-pie">
    <svg viewBox="0 0 200 200" role="img" aria-label="Share of measured rollout stage time">
      <circle cx="100" cy="100" r="76" fill="none" stroke="#292929" strokeWidth="28" />
      {slices.map(slice => <circle key={slice.name} cx="100" cy="100" r="76" fill="none"
        stroke={slice.color} strokeWidth="28" pathLength="100"
        strokeDasharray={`${slice.percent} ${100 - slice.percent}`}
        strokeDashoffset={-slice.offset} transform="rotate(-90 100 100)">
        <title>{labels[slice.name] ?? slice.name}: {slice.percent.toFixed(1)}% ({duration(slice.stage.seconds)})</title>
      </circle>)}
      <text x="100" y="98" className="timing-pie-total">{duration(total)}</text>
      <text x="100" y="117" className="timing-pie-caption">{gpu ? "per sampled step" : "avg per generation"}</text>
    </svg>
    {total === 0 ? <p className="timing-note">No measured stage time yet.</p> :
      <ul className="timing-pie-legend">{slices.map(slice => <li key={slice.name}>
        <span className="timing-pie-dot" style={{ background: slice.color }} />
        <span>{labels[slice.name] ?? slice.name}</span>
        <strong>{slice.percent < .1 ? "<0.1" : slice.percent.toFixed(1)}%</strong>
        <span className="timing-pie-value">{duration(slice.stage.seconds)}</span>
      </li>)}</ul>}
  </div>;
}

export function TimingHistoryChart({ history, scope }: { history: TimingEntry[]; scope: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(300);
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setContainerWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const [hovered, setHovered] = useState<number | null>(null);
  const gpu = scope.startsWith("gpu");
  const rows = history.flatMap(entry => {
    const report = timingReport(entry.timing, scope);
    return report ? [{generation: entry.generation, stages: report.stages}] : [];
  });
  const keys = [...new Set([...Object.keys(labels), ...rows.flatMap(row => Object.keys(row.stages))])];
  const totals = rows.map(row => Object.values(row.stages).reduce((sum, stage) => sum + stage.seconds, 0));
  const maximum = Math.max(gpu ? 1e-6 : 1, ...totals);
  const width = Math.max(containerWidth, rows.length * 8 + 48);
  const plotWidth = width - 48, plotHeight = 150, top = 16, bottom = top + plotHeight;
  const barWidth = plotWidth / Math.max(1, rows.length);
  const tickEvery = Math.max(1, Math.ceil(rows.length / 6));
  const active = rows.find(row => row.generation === hovered);
  return <div className="timing-history" ref={container}>
    {!rows.length && <p className="timing-note">No timing measurements yet for this run.</p>}
    <div className="timing-chart-scroll">
      <svg width={width} height="196" role="img" aria-label={gpu ? "Sampled GPU stage times by generation" : "Host stage times by generation"}>
        {[0, .5, 1].map(fraction => <g key={fraction}>
          <line x1="44" x2={width} y1={bottom - fraction * plotHeight} y2={bottom - fraction * plotHeight} stroke="#333" />
          <text x="40" y={bottom - fraction * plotHeight + 3} textAnchor="end" fill="#999" fontSize="9">{gpu ? duration(maximum * fraction).replace(" ", "") : `${(maximum * fraction).toFixed(maximum < 10 ? 1 : 0)}s`}</text>
        </g>)}
        {rows.map((row, index) => {
          let accumulated = 0;
          return <g key={row.generation} tabIndex={0} role="img"
            aria-label={`Generation ${row.generation}, measured stage time ${duration(totals[index])}`}
            onMouseEnter={() => setHovered(row.generation)} onFocus={() => setHovered(row.generation)}>
            {keys.map(name => {
              const seconds = row.stages[name]?.seconds ?? 0;
              const height = seconds / maximum * plotHeight;
              accumulated += height;
              return seconds > 0 ? <rect key={name} x={44 + index * barWidth} y={bottom - accumulated}
                width={Math.max(1, barWidth - 2)} height={height} fill={stageColors[name] ?? "#999"}>
                <title>{`Generation ${row.generation} · ${labels[name] ?? name}: ${duration(seconds)}`}</title>
              </rect> : null;
            })}
            {(index % tickEvery === 0 || index === rows.length - 1) && <text x={44 + (index + .5) * barWidth}
              y="183" textAnchor="middle" fill="#aaa" fontSize="9">{row.generation}</text>}
          </g>;
        })}
      </svg>
    </div>
    <div className="timing-chart-legend">{keys.filter(name => rows.some(row => (row.stages[name]?.seconds ?? 0) > 0)).map(name =>
      <span key={name}><i className="timing-chart-dot" style={{background:stageColors[name] ?? "#999"}} />{labels[name] ?? name}</span>)}</div>
    <p className="timing-note">Generation → · {gpu ? "GPU seconds per sampled macro step" : `host stage seconds ${scope === "all" ? "summed across workers" : "for the winner"}`}. Scroll horizontally for longer runs.</p>
    {active ? <div className="timing-chart-inspect">
      <strong>Generation {active.generation}</strong>
      {Object.entries(active.stages).sort((a,b) => b[1].seconds-a[1].seconds).map(([name, stage]) =>
        <div className="stat-row" key={name}><span><i className="timing-chart-dot" style={{background:stageColors[name] ?? "#999"}} />{labels[name] ?? name}</span><span>{duration(stage.seconds)}</span></div>)}
    </div> : <p className="timing-note">Hover or focus a bar to inspect that generation. The sidebar averages always cover the full run.</p>}
  </div>;
}

export function TrainingTimingPanel({ history, scope, onScopeChange }: { history: TimingEntry[]; scope: string; onScopeChange: (scope: string) => void }) {
  const timing = useMemo(() => averageTimings(history), [history]);
  if (!timing) return <section><h2>Training timing</h2><p className="timing-note">No timing data yet. Measurements appear after a generation completes on the updated trainer.</p></section>;
  const gpu = scope.startsWith("gpu");
  const report = timingReport(timing, scope);
  const stages = Object.entries(report?.stages ?? {}).sort((a, b) => b[1].seconds - a[1].seconds);
  const stageTotal = stages.reduce((sum, [, stage]) => sum + stage.seconds, 0);
  const phases: [string, number][] = [
    ["Worker pool", timing.poolSeconds], ["Selection + mutation", timing.selectionSeconds],
    ["Preview PNGs", timing.previewSeconds], ["Checkpoint files", timing.checkpointSeconds],
    ["Other preparation", timing.otherSeconds],
  ];
  const scopeCount = history.filter(entry => timingReport(entry.timing, scope)).length;
  return <section className="training-timing">
    <h2>Training timing</h2>
    <label className="timing-scope">Inspect <select value={scope} onChange={e => onScopeChange(e.target.value)}>
      <option value="all">CPU time + waits · all workers</option><option value="winner">CPU time + waits · winner</option>
      <option value="gpu-all">GPU pass details · all workers</option><option value="gpu-winner">GPU pass details · winner</option>
    </select></label>
    <h3 className="timing-average-heading">Run averages · {history.length} generations</h3>
    <p className="timing-note">All completed generations with timing data, including startup. Selecting a generation does not change these averages.</p>
    <div className="stat-row"><span>Average generation elapsed</span><strong>{duration(timing.seconds)}</strong></div>
    {phases.map(([label, seconds]) => <div className="stat-row" key={label}><span>{label}</span><span>{duration(seconds)}</span></div>)}
    <p className="timing-note">Through checkpoint saving; excludes history writing and WebSocket delivery. Worker pool includes scheduling, transfer and startup.</p>
    <div className="stat-row"><span>Rollouts per generation</span><span>{timing.rollouts.count.toLocaleString(undefined, {maximumFractionDigits: 1})}</span></div>
    <div className="stat-row"><span>Mean / longest recorded rollout</span><span>{duration(timing.rollouts.meanSeconds)} / {duration(timing.rollouts.maxSeconds)}</span></div>
    <p className="timing-note">Measurements cover {scopeCount} generations. {gpu
      ? `GPU intervals averaged over ${report?.gpu?.samples ?? 0} sampled macro steps. Includes scheduling within intervals; excludes CPU submission and readback. Do not add GPU and host times or subtract them to estimate overhead.`
      : "Host wall time includes GPU waits and transfer overhead. Worker times sum in parallel and can exceed elapsed time."}</p>
    {gpu && !report && <p className="timing-note">GPU timestamps unavailable or disabled in this history. Restart with the updated trainer on a supported adapter.</p>}
    {gpu && !!report?.gpu?.droppedIntervals && <p className="timing-note">{report.gpu.droppedIntervals} timestamp intervals were omitted; this breakdown is incomplete.</p>}
    <h3 className="timing-average-heading">{gpu ? "GPU pass breakdown" : "CPU time and waits"}</h3>
    {!gpu && <p className="timing-note">The completion/readback bucket is CPU waiting, not an individual GPU pass. <button type="button" onClick={() => onScopeChange(scope === "winner" ? "gpu-winner" : "gpu-all")}>Show GPU pass details</button></p>}
    {report && <TimingPie stages={stages} total={stageTotal} gpu={gpu} />}
    <details className="timing-call-details"><summary>Per-call measurements</summary>
    {stages.map(([name, stage]: [string, TimingStage]) => <div className="timing-stage" key={name}>
      <div className="stat-row"><span>{labels[name] ?? name}</span><strong>{duration(stage.seconds)} / {gpu ? "sampled step" : "gen"}</strong></div>
      <progress aria-label={`${labels[name] ?? name} share of average stage time`} value={stage.seconds} max={stageTotal || 1} />
      <div className="timing-note">{stage.count.toLocaleString(undefined, {maximumFractionDigits: 1})} calls/{gpu ? "sampled step" : "gen"} · avg per call {duration(stage.count ? stage.seconds / stage.count : 0)} · max observed {duration(stage.maxSeconds)}</div>
    </div>)}
    </details>
  </section>;
}

import { useState } from "react";
import type { PolarFitnessInfo } from "../gpu/types";
import { generationImageUrl, type GenerationImageKind } from "../net/images";

export function PolarFitnessPanel({ info, apiUrl, runId, generation }: {
  info: PolarFitnessInfo; apiUrl: string; runId: string; generation: number;
}) {
  const [alpha, setAlpha] = useState(false);
  const maxLoss = Math.max(1e-9, ...info.losses.flat());
  const curve = (values: number[]) => values.map((v, i) => `${10+300*i/(values.length-1)},${100-85*v/maxLoss}`).join(" ");
  const chosenLoss = info.losses[info.reflected ? 1 : 0][info.shift];
  const panels: [string, string][] = [["target", "Target (unrotated)"], ["candidate", "Candidate"], ["aligned", "Target (matched)"], ["diff", "Squared difference (all channels)"]];
  const download = `${apiUrl}/runs/${encodeURIComponent(runId)}/images/gen_${String(generation).padStart(5, "0")}_polar.npz`;
  return <section>
    <h2>Polar fitness</h2>
    <p className="hint">Pixel loss {chosenLoss.toPrecision(5)} · {(info.angle*180/Math.PI).toFixed(2)}° CCW · {info.reflected ? "Reflected target" : "Original target"}</p>
    <p className="hint">Horizontal: angle 0–360°. Vertical: radius 0–{info.radius.toFixed(3)}, increasing downward. {info.radialSamples} × {info.angularSamples} samples, {info.channels} channel{info.channels === 1 ? "" : "s"}.</p>
    {info.channels === 4 && <label className="checkbox-row"><input type="checkbox" checked={alpha} onChange={e => setAlpha(e.target.checked)} />Show occupancy channel</label>}
    <div className="snapshot-grid">
      {panels.map(([kind, label]) => <div className="snapshot-item snapshot-item-wide" key={kind}>
        <img style={{ width: "100%", height: "auto", imageRendering: "pixelated" }}
          src={generationImageUrl(apiUrl, runId, generation, `polar_${kind}${alpha && kind !== "diff" ? "_alpha" : ""}` as GenerationImageKind)} alt={label} />
        <span className="snapshot-label">{label}</span>
      </div>)}
    </div>
    <p className="hint">Sharpened values use a fixed −2 to 3 grayscale/RGB scale (zero = 40% gray). Squared difference uses 0–1 (larger errors appear white); black means an exact match.</p>
    <svg viewBox="0 0 320 120" role="img" aria-label="Loss across rotations: blue original target, orange reflected target" style={{ width: "100%" }}>
      <path d="M10 10V100H310" fill="none" stroke="currentColor" opacity=".3" />
      <polyline points={curve(info.losses[0])} fill="none" stroke="#66afff" strokeWidth="1.5" />
      <polyline points={curve(info.losses[1])} fill="none" stroke="#ffac60" strokeWidth="1.5" />
      <circle cx={10+300*info.shift/(info.angularSamples-1)} cy={100-85*chosenLoss/maxLoss} r="3" fill="#ff5555" />
      <text x="10" y="116" fontSize="9" fill="currentColor">0°</text><text x="289" y="116" fontSize="9" fill="currentColor">360°</text>
    </svg>
    <p className="hint">Blue: original. Orange: reflected. Red: selected minimum. Loss range 0–{maxLoss.toPrecision(4)}.</p>
    <a href={download} download>Download exact arrays and loss curves (.npz)</a>
  </section>;
}

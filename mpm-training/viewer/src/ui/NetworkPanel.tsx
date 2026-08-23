import type { PointerEvent as ReactPointerEvent } from "react"
import { useEffect, useMemo, useRef, useState } from "react"
import { evalPolicy } from "../gpu/policyEval"
import type { PhysicsSettings, SimulationConfig, UpdateRuleWeights } from "../gpu/types"
import { Slider } from "./Slider"

interface NetworkPanelProps {
  /** Only `channels`/`hiddenDim`/`weights` are actually read (a plain,
   * static forward-pass input — see this file's own module docstring),
   * but the whole config is simplest to thread through rather than
   * picking those three fields apart at every call site. null before
   * the first generation loads. */
  config: SimulationConfig | null
  /** maxEnvWrite/maxAngularAccel give their output bars' true domains.
   * The former strafe pair is now an unscaled [-1,1] growth-direction
   * and division-polarity signal; maxStrafe only controls its optional
   * physical acceleration. */
  physics: PhysicsSettings | null
}

const SPOT_LABELS = ["Front", "Left", "Back", "Right"]

// How many of the sensed channels get their own manual 2D vector pad —
// capped at 3 purely for sidebar space (each one is a whole square plus
// a value slider); the remaining channels' own value/gradient, and the
// agent's own spawn-relative position, stay pinned at zero in the input
// vector this panel builds (see buildInputVector()'s own comment).
const MANUAL_CHANNELS = 3

// Sweep/pad range for both the vector pads' own dx/dy axes and each
// channel's own raw-value slider — a representative window over the
// sensed-value range this policy was actually evolved against, not a
// hard bound (the chemical field itself is genuinely unbounded, same
// reasoning the old live-probe view's own auto-scaling comment gave).
const DOMAIN = 1

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Diverging cyan (positive) <-> orange (negative) around a dark neutral
 * midpoint — cyan is this app's own existing accent (#7dd3fc, see
 * style.css's own .is-active/.run-picker-item rules), paired with a warm
 * counter-hue for the negative pole rather than an arbitrary third color,
 * so the "brain" panel reads as part of the same UI, not a bolted-on
 * widget with its own palette. */
function activationColor(t: number): string {
  const c = clamp(t, -1, 1)
  if (c >= 0) {
    const r = Math.round(26 + (125 - 26) * c)
    const g = Math.round(26 + (211 - 26) * c)
    const b = Math.round(26 + (252 - 26) * c)
    return `rgb(${r},${g},${b})`
  }
  const s = -c
  const r = Math.round(26 + (251 - 26) * s)
  const g = Math.round(26 + (146 - 26) * s)
  const b = Math.round(26 + (60 - 26) * s)
  return `rgb(${r},${g},${b})`
}

// Matplotlib/ParaView's "coolwarm" (Moreland's diverging blue-white-red)
// — a handful of its own well-known anchor colors, piecewise-linearly
// interpolated, rather than the full 33-entry LUT: close enough by eye
// at the small pad sizes this actually renders at, and this is
// explicitly a rough, illustrative view ("doesn't give all insights"),
// not a publication-grade colorbar.
const COOLWARM_STOPS: Array<[number, number, number, number]> = [
  [0, 59, 76, 192],
  [0.25, 138, 176, 244],
  [0.5, 221, 221, 221],
  [0.75, 245, 156, 125],
  [1, 180, 4, 38],
]

function coolwarm(t: number): string {
  const u = clamp(t, -1, 1) * 0.5 + 0.5 // [-1,1] -> [0,1]
  let i = 0
  while (i < COOLWARM_STOPS.length - 2 && u > COOLWARM_STOPS[i + 1][0]) i++
  const [t0, r0, g0, b0] = COOLWARM_STOPS[i]
  const [t1, r1, g1, b1] = COOLWARM_STOPS[i + 1]
  const f = t1 > t0 ? (u - t0) / (t1 - t0) : 0
  const r = Math.round(r0 + (r1 - r0) * f)
  const g = Math.round(g0 + (g1 - g0) * f)
  const b = Math.round(b0 + (b1 - b0) * f)
  return `rgb(${r},${g},${b})`
}

/** A single signed value as a bipolar bar — fills from the track's own
 * center toward one edge, never the whole track, so magnitude AND sign
 * both read at a glance (a plain 0-to-max bar would need the reader to
 * already know the value's sign from somewhere else). */
function ActivationBar({
  label,
  value,
  domain,
}: {
  label: string
  value: number
  domain: number
}) {
  const t = domain > 0 ? clamp(value / domain, -1, 1) : 0
  const pct = Math.abs(t) * 50
  return (
    <div className="nn-bar-row">
      <span className="nn-bar-label">{label}</span>
      <div className="nn-bar-track">
        <div className="nn-bar-zero" />
        <div
          className="nn-bar-fill"
          style={{
            left: t >= 0 ? "50%" : `${50 - pct}%`,
            width: `${pct}%`,
            background: activationColor(t),
          }}
        />
      </div>
      <span className="nn-bar-value">{value.toFixed(3)}</span>
    </div>
  )
}

// Grid resolution for each pad's own background heatmap — the pad
// itself renders at VECTOR_PAD_SIZE_PX (style.css's own .vector-pad),
// so this is plenty dense for that size while keeping the per-pad
// forward-pass cost (RESOLUTION² evalPolicy() calls, one per cell)
// trivial to redo on every drag/slider tick.
const VECTOR_PAD_HEATMAP_RESOLUTION = 20
const VECTOR_PAD_SIZE_PX = 56

/** One channel's own manually-set (dx, dy) — the local-frame forward/
 * lateral gradient components evalPolicy() reads (see agents.wgsl's own
 * inputVec population: `gx*cosH+gy*sinH`/`-gx*sinH+gy*cosH` — this pad
 * IS that local frame directly, not a world-frame vector needing its
 * own rotation). Square, literally: dragging (or clicking) anywhere in
 * it sets both components at once from the pointer's own position,
 * clamped to [-domain, domain] per axis. Keeps its own drag state via
 * pointer capture (same pattern render/GridCanvas.tsx's own tool
 * handlers use) so a drag that strays outside the square's own bounds
 * mid-gesture still tracks correctly rather than stopping dead at the
 * edge.
 *
 * `heatmap`, when given, paints a coolwarm background UNDER the axes/
 * dot — see buildLeftChannelHeatmap()'s own comment for exactly what
 * it's a sweep of. Deliberately optional (rather than always required)
 * so this component stays reusable as a plain vector picker with no
 * output context at all. */
function VectorPad({
  x,
  y,
  domain,
  heatmap,
  heatmapDomain,
  onChange,
}: {
  x: number
  y: number
  domain: number
  heatmap?: Float32Array
  heatmapDomain?: number
  onChange: (x: number, y: number) => void
}) {
  const padRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !heatmap) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(VECTOR_PAD_SIZE_PX * dpr)
    canvas.height = Math.round(VECTOR_PAD_SIZE_PX * dpr)
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const dom = heatmapDomain && heatmapDomain > 0 ? heatmapDomain : 1
    const cell = VECTOR_PAD_SIZE_PX / VECTOR_PAD_HEATMAP_RESOLUTION
    for (let gy = 0; gy < VECTOR_PAD_HEATMAP_RESOLUTION; gy++) {
      for (let gx = 0; gx < VECTOR_PAD_HEATMAP_RESOLUTION; gx++) {
        const t = clamp(heatmap[gy * VECTOR_PAD_HEATMAP_RESOLUTION + gx] / dom, -1, 1)
        ctx.fillStyle = coolwarm(t)
        // +0.5 past each cell's own footprint — same hairline-seam fix
        // this file's own earlier heatmap rendering used.
        ctx.fillRect(gx * cell, gy * cell, cell + 0.5, cell + 0.5)
      }
    }
  }, [heatmap, heatmapDomain])

  const setFromPointer = (e: ReactPointerEvent) => {
    const pad = padRef.current
    if (!pad) return
    const rect = pad.getBoundingClientRect()
    const px = clamp((e.clientX - rect.left) / rect.width, 0, 1)
    const py = clamp((e.clientY - rect.top) / rect.height, 0, 1)
    // Y: top of the square = +domain, bottom = -domain — "up" reads as
    // positive, the conventional chart orientation (matches this file's
    // own bar fills: positive right/up-toned, negative the other way).
    onChange((px * 2 - 1) * domain, (1 - py * 2) * domain)
  }

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    setFromPointer(e)
  }
  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.buttons === 0) return
    setFromPointer(e)
  }

  const px = ((clamp(x, -domain, domain) / domain) * 0.5 + 0.5) * 100
  const py = (1 - ((clamp(y, -domain, domain) / domain) * 0.5 + 0.5)) * 100

  return (
    <div
      ref={padRef}
      className="vector-pad"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
    >
      {heatmap && <canvas ref={canvasRef} className="vector-pad-heatmap" />}
      <div className="vector-pad-axis vector-pad-axis-h" />
      <div className="vector-pad-axis vector-pad-axis-v" />
      <div
        className="vector-pad-dot"
        style={{ left: `${px}%`, top: `${py}%` }}
      />
    </div>
  )
}

interface ManualChannelInput {
  value: number
  dx: number
  dy: number
}

const ZERO_CHANNEL: ManualChannelInput = { value: 0, dx: 0, dy: 0 }

/** Builds the full IN_DIM input vector evalPolicy() expects
 * ([value×channels, gradForward×channels, gradLateral×channels, dx, dy]
 * — see agents.wgsl's own IN_DIM) from this panel's own manual per-
 * channel state: the first MANUAL_CHANNELS channels get their own
 * dialed-in value/dx/dy, every other channel (and the agent's own
 * spawn-relative position, appended after the per-channel triples)
 * stays pinned at exactly zero. */
function buildInputVector(
  channels: number,
  manual: ManualChannelInput[]
): Float32Array {
  const input = new Float32Array(channels * 3 + 2)
  for (let c = 0; c < Math.min(MANUAL_CHANNELS, channels); c++) {
    input[c] = manual[c].value
    input[channels + c] = manual[c].dx
    input[2 * channels + c] = manual[c].dy
  }
  return input
}

// Which spot's own env-write gets swept into each pad's own background
// heatmap — see buildLeftChannelHeatmaps()'s own comment. "Left" purely
// as a single, fixed, representative slice (out of env-write's full
// spot×channel space) simple enough to paint into a 56px square at a
// glance — not the only output that channel's gradient drives, just
// the one picked to illustrate it.
const LEFT_SPOT_INDEX = SPOT_LABELS.indexOf("Left")

/** For each manually-controlled channel, sweeps THAT channel's own
 * (dx, dy) across the full pad range [-domain, domain]² — holding
 * every other input (that channel's own "value" slider, every OTHER
 * channel's value/dx/dy, position) fixed at whatever `manual` currently
 * has — and records the Left spot's own env-write for that SAME
 * channel at each grid cell. Gives each pad a rough, at-a-glance
 * "what does dragging me actually do" picture — deliberately a single
 * fixed slice of the full output space (one spot, this channel only),
 * not a complete sensitivity map: "doesn't give all insights, but is a
 * simple way to get an understanding." One evalPolicy() call per grid
 * cell per channel (VECTOR_PAD_HEATMAP_RESOLUTION² × manualChannelCount
 * total), cheap enough to just redo on every drag/slider tick. */
function buildLeftChannelHeatmaps(
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number,
  manual: ManualChannelInput[],
  manualChannelCount: number,
  maxEnvWrite: number,
  maxAngularAccel: number,
  maxStrafe: number
): Float32Array[] {
  const res = VECTOR_PAD_HEATMAP_RESOLUTION
  return Array.from({ length: manualChannelCount }, (_, c) => {
    const grid = new Float32Array(res * res)
    const input = buildInputVector(channels, manual)
    for (let gy = 0; gy < res; gy++) {
      // Row 0 = top = +domain — matches VectorPad's own py mapping, so
      // the heatmap's own "up" lines up with where the dot actually
      // goes when dragged up.
      const dy = res > 1 ? DOMAIN - (2 * DOMAIN * gy) / (res - 1) : 0
      for (let gx = 0; gx < res; gx++) {
        const dx = res > 1 ? -DOMAIN + (2 * DOMAIN * gx) / (res - 1) : 0
        input[channels + c] = dx
        input[2 * channels + c] = dy
        const result = evalPolicy(input, weights, channels, hiddenDim, maxEnvWrite, maxAngularAccel, maxStrafe)
        grid[gy * res + gx] = result.envWrite[LEFT_SPOT_INDEX * channels + c]
      }
    }
    return grid
  })
}

/** "Brain" view of the CURRENT generation's own policy — NOT a live
 * particle probe: the first MANUAL_CHANNELS sensed channels are each
 * dialed in by hand here (a square pad for that channel's own local-
 * frame gradient, plus a slider for its own raw sensed value), every
 * other input stays at zero, and the whole output (env-write per spot/
 * channel + turn/strafe) is evalPolicy()'s (gpu/policyEval.ts,
 * mirroring core/agents.wgsl) own response to THAT exact vector,
 * recomputed live on every pad drag/slider tick — cheap enough (one
 * forward pass, no sweep) to do inline via useMemo, no probe timer
 * needed. Purely a display — nothing here feeds back into the sim. */
export function NetworkPanel({ config, physics }: NetworkPanelProps) {
  const channels = config?.channels ?? 0
  const hiddenDim = config?.hiddenDim ?? 0
  const maxEnvWrite = physics?.maxEnvWrite ?? 1
  const maxAngularAccel = physics?.maxAngularAccel ?? 1
  const maxStrafe = physics?.maxStrafe ?? 1

  const [manual, setManual] = useState<ManualChannelInput[]>(
    Array.from({ length: MANUAL_CHANNELS }, () => ({ ...ZERO_CHANNEL }))
  )
  const updateChannel = (c: number, patch: Partial<ManualChannelInput>) => {
    setManual((prev) =>
      prev.map((ch, i) => (i === c ? { ...ch, ...patch } : ch))
    )
  }

  const output = useMemo(() => {
    if (!config?.weights || channels < 1 || hiddenDim < 1) return null
    const input = buildInputVector(channels, manual)
    return evalPolicy(
      input,
      config.weights,
      channels,
      hiddenDim,
      maxEnvWrite,
      maxAngularAccel,
      maxStrafe
    )
  }, [
    config?.weights,
    channels,
    hiddenDim,
    manual,
    maxEnvWrite,
    maxAngularAccel,
    maxStrafe,
  ])

  const manualChannelCount = Math.min(MANUAL_CHANNELS, channels)

  const leftHeatmaps = useMemo(() => {
    if (!config?.weights || manualChannelCount < 1 || hiddenDim < 1) return null
    return buildLeftChannelHeatmaps(
      config.weights,
      channels,
      hiddenDim,
      manual,
      manualChannelCount,
      maxEnvWrite,
      maxAngularAccel,
      maxStrafe
    )
  }, [
    config?.weights,
    channels,
    hiddenDim,
    manual,
    manualChannelCount,
    maxEnvWrite,
    maxAngularAccel,
    maxStrafe,
  ])

  if (!config?.weights) {
    return (
      <section className="nn-panel">
        <h2>Network</h2>
        <p className="hint">Waiting for weights…</p>
      </section>
    )
  }

  return (
    <section className="nn-panel">
      <h2>Network</h2>
      <div className="nn-block">
        <h3>Input</h3>
        <p className="hint">
          Background: this channel's own Left-deposit output, swept across the pad (its own value + every other
          channel held as set). One slice, not the full picture.
        </p>
        {Array.from({ length: manualChannelCount }, (_, c) => (
          <div key={c} className="nn-vector-channel">
            <span className="nn-group-label">ch{c}</span>
            <div className="nn-vector-row">
              <VectorPad
                x={manual[c].dx}
                y={manual[c].dy}
                domain={DOMAIN}
                heatmap={leftHeatmaps?.[c]}
                heatmapDomain={maxEnvWrite}
                onChange={(dx, dy) => updateChannel(c, { dx, dy })}
              />
              <div className="nn-vector-readout">
                <div className="nn-vector-readout-row">
                  <span>dx</span>
                  <span className="nn-vector-readout-value">
                    {manual[c].dx.toFixed(2)}
                  </span>
                </div>
                <div className="nn-vector-readout-row">
                  <span>dy</span>
                  <span className="nn-vector-readout-value">
                    {manual[c].dy.toFixed(2)}
                  </span>
                </div>
                <label className="slider-row">
                  <span>value</span>
                  <Slider
                    min={-DOMAIN}
                    max={DOMAIN}
                    step={0.01}
                    value={manual[c].value}
                    onChange={(v) => updateChannel(c, { value: v })}
                  />
                  <span className="slider-value">
                    {manual[c].value.toFixed(2)}
                  </span>
                </label>
              </div>
            </div>
          </div>
        ))}
      </div>

      {output && (
        <>
          <div className="nn-block">
            <h3>Output — env write</h3>
            {SPOT_LABELS.map((label, s) => (
              <div key={label} className="nn-group">
                <span className="nn-group-label">{label} deposit</span>
                {Array.from({ length: channels }, (_, c) => (
                  <ActivationBar
                    key={c}
                    label={`ch${c}`}
                    value={output.envWrite[s * channels + c]}
                    domain={maxEnvWrite}
                  />
                ))}
              </div>
            ))}
          </div>

          <div className="nn-block">
            <h3>Output — growth direction</h3>
            <div className="nn-group">
              <ActivationBar
                label="turn"
                value={output.angularAccel}
                domain={maxAngularAccel}
              />
              <ActivationBar
                label="direction x"
                value={output.strafe[0]}
                domain={1}
              />
              <ActivationBar
                label="direction y"
                value={output.strafe[1]}
                domain={1}
              />
            </div>
          </div>
        </>
      )}
    </section>
  )
}

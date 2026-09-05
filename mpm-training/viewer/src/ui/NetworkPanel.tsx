import type { PointerEvent as ReactPointerEvent } from "react"
import { useEffect, useMemo, useRef, useState } from "react"
import { evalPolicy, policyWeightsShapeError } from "../gpu/policyEval"
import { policyHasRecurrence, type PhysicsSettings, type PolicyArchitecture, type SimulationConfig, type UpdateRuleWeights } from "../gpu/types"
import { Slider } from "./Slider"

interface NetworkPanelProps {
  /** Only `channels`/`hiddenDim`/`weights` are actually read (a plain,
   * static forward-pass input — see this file's own module docstring),
   * but the whole config is simplest to thread through rather than
   * picking those three fields apart at every call site. null before
   * the first generation loads. */
  config: SimulationConfig | null
  /** maxEnvWrite/maxAngularAccel give their output bars' true domains. */
  physics: PhysicsSettings | null
}

// How many of the sensed channels get their own manual 2D vector pad —
// capped at 3 purely for sidebar space (each one is a whole square plus
// a value slider); the remaining channels' values/gradients stay pinned
// at zero in the input vector this panel builds.
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

// Viridis anchors, piecewise-linearly interpolated. The signed neural
// response range [-1,1] maps across the sequential palette from purple to
// yellow; unlike coolwarm, zero is green/teal rather than nearly white.
const VIRIDIS_STOPS: Array<[number, number, number, number]> = [
  [0, 68, 1, 84],
  [0.25, 59, 82, 139],
  [0.5, 33, 145, 140],
  [0.75, 94, 201, 98],
  [1, 253, 231, 37],
]

function viridis(t: number): string {
  const u = clamp(t, -1, 1) * 0.5 + 0.5 // [-1,1] -> [0,1]
  let i = 0
  while (i < VIRIDIS_STOPS.length - 2 && u > VIRIDIS_STOPS[i + 1][0]) i++
  const [t0, r0, g0, b0] = VIRIDIS_STOPS[i]
  const [t1, r1, g1, b1] = VIRIDIS_STOPS[i + 1]
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
 * `heatmap`, when given, paints a Viridis background UNDER the axes/
 * dot — see buildChannelHeatmaps()'s own comment for exactly what
 * it's a sweep of. Deliberately optional (rather than always required)
 * so this component stays reusable as a plain vector picker with no
 * output context at all. */
function VectorPad({
  x,
  y,
  domain,
  heatmap,
  onChange,
}: {
  x: number
  y: number
  domain: number
  heatmap?: Float32Array
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
    // Per-preview contrast normalization: the old absolute
    // output/maxEnvWrite mapping compressed freshly initialized policies
    // into a nearly constant middle color. Center the observed range and
    // stretch its extrema across Viridis so the spatial response pattern is
    // visible. A truly constant map remains at the neutral midpoint.
    let minValue = Infinity
    let maxValue = -Infinity
    for (const value of heatmap) {
      minValue = Math.min(minValue, value)
      maxValue = Math.max(maxValue, value)
    }
    const midpoint = (minValue + maxValue) * 0.5
    const halfRange = (maxValue - minValue) * 0.5
    const cell = VECTOR_PAD_SIZE_PX / VECTOR_PAD_HEATMAP_RESOLUTION
    for (let gy = 0; gy < VECTOR_PAD_HEATMAP_RESOLUTION; gy++) {
      for (let gx = 0; gx < VECTOR_PAD_HEATMAP_RESOLUTION; gx++) {
        const value = heatmap[gy * VECTOR_PAD_HEATMAP_RESOLUTION + gx]
        const t = halfRange > 1e-7 ? clamp((value - midpoint) / halfRange, -1, 1) : 0
        ctx.fillStyle = viridis(t)
        // +0.5 past each cell's own footprint — same hairline-seam fix
        // this file's own earlier heatmap rendering used.
        ctx.fillRect(gx * cell, gy * cell, cell + 0.5, cell + 0.5)
      }
    }
  }, [heatmap])

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

interface ManualElasticInput {
  volume: number
  axial: number
  shear: number
}

const ZERO_CHANNEL: ManualChannelInput = { value: 0, dx: 0, dy: 0 }

/** Builds the full IN_DIM input vector evalPolicy() expects
 * ([value×channels, gradForward×channels, gradLateral×channels,
 * morphology×3, elastic Hencky strain×3] — see
 * agents.wgsl's own IN_DIM) from this panel's manual per-channel state.
 * The first MANUAL_CHANNELS channels get their own dialed-in value/dx/dy;
 * every other channel stays pinned at exactly zero. */
function buildInputVector(
  channels: number,
  manual: ManualChannelInput[],
  morphology: ManualChannelInput,
  elastic: ManualElasticInput
): Float32Array {
  const input = new Float32Array(channels * 3 + 6)
  for (let c = 0; c < Math.min(MANUAL_CHANNELS, channels); c++) {
    input[c] = manual[c].value
    input[channels + c] = manual[c].dx
    input[2 * channels + c] = manual[c].dy
  }
  input[3 * channels] = morphology.value
  input[3 * channels + 1] = morphology.dx
  input[3 * channels + 2] = morphology.dy
  input[3 * channels + 3] = elastic.volume
  input[3 * channels + 4] = elastic.axial
  input[3 * channels + 5] = elastic.shear
  return input
}

/** For each manually-controlled channel, sweeps THAT channel's own
 * (dx, dy) across the full pad range [-domain, domain]² — holding
 * every other input (that channel's own "value" slider and every OTHER
 * channel's value/dx/dy) fixed at whatever `manual` currently
 * has — and records the front env-write for that SAME channel at each
 * grid cell. Gives each pad a rough, at-a-glance
 * "what does dragging me actually do" picture. One evalPolicy() call per grid
 * cell per channel (VECTOR_PAD_HEATMAP_RESOLUTION² × manualChannelCount
 * total), cheap enough to just redo on every drag/slider tick. */
function buildChannelHeatmaps(
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number,
  manual: ManualChannelInput[],
  morphology: ManualChannelInput,
  elastic: ManualElasticInput,
  manualChannelCount: number,
  maxEnvWrite: number,
  maxAngularAccel: number,
  maxStrafe: number,
  architecture: PolicyArchitecture,
): Float32Array[] {
  const res = VECTOR_PAD_HEATMAP_RESOLUTION
  return Array.from({ length: manualChannelCount }, (_, c) => {
    const grid = new Float32Array(res * res)
    const input = buildInputVector(channels, manual, morphology, elastic)
    for (let gy = 0; gy < res; gy++) {
      // Row 0 = top = +domain — matches VectorPad's own py mapping, so
      // the heatmap's own "up" lines up with where the dot actually
      // goes when dragged up.
      const dy = res > 1 ? DOMAIN - (2 * DOMAIN * gy) / (res - 1) : 0
      for (let gx = 0; gx < res; gx++) {
        const dx = res > 1 ? -DOMAIN + (2 * DOMAIN * gx) / (res - 1) : 0
        input[channels + c] = dx
        input[2 * channels + c] = dy
        const result = evalPolicy(input, weights, channels, hiddenDim, maxEnvWrite, maxAngularAccel, maxStrafe, architecture)
        grid[gy * res + gx] = result.envWrite[c]
      }
    }
    return grid
  })
}

/** "Brain" view of the CURRENT generation's own policy — NOT a live
 * particle probe: the first MANUAL_CHANNELS policy-normalized channels are
 * each dialed in by hand here (a square pad for that channel's normalized
 * local-frame gradient, plus a slider for its normalized sensed value), every
 * other input stays at zero, and the whole output (signed chemical
 * deltas per channel + turn/growth controls + RGB) is evalPolicy()'s (gpu/policyEval.ts,
 * mirroring core/agents.wgsl) own response to THAT exact vector,
 * recomputed live on every pad drag/slider tick — cheap enough (one
 * forward pass, no sweep) to do inline via useMemo, no probe timer
 * needed. Purely a display — nothing here feeds back into the sim. */
export function NetworkPanel({ config, physics }: NetworkPanelProps) {
  const channels = config?.channels ?? 0
  const hiddenDim = config?.hiddenDim ?? 0
  const architecture = config?.policyArchitecture ?? "stateless-128"
  const maxEnvWrite = physics?.maxEnvWrite ?? 1
  const maxAngularAccel = physics?.maxAngularAccel ?? 1
  const maxStrafe = physics?.maxStrafe ?? 1
  const elasticInputsEnabled = config?.elasticStrainInputsEnabled ?? false
  const weightsError = config?.weights
    ? policyWeightsShapeError(config.weights, channels, hiddenDim, architecture)
    : null

  const [manual, setManual] = useState<ManualChannelInput[]>(
    Array.from({ length: MANUAL_CHANNELS }, () => ({ ...ZERO_CHANNEL }))
  )
  const [morphology, setMorphology] = useState<ManualChannelInput>({ ...ZERO_CHANNEL })
  const [elastic, setElastic] = useState<ManualElasticInput>({ volume: 0, axial: 0, shear: 0 })
  const updateChannel = (c: number, patch: Partial<ManualChannelInput>) => {
    setManual((prev) =>
      prev.map((ch, i) => (i === c ? { ...ch, ...patch } : ch))
    )
  }

  const output = useMemo(() => {
    if (!config?.weights || weightsError || channels < 1 || hiddenDim < 1) return null
    const input = buildInputVector(
      channels, manual, morphology,
      elasticInputsEnabled ? elastic : { volume: 0, axial: 0, shear: 0 }
    )
    return evalPolicy(
      input,
      config.weights,
      channels,
      hiddenDim,
      maxEnvWrite,
      maxAngularAccel,
      maxStrafe,
      architecture
    )
  }, [
    config?.weights,
    channels,
    hiddenDim,
    manual,
    morphology,
    elastic,
    elasticInputsEnabled,
    maxEnvWrite,
    maxAngularAccel,
    maxStrafe,
    architecture,
    weightsError,
  ])

  const manualChannelCount = Math.min(MANUAL_CHANNELS, channels)

  const channelHeatmaps = useMemo(() => {
    if (!config?.weights || weightsError || manualChannelCount < 1 || hiddenDim < 1) return null
    return buildChannelHeatmaps(
      config.weights,
      channels,
      hiddenDim,
      manual,
      morphology,
      elasticInputsEnabled ? elastic : { volume: 0, axial: 0, shear: 0 },
      manualChannelCount,
      maxEnvWrite,
      maxAngularAccel,
      maxStrafe,
      architecture
    )
  }, [
    config?.weights,
    channels,
    hiddenDim,
    manual,
    morphology,
    elastic,
    elasticInputsEnabled,
    manualChannelCount,
    maxEnvWrite,
    maxAngularAccel,
    maxStrafe,
    architecture,
    weightsError,
  ])

  if (!config?.weights) {
    return (
      <section className="nn-panel">
        <h2>Network</h2>
        <p className="hint">Waiting for weights…</p>
      </section>
    )
  }

  if (weightsError) {
    return (
      <section className="nn-panel">
        <h2>Network</h2>
        <p className="hint">{weightsError}</p>
      </section>
    )
  }

  return (
    <section className="nn-panel">
      <h2>Network</h2>
      <div className="nn-block">
        <h3>Input</h3>
        <p className="hint">
          Background: this channel's signed chemical delta, swept across the pad (its own value + every other
          channel held as set). Contrast is exaggerated independently per pad
          by stretching its observed min/max across Viridis. This shows
          response shape, not absolute output strength.
        </p>
        {Array.from({ length: manualChannelCount }, (_, c) => (
          <div key={c} className="nn-vector-channel">
            <span className="nn-group-label">ch{c}</span>
            <div className="nn-vector-row">
              <VectorPad
                x={manual[c].dx}
                y={manual[c].dy}
                domain={DOMAIN}
                heatmap={channelHeatmaps?.[c]}
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
        <div className="nn-vector-channel">
          <span className="nn-group-label">morphology</span>
          <div className="nn-vector-row">
            <VectorPad
              x={morphology.dx}
              y={morphology.dy}
              domain={1}
              onChange={(dx, dy) => setMorphology((prev) => ({ ...prev, dx, dy }))}
            />
            <div className="nn-vector-readout">
              <div className="nn-vector-readout-row"><span>forward</span><span className="nn-vector-readout-value">{morphology.dx.toFixed(2)}</span></div>
              <div className="nn-vector-readout-row"><span>lateral</span><span className="nn-vector-readout-value">{morphology.dy.toFixed(2)}</span></div>
              <label className="slider-row">
                <span>normalized occupancy</span>
                <Slider min={-1} max={1} step={0.01} value={morphology.value} onChange={(value) => setMorphology((prev) => ({ ...prev, value }))} />
                <span className="slider-value">{morphology.value.toFixed(2)}</span>
              </label>
            </div>
          </div>
        </div>
        <div className="nn-vector-channel">
          <span className="nn-group-label">elastic strain</span>
          <div className="nn-vector-readout">
            <label className="slider-row">
              <span>volume</span>
              <Slider min={-1} max={1} step={0.01} value={elastic.volume} disabled={!elasticInputsEnabled} onChange={(volume) => setElastic((prev) => ({ ...prev, volume }))} />
              <span className="slider-value">{elastic.volume.toFixed(2)}</span>
            </label>
            <label className="slider-row">
              <span>forward − lateral</span>
              <Slider min={-1} max={1} step={0.01} value={elastic.axial} disabled={!elasticInputsEnabled} onChange={(axial) => setElastic((prev) => ({ ...prev, axial }))} />
              <span className="slider-value">{elastic.axial.toFixed(2)}</span>
            </label>
            <label className="slider-row">
              <span>shear</span>
              <Slider min={-1} max={1} step={0.01} value={elastic.shear} disabled={!elasticInputsEnabled} onChange={(shear) => setElastic((prev) => ({ ...prev, shear }))} />
              <span className="slider-value">{elastic.shear.toFixed(2)}</span>
            </label>
          </div>
          <p className="hint">
            {elasticInputsEnabled
              ? "Normalized channel-7-gradient-frame Hencky strain as received by the policy."
              : "Temporarily unwired: all three policy lanes are forced to zero."}
          </p>
        </div>
      </div>

      {output && (
        <>
          <div className="nn-block">
            <h3>Output — chemical delta</h3>
            <div className="nn-group">
              <span className="nn-group-label">Under particle</span>
              {Array.from({ length: channels }, (_, c) => (
                <ActivationBar
                  key={c}
                  label={`ch${c}`}
                  value={output.envWrite[c]}
                  domain={maxEnvWrite}
                />
              ))}
            </div>
          </div>

          <div className="nn-block">
            <h3>Output — local growth vector</h3>
            <div className="nn-group">
              <ActivationBar
                label="local forward"
                value={output.growthVector[0]}
                domain={1}
              />
              <ActivationBar
                label="local lateral"
                value={output.growthVector[1]}
                domain={1}
              />
              <ActivationBar
                label="magnitude"
                value={Math.min(1, Math.hypot(...output.growthVector))}
                domain={1}
              />
            </div>
          </div>

          {policyHasRecurrence(architecture) && (
            <div className="nn-block">
              <h3>Output — private-state update</h3>
              <p className="hint">Candidate residuals and gates at zero private state.</p>
              <div className="nn-group">
                {Array.from(output.stateDelta, (value, i) => (
                  <ActivationBar key={`state-delta-${i}`} label={`Δs${i}`} value={value} domain={1} />
                ))}
                {Array.from(output.stateGate, (value, i) => (
                  <ActivationBar key={`state-gate-${i}`} label={`gate ${i}`} value={value} domain={1} />
                ))}
              </div>
            </div>
          )}

          <div className="nn-block">
            <h3>{policyHasRecurrence(architecture) ? "Derived color — zero private state" : "Output — cell color"}</h3>
            <div
              aria-label="Current neural RGB color"
              style={{
                height: "2rem",
                borderRadius: "0.35rem",
                backgroundColor: `rgb(${output.color.map((v) => Math.round(v * 255)).join(",")})`,
                border: "1px solid rgba(255,255,255,0.18)",
                marginBottom: "0.5rem",
              }}
            />
            <div className="nn-group">
              <ActivationBar label="red" value={output.color[0]} domain={1} />
              <ActivationBar label="green" value={output.color[1]} domain={1} />
              <ActivationBar label="blue" value={output.color[2]} domain={1} />
            </div>
          </div>
        </>
      )}
    </section>
  )
}

import { useEffect, useRef } from "react"
import type { NetworkProbe } from "../gpu/nnProbe"
import type { PhysicsSettings } from "../gpu/types"

interface NetworkPanelProps {
  /** Latest forward-pass snapshot (gpu/nnProbe.ts), refreshed on its own
   * timer by GridCanvas (see PROBE_INTERVAL_MS there) — null before the
   * first one resolves, or right after a run/generation change reshapes
   * channels/hiddenDim out from under a stale one (see TrainingView's
   * own reset alongside targetPoints/physicsOverride). */
  probe: NetworkProbe | null
  /** maxEnvWrite/maxAngularAccel/maxStrafe give the output bars their
   * TRUE domain — evalPolicy() (core/agents.wgsl) literally scales its
   * squashed output by these, so ±max is the real range those values can
   * ever reach, not an approximation picked after the fact. */
  physics: PhysicsSettings | null
}

const SPOT_LABELS = ["Front", "Left", "Back", "Right"]
const INPUT_GROUP_LABELS = ["Value", "Fwd grad", "Lat grad"]

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

function maxAbs(values: Iterable<number>, fallback: number): number {
  let m = 0
  for (const v of values) m = Math.max(m, Math.abs(v))
  return m > 1e-9 ? m : fallback
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

/** A single signed value as a bipolar bar — fills from the track's own
 * center toward one edge, never the whole track, so magnitude AND sign
 * both read at a glance (a plain 0-to-max bar would need the reader to
 * already know the value's sign from somewhere else). */
function ActivationBar({ label, value, domain }: { label: string; value: number; domain: number }) {
  const t = domain > 0 ? clamp(value / domain, -1, 1) : 0
  const pct = Math.abs(t) * 50
  return (
    <div className="nn-bar-row">
      <span className="nn-bar-label">{label}</span>
      <div className="nn-bar-track">
        <div className="nn-bar-zero" />
        <div
          className="nn-bar-fill"
          style={{ left: t >= 0 ? "50%" : `${50 - pct}%`, width: `${pct}%`, background: activationColor(t) }}
        />
      </div>
      <span className="nn-bar-value">{value.toFixed(3)}</span>
    </div>
  )
}

/** Hidden layer as a compact heatmap grid (a "cortex scan" more than a
 * wiring diagram) — with hiddenDim typically in the hundreds, drawing
 * every input↔hidden↔output edge (thousands of lines, redrawn on every
 * probe tick) reads as spaghetti and costs far more than it adds; a
 * colored-cell grid on a plain <canvas> (cheap per-cell fills, no DOM
 * nodes) is both cheaper and more legible for "what's lit up right now."
 * Roughly square (not a single tall column) purely so it reads as one
 * cohesive block rather than a scrollable strip. */
function HiddenGrid({ hidden }: { hidden: Float32Array }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const cols = Math.max(1, Math.ceil(Math.sqrt(hidden.length * 1.4)))
    const rows = Math.max(1, Math.ceil(hidden.length / cols))
    const dpr = window.devicePixelRatio || 1
    const cssWidth = canvas.clientWidth || 240
    const cssHeight = Math.round((cssWidth / cols) * rows)
    canvas.style.height = `${cssHeight}px`
    canvas.width = Math.round(cssWidth * dpr)
    canvas.height = Math.round(cssHeight * dpr)
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = "#0d0d0d"
    ctx.fillRect(0, 0, cssWidth, cssHeight)
    const cellW = cssWidth / cols
    const cellH = cssHeight / rows
    const pad = Math.max(0.5, Math.min(cellW, cellH) * 0.14)
    for (let i = 0; i < hidden.length; i++) {
      const col = i % cols
      const row = Math.floor(i / cols)
      // Hidden activations are safeTanh() outputs — already exactly
      // bounded to [-1,1], so no domain normalization needed here (unlike
      // ActivationBar's own inputs/outputs).
      ctx.fillStyle = activationColor(hidden[i])
      ctx.fillRect(col * cellW + pad, row * cellH + pad, cellW - pad * 2, cellH - pad * 2)
    }
  }, [hidden])

  return <canvas ref={canvasRef} className="nn-hidden-grid" />
}

/** Live "brain" view of the update rule's own forward pass for the
 * rollout's original seed particle (gpu/nnProbe.ts's own probe(), always
 * particle 0) — sensed input, hidden-layer activations, and the squashed/
 * scaled output the network actually acts on this instant, refreshed
 * every PROBE_INTERVAL_MS (see render/GridCanvas.tsx). Purely a display —
 * nothing here feeds back into the sim. */
export function NetworkPanel({ probe, physics }: NetworkPanelProps) {
  if (!probe) {
    return (
      <section>
        <h2>Network</h2>
        <p className="hint">Waiting for the first probe…</p>
      </section>
    )
  }

  // Sensed field value/gradient are genuinely unbounded (the chemical
  // field itself isn't clamped to any fixed range — see core/agents.wgsl's
  // own comment on MAX_ENV_WRITE), so the input bars auto-scale to
  // whatever's actually showing up right now rather than an arbitrary
  // fixed domain; the output groups use the REAL physics ceiling instead
  // (see this component's own props docstring) whenever it's known.
  const inputDomain = maxAbs(probe.input, 1)
  const maxEnvWrite = physics?.maxEnvWrite ?? maxAbs(probe.envWrite, 1)
  const maxAngularAccel = physics?.maxAngularAccel ?? maxAbs([probe.angularAccel], 1)
  const maxStrafe = physics?.maxStrafe ?? maxAbs(probe.strafe, 1)

  return (
    <section className="nn-panel">
      <h2>Network</h2>

      <div className="nn-block">
        <h3>Input (particle 0)</h3>
        {INPUT_GROUP_LABELS.map((label, g) => (
          <div key={label} className="nn-group">
            <span className="nn-group-label">{label}</span>
            {Array.from({ length: probe.channels }, (_, c) => (
              <ActivationBar key={c} label={`ch${c}`} value={probe.input[g * probe.channels + c]} domain={inputDomain} />
            ))}
          </div>
        ))}
      </div>

      <div className="nn-block">
        <h3>Hidden ({probe.hiddenDim})</h3>
        <HiddenGrid hidden={probe.hidden} />
      </div>

      <div className="nn-block">
        <h3>Output</h3>
        {SPOT_LABELS.map((label, s) => (
          <div key={label} className="nn-group">
            <span className="nn-group-label">{label} deposit</span>
            {Array.from({ length: probe.channels }, (_, c) => (
              <ActivationBar key={c} label={`ch${c}`} value={probe.envWrite[s * probe.channels + c]} domain={maxEnvWrite} />
            ))}
          </div>
        ))}
        <div className="nn-group">
          <span className="nn-group-label">Motion</span>
          <ActivationBar label="turn" value={probe.angularAccel} domain={maxAngularAccel} />
          <ActivationBar label="strafe x" value={probe.strafe[0]} domain={maxStrafe} />
          <ActivationBar label="strafe y" value={probe.strafe[1]} domain={maxStrafe} />
        </div>
      </div>

      <div className="stat-row">
        <span>Split probability</span>
        <span>{(probe.splitProb * 100).toFixed(1)}%</span>
      </div>
    </section>
  )
}

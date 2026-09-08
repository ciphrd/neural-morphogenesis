import { useMemo, useState } from "react"
import { MAX_SUPPORTED_DENSITY, MIN_SUPPORTED_DENSITY } from "../gpu/density"
import type { PhysicsSettings } from "../gpu/types"
import { GROWTH_SLIDER_SPECS } from "./GrowthPanel"
import { physicsSliderSpecsFor } from "./PhysicsPanel"

export type SweepParameterKey =
  | Exclude<keyof PhysicsSettings, "mpmEnabled" | "normalizeDepositsByLocalDensity">
  | "particleDensityMultiplier"
  | "substrateResolution"

export interface SweepAxis {
  key: SweepParameterKey
  label: string
  min: number
  max: number
  step: number
  /** Explicit values override min/max/step and preserve the entered order. */
  values?: number[]
}

export interface SampleSweepRequest {
  axes: SweepAxis[]
  steps: number
  includeJson: boolean
}

interface Props {
  current: PhysicsSettings
  currentDensity: number
  currentSubstrateResolution: number
  defaultSteps: number
  running: boolean
  completed: number
  total: number
  error: string | null
  onClose: () => void
  onRun: (request: SampleSweepRequest) => void
  onCancel: () => void
}

interface AxisDraft {
  key: SweepAxis["key"]
  mode: "range" | "values"
  min: string
  max: string
  step: string
  rawValues: string
}

const MAX_SAMPLES = 1000

function countValues(min: number, max: number, step: number): number {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(step)) return 0
  if (step <= 0 || max < min) return 0
  return Math.floor((max - min) / step + 1e-9) + 1
}

function parseNumber(value: string): number {
  return value.trim() === "" ? Number.NaN : Number(value)
}

export function sweepValues(axis: SweepAxis): number[] {
  if (axis.values) return [...axis.values]
  const count = countValues(axis.min, axis.max, axis.step)
  return Array.from({ length: count }, (_, index) => {
    const value = axis.min + index * axis.step
    return Math.abs(value) < 1e-14 ? 0 : Number(value.toPrecision(12))
  })
}

function parseRawValues(input: string): number[] {
  const tokens = input.trim().split(/[\s,;]+/).filter(Boolean)
  if (tokens.length === 0) return []
  const values = tokens.map(Number)
  return values.every(Number.isFinite) ? values : []
}

export function SampleSweepModal({
  current,
  currentDensity,
  currentSubstrateResolution,
  defaultSteps,
  running,
  completed,
  total,
  error,
  onClose,
  onRun,
  onCancel,
}: Props) {
  const specs = useMemo(() => [
    {
      key: "particleDensityMultiplier" as const,
      label: "Particle density",
      min: MIN_SUPPORTED_DENSITY,
      max: MAX_SUPPORTED_DENSITY,
      step: 0.5,
      group: "Density",
    },
    {
      key: "substrateResolution" as const,
      label: "Substrate resolution",
      min: 64,
      max: 2048,
      step: 64,
      group: "Simulation",
    },
    ...physicsSliderSpecsFor(current).map((spec) => ({ ...spec, group: "Physics" })),
    ...GROWTH_SLIDER_SPECS.map((spec) => ({ ...spec, group: "Growth" })),
  ], [current])
  const firstSpec = specs[0]
  const currentValue = (key: SweepAxis["key"]): number =>
    key === "particleDensityMultiplier"
      ? currentDensity
      : key === "substrateResolution"
        ? currentSubstrateResolution
        : current[key]
  const makeDraft = (key: SweepAxis["key"]): AxisDraft => {
    const spec = specs.find((candidate) => candidate.key === key) ?? firstSpec
    const value = currentValue(spec.key)
    return {
      key: spec.key,
      mode: "range",
      min: String(value),
      max: String(value),
      step: String(spec.step),
      rawValues: String(value),
    }
  }
  const [axes, setAxes] = useState<AxisDraft[]>(() => [makeDraft(firstSpec.key)])
  const [steps, setSteps] = useState(String(defaultSteps))
  const [includeJson, setIncludeJson] = useState(false)

  const parsedAxes = axes.map((axis): SweepAxis => {
    const spec = specs.find((candidate) => candidate.key === axis.key)!
    return {
      key: axis.key,
      label: spec.label,
      min: parseNumber(axis.min),
      max: parseNumber(axis.max),
      step: parseNumber(axis.step),
      values: axis.mode === "values" ? parseRawValues(axis.rawValues) : undefined,
    }
  })
  const axisValues = parsedAxes.map(sweepValues)
  const counts = axisValues.map((values) => values.length)
  const sampleCount = counts.reduce((product, count) => product * count, 1)
  const parsedSteps = parseNumber(steps)
  const duplicateKeys = new Set(axes.map((axis) => axis.key)).size !== axes.length
  const invalidExplicitValues = axes.some(
    (axis, index) => axis.mode === "values" && axisValues[index].length === 0
  )
  const validSubstrateValues = parsedAxes.every((axis, index) =>
    axis.key !== "substrateResolution"
      || axisValues[index].every((value) => Number.isInteger(value) && value > 0)
  )
  const valid = !duplicateKeys && !invalidExplicitValues && validSubstrateValues && counts.every((count) => count > 0) &&
    sampleCount <= MAX_SAMPLES && Number.isInteger(parsedSteps) && parsedSteps > 0

  const updateAxis = (index: number, patch: Partial<AxisDraft>) => {
    setAxes((old) => old.map((axis, candidate) => candidate === index ? { ...axis, ...patch } : axis))
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !running) onClose()
    }}>
      <div className="sample-modal" role="dialog" aria-modal="true" aria-labelledby="sample-modal-title">
        <div className="sample-modal-header">
          <div>
            <h2 id="sample-modal-title">Collect parameter samples</h2>
            <p className="hint">Sweep 1–3 live simulation settings with the currently loaded brain.</p>
          </div>
          <button className="icon-button" onClick={onClose} disabled={running} aria-label="Close">×</button>
        </div>

        <div className="sample-axis-list">
          {axes.map((axis, index) => (
            <div className="sample-axis" key={index}>
              <div className="sample-axis-heading">
                <strong>Parameter {index + 1}</strong>
                {axes.length > 1 && (
                  <button className="sample-link-button" onClick={() => setAxes((old) => old.filter((_, i) => i !== index))} disabled={running}>
                    Remove
                  </button>
                )}
              </div>
              <select
                className="select"
                value={axis.key}
                disabled={running}
                onChange={(event) => {
                  const key = event.currentTarget.value as SweepAxis["key"]
                  updateAxis(index, makeDraft(key))
                }}
              >
                {["Simulation", "Density", "Physics", "Growth"].map((group) => (
                  <optgroup label={group} key={group}>
                    {specs.filter((spec) => spec.group === group).map((spec) => (
                      <option value={spec.key} key={spec.key}>{spec.label}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={axis.mode === "values"}
                  disabled={running}
                  onChange={(event) => updateAxis(index, {
                    mode: event.currentTarget.checked ? "values" : "range",
                  })}
                />
                Enter explicit values
              </label>
              {axis.mode === "values" ? (
                <label className="sample-values-row">
                  <span>Values (comma or space separated)</span>
                  <input
                    className="number-input"
                    type="text"
                    value={axis.rawValues}
                    placeholder="64, 128, 256, 512"
                    disabled={running}
                    onChange={(event) => updateAxis(index, { rawValues: event.currentTarget.value })}
                  />
                </label>
              ) : (
                <div className="sample-range-grid">
                  {(["min", "max", "step"] as const).map((field) => (
                    <label key={field}>
                      <span>{field[0].toUpperCase() + field.slice(1)}</span>
                      <input
                        className="number-input"
                        type="number"
                        value={axis[field]}
                        disabled={running}
                        onChange={(event) => updateAxis(index, { [field]: event.currentTarget.value })}
                      />
                    </label>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {axes.length < 3 && (
          <button
            className="sample-add-button"
            disabled={running || axes.length >= specs.length}
            onClick={() => {
              const used = new Set(axes.map((axis) => axis.key))
              const next = specs.find((spec) => !used.has(spec.key))
              if (next) setAxes((old) => [...old, makeDraft(next.key)])
            }}
          >
            + Add parameter
          </button>
        )}

        <label className="sample-steps-row">
          <span>Simulation steps</span>
          <input className="number-input" type="number" min={1} step={1} value={steps} disabled={running} onChange={(event) => setSteps(event.currentTarget.value)} />
        </label>
        <label className="checkbox-row sample-json-row">
          <input
            type="checkbox"
            checked={includeJson}
            disabled={running}
            onChange={(event) => setIncludeJson(event.currentTarget.checked)}
          />
          Include JSON outputs
        </label>

        <div className="sample-summary">
          {duplicateKeys
            ? "Choose each parameter only once."
            : invalidExplicitValues
              ? "Enter at least one valid explicit numeric value."
            : !validSubstrateValues
              ? "Substrate resolutions must be positive whole numbers."
            : sampleCount > MAX_SAMPLES
              ? `${sampleCount.toLocaleString()} combinations exceeds the ${MAX_SAMPLES.toLocaleString()}-sample limit.`
              : `${sampleCount.toLocaleString()} capture${sampleCount === 1 ? "" : "s"} × ${Number.isFinite(parsedSteps) ? parsedSteps.toLocaleString() : "—"} steps`}
        </div>
        {running && (
          <div className="sample-progress">
            <progress max={Math.max(total, 1)} value={completed} />
            <span>{completed} / {total}</span>
          </div>
        )}
        {error && <p className="sample-error">{error}</p>}

        <div className="sample-modal-actions">
          <button className="playback-button" onClick={running ? onCancel : onClose}>{running ? "Cancel" : "Close"}</button>
          <button
            className="playback-button sample-run-button"
            disabled={!valid || running}
            onClick={() => onRun({ axes: parsedAxes, steps: parsedSteps, includeJson })}
          >
            Run & download ZIP
          </button>
        </div>
      </div>
    </div>
  )
}

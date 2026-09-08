import { useState } from "react"
import type { PhysicsSettings } from "../gpu/types"
import { Slider } from "./Slider"

interface GrowthPanelProps {
  value: PhysicsSettings
  onChange: (next: PhysicsSettings) => void
  isOverridden: boolean
  onReset: () => void
}

export type GrowthKey = Extract<
  keyof PhysicsSettings,
  "growthDuration" | "growthSpeedMultiplier" | "growthCompressionStart" | "growthCompressionStop" | "growthCompressionFeedback" | "neuralUpdatesPerMacro" | "communicationSpeed" | "internalStateSpeed"
>

export interface GrowthSliderSpec {
  key: GrowthKey
  label: string
  hint: string
  min: number
  max: number
  step: number
  format: (v: number) => string
}

// All absolute ranges, deliberately NOT PhysicsPanel's own
// scaledRange(trained, N): every knob here is bounded by real physics or
// real semantics, none of which depend on whatever value the
// run happened to be trained with.
export const GROWTH_SLIDER_SPECS: GrowthSliderSpec[] = [
  {
    key: "neuralUpdatesPerMacro",
    label: "Communication ticks / frame",
    hint: "Each tick evaluates the neural policy and updates chemistry. Persistent substrate diffuses and decays between evaluations. Total communication time stays fixed; growth commits on the final tick.",
    min: 1,
    max: 16,
    step: 1,
    format: (v) => `${Math.round(v)} rounds`,
  },
  {
    key: "communicationSpeed",
    label: "Communication speed",
    hint: "Chemical and memory time per mechanical frame, divided across communication ticks. Increase this to let communication evolve further before the shape moves.",
    min: 0,
    max: 4,
    step: 0.05,
    format: (v) => `${v.toFixed(2)}×`,
  },
  {
    key: "internalStateSpeed",
    label: "Internal state speed",
    hint: "Multiplier for gated private-state updates only. 1× preserves the default dynamics; 0 freezes internal state.",
    min: 0,
    max: 4,
    step: 0.05,
    format: (v) => `${v.toFixed(2)}×`,
  },
  {
    key: "growthSpeedMultiplier",
    label: "Growth speed",
    hint: "Live multiplier on material growth. 2× halves the time needed to add the same rest area; 0× pauses growth without changing the policy output.",
    min: 0,
    max: 8,
    step: 0.1,
    format: (v) => `${v.toFixed(1)}×`,
  },
  {
    key: "growthDuration",
    label: "Growth timescale",
    hint: "Mechanical ticks for a unit-magnitude field to double stress-free material area. Subdivision itself does not grow material.",
    min: 0,
    max: 160,
    step: 1,
    format: (v) => `${v.toFixed(0)} ticks`,
  },
  {
    key: "growthCompressionFeedback",
    label: "Growth blockage",
    hint: "How much inward compression blocks expansion. 0: no blockage, so growth pushes against compression. 1: current compression slowdown and arrest behavior. Intermediate values blend between the two.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => v.toFixed(2),
  },
  {
    key: "growthCompressionStart",
    label: "Compression slowdown",
    hint: "Elastic areal compression at which proliferation starts slowing. When equal to the arrest threshold, this becomes a hard cutoff.",
    min: 0,
    max: 0.2,
    step: 0.002,
    format: (v) => `${(100 * v).toFixed(1)}%`,
  },
  {
    key: "growthCompressionStop",
    label: "Compression arrest",
    hint: "Elastic areal compression at which proliferation is fully paused. Growth resumes smoothly after pressure releases.",
    min: 0.002,
    max: 0.3,
    step: 0.002,
    format: (v) => `${(100 * v).toFixed(1)}%`,
  },
]

/** Live controls for continuous material growth and communication cadence. */
export function GrowthPanel({
  value,
  onChange,
  isOverridden,
  onReset,
}: GrowthPanelProps) {
  const [open, setOpen] = useState(false)
  return (
    <section>
      <div className="physics-panel-header">
        <button
          className="physics-panel-toggle"
          onClick={() => setOpen((o) => !o)}
        >
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>
            ▸
          </span>
          <h2>Growth</h2>
        </button>
        <button
          className="icon-button"
          onClick={onReset}
          disabled={!isOverridden}
          title="Reset to training values"
          aria-label="Reset to training values"
        >
          ↺
        </button>
      </div>
      {open && (
        <div className="physics-panel-body">
          {GROWTH_SLIDER_SPECS.map((spec) => (
            <label data-audio-target={`physics.${spec.key}`} key={spec.key} className="slider-row" title={spec.hint}>
              <span>{spec.label}</span>
              <Slider min={spec.min} max={spec.max} step={spec.step}
                value={value[spec.key]}
                onChange={(v) => onChange({ ...value, [spec.key]: v })} />
              <span className="slider-value">{spec.format(value[spec.key])}</span>
            </label>
          ))}
        </div>
      )}
    </section>
  )
}

import { useState } from "react"
import type { PhysicsSettings } from "../gpu/types"
import { Slider } from "./Slider"

interface GrowthPanelProps {
  /** This generation's own trained values — the reset target, same
   * contract PhysicsPanel has. Growth knobs live in PhysicsSettings
   * alongside every other live-adjustable setting rather than in a
   * parallel state object, so this panel reuses that whole
   * value/onChange/isOverridden/onReset plumbing untouched. */
  trained: PhysicsSettings
  value: PhysicsSettings
  onChange: (next: PhysicsSettings) => void
  isOverridden: boolean
  onReset: () => void
}

export type GrowthKey = Extract<
  keyof PhysicsSettings,
  "growthDuration" | "growthAnisotropy" | "divisionDirectionality" | "boundaryTangentMinGradient" | "neuralUpdatesPerMacro" | "communicationSpeed" | "internalStateSpeed"
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
    label: "Neural updates / tick",
    hint: "Neural evaluations before one MLS-MPM update. Chemical and turning dynamics are timestep-scaled, so this raises temporal resolution rather than raw speed. Lifecycle and division commit only on the final round.",
    min: 1,
    max: 16,
    step: 1,
    format: (v) => `${Math.round(v)} rounds`,
  },
  {
    key: "communicationSpeed",
    label: "Communication speed",
    hint: "Cell-chemical updates and orientation time per mechanical tick. Neural updates control resolution; this controls elapsed communication time.",
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
    key: "growthDuration",
    label: "Growth duration",
    hint: "Approximate mechanical ticks required to double stress-free area. Larger values give agents more communication rounds before division. 0 = growth off.",
    min: 0,
    max: 160,
    step: 1,
    format: (v) => `${v.toFixed(0)} ticks`,
  },
  {
    key: "growthAnisotropy",
    label: "Growth anisotropy",
    hint: "Caps policy-directed rest-growth elongation. 1× grants full policy authority; 0 makes mechanical growth isotropic.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => `${v.toFixed(2)}×`,
  },
  {
    key: "divisionDirectionality",
    label: "Division directionality",
    hint: "Caps one-sided daughter placement. 1× grants full policy authority; 0 keeps every split center-preserving and symmetric.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => `${v.toFixed(2)}×`,
  },
  {
    key: "boundaryTangentMinGradient",
    label: "Tangent flat-gradient threshold",
    hint: "Morphology-gradient magnitudes at or below this value are treated as flat interiors and fall back to the neural division direction. 0 uses the tangent for every nonzero gradient.",
    min: 0,
    max: 0.05,
    step: 0.000001,
    format: (v) => v.toExponential(2),
  },
]

/** Collapsible "Growth" section (default closed), sibling to
 * PhysicsPanel — the knobs behind this project's own kinematic growth
 * model (the multiplicative decomposition F = Fe*Fg, see
 * ../../../core/g2p.wgsl's own substrate-driven growth block and
 * ../../../core/agents.wgsl's own ParticleRest.growthF field).
 *
 * Split into its own section rather than appended to PhysicsPanel's own
 * flat list because these controls behave as a group:
 * neuralUpdatesPerMacro controls communication cadence relative to mechanics;
 * growthDuration controls the substrate-driven cell cycle;
 * growthAnisotropy and divisionDirectionality cap directional authority;
 * boundaryTangentMinGradient controls where the hardcoded tangent rule yields
 * back to neural division orientation.
 * Same live-uniform-write path as every
 * PhysicsPanel knob (gpu/simulation.ts's own applyPhysics()), so moving
 * any of these never disturbs the rollout in flight and never affects
 * training itself — playback only. */
export function GrowthPanel({
  trained,
  value,
  onChange,
  isOverridden,
  onReset,
}: GrowthPanelProps) {
  const [open, setOpen] = useState(false)
  void trained

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
            <label key={spec.key} className="slider-row" title={spec.hint}>
              <span>{spec.label}</span>
              <Slider
                min={spec.min}
                max={spec.max}
                step={spec.step}
                value={value[spec.key]}
                onChange={(v) => onChange({ ...value, [spec.key]: v })}
              />
              <span className="slider-value">
                {spec.format(value[spec.key])}
              </span>
            </label>
          ))}
        </div>
      )}
    </section>
  )
}

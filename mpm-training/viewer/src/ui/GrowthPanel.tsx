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

type GrowthKey = Extract<
  keyof PhysicsSettings,
  "growthDuration" | "growthThreshold" | "growthAnisotropy"
>

interface GrowthSliderSpec {
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
// real semantics (a division factor must exceed 1 and the compression
// compression reference is an elastic-area ratio), none of which depend on whatever value the
// run happened to be trained with.
const SPECS: GrowthSliderSpec[] = [
  {
    key: "growthDuration",
    label: "Growth duration",
    hint: "Approximate neural/chemical updates required to double stress-free area. Larger values give agents more time to coordinate. 0 = growth off.",
    min: 0,
    max: 160,
    step: 1,
    format: (v) => `${v.toFixed(0)} ticks`,
  },
  {
    key: "growthThreshold",
    label: "Compression reference",
    hint: "Below this elastic area ratio, compression continuously slows growth. 0 disables mechanical inhibition.",
    min: 0,
    max: 1,
    step: 0.001,
    format: (v) => v.toFixed(3),
  },
  {
    key: "growthAnisotropy",
    label: "Anisotropy",
    hint: "Global multiplier on the neural anisotropy output. 0 forces isotropic blob growth; 1 leaves the learned directional growth unchanged.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => v.toFixed(2),
  },
]

/** Collapsible "Growth" section (default closed), sibling to
 * PhysicsPanel — the knobs behind this project's own kinematic growth
 * model (the multiplicative decomposition F = Fe*Fg, see
 * ../../../core/g2p.wgsl's own substrate-driven growth block and
 * ../../../core/agents.wgsl's own ParticleRest.growthF field).
 *
 * Split into its own section rather than appended to PhysicsPanel's own
 * flat list because these four behave as a group and are tuned together:
 * growthDuration/growthThreshold control the substrate-driven cell cycle and
 * its optional continuous mechanical feedback; growthAnisotropy scales the
 * policy's per-particle anisotropy without replacing its spatial variation. Same
 * live-uniform-write path as every
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
          {SPECS.map((spec) => (
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

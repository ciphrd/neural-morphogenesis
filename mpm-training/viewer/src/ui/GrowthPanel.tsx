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
  "growthDuration" | "growthAnisotropy" | "growthCompressionStart" | "growthCompressionStop" | "growthCompressionFeedback" | "divisionDirectionality" | "divisionDriveBoost" | "boundaryTangentMinGradient" | "neuralUpdatesPerMacro" | "communicationSpeed" | "internalStateSpeed"
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

// Morphology occupancy is clamped to [0,1] before its centered finite
// difference is measured, so its gradient magnitude cannot reach 1. Using 1
// as the live threshold therefore disables every boundary-tangent branch
// without requiring another GPU-uniform field.
const TANGENT_DISABLED_THRESHOLD = 1
const TANGENT_SLIDER_MAX = 0.05
const DEFAULT_ACTIVE_TANGENT_THRESHOLD = 0.008

function activeTangentThreshold(value: number): number {
  return value < TANGENT_DISABLED_THRESHOLD
    ? Math.min(value, TANGENT_SLIDER_MAX)
    : DEFAULT_ACTIVE_TANGENT_THRESHOLD
}

// All absolute ranges, deliberately NOT PhysicsPanel's own
// scaledRange(trained, N): every knob here is bounded by real physics or
// real semantics, none of which depend on whatever value the
// run happened to be trained with.
export const GROWTH_SLIDER_SPECS: GrowthSliderSpec[] = [
  {
    key: "divisionDriveBoost",
    label: "Division chance boost",
    hint: "Blends the signed neural division drive toward a probability mapping. At 0, outputs at or below zero cannot start growth. At 1, neural [-1,1] maps to [0,1], so a zero output gives 50% division chance per tick.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => `${(100 * v).toFixed(0)}%`,
  },
  {
    key: "neuralUpdatesPerMacro",
    label: "Neural updates / tick",
    hint: "Neural evaluations before one MLS-MPM update. Memory and turning are timestep-scaled. Persistent substrate stays frozen and uses only the final output; cell-owned chemistry can evolve between rounds. Lifecycle and division commit only on the final round.",
    min: 1,
    max: 16,
    step: 1,
    format: (v) => `${Math.round(v)} rounds`,
  },
  {
    key: "communicationSpeed",
    label: "Communication speed",
    hint: "Communication time per mechanical tick. It scales cell-owned chemistry, memory, turning, and the one end-of-tick persistent-field evolution; neural updates only divide the agent-state timestep.",
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
    key: "growthCompressionFeedback",
    label: "Compression feedback",
    hint: "Strength of physical contact inhibition. 1× fully applies the compression gate; 0 restores pressure-independent growth.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => `${v.toFixed(2)}×`,
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
  {
    key: "divisionDirectionality",
    label: "Division polarization",
    hint: "Controls one-sided placement along the NN growth direction. 1× can keep the parent fixed and place the daughter fully toward that output; 0 keeps the pair centered and symmetric.",
    min: 0,
    max: 1,
    step: 0.01,
    format: (v) => `${v.toFixed(2)}×`,
  },
  {
    key: "boundaryTangentMinGradient",
    label: "Tangent flat-gradient threshold",
    hint: "For Lab tangent-growth scenarios, morphology gradients at or below this value are treated as flat. Above it, the tangent directs both growth and spawn placement.",
    min: 0,
    max: TANGENT_SLIDER_MAX,
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
 * growthDuration controls the policy-driven cell cycle;
 * growthAnisotropy controls tensor directionality and divisionDirectionality
 * controls how strongly growth-directed splits are polarized;
 * boundaryTangentMinGradient controls Lab tangent-growth diagnostics.
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
  const [rememberedTangentThreshold, setRememberedTangentThreshold] = useState(
    activeTangentThreshold(
      value.boundaryTangentMinGradient < TANGENT_DISABLED_THRESHOLD
        ? value.boundaryTangentMinGradient
        : trained.boundaryTangentMinGradient
    )
  )
  const tangentEnabled = value.boundaryTangentMinGradient < TANGENT_DISABLED_THRESHOLD

  const setTangentEnabled = (enabled: boolean) => {
    if (enabled) {
      const trainedThreshold = activeTangentThreshold(trained.boundaryTangentMinGradient)
      const restoredThreshold = Number.isFinite(rememberedTangentThreshold)
        ? rememberedTangentThreshold
        : trainedThreshold
      onChange({
        ...value,
        boundaryTangentMinGradient: restoredThreshold,
      })
      return
    }
    if (tangentEnabled) {
      setRememberedTangentThreshold(value.boundaryTangentMinGradient)
    }
    onChange({
      ...value,
      boundaryTangentMinGradient: TANGENT_DISABLED_THRESHOLD,
    })
  }

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
          {GROWTH_SLIDER_SPECS.map((spec) => {
            const isTangentThreshold = spec.key === "boundaryTangentMinGradient"
            const displayedValue = isTangentThreshold && !tangentEnabled
              ? rememberedTangentThreshold
              : value[spec.key]
            return (
              <div key={spec.key}>
                {isTangentThreshold && (
                  <label
                    className="checkbox-row"
                    title="When disabled, diagnostic growth and spawning use the neural growth axis instead of the morphology-boundary tangent."
                  >
                    <input
                      type="checkbox"
                      checked={tangentEnabled}
                      onChange={(event) => setTangentEnabled(event.target.checked)}
                    />
                    Use boundary-tangent direction
                  </label>
                )}
                <label className="slider-row" title={spec.hint}>
                  <span>{spec.label}</span>
                  <Slider
                    min={spec.min}
                    max={spec.max}
                    step={spec.step}
                    value={displayedValue}
                    disabled={isTangentThreshold && !tangentEnabled}
                    onChange={(v) => {
                      if (isTangentThreshold) setRememberedTangentThreshold(v)
                      onChange({ ...value, [spec.key]: v })
                    }}
                  />
                  <span className="slider-value">
                    {isTangentThreshold && !tangentEnabled
                      ? "Disabled"
                      : spec.format(displayedValue)}
                  </span>
                </label>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

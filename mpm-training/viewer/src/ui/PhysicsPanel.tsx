import { useState } from "react"
import type { PhysicsSettings } from "../gpu/types"
import { Slider } from "./Slider"

interface PhysicsPanelProps {
  /** This generation's own trained values (train_server.py's broadcast
   * message at training time) — the reset target, and what the slider
   * ranges scale around. */
  trained: PhysicsSettings
  /** What's currently applied to the live replay — either `trained`
   * itself (nothing overridden yet) or the user's in-progress tweak. */
  value: PhysicsSettings
  onChange: (next: PhysicsSettings) => void
  /** Whether `value` actually differs from `trained` right now — passed
   * in rather than recomputed here so this component doesn't need its
   * own opinion on float equality; TrainingView already knows this for
   * free (it's exactly "is there an override object at all"). */
  isOverridden: boolean
  onReset: () => void
}

// Excludes mpmEnabled — the one boolean field in PhysicsSettings, its own
// checkbox row below rather than a numeric-range SliderSpec.
export type PhysicsSliderKey = Exclude<keyof PhysicsSettings, "mpmEnabled">

export interface PhysicsSliderSpec {
  key: PhysicsSliderKey
  label: string
  min: number
  max: number
  step: number
  format: (v: number) => string
}

/** Scales a slider's range around this generation's own trained value —
 * self-adjusting rather than a hardcoded absolute range, since these are
 * plain module constants (trainer/simulation_settings.py) or CLI defaults
 * (evolve.py), not something with a fixed "sensible" domain baked in
 * anywhere. A trained value of exactly 0 still gets a small explorable
 * range instead of a degenerate zero-width slider. Mirrors
 * envnca/frontend/src/ui/PhysicsPanel.tsx's own scaledRange() exactly. */
function scaledRange(
  trained: number,
  multiplier: number
): { min: number; max: number; step: number } {
  const max = Math.max(trained * multiplier, 0.05)
  return { min: 0, max, step: max / 200 }
}

// [0,1]-bounded fraction sliders share the same shape —
// physically bounded, not scaled from the trained value like the others.
const FRACTION_RANGE = { min: 0, max: 1, step: 0.001 } as const

export function physicsSliderSpecsFor(
  trained: PhysicsSettings
): PhysicsSliderSpec[] {
  return [
    {
      key: "gravity",
      label: "Gravity",
      ...scaledRange(trained.gravity, 3),
      format: (v) => v.toFixed(1),
    },
    {
      key: "damping",
      label: "Damping",
      ...FRACTION_RANGE,
      format: (v) => v.toFixed(3),
    },
    {
      key: "materialE",
      label: "Material E",
      ...scaledRange(trained.materialE, 3),
      format: (v) => v.toFixed(0),
    },
    {
      key: "materialNu",
      label: "Material ν (Poisson)",
      // Must stay below 0.5 — lambda0 = E*nu/((1+nu)*(1-2*nu)) blows up
      // as nu approaches it (see mpm_core.py's own lame_params()).
      min: 0,
      max: 0.49,
      step: 0.001,
      format: (v) => v.toFixed(3),
    },
    {
      key: "materialHardening",
      label: "Material hardening",
      ...scaledRange(trained.materialHardening, 3),
      format: (v) => v.toFixed(2),
    },
    {
      key: "materialElasticity",
      label: "Material elasticity",
      // Physically bounded — mpm_core.py's own yield_bounds() clamps this
      // to [0,1] itself (0 = snow-like, 1 = wide/soft yield bounds).
      min: 0,
      max: 1,
      step: 0.01,
      format: (v) => v.toFixed(2),
    },
    {
      key: "materialFluidity",
      label: "Fluidity (shear relaxation)",
      ...FRACTION_RANGE,
      format: (v) => v.toFixed(3),
    },
    {
      key: "decay",
      label: "Substrate decay",
      ...FRACTION_RANGE,
      format: (v) => v.toFixed(3),
    },
    {
      key: "maxAccel",
      label: "Max accel",
      ...scaledRange(trained.maxAccel, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "maxStrafe",
      label: "Physical strafe scale",
      ...scaledRange(trained.maxStrafe, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "friction",
      label: "Friction",
      ...FRACTION_RANGE,
      format: (v) => v.toFixed(3),
    },
    {
      key: "maxEnvWrite",
      label: "Max env write",
      ...scaledRange(trained.maxEnvWrite, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "maxAngularAccel",
      label: "Max angular accel",
      min: 0,
      max: 1.8,
      step: 0.0001,
      format: (v) => v.toFixed(3),
    },
    {
      key: "angularDamping",
      label: "Angular damping",
      ...FRACTION_RANGE,
      format: (v) => v.toFixed(3),
    },
    {
      key: "maxAngularVelocity",
      label: "Max angular velocity",
      ...scaledRange(trained.maxAngularVelocity, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "depositSigma",
      label: "Deposit splat radius",
      ...scaledRange(trained.depositSigma, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "splitDisplacement",
      label: "Split displacement",
      ...scaledRange(trained.splitDisplacement, 3),
      format: (v) => v.toFixed(4),
    },
    {
      key: "divisionCooldown",
      label: "Division cooldown",
      ...scaledRange(trained.divisionCooldown, 3),
      format: (v) => v.toFixed(1),
    },
    {
      key: "splatRadius",
      label: "Density splat radius",
      ...scaledRange(trained.splatRadius, 3),
      format: (v) => v.toFixed(4),
    },
    {
      key: "morphologyBlurSigma",
      label: "Morphology blur sigma",
      ...scaledRange(trained.morphologyBlurSigma, 3),
      format: (v) => v.toFixed(4),
    },
    {
      key: "morphologyDensityReference",
      label: "Morphology density reference",
      min: 0.001,
      max: Math.max(3, trained.morphologyDensityReference * 3),
      step: 0.001,
      format: (v) => v.toFixed(3),
    },
    {
      key: "repulsionStrength",
      label: "Repulsion strength",
      min: 0,
      max: 1_000,
      step: 0.001,
      format: (v) => v.toFixed(3),
    },
    {
      key: "repulsionMaxDelta",
      label: "Repulsion max delta",
      // Explicit absolute range, not scaledRange() — this bounds a
      // per-substep VELOCITY delta against MLS-MPM's own stability
      // limit (core/repulsion.wgsl's own RepulsionParams.maxDelta field
      // comment: DX/DT = 250 is the theoretical "moves exactly one grid
      // cell in one substep" bound), a fixed physical ceiling regardless
      // of the trained repulsionStrength value.
      min: 1,
      max: 250,
      step: 0.5,
      format: (v) => v.toFixed(1),
    },
  ]
}

/** Collapsible "Physics" section (default closed) exposing every
 * live-adjustable simulation setting as a slider — gravity, damping, MPM
 * material (E/nu/hardening/elasticity), persistent substrate decay,
 * cell-chemical delta magnitude,
 * strafe's own maxAccel/maxStrafe/friction (strafe
 * drives MpmCore's own velocity directly — an acceleration, damped by
 * friction — see agents.wgsl's own module docstring for the full
 * history), the heading integrator's maxAngularAccel/angularDamping/
 * maxAngularVelocity, and depositSigma (the cell-state Gaussian splat radius — see
 * agents.wgsl's own depositGaussian() for the exact kernel this drives,
 * replacing that shader's old flat 4-corner bilinear scatter),
 * growth's own splitDisplacement (daughter separation; the signed growth
 * vector biases the new daughter and pair center toward +n) and
 * divisionCooldown (macro steps a particle refuses
 * to split again for, right after splitting, whether as parent or child
 * — see agents.wgsl's own module docstring for the full growth design;
 * the growth cap itself is `particles` — see types.ts's own
 * SimulationConfig.particles docstring for why that's a CAP now, not a
 * starting count, and not a slider here since it's rebuild-triggering,
 * same as channels/fieldN/hiddenDim), density's splatRadius,
 * morphologyBlurSigma, and morphologyDensityReference, repulsion strength,
 * and mpmEnabled (a checkbox, not a slider — off skips MpmCore's own
 * physics substeps entirely each macro step, a debug/testing aid to
 * isolate sensing/communication/growth/chirality from elastic material
 * response/gravity/repulsion — see types.ts's own
 * RunSettings.mpmEnabled docstring for the full reasoning).
 * See gpu/simulation.ts's applyPhysics() for why moving any of these
 * never disturbs the rollout currently in flight (a plain uniform-buffer
 * write, not a pipeline rebuild) — this is playback-only, doesn't affect
 * training itself. */
export function PhysicsPanel({
  trained,
  value,
  onChange,
  isOverridden,
  onReset,
}: PhysicsPanelProps) {
  const [open, setOpen] = useState(false)
  const specs = physicsSliderSpecsFor(trained)

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
          <h2>Advanced physics overrides</h2>
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
          {/* Boolean, not a slider — kept out of specsFor()'s own
           * SliderSpec list (which assumes a numeric range) and rendered
           * as its own checkbox row instead (same label-left/checkbox-
           * right layout the "Point particles toward heading" row uses
           * — see style.css's own .checkbox-row). See gpu/types.ts's own
           * RunSettings.mpmEnabled docstring for exactly what turning
           * this off does. */}
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={value.mpmEnabled}
              onChange={(e) =>
                onChange({ ...value, mpmEnabled: e.target.checked })
              }
            />
            MPM physics
          </label>
          {specs.map((spec) => (
            <label key={spec.key} className="slider-row">
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

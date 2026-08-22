import { useState } from "react";
import type { PhysicsSettings } from "../gpu/types";

interface PhysicsPanelProps {
  /** This generation's own trained values (constants.py at training
   * time) — the reset target, and what the slider ranges scale around. */
  trained: PhysicsSettings;
  /** What's currently applied to the live replay — either `trained`
   * itself (nothing overridden yet) or the user's in-progress tweak. */
  value: PhysicsSettings;
  onChange: (next: PhysicsSettings) => void;
  /** Whether `value` actually differs from `trained` right now — passed
   * in rather than recomputed here so this component doesn't need its
   * own opinion on float equality; TrainingView already knows this for
   * free (it's exactly "is there an override object at all"). */
  isOverridden: boolean;
  onReset: () => void;
}

interface SliderSpec {
  key: keyof PhysicsSettings;
  label: string;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
}

/** Scales a slider's range around this generation's own trained value —
 * self-adjusting rather than a hardcoded absolute range, since these are
 * plain module constants (constants.py), not something with a fixed
 * "sensible" domain baked in anywhere; whatever a training run actually
 * used should sit comfortably inside the range its own slider offers. A
 * trained value of exactly 0 still gets a small explorable range instead
 * of a degenerate zero-width slider. */
function scaledRange(trained: number, multiplier: number): { min: number; max: number; step: number } {
  const max = Math.max(trained * multiplier, 0.05);
  return { min: 0, max, step: max / 200 };
}

function specsFor(trained: PhysicsSettings): SliderSpec[] {
  return [
    {
      key: "decay",
      label: "Decay",
      // Physically bounded, not scaled from the trained value like the
      // others below — decay > 1 makes the substrate grow without bound
      // step over step, so 1.0 is a real, meaningful ceiling here, not
      // an arbitrary one.
      min: 0,
      max: 1,
      step: 0.001,
      format: (v) => v.toFixed(3),
    },
    { key: "maxSpeed", label: "Max speed", ...scaledRange(trained.maxSpeed, 3), format: (v) => v.toFixed(3) },
    { key: "maxAccel", label: "Max accel", ...scaledRange(trained.maxAccel, 3), format: (v) => v.toFixed(3) },
    { key: "maxStrafe", label: "Max strafe", ...scaledRange(trained.maxStrafe, 3), format: (v) => v.toFixed(3) },
    {
      key: "maxEnvWrite",
      label: "Max env write",
      ...scaledRange(trained.maxEnvWrite, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "repulsionSigma",
      label: "Repulsion sigma",
      ...scaledRange(trained.repulsionSigma, 3),
      format: (v) => v.toFixed(3),
    },
    {
      key: "repulsionStrength",
      label: "Repulsion strength",
      ...scaledRange(trained.repulsionStrength, 3),
      format: (v) => v.toFixed(4),
    },
  ];
}

/** Collapsible "Physics" section (default closed) exposing
 * decay/maxSpeed/maxAccel/maxStrafe/maxEnvWrite/repulsionSigma/
 * repulsionStrength as live sliders — see
 * gpu/simulation.ts's setPhysics() for why moving one never disturbs the
 * rollout currently in flight (a plain uniform-buffer write, not a
 * pipeline rebuild). Deliberately doesn't include hiddenDim — see
 * PhysicsSettings' own docstring for why that one isn't a physics knob
 * at all. */
export function PhysicsPanel({ trained, value, onChange, isOverridden, onReset }: PhysicsPanelProps) {
  const [open, setOpen] = useState(false);
  const specs = specsFor(trained);

  return (
    <section>
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen((o) => !o)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>▸</span>
          <h2>Physics</h2>
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
          {specs.map((spec) => (
            <label key={spec.key} className="slider-row">
              <span>{spec.label}</span>
              <input
                type="range"
                min={spec.min}
                max={spec.max}
                step={spec.step}
                value={value[spec.key]}
                onChange={(e) => onChange({ ...value, [spec.key]: Number(e.target.value) })}
              />
              <span className="slider-value">{spec.format(value[spec.key])}</span>
            </label>
          ))}
          <p className="hint">
            {isOverridden
              ? "Overriding this generation's own trained values — playback only, doesn't affect training."
              : "Showing this generation's own trained values."}
          </p>
        </div>
      )}
    </section>
  );
}

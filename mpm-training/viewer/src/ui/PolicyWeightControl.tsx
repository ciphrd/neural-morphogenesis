import { Slider } from "./Slider"

interface PolicyWeightControlProps {
  value: number
  disabled?: boolean
  onChange: (value: number) => void
}

const PRESETS = [0, 0.5, 1, 1.5, 2] as const

export function PolicyWeightControl({
  value,
  disabled = false,
  onChange,
}: PolicyWeightControlProps) {
  return (
    <section className="policy-weight-control">
      <div className="policy-weight-header">
        <h2>Policy weight gain</h2>
        <button
          className="icon-button"
          type="button"
          onClick={() => onChange(1)}
          disabled={disabled || value === 1}
          title="Reset policy weight gain to 1×"
          aria-label="Reset policy weight gain"
        >
          ↺
        </button>
      </div>
      <label className="policy-weight-slider">
        <span className="sr-only">Policy weight gain</span>
        <Slider
          min={0}
          max={3}
          step={0.01}
          value={value}
          disabled={disabled}
          onChange={onChange}
        />
        <input
          className="number-input policy-weight-input"
          type="number"
          min={0}
          max={3}
          step={0.01}
          value={value}
          disabled={disabled}
          aria-label="Policy weight gain exact value"
          onChange={(event) => {
            const next = event.currentTarget.valueAsNumber
            if (Number.isFinite(next)) onChange(Math.min(3, Math.max(0, next)))
          }}
        />
        <span className="policy-weight-unit">×</span>
      </label>
      <div className="policy-weight-presets" aria-label="Policy weight gain presets">
        {PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            className={value === preset ? "is-active" : ""}
            disabled={disabled}
            onClick={() => onChange(preset)}
          >
            {preset}×
          </button>
        ))}
      </div>
      <p className="hint">
        Scales both learned weight matrices; biases stay unchanged. The rollout
        restarts so each setting is directly comparable.
      </p>
    </section>
  )
}

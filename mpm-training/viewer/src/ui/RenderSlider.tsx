import { Slider } from "./Slider"

interface RenderSliderProps {
  audioTarget: string
  label: string
  value: number
  min: number
  max: number
  step: number
  display: string
  disabled?: boolean
  onChange: (value: number) => void
}

export function RenderSlider({ audioTarget, label, value, min, max, step, display, disabled, onChange }: RenderSliderProps) {
  return (
    <label data-audio-target={audioTarget} className="slider-row">
      <span>{label}</span>
      <Slider min={min} max={max} step={step} value={value} disabled={disabled} onChange={onChange} />
      <span className="slider-value">{display}</span>
    </label>
  )
}


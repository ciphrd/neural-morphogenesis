// Custom-styled slider — the plain browser `<input type="range">` this
// replaces has no reliable, cross-browser CSS path to a colored
// fill-to-value track (::-webkit-slider-runnable-track and
// ::-moz-range-progress are two different, inconsistent APIs for the
// same idea, and neither lets a single stylesheet rule express "the
// portion left of the thumb is one color, the rest is another" the way
// a plain background-color / gradient can), so this draws the visible
// track/fill/thumb itself as plain positioned <div>s, computed from
// value/min/max here in JS, and keeps the REAL <input type="range">
// underneath — same size, fully interactive, just visually
// transparent — so dragging, click-to-seek, keyboard arrows, and
// screen-reader semantics all keep working exactly as they did before;
// only the paint changes.
//
// The block of constants right below is the actual "scaffolding" this
// component exists for — every visual knob (track/fill/thumb size,
// color, radius) lives in exactly this one place, not scattered across
// style.css selectors, so retuning the look later is a one-file edit.
// Tweak freely; nothing here is load-bearing for behavior.

const TRACK_HEIGHT_PX = 4
const TRACK_COLOR = "#2a2a2a"
const TRACK_RADIUS_PX = 2

const FILL_COLOR = "#7dd3fc" // this app's own existing accent — see style.css's own .is-active/.run-picker-item rules
const FILL_COLOR_DISABLED = "#4a4a4a"

const THUMB_SIZE_PX = 13
const THUMB_COLOR = "#7dd3fc"
const THUMB_COLOR_DISABLED = "#6a6a6a"
const THUMB_BORDER_COLOR = "#0d0d0d" // matches .controls/.controls-right's own panel background, see style.css
const THUMB_BORDER_WIDTH_PX = 2

// Total hit height for the invisible real <input> — generous relative
// to TRACK_HEIGHT_PX above on purpose, a thin track shouldn't also mean
// a thin, hard-to-grab click/touch target.
const HIT_HEIGHT_PX = 18

interface SliderProps {
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  disabled?: boolean
}

export function Slider({ value, min, max, step, onChange, disabled = false }: SliderProps) {
  const clamped = Math.min(max, Math.max(min, value))
  const pct = max > min ? ((clamped - min) / (max - min)) * 100 : 0

  return (
    <div className="slider" style={{ height: HIT_HEIGHT_PX }}>
      <div
        className="slider-track"
        style={{ height: TRACK_HEIGHT_PX, borderRadius: TRACK_RADIUS_PX, background: TRACK_COLOR }}
      />
      <div
        className="slider-fill"
        style={{
          width: `${pct}%`,
          height: TRACK_HEIGHT_PX,
          borderRadius: TRACK_RADIUS_PX,
          background: disabled ? FILL_COLOR_DISABLED : FILL_COLOR,
        }}
      />
      <div
        className="slider-thumb"
        style={{
          left: `${pct}%`,
          width: THUMB_SIZE_PX,
          height: THUMB_SIZE_PX,
          background: disabled ? THUMB_COLOR_DISABLED : THUMB_COLOR,
          border: `${THUMB_BORDER_WIDTH_PX}px solid ${THUMB_BORDER_COLOR}`,
        }}
      />
      <input
        className="slider-input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

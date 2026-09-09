import { useRef, useState, type PointerEvent } from "react"
import { RenderSlider } from "./RenderSlider"

export interface SineWave {
  id: string
  label: string
  color: string
  frequency: number
  exponent: number
  period: number
  speed: number
  targets: Record<WaveParameter, string>
}
type WaveParameter = "frequency" | "exponent" | "period" | "speed"
interface Props {
  label: string
  waves: SineWave[]
  domain: "space" | "time"
  onChange: (id: string, patch: Partial<Record<WaveParameter, number>>) => void
}
const WIDTH = 480, HEIGHT = 180, TOP = 12, BOTTOM = 156
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** Shared spatial/temporal oscillator editor; values remain owned by the effect. */
export function SineWaveEditor({ label, waves, domain, onChange }: Props) {
  const [selected, setSelected] = useState(waves[0]?.id ?? "")
  const [mode, setMode] = useState<"shape" | "motion">("shape")
  const [seconds, setSeconds] = useState(1)
  const drag = useRef<{ pointer: number; x: number; y: number; wave: SineWave; mode: "shape" | "motion" } | null>(null)
  const wave = waves.find(item => item.id === selected) ?? waves[0]
  if (!wave) return null
  const begin = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    const hit = (event.target as Element).getAttribute("data-wave")
    const current = waves.find(item => item.id === hit) ?? wave
    setSelected(current.id)
    drag.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY, wave: { ...current }, mode }
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }
  const move = (event: PointerEvent<SVGSVGElement>) => {
    const start = drag.current
    if (!start || start.pointer !== event.pointerId) return
    const box = event.currentTarget.getBoundingClientRect()
    const dx = (event.clientX - start.x) / box.width
    const dy = (start.y - event.clientY) / box.height
    if (start.mode === "shape") {
      onChange(start.wave.id, {
        frequency: clamp(Math.max(0.01, start.wave.frequency) * 2 ** (-dx * 4), 0, 40),
        exponent: clamp(start.wave.exponent * 2 ** (dy * 4), 0.1, 64),
      })
    } else {
      onChange(start.wave.id, {
        period: clamp(start.wave.period * 2 ** (dx * 4), 0.01, 60),
        speed: clamp(start.wave.speed + dy * Math.max(1, Math.abs(start.wave.speed)) * 4, -1000, 1000),
      })
    }
  }
  const end = () => { drag.current = null }
  const temporal = domain === "time" || mode === "motion"
  // Match the renderer's sine remapped to [0, 1], then raised to the exponent.
  const path = (item: SineWave) => {
    const rate = temporal ? (domain === "time" ? item.frequency : 1) * item.speed / item.period : item.frequency
    // Average subpixel cycles to avoid misleading alias patterns at high strobe rates.
    const samples = Math.min(64, Math.max(1, Math.ceil(Math.abs(rate) * (temporal ? seconds : 1) / WIDTH * 8)))
    return Array.from({ length: WIDTH + 1 }, (_, x) => {
      let value = 0
      for (let sample = 0; sample < samples; sample++) {
        const position = (x + sample / samples) / WIDTH
        const phase = temporal ? position * seconds * rate : (position - 0.5) * rate
        value += (0.5 + 0.5 * Math.sin(2 * Math.PI * phase)) ** item.exponent / samples
      }
      return `${x ? "L" : "M"}${x},${(BOTTOM - value * (BOTTOM - TOP)).toFixed(2)}`
    }).join(" ")
  }
  return <div className="sine-wave-editor" role="group" aria-label={`${label} wave editor`}>
    <div className="sine-wave-tools">
      <div className="sine-wave-channels">{waves.map(item => <button key={item.id} type="button" style={{ color: item.color }} aria-label={`${label}: edit ${item.label}`} aria-pressed={wave.id === item.id} onClick={() => setSelected(item.id)}>{item.label[0]}</button>)}</div>
      <select aria-label={`${label} drag mode`} value={mode} onChange={event => setMode(event.target.value as typeof mode)}>
        <option value="shape">Shape</option><option value="motion">Motion</option>
      </select>
      {temporal && <select aria-label={`${label} time window`} value={seconds} onChange={event => setSeconds(Number(event.target.value))}>
        {[0.01, 0.1, 0.25, 1, 4].map(value => <option key={value} value={value}>{value}s</option>)}
      </select>}
    </div>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="sine-wave-plot" role="img" aria-label={`${label}: red, green and blue ${temporal ? "pulses over time" : "waves across the image"}. Drag to edit ${wave.label}.`} onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}>
      {[0, 0.25, 0.5, 0.75, 1].map(value => <g key={value} className="sine-wave-grid"><line x1={value * WIDTH} x2={value * WIDTH} y1={TOP} y2={BOTTOM} /><line x1={0} x2={WIDTH} y1={TOP + value * (BOTTOM - TOP)} y2={TOP + value * (BOTTOM - TOP)} /></g>)}
      {waves.map(item => <path key={item.id} d={path(item)} stroke={item.color} strokeDasharray={item.id === waves[0]?.id ? undefined : item.id === waves[1]?.id ? "7 5" : "2 4"} strokeWidth={item.id === wave.id ? 2 : 1.5} opacity={item.id === wave.id ? 1 : 0.65} fill="none" pointerEvents="none" />)}
      {waves.map(item => <path key={item.id} data-wave={item.id} d={path(item)} stroke="transparent" strokeWidth={10} fill="none" />)}
      <text x={4} y={175}>{temporal ? "0s" : "−½ width"}</text>
      <text x={WIDTH - 4} y={175} textAnchor="end">{temporal ? `${seconds}s` : "+½ width"}</text>
    </svg>
    <p className="hint">{mode === "shape" ? "Drag ↔ to widen / tighten waves; ↕ to sharpen / soften peaks." : "Drag ↔ to change period; ↕ to change speed and direction."}</p>
    <div className="sine-wave-values">
      {([['frequency', 'Frequency', 0, 40, 0.01], ['exponent', 'Sharpness', 0.1, 64, 0.1], ['period', 'Period', 0.01, 60, 0.01], ['speed', 'Speed', -1000, 1000, 0.05]] as const).map(([key, name, min, max, step]) =>
        <RenderSlider key={`${wave.id}-${key}`} audioTarget={wave.targets[key]} label={`${wave.label} ${name.toLowerCase()}`} min={min} max={max} step={step} value={wave[key]} display={`${wave[key].toFixed(2)}${key === 'period' ? 's' : key === 'speed' ? '×' : ''}`} onChange={value => onChange(wave.id, { [key]: value })} />)}
    </div>
  </div>
}

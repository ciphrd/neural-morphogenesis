import { SineWaveEditor, type SineWave } from "./SineWaveEditor"
import { useState } from "react"
import { POST_EFFECT_DEFAULTS } from "../gpu/postEffects"
import type { PerformanceRenderSettings } from "../performance/types"
import { RenderSlider } from "./RenderSlider"
import { ToggleButton } from "./ToggleButton"

interface Props {
  value: PerformanceRenderSettings
  onChange: (value: PerformanceRenderSettings) => void
}

export function PostProcessingPanel({ value, onChange }: Props) {
  const [open, setOpen] = useState(true)
  const fx = { ...POST_EFFECT_DEFAULTS, ...value.bloom }
  const patch = (key: "bloom", next: PerformanceRenderSettings["bloom"]) => onChange({ ...value, [key]: next })
  const patchFx = (next: Partial<typeof POST_EFFECT_DEFAULTS>) => patch("bloom", { ...fx, ...next })
  const channels = ["red", "green", "blue"] as const
  const parameterKey = (channel: string, parameter: string, strobe: boolean) =>
    `${strobe ? `strobe${channel[0].toUpperCase()}${channel.slice(1)}` : channel}${parameter[0].toUpperCase()}${parameter.slice(1)}` as keyof typeof POST_EFFECT_DEFAULTS
  const waves = (strobe: boolean): SineWave[] => channels.map((channel, index) => ({
    id: channel, label: channel[0].toUpperCase() + channel.slice(1), color: ["#ff6b75", "#6cdb91", "#709eff"][index],
    frequency: fx[parameterKey(channel, "frequency", strobe)] as number,
    exponent: fx[parameterKey(channel, "exponent", strobe)] as number,
    period: fx[parameterKey(channel, "period", strobe)] as number,
    speed: fx[parameterKey(channel, "speed", strobe)] as number,
    targets: Object.fromEntries(["frequency", "exponent", "period", "speed"].map(parameter => [parameter, `render.bloom.${parameterKey(channel, parameter, strobe)}`])) as SineWave["targets"],
  }))
  const patchWave = (channel: string, next: Partial<Pick<SineWave, "frequency" | "exponent" | "period" | "speed">>, strobe: boolean) =>
    patchFx(Object.fromEntries(Object.entries(next).map(([parameter, value]) => [parameterKey(channel, parameter, strobe), value])))
  return (
    <section>
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen(current => !current)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>▸</span>
          <h2>Post-processing</h2>
        </button>
      </div>
      {open && <div className="physics-panel-body performance-rendering-body">
            <ToggleButton className="toggle-row" label="Bloom" checked={value.bloom.enabled}
              onChange={(enabled) => patch("bloom", { ...value.bloom, enabled })} />
            {value.bloom.enabled && (
              <>
                <RenderSlider audioTarget="render.bloom.intensity" label="Bloom intensity" min={0} max={3} step={0.05} value={value.bloom.intensity} display={value.bloom.intensity.toFixed(2)} onChange={(intensity) => patch("bloom", { ...value.bloom, intensity })} />
                <RenderSlider audioTarget="render.bloom.threshold" label="Bloom threshold" min={0} max={1} step={0.01} value={value.bloom.threshold} display={value.bloom.threshold.toFixed(2)} onChange={(threshold) => patch("bloom", { ...value.bloom, threshold })} />
                <RenderSlider audioTarget="render.bloom.radiusPx" label="Bloom radius" min={0.25} max={8} step={0.25} value={value.bloom.radiusPx} display={`${value.bloom.radiusPx.toFixed(2)}px`} onChange={(radiusPx) => patch("bloom", { ...value.bloom, radiusPx })} />
                <RenderSlider audioTarget="render.bloom.levels" label="Bloom levels" min={2} max={10} step={1} value={value.bloom.levels} display={String(value.bloom.levels)} onChange={(levels) => patch("bloom", { ...value.bloom, levels: Math.round(levels) })} />
                <RenderSlider audioTarget="render.bloom.scatter" label="Bloom scatter" min={0} max={1} step={0.01} value={value.bloom.scatter} display={value.bloom.scatter.toFixed(2)} onChange={(scatter) => patch("bloom", { ...value.bloom, scatter })} />
              </>
            )}
            <ToggleButton className="toggle-row" label="Depth of field" checked={fx.dofEnabled} onChange={dofEnabled => patchFx({ dofEnabled })} />
            {fx.dofEnabled && <>
              {([["dofFocus", "Focus position", 0, 1, 0.01], ["dofWidth", "Sharp band", 0, 1, 0.01], ["dofTop", "Top blur", 0, 32, 0.5], ["dofFront", "Foreground blur", 0, 32, 0.5]] as const).map(([key, label, min, max, step]) =>
                <RenderSlider key={key} audioTarget={`render.bloom.${key}`} label={label} min={min} max={max} step={step} value={fx[key]} display={fx[key].toFixed(2)} onChange={v => patchFx({ [key]: v })} />)}
            </>}
            <ToggleButton className="toggle-row" label="RGB slicer" checked={fx.rgbEnabled} onChange={rgbEnabled => patchFx({ rgbEnabled })} />
            {fx.rgbEnabled && <>
              <RenderSlider audioTarget="render.bloom.rgbMix" label="RGB mix" min={0} max={1} step={0.01} value={fx.rgbMix} display={fx.rgbMix.toFixed(2)} onChange={rgbMix => patchFx({ rgbMix })} />
              <RenderSlider audioTarget="render.bloom.rgbAngle" label="Slice angle" min={0} max={360} step={1} value={fx.rgbAngle} display={`${fx.rgbAngle.toFixed(0)}°`} onChange={rgbAngle => patchFx({ rgbAngle })} />
              <SineWaveEditor label="RGB slicer" domain="space" waves={waves(false)} onChange={(channel, next) => patchWave(channel, next, false)} />
            </>}
            <ToggleButton className="toggle-row" label="Stroboscope" checked={fx.strobeEnabled} onChange={strobeEnabled => patchFx({ strobeEnabled })} />
            {fx.strobeEnabled && <>
              <RenderSlider audioTarget="render.bloom.strobeMix" label="Strobe mix" min={0} max={1} step={0.01} value={fx.strobeMix} display={fx.strobeMix.toFixed(2)} onChange={strobeMix => patchFx({ strobeMix })} />
              <SineWaveEditor label="Stroboscope" domain="time" waves={waves(true)} onChange={(channel, next) => patchWave(channel, next, true)} />
            </>}
      </div>}
    </section>
  )
}

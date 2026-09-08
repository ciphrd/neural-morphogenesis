import { ToggleButton } from "./ToggleButton"
import { useState } from "react"
import type { FieldMode, ParticleShape, ParticleColorMode } from "../gpu/render"
import {
  cellMemoryFromConfig,
  chemicalCommunicationArchitectureFromConfig,
  type SimulationConfig,
} from "../gpu/types"
import type { PerformanceRenderSettings } from "../performance/types"
import { ChannelWindowSlider } from "./ChannelWindowSlider"
import { Slider } from "./Slider"

interface PerformanceRenderingPanelProps {
  config: SimulationConfig | null
  value: PerformanceRenderSettings
  onChange: (value: PerformanceRenderSettings) => void
}

interface RenderSliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  display: string
  disabled?: boolean
  onChange: (value: number) => void
}

function RenderSlider({ label, value, min, max, step, display, disabled, onChange }: RenderSliderProps) {
  return (
    <label className="slider-row">
      <span>{label}</span>
      <Slider min={min} max={max} step={step} value={value} disabled={disabled} onChange={onChange} />
      <span className="slider-value">{display}</span>
    </label>
  )
}

/** Rendering controls for the projection snapshot. These deliberately edit
 * the same object used by scene recall and audio mapping, so the controller
 * and output window cannot drift into separate render configurations. */
export function PerformanceRenderingPanel({ config, value, onChange }: PerformanceRenderingPanelProps) {
  const [open, setOpen] = useState(true)
  const patch = <Key extends keyof PerformanceRenderSettings>(
    key: Key,
    next: PerformanceRenderSettings[Key],
  ) => onChange({ ...value, [key]: next })
  const cellMemoryEnabled = !config || cellMemoryFromConfig(config) === "recurrent"
  const chemicalLevelsActive = !config
    || chemicalCommunicationArchitectureFromConfig(config) === "cell-owned-projection"

  return (
    <div className="performance-rendering-panel">
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen((current) => !current)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>▸</span>
          <h2>Rendering</h2>
        </button>
      </div>
      {open && (
        <div className="physics-panel-body performance-rendering-body">
          <div className="slider-row">
            <span>Zoom</span>
            <ToggleButton label="Auto zoom" checked={value.autoZoom.enabled}
              onChange={(enabled) => patch("autoZoom", { ...value.autoZoom, enabled })} />
            <Slider
              min={0.25}
              max={8}
              step={0.05}
              value={value.zoom}
              disabled={value.autoZoom.enabled}
              onChange={(next) => patch("zoom", next)}
            />
            <span className="slider-value">{value.autoZoom.enabled ? "Auto" : `${value.zoom.toFixed(2)}×`}</span>
          </div>

          <details className="settings-category">
            <summary>Post-processing</summary>
            <ToggleButton className="toggle-row" label="Bloom" checked={value.bloom.enabled}
              onChange={(enabled) => patch("bloom", { ...value.bloom, enabled })} />
            {value.bloom.enabled && (
              <>
                <RenderSlider label="Bloom intensity" min={0} max={3} step={0.05} value={value.bloom.intensity} display={value.bloom.intensity.toFixed(2)} onChange={(intensity) => patch("bloom", { ...value.bloom, intensity })} />
                <RenderSlider label="Bloom threshold" min={0} max={1} step={0.01} value={value.bloom.threshold} display={value.bloom.threshold.toFixed(2)} onChange={(threshold) => patch("bloom", { ...value.bloom, threshold })} />
                <RenderSlider label="Bloom radius" min={0.25} max={8} step={0.25} value={value.bloom.radiusPx} display={`${value.bloom.radiusPx.toFixed(2)}px`} onChange={(radiusPx) => patch("bloom", { ...value.bloom, radiusPx })} />
                <RenderSlider label="Bloom levels" min={2} max={10} step={1} value={value.bloom.levels} display={String(value.bloom.levels)} onChange={(levels) => patch("bloom", { ...value.bloom, levels: Math.round(levels) })} />
                <RenderSlider label="Bloom scatter" min={0} max={1} step={0.01} value={value.bloom.scatter} display={value.bloom.scatter.toFixed(2)} onChange={(scatter) => patch("bloom", { ...value.bloom, scatter })} />
              </>
            )}
          </details>

          <RenderSlider label="Particle size" min={0.25} max={16} step={0.25} value={value.particleRadiusPx} display={`${value.particleRadiusPx.toFixed(2)}px`} onChange={(next) => patch("particleRadiusPx", next)} />
          <label className="slider-row">
            <span>Shape</span>
            <select className="select" value={value.particleShape} onChange={(event) => patch("particleShape", event.target.value as ParticleShape)}>
              <option value="dot">Dots</option><option value="triangle">Triangles</option><option value="domain">Material domain</option>
            </select>
          </label>
          <label className="slider-row">
            <span>Color</span>
            <select className="select" value={value.particleColorMode} onChange={(event) => patch("particleColorMode", event.target.value as ParticleColorMode)}>
              <option value="white">White</option><option value="neural-color">Neural RGB</option>
              <option value="growth-magnitude">Growth magnitude</option><option value="neural-memory">Neural memory</option>
              <option value="chemical-memory">Chemical memory</option><option value="boundary-value">Boundary value</option>
              <option value="neurons">Neurons</option>
            </select>
          </label>
          <RenderSlider label="Opacity" min={0} max={1} step={0.01} value={value.particleAlpha} display={value.particleAlpha.toFixed(2)} onChange={(next) => patch("particleAlpha", next)} />
          <ToggleButton className="toggle-row" label="Domain" checked={value.domainVisible} onChange={(next) => patch("domainVisible", next)} />
          <ToggleButton className="toggle-row" label="Alignment lines" checked={value.directionalLineVisible} onChange={(next) => patch("directionalLineVisible", next)} />
          <ToggleButton className="toggle-row" label="Growth lines" checked={value.growthLineVisible} onChange={(next) => patch("growthLineVisible", next)} />
          {(value.particleColorMode === "neural-memory" || value.particleColorMode === "chemical-memory") && <>
            <ChannelWindowSlider channels={value.particleColorMode === "neural-memory" ? 8 : (config?.channels ?? 8)} value={value.internalStateChannelStart} onChange={(next) => patch("internalStateChannelStart", next)} />
            <RenderSlider label="Opponent subtraction" min={0} max={1} step={0.01} value={value.chemicalMemoryOpponentSubtraction} display={value.chemicalMemoryOpponentSubtraction.toFixed(2)} onChange={(next) => patch("chemicalMemoryOpponentSubtraction", next)} />
            {value.particleColorMode === "neural-memory" && !cellMemoryEnabled && <p className="hint">Enable cell memory to use these channels.</p>}
            {value.particleColorMode === "chemical-memory" && !chemicalLevelsActive && <p className="hint">Chemical memory is inactive in environment mode.</p>}
          </>}
          {value.particleColorMode === "boundary-value" && <RenderSlider label="Boundary scale" min={0.001} max={0.1} step={0.001} value={value.boundaryGradientScale} display={value.boundaryGradientScale.toFixed(3)} onChange={(next) => patch("boundaryGradientScale", next)} />}

          <label className="slider-row">
            <span>Background</span>
            <select className="select" value={value.fieldMode} onChange={(event) => patch("fieldMode", event.target.value as FieldMode)}>
              <option value="none">None</option>
              <option value="density">Density</option>
              <option value="speed">Speed</option>
              <option value="deformation">Deformation</option>
              <option value="pressure">Pressure</option>
              <option value="shear">Shear</option>
              <option value="repulsion">Repulsion field</option>
              <option value="morphology">Policy morphology</option>
              <option value="substrate">Substrate</option>
              <option value="growth">Growth (cividis)</option>
              <option value="orientation">Orientation</option>
              <option value="gradient">Boundary gradient</option>
            </select>
          </label>
          {value.fieldMode === "morphology" && (
            <>
              <ToggleButton className="toggle-row" label="Show morphology gradient (R/G)" checked={value.morphologyGradientVisible} onChange={(next) => patch("morphologyGradientVisible", next)} />
              <ToggleButton className="toggle-row" label="Show morphology density (B)" checked={value.morphologyDensityVisible} onChange={(next) => patch("morphologyDensityVisible", next)} />
            </>
          )}
          {value.fieldMode === "substrate" && config && (
            <div className="channel-window-control">
              <div className="channel-window-label"><span>RGB channels</span><span>{value.substrateChannelStart}–{Math.min(config.channels - 1, value.substrateChannelStart + 2)}</span></div>
              <ChannelWindowSlider channels={config.channels} value={value.substrateChannelStart} onChange={(next) => patch("substrateChannelStart", next)} />
            </div>
          )}
          <ToggleButton className="toggle-row" label="Zero substrate is black" checked={value.substrateZeroIsBlack} onChange={(next) => patch("substrateZeroIsBlack", next)} />
          <ToggleButton className="toggle-row" label="Zero boundary gradient is black" checked={value.boundaryGradientZeroIsBlack} onChange={(next) => patch("boundaryGradientZeroIsBlack", next)} />
          <RenderSlider label="Accent" min={-2} max={2} step={0.01} value={value.accent} display={value.accent.toFixed(2)} onChange={(next) => patch("accent", next)} />
          {value.fieldMode === "gradient" && (
            <>
              <RenderSlider label="Blur" min={0} max={2} step={0.01} value={value.blur} display={value.blur.toFixed(2)} onChange={(next) => patch("blur", next)} />
              <RenderSlider label="Gradient exponent" min={0.25} max={4} step={0.05} value={value.gradientExponent} display={value.gradientExponent.toFixed(2)} onChange={(next) => patch("gradientExponent", next)} />
            </>
          )}
        </div>
      )}
    </div>
  )
}

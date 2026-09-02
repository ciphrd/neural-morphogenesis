import { useState } from "react"
import type { FieldMode, ParticleRenderMode } from "../gpu/render"
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
            <label className="auto-zoom-toggle">
              <input
                type="checkbox"
                checked={value.autoZoom.enabled}
                onChange={(event) => patch("autoZoom", { ...value.autoZoom, enabled: event.target.checked })}
              />
              Auto
            </label>
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
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={value.bloom.enabled}
                onChange={(event) => patch("bloom", { ...value.bloom, enabled: event.target.checked })}
              />
              Bloom
            </label>
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
            <span>Particles</span>
            <select className="select" value={value.particleRenderMode} onChange={(event) => patch("particleRenderMode", event.target.value as ParticleRenderMode)}>
              <option value="dots-white">Dots (white)</option>
              <option value="dots-neural-color">Dots (neural RGB)</option>
              <option value="dots-internal-state">Chemical memory</option>
              <option value="dots-chemical-levels">Chemical levels</option>
              <option value="dots-boundary-value">Boundary value</option>
              <option value="dots-activation">Dots (neurons)</option>
              <option value="dots-activation-translucent">Dots (translucent neurons)</option>
              <option value="directional-arrows">Directional triangles</option>
            </select>
          </label>

          {value.particleRenderMode === "dots-white" && (
            <RenderSlider label="Dots alpha" min={0} max={1} step={0.01} value={value.whiteDotsAlpha} display={value.whiteDotsAlpha.toFixed(2)} onChange={(next) => patch("whiteDotsAlpha", next)} />
          )}
          {value.particleRenderMode === "dots-neural-color" && (
            <RenderSlider label="Neural RGB alpha" min={0} max={1} step={0.01} value={value.neuralColorAlpha} display={value.neuralColorAlpha.toFixed(2)} onChange={(next) => patch("neuralColorAlpha", next)} />
          )}
          {(value.particleRenderMode === "dots-internal-state" || value.particleRenderMode === "dots-chemical-levels") && (
            <>
              <RenderSlider
                label={value.particleRenderMode === "dots-internal-state" ? "Chemical memory alpha" : "Chemical levels alpha"}
                min={0}
                max={1}
                step={0.01}
                value={value.internalStateAlpha}
                display={value.internalStateAlpha.toFixed(2)}
                onChange={(next) => patch("internalStateAlpha", next)}
              />
              <div className="channel-window-control">
                <div className="channel-window-label">
                  <span>Channels</span>
                  <span>{value.internalStateChannelStart}–{value.internalStateChannelStart + 2}</span>
                </div>
                <ChannelWindowSlider
                  channels={value.particleRenderMode === "dots-internal-state" ? 8 : (config?.channels ?? 8)}
                  value={value.internalStateChannelStart}
                  onChange={(next) => patch("internalStateChannelStart", next)}
                  channelKind={value.particleRenderMode === "dots-internal-state" ? "chemical memory" : "chemical levels"}
                />
              </div>
              {value.particleRenderMode === "dots-internal-state" && (
                <RenderSlider label="Opponent subtraction" min={0} max={1} step={0.01} value={value.chemicalMemoryOpponentSubtraction} display={value.chemicalMemoryOpponentSubtraction.toFixed(2)} onChange={(next) => patch("chemicalMemoryOpponentSubtraction", next)} />
              )}
              {value.particleRenderMode === "dots-internal-state" && !cellMemoryEnabled && <p className="hint">Cell memory is disabled, so these channels remain zero.</p>}
              {value.particleRenderMode === "dots-chemical-levels" && !chemicalLevelsActive && <p className="hint">Chemical levels are inactive in persistent-environment mode.</p>}
            </>
          )}
          {value.particleRenderMode === "dots-boundary-value" && (
            <>
              <RenderSlider label="Boundary g0" min={0.001} max={0.1} step={0.001} value={value.boundaryGradientScale} display={value.boundaryGradientScale.toFixed(3)} onChange={(next) => patch("boundaryGradientScale", next)} />
              <p className="hint">Cell color shows boundary-gradient strength from dark blue to yellow.</p>
            </>
          )}
          {value.particleRenderMode === "dots-activation-translucent" && (
            <RenderSlider label="Activation alpha" min={0} max={1} step={0.01} value={value.activationAlpha} display={value.activationAlpha.toFixed(2)} onChange={(next) => patch("activationAlpha", next)} />
          )}
          {value.particleRenderMode === "directional-arrows" && (
            <RenderSlider label="Triangle size" min={8} max={80} step={1} value={value.growthAxisLengthPx} display={`${value.growthAxisLengthPx}px`} onChange={(next) => patch("growthAxisLengthPx", next)} />
          )}

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
              <option value="gradient">Boundary gradient</option>
            </select>
          </label>
          {value.fieldMode === "morphology" && (
            <>
              <label className="checkbox-row"><input type="checkbox" checked={value.morphologyGradientVisible} onChange={(event) => patch("morphologyGradientVisible", event.target.checked)} />Show morphology gradient (R/G)</label>
              <label className="checkbox-row"><input type="checkbox" checked={value.morphologyDensityVisible} onChange={(event) => patch("morphologyDensityVisible", event.target.checked)} />Show morphology density (B)</label>
            </>
          )}
          {value.fieldMode === "substrate" && config && (
            <div className="channel-window-control">
              <div className="channel-window-label"><span>RGB channels</span><span>{value.substrateChannelStart}–{Math.min(config.channels - 1, value.substrateChannelStart + 2)}</span></div>
              <ChannelWindowSlider channels={config.channels} value={value.substrateChannelStart} onChange={(next) => patch("substrateChannelStart", next)} />
            </div>
          )}
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

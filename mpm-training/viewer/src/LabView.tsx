import { useEffect, useMemo, useRef, useState } from "react"
import { randomWeights } from "./gpu/agents"
import { configAtDensity } from "./gpu/density"
import type { FieldMode, ParticleColorMode, ParticleShape } from "./gpu/render"
import type { CellMemory, ChemicalCommunicationArchitecture, PhysicsSettings } from "./gpu/types"
import {
  cellMemoryFromConfig,
  chemicalCommunicationArchitectureFromConfig,
  hiddenLayersFromConfig,
  physicsSettingsFromConfig,
  policyArchitectureForCellMemory,
} from "./gpu/types"
import type { SimulationScenario } from "./gpu/simulation"
import { useTrainingSocket } from "./net/trainingSocket"
import { pickRecordingFormat } from "./render/canvasRecorder"
import type { DeformSettings, GridCanvasHandle, Tool } from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"
import { ChannelWindowSlider } from "./ui/ChannelWindowSlider"
import { DevelopmentalFieldsPanel } from "./ui/DevelopmentalFieldsPanel"
import { GrowthPanel } from "./ui/GrowthPanel"
import { PhysicsPanel } from "./ui/PhysicsPanel"
import { Slider } from "./ui/Slider"
import { VIEWER_DEFAULTS } from "./viewerConfig"

const TRAIN_API_URL = "http://localhost:8003"
const TRAIN_WS_URL = "ws://localhost:8003/ws"
const RECORDING_FORMAT = pickRecordingFormat()

function explorationBrainSeed(seed: number, variant: number): number {
  if (variant === 0) return seed >>> 0
  let x = ((seed >>> 0) ^ Math.imul(variant, 0x9e3779b9)) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0
  return (x ^ (x >>> 16)) >>> 0
}

const BOUNDARY_TANGENT_SCENARIO: SimulationScenario = {
  initialLayout: { kind: "rows", rows: 2, columns: 9 },
  // seedRows is bottom-to-top, left-to-right: index 13 is top-middle.
  // Slot 13 remains the original particle's daughter after the first split,
  // so targeting it again exercises the exact same lineage cell at step 200.
  events: [
    { step: 50, type: "split", particleIndex: 13 },
    { step: 200, type: "split", particleIndex: 13 },
  ],
  suppressNaturalGrowth: true,
}

const VERTICAL_SPLIT_SCENARIO: SimulationScenario = {
  initialLayout: { kind: "rows", rows: 2, columns: 9 },
  // Fixed world-up division axis. Slot 13 is retained as one daughter,
  // allowing the second event to target the same lineage cell again.
  events: [
    { step: 50, type: "split", particleIndex: 13, direction: [0, 1] },
    { step: 200, type: "split", particleIndex: 13, direction: [0, 1] },
  ],
  suppressNaturalGrowth: true,
}

const REPEATED_TOP_ROW_SPLIT_SCENARIO: SimulationScenario = {
  initialLayout: { kind: "rows", rows: 2, columns: 9 },
  // Each synchronized split creates nine children in the next contiguous
  // range. Those upward daughters become the top row targeted next time.
  events: Array.from({ length: 7 }, (_, cycle) => ({
    step: (cycle + 1) * 100,
    type: "split" as const,
    particleIndex: 9 + cycle * 9,
    particleCount: 9,
    direction: [0, 1] as const,
  })),
  suppressNaturalGrowth: true,
}

type LabScenarioId = "boundary-tangent" | "vertical" | "repeated-top-row"

export function LabView() {
  const { latest } = useTrainingSocket(TRAIN_WS_URL, TRAIN_API_URL)
  const [scenarioId, setScenarioId] = useState<LabScenarioId>(
    VIEWER_DEFAULTS.lab.scenario
  )
  const scenario = scenarioId === "repeated-top-row"
    ? REPEATED_TOP_ROW_SPLIT_SCENARIO
    : scenarioId === "vertical"
      ? VERTICAL_SPLIT_SCENARIO
      : BOUNDARY_TANGENT_SCENARIO
  const scenarioParticleCap = scenarioId === "repeated-top-row" ? 81 : 20
  const scenarioMinimumSteps = scenarioId === "repeated-top-row" ? 750 : 240
  const scenarioAxisLabel = scenarioId === "boundary-tangent"
    ? "Local boundary-tangent axis"
    : "Fixed vertical axis"
  const baseConfig = useMemo(() => latest ? {
    ...latest,
    particles: scenarioParticleCap,
    initialParticleCount: 18,
    macroSteps: Math.max(scenarioMinimumSteps, latest.macroSteps),
  } : null, [latest, scenarioParticleCap, scenarioMinimumSteps])
  const [chemicalArchitectureOverride, setChemicalArchitectureOverride] =
    useState<ChemicalCommunicationArchitecture | null>(null)
  const [chiralityOverride, setChiralityOverride] = useState<boolean | null>(null)
  const [particleDensityOverride, setParticleDensityOverride] = useState<number | null>(null)
  const [substrateResolutionOverride, setSubstrateResolutionOverride] = useState<number | null>(null)
  const [policyExploration, setPolicyExploration] = useState<{
    cellMemory: CellMemory
    hiddenWidth: number
    variant: number
  } | null>(null)
  const [physicsOverride, setPhysicsOverride] = useState<PhysicsSettings | null>(null)
  useEffect(() => {
    setChemicalArchitectureOverride(null)
    setChiralityOverride(null)
    setParticleDensityOverride(null)
    setSubstrateResolutionOverride(null)
    setPolicyExploration(null)
    setPhysicsOverride(null)
  }, [latest?.generation])
  const defaultParticleDensity =
    VIEWER_DEFAULTS.playback.particleDensityMultiplier ??
    baseConfig?.particleDensityMultiplier ??
    1
  const effectiveParticleDensity =
    particleDensityOverride ?? defaultParticleDensity
  const defaultSubstrateResolution =
    VIEWER_DEFAULTS.playback.substrateResolution ?? baseConfig?.fieldN ?? 256
  const effectiveSubstrateResolution =
    substrateResolutionOverride ?? defaultSubstrateResolution
  const effectiveChirality = chiralityOverride ?? baseConfig?.chirality ?? true
  const playbackConfig = useMemo(() => {
    if (!baseConfig) return null
    const densityResolved = configAtDensity({
      ...baseConfig,
      fieldN: effectiveSubstrateResolution,
      chemicalCommunicationArchitecture:
        chemicalArchitectureOverride ?? chemicalCommunicationArchitectureFromConfig(baseConfig),
      chirality: effectiveChirality,
    }, effectiveParticleDensity)
    return { ...densityResolved, particles: scenarioParticleCap, initialParticleCount: 18 }
  }, [baseConfig, chemicalArchitectureOverride, effectiveChirality, effectiveParticleDensity, effectiveSubstrateResolution, scenarioParticleCap])
  const config = useMemo(() => {
    if (!playbackConfig || !policyExploration) return playbackConfig
    const policyArchitecture = policyArchitectureForCellMemory(policyExploration.cellMemory)
    return {
      ...playbackConfig,
      cellMemory: policyExploration.cellMemory,
      hiddenDim: policyExploration.hiddenWidth,
      hiddenLayers: [policyExploration.hiddenWidth],
      policyArchitecture,
      weights: randomWeights(
        playbackConfig.channels,
        policyExploration.hiddenWidth,
        policyArchitecture,
        explorationBrainSeed(playbackConfig.seed, policyExploration.variant),
      ),
    }
  }, [playbackConfig, policyExploration])
  const displayedCellMemory = policyExploration?.cellMemory
    ?? (baseConfig ? cellMemoryFromConfig(baseConfig) : "recurrent")
  const displayedHiddenWidth = policyExploration?.hiddenWidth
    ?? (baseConfig ? hiddenLayersFromConfig(baseConfig)[0] : 128)
  const trainedPhysics = useMemo(
    () => playbackConfig ? physicsSettingsFromConfig(playbackConfig) : null,
    [playbackConfig],
  )
  const physics = physicsOverride ?? trainedPhysics
  const canvasRef = useRef<GridCanvasHandle>(null)
  const [paused, setPaused] = useState(VIEWER_DEFAULTS.playback.paused)
  const [recording, setRecording] = useState(false)
  const [step, setStep] = useState(0)
  const [cellCount, setCellCount] = useState(0)
  const [tool, setTool] = useState<Tool>(VIEWER_DEFAULTS.tools.selected)
  const [zoom, setZoom] = useState(VIEWER_DEFAULTS.rendering.zoom)
  const [autoZoomEnabled, setAutoZoomEnabled] = useState(VIEWER_DEFAULTS.rendering.autoZoom.enabled)
  const [effectiveZoom, setEffectiveZoom] = useState(VIEWER_DEFAULTS.rendering.zoom)
  const autoZoomSettings = useMemo(() => ({
    ...VIEWER_DEFAULTS.rendering.autoZoom,
    enabled: autoZoomEnabled,
  }), [autoZoomEnabled])
  const [bloom, setBloom] = useState(() => ({ ...VIEWER_DEFAULTS.rendering.bloom }))
  const [particleRadiusPx, setParticleRadiusPx] = useState(VIEWER_DEFAULTS.rendering.particleRadiusPx)
  const [particleShape, setParticleShape] = useState<ParticleShape>(VIEWER_DEFAULTS.rendering.particleShape)
  const [particleColorMode, setParticleColorMode] = useState<ParticleColorMode>(VIEWER_DEFAULTS.rendering.particleColorMode)
  const [particleAlpha, setParticleAlpha] = useState(VIEWER_DEFAULTS.rendering.particleAlpha)
  const [directionalLineVisible, setDirectionalLineVisible] = useState(VIEWER_DEFAULTS.rendering.directionalLineVisible)
  const [splitDirectionLineVisible, setSplitDirectionLineVisible] = useState(VIEWER_DEFAULTS.rendering.splitDirectionLineVisible)
  const [mitosisSignalBoost, setMitosisSignalBoost] = useState(VIEWER_DEFAULTS.rendering.mitosisSignalBoost)
  const [internalStateChannelStart, setInternalStateChannelStart] = useState(VIEWER_DEFAULTS.rendering.internalStateChannelStart)
  const [chemicalMemoryOpponentSubtraction, setChemicalMemoryOpponentSubtraction] = useState(VIEWER_DEFAULTS.rendering.chemicalMemoryOpponentSubtraction)
  const [boundaryGradientScale, setBoundaryGradientScale] = useState(VIEWER_DEFAULTS.rendering.boundaryGradientScale)
  const [fieldMode, setFieldMode] = useState<FieldMode>(VIEWER_DEFAULTS.rendering.fieldMode)
  const [substrateChannelStart, setSubstrateChannelStart] = useState(VIEWER_DEFAULTS.rendering.substrateChannelStart)
  const [substrateZeroIsBlack, setSubstrateZeroIsBlack] = useState(VIEWER_DEFAULTS.rendering.substrateZeroIsBlack)
  const [boundaryGradientZeroIsBlack, setBoundaryGradientZeroIsBlack] = useState(VIEWER_DEFAULTS.rendering.boundaryGradientZeroIsBlack)
  const [morphologyGradientVisible, setMorphologyGradientVisible] = useState(VIEWER_DEFAULTS.rendering.morphologyGradientVisible)
  const [morphologyDensityVisible, setMorphologyDensityVisible] = useState(VIEWER_DEFAULTS.rendering.morphologyDensityVisible)
  const [accent, setAccent] = useState(VIEWER_DEFAULTS.rendering.accent)
  const [blur, setBlur] = useState(VIEWER_DEFAULTS.rendering.blur)
  const [gradientExponent, setGradientExponent] = useState(VIEWER_DEFAULTS.rendering.gradientExponent)
  const [developmentalSettings, setDevelopmentalSettings] = useState(() => ({ ...VIEWER_DEFAULTS.developmental }))
  const [deformSettings, setDeformSettings] = useState<DeformSettings>(() => ({
    ...VIEWER_DEFAULTS.tools.deform,
  }))

  const toggleRecording = () => {
    if (!RECORDING_FORMAT) return
    if (recording) {
      void canvasRef.current?.stopRecording().finally(() => setRecording(false))
    } else {
      canvasRef.current?.startRecording()
      setRecording(true)
    }
  }

  const particleStateChannelCount = particleColorMode === "neural-memory"
    ? 8
    : (config?.channels ?? 1)
  const particleStateChannelStart = Math.min(
    Math.max(0, particleStateChannelCount - 3),
    internalStateChannelStart,
  )

  return (
    <div className="training-layout lab-layout">
      <aside className="controls lab-controls">
        <h1>Simulation lab</h1>
        <section>
          <h2>Scenario</h2>
          <select
            className="select lab-scenario-select"
            value={scenarioId}
            onChange={(event) => {
              setScenarioId(event.target.value as LabScenarioId)
              setStep(0)
              setCellCount(0)
            }}
            aria-label="Lab scenario"
          >
            <option value="boundary-tangent">2 × 9 — boundary-tangent splits</option>
            <option value="vertical">2 × 9 — vertical splits</option>
            <option value="repeated-top-row">2 × 9 — grow seven vertical rows</option>
          </select>
        </section>
        <section>
          <h2>Initial state</h2>
          <div className="stat-row"><span>Cells</span><span>18</span></div>
          <div className="stat-row"><span>Layout</span><span>2 rows × 9</span></div>
          <div className="stat-row"><span>Spacing</span><span>Split distance</span></div>
        </section>
        <section>
          <details className="settings-category">
          <summary>Simulation</summary>
          <div className="stat-row">
            <span>Cell memory</span>
            <select
              className="select"
              aria-label="Cell memory"
              value={displayedCellMemory}
              disabled={!baseConfig}
              onChange={(event) => setPolicyExploration({
                cellMemory: event.target.value as CellMemory,
                hiddenWidth: displayedHiddenWidth,
                variant: 0,
              })}
            >
              <option value="none">None</option>
              <option value="recurrent">Recurrent</option>
            </select>
          </div>
          <div className="stat-row">
            <span>Hidden layer</span>
            <select
              className="select"
              aria-label="Hidden layer width"
              value={displayedHiddenWidth}
              disabled={!baseConfig}
              onChange={(event) => setPolicyExploration({
                cellMemory: displayedCellMemory,
                hiddenWidth: Number(event.target.value),
                variant: 0,
              })}
            >
              {[16, 32, 64, 128, 256].map((width) => (
                <option key={width} value={width}>{width}</option>
              ))}
            </select>
          </div>
          <div className="stat-row">
            <span>Brain source</span>
            <button
              className="select simulation-control-button"
              disabled={!policyExploration}
              onClick={() => setPolicyExploration(null)}
              title="Restore the current generation's trained brain"
            >
              {policyExploration
                ? `Seeded random #${policyExploration.variant + 1} ↺`
                : "Trained"}
            </button>
          </div>
          <div className="stat-row">
            <span>Chemical architecture</span>
            <select
              className="select"
              aria-label="Chemical architecture"
              value={config?.chemicalCommunicationArchitecture ?? "cell-owned-projection"}
              disabled={!baseConfig}
              onChange={(event) => {
                const selected = event.target.value as ChemicalCommunicationArchitecture
                const trained = baseConfig
                  ? chemicalCommunicationArchitectureFromConfig(baseConfig)
                  : "cell-owned-projection"
                setChemicalArchitectureOverride(selected === trained ? null : selected)
              }}
            >
              <option value="cell-owned-projection">Cell-owned projection</option>
              <option value="persistent-environment">Persistent environment</option>
            </select>
          </div>
          <div className="stat-row">
            <span>Particle density</span>
            <select
              className="select"
              aria-label="Particle density"
              value={effectiveParticleDensity}
              disabled={!baseConfig}
              onChange={(event) => {
                const selected = Number(event.target.value)
                setParticleDensityOverride(selected === defaultParticleDensity ? null : selected)
                setPhysicsOverride(null)
              }}
            >
              {Array.from(new Set([0.5, 1, 2, 4, effectiveParticleDensity]))
                .sort((a, b) => a - b)
                .map((density) => (
                  <option key={density} value={density}>{density}×</option>
                ))}
            </select>
          </div>
          <div className="stat-row">
            <span>Substrate resolution</span>
            <select
              className="select"
              aria-label="Substrate resolution"
              value={effectiveSubstrateResolution}
              disabled={!baseConfig}
              title="Changing substrate resolution rebuilds and restarts the Lab scenario"
              onChange={(event) => {
                const selected = Number(event.target.value)
                setSubstrateResolutionOverride(
                  selected === defaultSubstrateResolution ? null : selected
                )
              }}
            >
              {Array.from(new Set([64, 128, 256, 512, 1024, 2048, effectiveSubstrateResolution]))
                .sort((a, b) => a - b)
                .map((resolution) => (
                  <option key={resolution} value={resolution}>{resolution}×{resolution}</option>
                ))}
            </select>
          </div>
          <label className="checkbox-row" title="Changing chirality restarts the Lab scenario">
            <input
              type="checkbox"
              checked={effectiveChirality}
              disabled={!baseConfig}
              onChange={(event) => {
                const selected = event.target.checked
                const trained = baseConfig?.chirality ?? true
                setChiralityOverride(selected === trained ? null : selected)
              }}
            />
            Chirality
          </label>
          </details>
        </section>
        <section>
          <h2>Events</h2>
          {scenario.events.map((event, eventIndex) => (
            <div key={`${event.step}-${event.particleIndex}`} className={`lab-event${step >= event.step ? " is-complete" : ""}`}>
              <span className="lab-event-step">{event.step}</span>
              <span>
                {scenarioId === "repeated-top-row"
                  ? `Split all nine top cells — cycle ${eventIndex + 1}/7`
                  : eventIndex === 0
                    ? "Split top-middle cell"
                    : "Split the same lineage cell again"}
                <small>{scenarioAxisLabel}</small>
              </span>
            </div>
          ))}
        </section>
        <DevelopmentalFieldsPanel
          value={developmentalSettings}
          onChange={setDevelopmentalSettings}
          onReseed={() => canvasRef.current?.reseedDevelopmentalFields()}
        />
        <section>
          <h2>Rendering</h2>
          <div className="slider-row">
            <span>Zoom</span>
            <label className="auto-zoom-toggle"><input type="checkbox" checked={autoZoomEnabled} onChange={(event) => setAutoZoomEnabled(event.target.checked)} />Auto</label>
            <Slider min={1} max={8} step={0.05} value={zoom} disabled={autoZoomEnabled} onChange={(value) => { setZoom(value); setEffectiveZoom(value) }} />
            <span className="slider-value">{(autoZoomEnabled ? effectiveZoom : zoom).toFixed(2)}×</span>
          </div>
          <details className="settings-category">
            <summary>Post-processing</summary>
            <label className="checkbox-row"><input type="checkbox" checked={bloom.enabled} onChange={(event) => setBloom((value) => ({ ...value, enabled: event.target.checked }))} />Bloom</label>
            {bloom.enabled && <>
              <label className="slider-row"><span>Bloom intensity</span><Slider min={0} max={3} step={0.05} value={bloom.intensity} onChange={(intensity) => setBloom((value) => ({ ...value, intensity }))} /><span className="slider-value">{bloom.intensity.toFixed(2)}</span></label>
              <label className="slider-row"><span>Bloom threshold</span><Slider min={0} max={1} step={0.01} value={bloom.threshold} onChange={(threshold) => setBloom((value) => ({ ...value, threshold }))} /><span className="slider-value">{bloom.threshold.toFixed(2)}</span></label>
              <label className="slider-row"><span>Bloom radius</span><Slider min={0.25} max={8} step={0.25} value={bloom.radiusPx} onChange={(radiusPx) => setBloom((value) => ({ ...value, radiusPx }))} /><span className="slider-value">{bloom.radiusPx.toFixed(2)}px</span></label>
              <label className="slider-row"><span>Bloom levels</span><Slider min={2} max={10} step={1} value={bloom.levels} onChange={(levels) => setBloom((value) => ({ ...value, levels }))} /><span className="slider-value">{bloom.levels}</span></label>
              <label className="slider-row"><span>Bloom scatter</span><Slider min={0} max={1} step={0.01} value={bloom.scatter} onChange={(scatter) => setBloom((value) => ({ ...value, scatter }))} /><span className="slider-value">{bloom.scatter.toFixed(2)}</span></label>
            </>}
          </details>
          <label className="slider-row">
            <span>Particle size</span>
            <Slider min={1} max={16} step={1} value={particleRadiusPx} onChange={setParticleRadiusPx} />
            <span className="slider-value">{particleRadiusPx}px</span>
          </label>
          <label className="slider-row"><span>Shape</span><select className="select" value={particleShape} onChange={(event) => setParticleShape(event.target.value as ParticleShape)}><option value="dot">Dot</option><option value="triangle">Triangle</option></select></label>
          <label className="slider-row"><span>Color</span><select className="select" value={particleColorMode} onChange={(event) => setParticleColorMode(event.target.value as ParticleColorMode)}><option value="white">White</option><option value="neural-color">Neural RGB</option><option value="mitosis-drive">Mitosis drive</option><option value="neural-memory">Neural memory</option><option value="chemical-memory">Chemical memory</option><option value="boundary-value">Boundary value</option><option value="neurons">Neurons</option></select></label>
          <label className="slider-row"><span>Alpha</span><Slider min={0} max={1} step={0.01} value={particleAlpha} onChange={setParticleAlpha} /><span className="slider-value">{particleAlpha.toFixed(2)}</span></label>
          <label className="checkbox-row"><input type="checkbox" checked={directionalLineVisible} onChange={(event) => setDirectionalLineVisible(event.target.checked)} />Heading direction</label>
          <label className="checkbox-row"><input type="checkbox" checked={splitDirectionLineVisible} onChange={(event) => setSplitDirectionLineVisible(event.target.checked)} />Split direction</label>
          {particleColorMode === "mitosis-drive" && (
            <label className="slider-row"><span>Signal boost</span><Slider min={1} max={10} step={0.1} value={mitosisSignalBoost} onChange={setMitosisSignalBoost} /><span className="slider-value">{mitosisSignalBoost.toFixed(1)}×</span></label>
          )}
          {(particleColorMode === "neural-memory" || particleColorMode === "chemical-memory") && (
            <>
              <div className="channel-window-control">
                <div className="channel-window-label"><span>Channels</span><span>{particleStateChannelStart}–{particleStateChannelStart + 2}</span></div>
                <ChannelWindowSlider channels={particleStateChannelCount} value={particleStateChannelStart} onChange={setInternalStateChannelStart} />
              </div>
              {particleColorMode === "neural-memory" && (
                <label className="slider-row"><span>Opponent subtraction</span><Slider min={0} max={1} step={0.01} value={chemicalMemoryOpponentSubtraction} onChange={setChemicalMemoryOpponentSubtraction} /><span className="slider-value">{chemicalMemoryOpponentSubtraction.toFixed(2)}</span></label>
              )}
            </>
          )}
          {particleColorMode === "boundary-value" && (
            <label className="slider-row"><span>Boundary g0</span><Slider min={0.001} max={0.1} step={0.001} value={boundaryGradientScale} onChange={setBoundaryGradientScale} /><span className="slider-value">{boundaryGradientScale.toFixed(3)}</span></label>
          )}
          <label className="slider-row">
            <span>Background</span>
            <select className="select" value={fieldMode} onChange={(event) => setFieldMode(event.target.value as FieldMode)}>
              <option value="none">None</option><option value="density">Density</option><option value="speed">Speed</option><option value="deformation">Deformation</option><option value="pressure">Pressure</option><option value="shear">Shear</option><option value="repulsion">Repulsion field</option><option value="morphology">Policy morphology</option><option value="substrate">Substrate</option><option value="gradient">Boundary gradient</option><option value="developmental-ap">Developmental AP coordinate</option><option value="developmental-anterior">Developmental anterior</option><option value="developmental-posterior">Developmental posterior</option><option value="developmental-inhibitor">Developmental inhibitor</option>
            </select>
          </label>
          {fieldMode === "morphology" && <>
            <label className="checkbox-row"><input type="checkbox" checked={morphologyGradientVisible} onChange={(event) => setMorphologyGradientVisible(event.target.checked)} />Show morphology gradient (R/G)</label>
            <label className="checkbox-row"><input type="checkbox" checked={morphologyDensityVisible} onChange={(event) => setMorphologyDensityVisible(event.target.checked)} />Show morphology density (B)</label>
          </>}
          {fieldMode === "substrate" && config && <><div className="channel-window-control"><div className="channel-window-label"><span>RGB channels</span><span>{substrateChannelStart}–{Math.min(config.channels - 1, substrateChannelStart + 2)}</span></div><ChannelWindowSlider channels={config.channels} value={substrateChannelStart} onChange={setSubstrateChannelStart} /></div><label className="checkbox-row"><input type="checkbox" checked={substrateZeroIsBlack} onChange={(event) => setSubstrateZeroIsBlack(event.target.checked)} />Zero is black</label></>}
          <label className="slider-row"><span>Accent</span><Slider min={-2} max={2} step={0.01} value={accent} onChange={setAccent} /><span className="slider-value">{accent.toFixed(2)}</span></label>
          {fieldMode === "gradient" && <>
            <label className="checkbox-row"><input type="checkbox" checked={boundaryGradientZeroIsBlack} onChange={(event) => setBoundaryGradientZeroIsBlack(event.target.checked)} />Zero is black</label>
            <label className="slider-row"><span>Blur</span><Slider min={0} max={2} step={0.01} value={blur} onChange={setBlur} /><span className="slider-value">{blur.toFixed(2)}</span></label>
            <label className="slider-row"><span>Gradient exponent</span><Slider min={0.25} max={4} step={0.05} value={gradientExponent} onChange={setGradientExponent} /><span className="slider-value">{gradientExponent.toFixed(2)}</span></label>
          </>}
        </section>
        {trainedPhysics && physics && (
          <>
            <PhysicsPanel
              trained={trainedPhysics}
              value={physics}
              onChange={setPhysicsOverride}
              isOverridden={physicsOverride !== null}
              onReset={() => setPhysicsOverride(null)}
            />
            <GrowthPanel
              trained={trainedPhysics}
              value={physics}
              onChange={setPhysicsOverride}
              isOverridden={physicsOverride !== null}
              onReset={() => setPhysicsOverride(null)}
            />
          </>
        )}
        <p className="hint">
          The current training configuration supplies policy and physics. Lab
          particles and scheduled events remain isolated from training.
        </p>
      </aside>

      <main className="center-column">
        <div className="viewport">
          <GridCanvas
            ref={canvasRef}
            config={config}
            scenario={scenario}
            targetPoints={null}
            targetVisible={false}
            physics={physics}
            developmentalSettings={developmentalSettings}
            zoom={zoom}
            autoZoom={autoZoomSettings}
            onEffectiveZoomChange={setEffectiveZoom}
            bloom={bloom}
            particleRadiusPx={particleRadiusPx}
            particleShape={particleShape}
            particleColorMode={particleColorMode}
            particleAlpha={particleAlpha}
            directionalLineVisible={directionalLineVisible}
            splitDirectionLineVisible={splitDirectionLineVisible}
            mitosisSignalBoost={mitosisSignalBoost}
            internalStateChannelStart={internalStateChannelStart}
            chemicalMemoryOpponentSubtraction={chemicalMemoryOpponentSubtraction}
            boundaryGradientScale={boundaryGradientScale}
            fieldMode={fieldMode}
            substrateChannelStart={substrateChannelStart}
            substrateZeroIsBlack={substrateZeroIsBlack}
            boundaryGradientZeroIsBlack={boundaryGradientZeroIsBlack}
            morphologyGradientVisible={morphologyGradientVisible}
            morphologyDensityVisible={morphologyDensityVisible}
            accent={accent}
            blur={blur}
            gradientExponent={gradientExponent}
            particleCap={scenarioParticleCap}
            initialParticleCount={18}
            deformSettings={deformSettings}
            tool={tool}
            loopAtTrainedSteps={false}
            paused={paused}
            onStep={(nextStep, particles) => {
              setStep(nextStep)
              setCellCount(particles)
            }}
          />
          <div className="viewport-telemetry" aria-label="Scenario status">
            <span>{config ? `${step} steps` : "Waiting for training config…"}</span>
            <span>{config ? `${cellCount} / ${scenarioParticleCap} cells` : "— cells"}</span>
          </div>
        </div>

        <div className="toolbar">
          <div className="tool-buttons-wrap">
            {tool === "deform" && (
              <div className="tool-settings-panel">
                <h3>Deform</h3>
                <>
                    <label className="checkbox-row">
                      <input type="checkbox" checked={deformSettings.direction === "outward"} onChange={(event) => setDeformSettings((value) => ({ ...value, direction: event.target.checked ? "outward" : "inward" }))} />
                      {deformSettings.direction === "outward" ? "Push outward (explode)" : "Pull inward (implode)"}
                    </label>
                    <label className="slider-row"><span>Strength</span><input type="range" min="0" max="2" step="0.01" value={deformSettings.strength} onChange={(event) => setDeformSettings((value) => ({ ...value, strength: Number(event.target.value) }))} /><span className="slider-value">{deformSettings.strength.toFixed(2)}</span></label>
                    <label className="slider-row"><span>Radius</span><input type="range" min="0.01" max="0.5" step="0.01" value={deformSettings.radius} onChange={(event) => setDeformSettings((value) => ({ ...value, radius: Number(event.target.value) }))} /><span className="slider-value">{deformSettings.radius.toFixed(2)}</span></label>
                    <label className="checkbox-row"><input type="checkbox" checked={deformSettings.mode === "deformation"} onChange={(event) => setDeformSettings((value) => ({ ...value, mode: event.target.checked ? "deformation" : "velocity" }))} />Direct deformation (F) edit</label>
                </>
              </div>
            )}
            <div className="tool-buttons">
              <button className={`icon-button${tool === "add" ? " is-active" : ""}`} onClick={() => setTool((value) => value === "add" ? "none" : "add")} aria-label="Add particle" aria-pressed={tool === "add"}>＋</button>
              <button className={`icon-button${tool === "move" ? " is-active" : ""}`} onClick={() => setTool((value) => value === "move" ? "none" : "move")} aria-label="Move particles" aria-pressed={tool === "move"}>✥</button>
              <button className={`icon-button${tool === "deform" ? " is-active" : ""}`} onClick={() => setTool((value) => value === "deform" ? "none" : "deform")} title="Deform" aria-label="Deform" aria-pressed={tool === "deform"}>⤢</button>
            </div>
          </div>
          <div className="toolbar-actions">
            <label className="toolbar-checkbox" title="Lab scenarios run past the configured event"><input type="checkbox" checked={false} readOnly />Loop</label>
            <button className="icon-button" disabled title="Parameter sweeps are unavailable in lab scenarios" aria-label="Collect parameter samples">◫</button>
            <button className="icon-button" onClick={() => canvasRef.current?.randomizeWeights()} disabled={!config} title="Load random brain" aria-label="Load random brain">🎲</button>
            <button className="icon-button" onClick={() => setPaused((value) => !value)} title={paused ? "Play" : "Pause"} aria-label={paused ? "Play" : "Pause"}>{paused ? "▶" : "⏸"}</button>
            <button className="icon-button" onClick={() => canvasRef.current?.restart()} title="Restart scenario" aria-label="Restart">↺</button>
            <button className={`icon-button${recording ? " is-recording" : ""}`} onClick={toggleRecording} disabled={!RECORDING_FORMAT} title={recording ? "Stop recording" : "Record"} aria-label={recording ? "Stop recording" : "Record"}>{recording ? "●" : "⏺"}</button>
          </div>
        </div>
      </main>
    </div>
  )
}

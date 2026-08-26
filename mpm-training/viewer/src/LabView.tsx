import { useMemo, useRef, useState } from "react"
import type { FieldMode, ParticleRenderMode } from "./gpu/render"
import { physicsSettingsFromConfig } from "./gpu/types"
import type { SimulationScenario } from "./gpu/simulation"
import { useTrainingSocket } from "./net/trainingSocket"
import { pickRecordingFormat } from "./render/canvasRecorder"
import type { DeformSettings, GridCanvasHandle, Tool } from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"
import { ChannelWindowSlider } from "./ui/ChannelWindowSlider"
import { Slider } from "./ui/Slider"

const TRAIN_API_URL = "http://localhost:8003"
const TRAIN_WS_URL = "ws://localhost:8003/ws"
const RECORDING_FORMAT = pickRecordingFormat()

const TWO_BY_FIVE_SCENARIO: SimulationScenario = {
  initialLayout: { kind: "rows", rows: 2, columns: 5 },
  // seedRows is bottom-to-top, left-to-right: index 7 is top-middle.
  events: [{ step: 50, type: "split", particleIndex: 7, direction: [1, 0] }],
  suppressNaturalGrowth: true,
}

export function LabView() {
  const { latest } = useTrainingSocket(TRAIN_WS_URL, TRAIN_API_URL)
  const config = useMemo(() => latest ? {
    ...latest,
    particles: 11,
    initialParticleCount: 10,
    macroSteps: Math.max(120, latest.macroSteps),
  } : null, [latest])
  const physics = useMemo(
    () => config ? physicsSettingsFromConfig(config) : null,
    [config],
  )
  const canvasRef = useRef<GridCanvasHandle>(null)
  const [paused, setPaused] = useState(false)
  const [recording, setRecording] = useState(false)
  const [step, setStep] = useState(0)
  const [cellCount, setCellCount] = useState(0)
  const [tool, setTool] = useState<Tool>("none")
  const [zoom, setZoom] = useState(1)
  const [particleRadiusPx, setParticleRadiusPx] = useState(4)
  const [particleRenderMode, setParticleRenderMode] = useState<ParticleRenderMode>("dots-white")
  const [whiteDotsAlpha, setWhiteDotsAlpha] = useState(1)
  const [neuralColorAlpha, setNeuralColorAlpha] = useState(1)
  const [internalStateAlpha, setInternalStateAlpha] = useState(1)
  const [activationAlpha, setActivationAlpha] = useState(0.2)
  const [internalStateChannelStart, setInternalStateChannelStart] = useState(0)
  const [boundaryGradientScale, setBoundaryGradientScale] = useState(0.01)
  const [growthAxisLengthPx, setGrowthAxisLengthPx] = useState(28)
  const [fieldMode, setFieldMode] = useState<FieldMode>("none")
  const [substrateChannelStart, setSubstrateChannelStart] = useState(0)
  const [morphologyGradientVisible, setMorphologyGradientVisible] = useState(true)
  const [morphologyDensityVisible, setMorphologyDensityVisible] = useState(true)
  const [accent, setAccent] = useState(0)
  const [blur, setBlur] = useState(0)
  const [gradientExponent, setGradientExponent] = useState(1)
  const [deformSettings, setDeformSettings] = useState<DeformSettings>({
    direction: "outward",
    strength: 1,
    radius: 0.08,
    mode: "velocity",
  })

  const toggleRecording = () => {
    if (!RECORDING_FORMAT) return
    if (recording) {
      void canvasRef.current?.stopRecording().finally(() => setRecording(false))
    } else {
      canvasRef.current?.startRecording()
      setRecording(true)
    }
  }

  return (
    <div className="training-layout lab-layout">
      <aside className="controls lab-controls">
        <h1>Simulation lab</h1>
        <section>
          <h2>Scenario</h2>
          <select className="select lab-scenario-select" value="two-rows-split" onChange={() => undefined} aria-label="Lab scenario">
            <option value="two-rows-split">2 × 5 — middle split</option>
          </select>
        </section>
        <section>
          <h2>Initial state</h2>
          <div className="stat-row"><span>Cells</span><span>10</span></div>
          <div className="stat-row"><span>Layout</span><span>2 rows × 5</span></div>
          <div className="stat-row"><span>Spacing</span><span>Split distance</span></div>
        </section>
        <section>
          <h2>Events</h2>
          <div className={`lab-event${step >= 50 ? " is-complete" : ""}`}>
            <span className="lab-event-step">50</span>
            <span>
              Split top-middle cell
              <small>Normal simulation division path</small>
            </span>
          </div>
        </section>
        <section>
          <h2>Rendering</h2>
          <label className="slider-row">
            <span>Zoom</span>
            <Slider min={1} max={8} step={0.05} value={zoom} onChange={setZoom} />
            <span className="slider-value">{zoom.toFixed(2)}×</span>
          </label>
          <label className="slider-row">
            <span>Particle size</span>
            <Slider min={1} max={16} step={1} value={particleRadiusPx} onChange={setParticleRadiusPx} />
            <span className="slider-value">{particleRadiusPx}px</span>
          </label>
          <label className="slider-row">
            <span>Particles</span>
            <select className="select" value={particleRenderMode} onChange={(event) => setParticleRenderMode(event.target.value as ParticleRenderMode)}>
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
          {particleRenderMode === "dots-white" && (
            <label className="slider-row"><span>Dots alpha</span><Slider min={0} max={1} step={0.01} value={whiteDotsAlpha} onChange={setWhiteDotsAlpha} /><span className="slider-value">{whiteDotsAlpha.toFixed(2)}</span></label>
          )}
          {particleRenderMode === "dots-neural-color" && (
            <label className="slider-row"><span>Neural RGB alpha</span><Slider min={0} max={1} step={0.01} value={neuralColorAlpha} onChange={setNeuralColorAlpha} /><span className="slider-value">{neuralColorAlpha.toFixed(2)}</span></label>
          )}
          {(particleRenderMode === "dots-internal-state" || particleRenderMode === "dots-chemical-levels") && (
            <>
              <label className="slider-row"><span>{particleRenderMode === "dots-internal-state" ? "Chemical memory alpha" : "Chemical levels alpha"}</span><Slider min={0} max={1} step={0.01} value={internalStateAlpha} onChange={setInternalStateAlpha} /><span className="slider-value">{internalStateAlpha.toFixed(2)}</span></label>
              <div className="channel-window-control">
                <div className="channel-window-label"><span>Channels</span><span>{internalStateChannelStart}–{internalStateChannelStart + 2}</span></div>
                <ChannelWindowSlider channels={particleRenderMode === "dots-internal-state" ? 8 : (config?.channels ?? 8)} value={internalStateChannelStart} onChange={setInternalStateChannelStart} />
              </div>
            </>
          )}
          {particleRenderMode === "dots-boundary-value" && (
            <label className="slider-row"><span>Boundary g0</span><Slider min={0.001} max={0.1} step={0.001} value={boundaryGradientScale} onChange={setBoundaryGradientScale} /><span className="slider-value">{boundaryGradientScale.toFixed(3)}</span></label>
          )}
          {particleRenderMode === "dots-activation-translucent" && (
            <label className="slider-row"><span>Activation alpha</span><Slider min={0} max={1} step={0.01} value={activationAlpha} onChange={setActivationAlpha} /><span className="slider-value">{activationAlpha.toFixed(2)}</span></label>
          )}
          {particleRenderMode === "directional-arrows" && (
            <label className="slider-row"><span>Triangle size</span><Slider min={8} max={80} step={1} value={growthAxisLengthPx} onChange={setGrowthAxisLengthPx} /><span className="slider-value">{growthAxisLengthPx}px</span></label>
          )}
          <label className="slider-row">
            <span>Background</span>
            <select className="select" value={fieldMode} onChange={(event) => setFieldMode(event.target.value as FieldMode)}>
              <option value="none">None</option><option value="density">Density</option><option value="speed">Speed</option><option value="deformation">Deformation</option><option value="pressure">Pressure</option><option value="shear">Shear</option><option value="repulsion">Repulsion field</option><option value="morphology">Policy morphology (gradient + density)</option><option value="substrate">Substrate</option><option value="growth">Growth (cividis)</option><option value="gradient">Boundary gradient</option>
            </select>
          </label>
          {fieldMode === "morphology" && <>
            <label className="checkbox-row"><input type="checkbox" checked={morphologyGradientVisible} onChange={(event) => setMorphologyGradientVisible(event.target.checked)} />Show morphology gradient (R/G)</label>
            <label className="checkbox-row"><input type="checkbox" checked={morphologyDensityVisible} onChange={(event) => setMorphologyDensityVisible(event.target.checked)} />Show morphology density (B)</label>
          </>}
          {fieldMode === "substrate" && config && (
            <div className="channel-window-control"><div className="channel-window-label"><span>RGB channels</span><span>{substrateChannelStart}–{Math.min(config.channels - 1, substrateChannelStart + 2)}</span></div><ChannelWindowSlider channels={config.channels} value={substrateChannelStart} onChange={setSubstrateChannelStart} /></div>
          )}
          <label className="slider-row"><span>Accent</span><Slider min={-2} max={2} step={0.01} value={accent} onChange={setAccent} /><span className="slider-value">{accent.toFixed(2)}</span></label>
          {fieldMode === "gradient" && <>
            <label className="slider-row"><span>Blur</span><Slider min={0} max={2} step={0.01} value={blur} onChange={setBlur} /><span className="slider-value">{blur.toFixed(2)}</span></label>
            <label className="slider-row"><span>Gradient exponent</span><Slider min={0.25} max={4} step={0.05} value={gradientExponent} onChange={setGradientExponent} /><span className="slider-value">{gradientExponent.toFixed(2)}</span></label>
          </>}
        </section>
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
            scenario={TWO_BY_FIVE_SCENARIO}
            targetPoints={null}
            targetVisible={false}
            physics={physics}
            zoom={zoom}
            particleRadiusPx={particleRadiusPx}
            particleRenderMode={particleRenderMode}
            whiteDotsAlpha={whiteDotsAlpha}
            neuralColorAlpha={neuralColorAlpha}
            internalStateAlpha={internalStateAlpha}
            activationAlpha={activationAlpha}
            internalStateChannelStart={internalStateChannelStart}
            boundaryGradientScale={boundaryGradientScale}
            growthAxisLengthPx={growthAxisLengthPx}
            fieldMode={fieldMode}
            substrateChannelStart={substrateChannelStart}
            morphologyGradientVisible={morphologyGradientVisible}
            morphologyDensityVisible={morphologyDensityVisible}
            accent={accent}
            blur={blur}
            gradientExponent={gradientExponent}
            particleCap={11}
            initialParticleCount={10}
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
            <span>{config ? `${cellCount} / 11 cells` : "— cells"}</span>
          </div>
        </div>

        <div className="toolbar">
          <div className="tool-buttons-wrap">
            {tool !== "none" && (
              <div className="tool-settings-panel">
                <h3>{tool === "add" ? "Add" : tool === "move" ? "Move" : "Deform"}</h3>
                {tool !== "deform" ? (
                  <p className="hint">{tool === "add" ? "Click the sim to add a particle." : "Drag a particle to move it."}</p>
                ) : (
                  <>
                    <label className="checkbox-row">
                      <input type="checkbox" checked={deformSettings.direction === "outward"} onChange={(event) => setDeformSettings((value) => ({ ...value, direction: event.target.checked ? "outward" : "inward" }))} />
                      {deformSettings.direction === "outward" ? "Push outward (explode)" : "Pull inward (implode)"}
                    </label>
                    <label className="slider-row"><span>Strength</span><input type="range" min="0" max="2" step="0.01" value={deformSettings.strength} onChange={(event) => setDeformSettings((value) => ({ ...value, strength: Number(event.target.value) }))} /><span className="slider-value">{deformSettings.strength.toFixed(2)}</span></label>
                    <label className="slider-row"><span>Radius</span><input type="range" min="0.01" max="0.5" step="0.01" value={deformSettings.radius} onChange={(event) => setDeformSettings((value) => ({ ...value, radius: Number(event.target.value) }))} /><span className="slider-value">{deformSettings.radius.toFixed(2)}</span></label>
                    <label className="checkbox-row"><input type="checkbox" checked={deformSettings.mode === "deformation"} onChange={(event) => setDeformSettings((value) => ({ ...value, mode: event.target.checked ? "deformation" : "velocity" }))} />Direct deformation (F) edit</label>
                  </>
                )}
              </div>
            )}
            <div className="tool-buttons">
              <button className={`icon-button${tool === "add" ? " is-active" : ""}`} onClick={() => setTool((value) => value === "add" ? "none" : "add")} title="Add particle — click the sim to place one" aria-label="Add particle" aria-pressed={tool === "add"}>＋</button>
              <button className={`icon-button${tool === "move" ? " is-active" : ""}`} onClick={() => setTool((value) => value === "move" ? "none" : "move")} title="Move particles — drag one in the sim" aria-label="Move particles" aria-pressed={tool === "move"}>✥</button>
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

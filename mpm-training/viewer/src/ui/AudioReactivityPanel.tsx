import { useEffect, useMemo, useState } from "react"
import {
  AUDIO_TARGETS,
  applyAudioMappings,
  type AudioMapping,
  type AudioTarget,
} from "../audio/audioReactivity"
import { useAudioInput } from "../audio/useAudioInput"
import type { PhysicsSettings } from "../gpu/types"
import { AudioSignalVisualizer } from "./AudioSignalVisualizer"
import { Slider } from "./Slider"

interface AudioReactivityPanelProps {
  basePhysics: PhysicsSettings | null
  onOutputChange: (physics: PhysicsSettings | null) => void
}

let nextMappingId = 1

export function AudioReactivityPanel({
  basePhysics,
  onOutputChange,
}: AudioReactivityPanelProps) {
  const [open, setOpen] = useState(true)
  const [enabled, setEnabled] = useState(false)
  const [deviceId, setDeviceId] = useState("")
  const [gain, setGain] = useState(8)
  const [threshold, setThreshold] = useState(0.01)
  const [smoothing, setSmoothing] = useState(0.75)
  const [mappings, setMappings] = useState<AudioMapping[]>([])
  const { devices, energy, status, error, analysis } = useAudioInput({
    enabled,
    deviceId,
    gain,
    threshold,
    smoothing,
  })

  const active = enabled && status === "active"
  const output = useMemo(
    () => applyAudioMappings(basePhysics, mappings, energy, active),
    [active, basePhysics, energy, mappings],
  )
  useEffect(() => {
    onOutputChange(output)
  }, [onOutputChange, output])

  const addMapping = () => {
    const spec = AUDIO_TARGETS.find(({ key }) => !mappings.some((mapping) => mapping.target === key))
      ?? AUDIO_TARGETS[0]
    setMappings((current) => [
      ...current,
      { id: nextMappingId++, target: spec.key, min: spec.min, max: spec.max },
    ])
  }

  const updateMapping = (id: number, patch: Partial<AudioMapping>) => {
    setMappings((current) => current.map((mapping) =>
      mapping.id === id ? { ...mapping, ...patch } : mapping
    ))
  }

  return (
    <section>
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen((value) => !value)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>▸</span>
          <h2>Audio reactivity</h2>
        </button>
        <label className="audio-enable" title="Enable audio input">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => setEnabled(event.target.checked)}
          />
          On
        </label>
      </div>
      {open && (
        <div className="physics-panel-body audio-panel-body">
          <label className="audio-field">
            <span>Input</span>
            <select
              className="select"
              value={deviceId}
              onChange={(event) => setDeviceId(event.target.value)}
            >
              <option value="">System default</option>
              {devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>{device.label}</option>
              ))}
            </select>
          </label>

          <div className="audio-meter-row" aria-label={`Audio energy ${Math.round(energy * 100)} percent`}>
            <span>Energy</span>
            <div className="audio-meter"><span style={{ width: `${energy * 100}%` }} /></div>
            <span>{Math.round(energy * 100)}%</span>
          </div>
          <div className={`audio-status is-${status}`}>
            {error ?? (status === "requesting" ? "Waiting for microphone permission…" : status)}
          </div>

          <AudioSignalVisualizer
            analysis={analysis}
            active={active}
            gain={gain}
            threshold={threshold}
          />

          <label className="slider-row">
            <span>Gain</span>
            <Slider min={1} max={40} step={0.5} value={gain} onChange={setGain} />
            <span className="slider-value">{gain.toFixed(1)}×</span>
          </label>
          <label className="slider-row">
            <span>Noise gate</span>
            <Slider min={0} max={0.1} step={0.001} value={threshold} onChange={setThreshold} />
            <span className="slider-value">{threshold.toFixed(3)}</span>
          </label>
          <label className="slider-row">
            <span>Smoothing</span>
            <Slider min={0} max={0.98} step={0.01} value={smoothing} onChange={setSmoothing} />
            <span className="slider-value">{smoothing.toFixed(2)}</span>
          </label>

          <div className="audio-mappings-header">
            <span>Mappings</span>
            <button className="icon-button" onClick={addMapping} title="Add mapping" aria-label="Add audio mapping">+</button>
          </div>
          {mappings.length === 0 && <p className="hint">Add a mapping to drive a simulation parameter.</p>}
          {mappings.map((mapping) => (
            <div className="audio-mapping" key={mapping.id}>
              <select
                className="select"
                value={mapping.target}
                aria-label="Mapped simulation parameter"
                onChange={(event) => {
                  const target = event.target.value as AudioTarget
                  const spec = AUDIO_TARGETS.find((candidate) => candidate.key === target)!
                  updateMapping(mapping.id, { target, min: spec.min, max: spec.max })
                }}
              >
                {AUDIO_TARGETS.map((target) => (
                  <option key={target.key} value={target.key}>{target.label}</option>
                ))}
              </select>
              <div className="audio-range">
                <label>Quiet <input className="number-input" type="number" value={mapping.min} onChange={(event) => updateMapping(mapping.id, { min: Number(event.target.value) })} /></label>
                <span>→</span>
                <label>Loud <input className="number-input" type="number" value={mapping.max} onChange={(event) => updateMapping(mapping.id, { max: Number(event.target.value) })} /></label>
              </div>
              <button className="audio-remove" onClick={() => setMappings((current) => current.filter(({ id }) => id !== mapping.id))} aria-label="Remove audio mapping">Remove</button>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

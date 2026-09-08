import { appendAudioMapping } from "../audio/mappingState"
import type { AudioSettings } from "../performance/workspacePresets"
import { ToggleButton } from "./ToggleButton"
import { useEffect, useMemo, useState, type ReactNode, type Dispatch, type SetStateAction } from "react"
import {
  applyAudioMappings,
  audioTargetSpecsFor,
  type AudioMapping,
  type AudioTarget,
  type AudioTargetGroup,
} from "../audio/audioReactivity"
import { useAudioInput } from "../audio/useAudioInput"
import type { PerformanceSnapshot } from "../performance/types"
import { AudioSignalVisualizer } from "./AudioSignalVisualizer"
import { ParameterPicker } from "./ParameterPicker"
import { Slider } from "./Slider"

interface AudioReactivityPanelProps {
  settings: AudioSettings
  onSettingsChange: Dispatch<SetStateAction<AudioSettings>>
  baseSnapshot: PerformanceSnapshot
  renderPanels: (panels: { audio: ReactNode; mappings: ReactNode }) => ReactNode
  onOutputChange: (snapshot: PerformanceSnapshot | null) => void
}

const TARGET_GROUPS: AudioTargetGroup[] = [
  "Actuators",
  "Simulation",
  "Rendering",
  "Auto zoom",
  "Post-processing",
  "Displacement",
]

interface DecimalInputProps {
  label: string
  value: number
  onChange: (value: number) => void
}

/** Keeps intermediate decimal text ("0.", "-", etc.) intact while the
 * controlled mapping value continues to update whenever the draft is a valid
 * number. Without a local draft, React immediately rewrites "0." as "0" and
 * makes typing fractional ranges impossible. */
function DecimalInput({ label, value, onChange }: DecimalInputProps) {
  const [draft, setDraft] = useState(String(value))
  const [editing, setEditing] = useState(false)

  useEffect(() => {
    if (!editing) setDraft(String(value))
  }, [editing, value])

  return (
    <input
      className="number-input"
      type="text"
      inputMode="decimal"
      autoComplete="off"
      spellCheck={false}
      aria-label={label}
      value={draft}
      onFocus={() => setEditing(true)}
      onChange={(event) => {
        const nextDraft = event.target.value
        setDraft(nextDraft)
        const parsed = Number(nextDraft)
        if (nextDraft.trim() !== "" && Number.isFinite(parsed)) onChange(parsed)
      }}
      onBlur={() => {
        setEditing(false)
        const parsed = Number(draft)
        if (draft.trim() !== "" && Number.isFinite(parsed)) {
          onChange(parsed)
          setDraft(String(parsed))
        } else {
          setDraft(String(value))
        }
      }}
    />
  )
}

export function AudioReactivityPanel({
  settings, onSettingsChange,
  baseSnapshot,
  renderPanels,
  onOutputChange,
}: AudioReactivityPanelProps) {
  const [open, setOpen] = useState(true)
  const [mappingsOpen, setMappingsOpen] = useState(true)
  const { enabled, deviceId, gain, threshold, smoothing, mappings } = settings
  const setEnabled = (enabled: boolean) => onSettingsChange(current => ({ ...current, enabled }))
  const setDeviceId = (deviceId: string) => onSettingsChange(current => ({ ...current, deviceId }))
  const setGain = (gain: number) => onSettingsChange(current => ({ ...current, gain }))
  const setThreshold = (threshold: number) => onSettingsChange(current => ({ ...current, threshold }))
  const setSmoothing = (smoothing: number) => onSettingsChange(current => ({ ...current, smoothing }))
  const setMappings = (update: (current: AudioMapping[]) => AudioMapping[]) => onSettingsChange(current => ({ ...current, mappings: update(current.mappings) }))
  const { devices, energy, status, error, analysis, energyHistory } = useAudioInput({
    enabled,
    deviceId,
    gain,
    threshold,
    smoothing,
  })

  const active = enabled && status === "active"
  const targets = useMemo(() => audioTargetSpecsFor(baseSnapshot), [baseSnapshot])
  const output = useMemo(
    () => active ? { ...applyAudioMappings(baseSnapshot, mappings, energy, true), audioEnergy: energy } : null,
    [active, baseSnapshot, energy, mappings],
  )
  useEffect(() => {
    onOutputChange(output)
  }, [onOutputChange, output])

  const addMapping = () => {
    setMappings(current => appendAudioMapping(current, targets))
  }

  const updateMapping = (id: number, patch: Partial<AudioMapping>) => {
    setMappings((current) => current.map((mapping) =>
      mapping.id === id ? { ...mapping, ...patch } : mapping
    ))
  }

  const audio = (
    <section>
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen((value) => !value)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>▸</span>
          <h2>Audio reactivity</h2>
        </button>
        <ToggleButton label="Audio input" hideLabel checked={enabled} onChange={(next) => {
          setEnabled(next)
          if (next) setOpen(true)
        }} title="Enable audio input" />
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

          {status !== "active" && (
            <div className={`audio-status is-${status}`}>
              {error ?? (status === "requesting" ? "Waiting for microphone permission…" : status)}
            </div>
          )}

          <AudioSignalVisualizer
            analysis={analysis}
            energyHistory={energyHistory}
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


        </div>
      )}
    </section>
  )
  const mappingsPanel = (
    <section>
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setMappingsOpen((value) => !value)}>
          <span className={"physics-panel-chevron" + (mappingsOpen ? " is-open" : "")}>▸</span>
          <h2>Mappings</h2>
        </button>
        <button className="icon-button" onClick={() => { setMappingsOpen(true); addMapping() }} title="Add mapping" aria-label="Add audio mapping">+</button>
      </div>
      {mappingsOpen && <div className="physics-panel-body audio-panel-body">
          {mappings.length === 0 && <p className="hint">Add a mapping to drive a simulation parameter.</p>}
          {mappings.map((mapping) => (
            <div className={"audio-mapping" + (mapping.enabled === false ? " is-disabled" : "")} key={mapping.id}>
              <div className="audio-mapping-top-row">
                <ToggleButton label={`Audio mapping ${mapping.id}`} hideLabel checked={mapping.enabled !== false}
                  onChange={(enabled) => updateMapping(mapping.id, { enabled })} title="Enable or disable this mapping" />
                <select
                  className="select"
                  value={mapping.target}
                  aria-label="Mapped simulation parameter"
                  onChange={(event) => {
                    const target = event.target.value as AudioTarget
                    const spec = targets.find((candidate) => candidate.key === target)!
                    updateMapping(mapping.id, { target, min: spec.min, max: spec.max })
                  }}
                >
                  {TARGET_GROUPS.map((group) => (
                    <optgroup key={group} label={group}>
                      {targets.filter((target) => target.group === group).map((target) => (
                        <option key={target.key} value={target.key}>{target.label}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <ParameterPicker targets={targets} onPick={(target) => {
                  const spec = targets.find(candidate => candidate.key === target)!
                  updateMapping(mapping.id, { target, min: spec.min, max: spec.max })
                }} />
              </div>
              <div className="audio-range">
                <label>
                  Quiet
                  <DecimalInput
                    label="Quiet mapping value"
                    value={mapping.min}
                    onChange={(min) => updateMapping(mapping.id, { min })}
                  />
                </label>
                <span>→</span>
                <label>
                  Loud
                  <DecimalInput
                    label="Loud mapping value"
                    value={mapping.max}
                    onChange={(max) => updateMapping(mapping.id, { max })}
                  />
                </label>
              </div>
              <button className="audio-remove" onClick={() => setMappings((current) => current.filter(({ id }) => id !== mapping.id))} aria-label="Remove audio mapping">Remove</button>
            </div>
          ))}
      </div>}
    </section>
  )
  return renderPanels({ audio, mappings: mappingsPanel })

}

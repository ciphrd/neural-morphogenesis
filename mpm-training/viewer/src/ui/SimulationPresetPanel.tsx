import { useEffect, useState } from "react"
import type {
  CellMemory,
  ChemicalCommunicationArchitecture,
  PhysicsSettings,
} from "../gpu/types"

const STORAGE_KEY = "mpm-training-simulation-presets-v1"

export interface SimulationPresetValue {
  physics: PhysicsSettings
  particleCap: number
  initialParticleCount: number
  noiseDisplacementStrength: number
  particleDensityMultiplier: number
  chirality: boolean
  chemicalArchitecture: ChemicalCommunicationArchitecture
  policyExploration: {
    cellMemory: CellMemory
    hiddenWidth: number
    variant: number
  } | null
}

interface SimulationPreset {
  id: string
  name: string
  value: SimulationPresetValue
}

interface SimulationPresetPanelProps {
  value: SimulationPresetValue | null
  onLoad: (value: SimulationPresetValue) => void
}

function loadPresets(): SimulationPreset[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]")
    return Array.isArray(parsed) ? parsed as SimulationPreset[] : []
  } catch {
    return []
  }
}

export function SimulationPresetPanel({ value, onLoad }: SimulationPresetPanelProps) {
  const [name, setName] = useState("")
  const [presets, setPresets] = useState<SimulationPreset[]>(loadPresets)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets))
  }, [presets])

  const save = () => {
    if (!value) return
    const trimmed = name.trim()
    const presetName = trimmed || `Preset ${presets.length + 1}`
    const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
    setPresets((current) => [...current, { id, name: presetName, value }])
    setName("")
  }

  return (
    <section className="simulation-presets">
      <h2>Saved settings</h2>
      <div className="simulation-preset-save">
        <input
          className="number-input"
          value={name}
          placeholder="Preset name"
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") save() }}
        />
        <button className="select" disabled={!value} onClick={save}>Save</button>
      </div>
      {presets.length === 0 && <p className="hint">No saved simulation settings.</p>}
      <div className="simulation-preset-list">
        {presets.map((preset) => (
          <div className="simulation-preset" key={preset.id}>
            <button onClick={() => onLoad(preset.value)}>{preset.name}</button>
            <button
              className="audio-remove"
              aria-label={`Delete ${preset.name}`}
              onClick={() => setPresets((current) => current.filter(({ id }) => id !== preset.id))}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </section>
  )
}

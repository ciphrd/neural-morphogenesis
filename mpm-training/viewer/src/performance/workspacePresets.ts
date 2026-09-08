import type { SimulationConfig } from "../gpu/types"
import type { AudioMapping } from "../audio/audioReactivity"
import type { PerformanceSnapshot } from "./types"
import { SEQUENCE_ACTIONS, type PerformanceSequence } from "./sequences"
export interface AudioSettings {
  enabled: boolean; deviceId: string; gain: number; threshold: number; smoothing: number; mappings: AudioMapping[]
}
export const DEFAULT_AUDIO_SETTINGS: AudioSettings = { enabled: false, deviceId: "", gain: 8, threshold: .01, smoothing: .75, mappings: [] }
export interface PerformanceControls { previewEnabled?: boolean; cutEnabled?: boolean }
export interface WorkspacePresetValue {
  config: SimulationConfig
  snapshot: PerformanceSnapshot
  density: number
  audio: AudioSettings
  sequences: PerformanceSequence[]
  controls: PerformanceControls
}
export interface WorkspacePreset { version: 1; id: string; name: string; savedAt: string; value: WorkspacePresetValue }
export const PRESETS_KEY = "mpm-training-workspace-presets-v1"
export const LAST_PRESET_KEY = "mpm-training-workspace-last-preset-v1"
export function readWorkspacePresets(): WorkspacePreset[] {
  try {
    const data = JSON.parse(localStorage.getItem(PRESETS_KEY) ?? "[]")
    return Array.isArray(data) ? data.filter(p => p?.version === 1 && typeof p.id === "string" && typeof p.name === "string"
      && p.value?.config?.weights && p.value.snapshot?.render && p.value.snapshot?.physics
      && Number.isFinite(p.value.density) && p.value.density > 0
      && Array.isArray(p.value.audio?.mappings) && Array.isArray(p.value.sequences) && p.value.controls)
      .map(p => ({ ...p, value: { ...p.value, sequences: p.value.sequences.filter((sequence: PerformanceSequence) => SEQUENCE_ACTIONS.some(action => action.value === sequence.action)) } })) : []
  } catch { return [] }
}
export function readLastWorkspacePreset(): WorkspacePreset | null {
  try { const id = localStorage.getItem(LAST_PRESET_KEY); return readWorkspacePresets().find(p => p.id === id) ?? null } catch { return null }
}

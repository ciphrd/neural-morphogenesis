import type { PhysicsSettings } from "../gpu/types"

export type AudioTarget = Exclude<keyof PhysicsSettings, "mpmEnabled">

export interface AudioTargetSpec {
  key: AudioTarget
  label: string
  min: number
  max: number
}

export interface AudioMapping {
  id: number
  target: AudioTarget
  min: number
  max: number
}

export const AUDIO_TARGETS: AudioTargetSpec[] = [
  { key: "gravity", label: "Gravity", min: 0, max: 20 },
  { key: "damping", label: "Damping", min: 0, max: 1 },
  { key: "maxAccel", label: "Max accel", min: 0, max: 1 },
  { key: "maxStrafe", label: "Physical strafe", min: 0, max: 1 },
  { key: "maxAngularVelocity", label: "Angular velocity", min: 0, max: 20 },
  { key: "communicationSpeed", label: "Communication speed", min: 0, max: 4 },
  { key: "growthDuration", label: "Growth duration", min: 160, max: 8 },
  { key: "growthAnisotropy", label: "Growth anisotropy", min: 0, max: 1 },
  { key: "divisionDirectionality", label: "Division directionality", min: 0, max: 1 },
  { key: "repulsionStrength", label: "Repulsion strength", min: 0, max: 1000 },
]

export function applyAudioMappings(
  base: PhysicsSettings | null,
  mappings: readonly AudioMapping[],
  energy: number,
  active: boolean,
): PhysicsSettings | null {
  if (!base || !active || mappings.length === 0) return base
  const next = { ...base }
  const normalizedEnergy = Math.max(0, Math.min(1, energy))
  for (const mapping of mappings) {
    next[mapping.target] = mapping.min + (mapping.max - mapping.min) * normalizedEnergy
  }
  return next
}

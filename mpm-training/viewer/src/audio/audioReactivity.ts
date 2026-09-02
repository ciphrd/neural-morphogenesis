import type { PhysicsSettings } from "../gpu/types"
import type {
  PerformanceAutoZoomSettings,
  PerformanceBloomSettings,
  PerformanceRenderSettings,
  PerformanceSnapshot,
} from "../performance/types"

type NumericKey<T> = {
  [Key in keyof T]-?: T[Key] extends number ? Key : never
}[keyof T] & string

export type AudioTarget =
  | `physics.${NumericKey<PhysicsSettings>}`
  | `render.${NumericKey<PerformanceRenderSettings>}`
  | `render.autoZoom.${NumericKey<PerformanceAutoZoomSettings>}`
  | `render.bloom.${NumericKey<PerformanceBloomSettings>}`
  | "noiseDisplacementStrength"

export type AudioTargetGroup = "Simulation" | "Rendering" | "Auto zoom" | "Bloom" | "Displacement"

export interface AudioTargetSpec {
  key: AudioTarget
  label: string
  group: AudioTargetGroup
  min: number
  max: number
}

export interface AudioMapping {
  id: number
  /** Absent in legacy mappings means enabled. */
  enabled?: boolean
  target: AudioTarget
  min: number
  max: number
}

const FIXED_RANGES: Partial<Record<AudioTarget, readonly [number, number]>> = {
  "physics.damping": [0, 1],
  "physics.materialNu": [0, 0.49],
  "physics.materialElasticity": [0, 1],
  "physics.materialFluidity": [0, 1],
  "physics.decay": [0, 1],
  "physics.friction": [0, 1],
  "physics.steeringStrength": [0, 1],
  "physics.angularDamping": [0, 1],
  "physics.growthAnisotropy": [0, 1],
  "physics.growthCompressionFeedback": [0, 1],
  "physics.growthDrive": [0, 1],
  "physics.divisionDirectionality": [0, 1],
  "physics.deathRate": [0, 1],
  "physics.neuralUpdatesPerMacro": [1, 16],
  "physics.communicationSpeed": [0, 4],
  "physics.internalStateSpeed": [0, 4],
  "physics.growthDuration": [0, 160],
  "physics.growthCompressionStart": [0, 0.2],
  "physics.growthCompressionStop": [0.002, 0.3],
  "physics.boundaryTangentMinGradient": [0, 0.05],
  "physics.repulsionStrength": [0, 1000],
  "physics.repulsionMaxDelta": [1, 250],
  "render.zoom": [0.25, 8],
  "render.particleRadiusPx": [0.25, 16],
  "render.substrateChannelStart": [0, 32],
  "render.accent": [-2, 2],
  "render.blur": [0, 2],
  "render.gradientExponent": [0.25, 4],
  "render.whiteDotsAlpha": [0, 1],
  "render.activationAlpha": [0, 1],
  "render.neuralColorAlpha": [0, 1],
  "render.internalStateAlpha": [0, 1],
  "render.internalStateChannelStart": [0, 32],
  "render.boundaryGradientScale": [0, 0.1],
  "render.chemicalMemoryOpponentSubtraction": [0, 1],
  "render.growthAxisLengthPx": [0, 80],
  "render.autoZoom.sampleEveryFrames": [1, 120],
  "render.autoZoom.maxSamples": [16, 1024],
  "render.autoZoom.fitFraction": [0.1, 1],
  "render.autoZoom.padding": [1, 3],
  "render.autoZoom.smoothing": [0.001, 1],
  "render.bloom.intensity": [0, 5],
  "render.bloom.threshold": [0, 2],
  "render.bloom.radiusPx": [0.25, 16],
  "render.bloom.scatter": [0, 1],
  "render.bloom.levels": [2, 10],
  noiseDisplacementStrength: [0, 2],
}

const INTEGER_TARGETS = new Set<AudioTarget>([
  "physics.neuralUpdatesPerMacro",
  "render.substrateChannelStart",
  "render.internalStateChannelStart",
  "render.autoZoom.sampleEveryFrames",
  "render.autoZoom.maxSamples",
  "render.bloom.levels",
])

const TARGET_LABELS: Partial<Record<AudioTarget, string>> = {
  "physics.gravity": "Center gravity",
}

function humanize(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/Px\b/g, "px")
    .replace(/^./, (letter) => letter.toUpperCase())
}

function inferredRange(target: AudioTarget, current: number): readonly [number, number] {
  const fixed = FIXED_RANGES[target]
  if (fixed) return fixed
  if (current < 0) {
    const magnitude = Math.max(Math.abs(current) * 3, 1)
    return [-magnitude, magnitude]
  }
  return [0, Math.max(current * 3, 0.05)]
}

function numericSpecs<T extends object>(
  value: T,
  prefix: string,
  group: AudioTargetGroup,
): AudioTargetSpec[] {
  return Object.entries(value).flatMap(([key, current]) => {
    if (typeof current !== "number") return []
    const target = `${prefix}.${key}` as AudioTarget
    const [min, max] = inferredRange(target, current)
    return [{ key: target, label: TARGET_LABELS[target] ?? humanize(key), group, min, max }]
  })
}

export function audioTargetSpecsFor(base: PerformanceSnapshot): AudioTargetSpec[] {
  const targets = base.physics ? numericSpecs(base.physics, "physics", "Simulation") : []
  targets.push(...numericSpecs(base.render, "render", "Rendering"))
  targets.push(...numericSpecs(base.render.autoZoom, "render.autoZoom", "Auto zoom"))
  targets.push(...numericSpecs(base.render.bloom, "render.bloom", "Bloom"))
  const [min, max] = inferredRange("noiseDisplacementStrength", base.noiseDisplacementStrength)
  targets.push({
    key: "noiseDisplacementStrength",
    label: "Simplex noise strength",
    group: "Displacement",
    min,
    max,
  })
  return targets
}

export function applyAudioMappings(
  base: PerformanceSnapshot,
  mappings: readonly AudioMapping[],
  energy: number,
  active: boolean,
): PerformanceSnapshot {
  if (!active || !mappings.some((mapping) => mapping.enabled !== false)) return base
  const next: PerformanceSnapshot = {
    ...base,
    physics: base.physics ? { ...base.physics } : null,
    render: {
      ...base.render,
      autoZoom: { ...base.render.autoZoom },
      bloom: { ...base.render.bloom },
    },
  }
  const normalizedEnergy = Math.max(0, Math.min(1, energy))
  for (const mapping of mappings) {
    if (mapping.enabled === false) continue
    let value = mapping.min + (mapping.max - mapping.min) * normalizedEnergy
    if (INTEGER_TARGETS.has(mapping.target)) value = Math.round(value)
    const parts = mapping.target.split(".")
    if (parts[0] === "physics" && next.physics) {
      Object.assign(next.physics, { [parts[1]]: value })
    } else if (parts[0] === "render" && parts.length === 2) {
      Object.assign(next.render, { [parts[1]]: value })
    } else if (parts[0] === "render" && parts[1] === "autoZoom") {
      Object.assign(next.render.autoZoom, { [parts[2]]: value })
    } else if (parts[0] === "render" && parts[1] === "bloom") {
      Object.assign(next.render.bloom, { [parts[2]]: value })
    } else if (mapping.target === "noiseDisplacementStrength") {
      next.noiseDisplacementStrength = value
    }
  }
  return next
}

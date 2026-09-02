import type { FieldMode, ParticleRenderMode } from "../gpu/render"
import type { PhysicsSettings, SimulationConfig } from "../gpu/types"

export interface PerformanceAutoZoomSettings {
  enabled: boolean
  sampleEveryFrames: number
  maxSamples: number
  fitFraction: number
  padding: number
  smoothing: number
}

export interface PerformanceBloomSettings {
  enabled: boolean
  intensity: number
  threshold: number
  radiusPx: number
  scatter: number
  levels: number
}

export interface PerformanceRenderSettings {
  zoom: number
  particleRadiusPx: number
  particleRenderMode: ParticleRenderMode
  fieldMode: FieldMode
  substrateChannelStart: number
  accent: number
  blur: number
  gradientExponent: number
  whiteDotsAlpha: number
  activationAlpha: number
  neuralColorAlpha: number
  internalStateAlpha: number
  internalStateChannelStart: number
  boundaryGradientScale: number
  chemicalMemoryOpponentSubtraction: number
  growthAxisLengthPx: number
  morphologyGradientVisible: boolean
  morphologyDensityVisible: boolean
  autoZoom: PerformanceAutoZoomSettings
  bloom: PerformanceBloomSettings
}

export interface PerformanceSnapshot {
  physics: PhysicsSettings | null
  render: PerformanceRenderSettings
  particleCap: number
  initialParticleCount: number
  noiseDisplacementStrength: number
  paused: boolean
  loopAtTrainedSteps: boolean
  blackout: boolean
}

export interface PerformanceScene {
  id: string
  name: string
  snapshot: PerformanceSnapshot
}

export interface ProjectionTelemetry {
  step: number
  particleCount: number
  fps: number
  updatedAt: number
}

export type ControllerToProjectionMessage =
  | { type: "config"; config: SimulationConfig | null }
  | { type: "snapshot"; snapshot: PerformanceSnapshot }
  | { type: "auto-prune"; fraction: number | null; delayMs: number }
  | { type: "auto-randomize"; intervalMs: number | null }
  | { type: "auto-reset"; intervalMs: number | null }
  | {
      type: "command"
      command: "restart" | "randomize" | "randomize-and-restart" | "kill-20-percent" | "kill-80-percent"
    }

export type ProjectionToControllerMessage =
  | { type: "hello" }
  | { type: "telemetry"; telemetry: ProjectionTelemetry }

export const PERFORMANCE_CHANNEL_NAME = "mpm-training-performance-v1"

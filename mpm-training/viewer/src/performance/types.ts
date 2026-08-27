import type { FieldMode, ParticleRenderMode } from "../gpu/render"
import type { PhysicsSettings, SimulationConfig } from "../gpu/types"

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
  growthAxisLengthPx: number
  morphologyGradientVisible: boolean
  morphologyDensityVisible: boolean
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
  | { type: "command"; command: "restart" }

export type ProjectionToControllerMessage =
  | { type: "hello" }
  | { type: "telemetry"; telemetry: ProjectionTelemetry }

export const PERFORMANCE_CHANNEL_NAME = "mpm-training-performance-v1"

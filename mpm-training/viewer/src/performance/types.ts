import type { AttractorSettings, AttractorPosition } from "./actuators"
import type { FieldMode, ParticleShape, ParticleColorMode } from "../gpu/render"
import type { PhysicsSettings, SimulationConfig } from "../gpu/types"

export interface PerformanceAutoZoomSettings {
  enabled: boolean
  sampleEveryFrames: number
  maxSamples: number
  fitFraction: number
  padding: number
  smoothing: number
}

export interface PerformanceBloomSettings extends Partial<import("../gpu/postEffects").PostEffectSettings> {
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
  particleShape: ParticleShape
  particleColorMode: ParticleColorMode
  centerDotSize: number
  particleAlpha: number
  directionalLineVisible: boolean
  domainVisible: boolean
  growthLineVisible: boolean
  substrateZeroIsBlack: boolean
  boundaryGradientZeroIsBlack: boolean
  fieldMode: FieldMode
  substrateChannelStart: number
  accent: number
  blur: number
  gradientExponent: number
  internalStateChannelStart: number
  boundaryGradientScale: number
  chemicalMemoryOpponentSubtraction: number
  morphologyGradientVisible: boolean
  morphologyDensityVisible: boolean
  autoZoom: PerformanceAutoZoomSettings
  bloom: PerformanceBloomSettings
}

export interface PerformanceSnapshot {
  autoPruneCircle?: boolean
  /** Live normalized NN input; absent or disabled audio is silence. */
  audioEnergy?: number
  attractor?: AttractorSettings
  physics: PhysicsSettings | null
  render: PerformanceRenderSettings
  particleCap: number
  initialParticleCount: number
  noiseDisplacementStrength: number
  paused: boolean
  loopAtTrainedSteps: boolean
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
  | {
      type: "command"
      command: "restart" | "randomize" | "randomize-and-restart" | "kill-20-percent" | "kill-80-percent" | "keep-center-circle" | "prune"
    }

export type ProjectionToControllerMessage =
  | { type: "policy-weights"; weights: SimulationConfig["weights"] }
  | { type: "attractor"; position: AttractorPosition }
  | { type: "hello" }
  | { type: "telemetry"; telemetry: ProjectionTelemetry }

export const PERFORMANCE_CHANNEL_NAME = "mpm-training-performance-v1"

import type { FieldMode, ParticleRenderMode } from "./gpu/render"
import type { PhysicsSettings } from "./gpu/types"
import type { DeformSettings, Tool } from "./render/GridCanvas"

interface ViewerDefaults {
  playback: {
    loopAtTrainedSteps: boolean
    paused: boolean
    /** null follows the particle cap stored in the selected training run. */
    particleCap: number | null
    /** null follows the initial count stored in the selected training run. */
    initialParticleCount: number | null
    /** null follows the density stored in the selected training run. */
    particleDensityMultiplier: number | null
    /** Values that replace the selected run's physics settings in playback. */
    physicsOverrides: Partial<PhysicsSettings>
  }
  rendering: {
    fieldMode: FieldMode
    substrateChannelStart: number
    morphologyGradientVisible: boolean
    morphologyDensityVisible: boolean
    accent: number
    blur: number
    gradientExponent: number
    particleRenderMode: ParticleRenderMode
    boundaryGradientScale: number
    zoom: number
    autoZoom: {
      enabled: boolean
      sampleEveryFrames: number
      maxSamples: number
      /** Fraction of the canvas side occupied by the sampled bounds. */
      fitFraction: number
      /** Extra space around the sampled bounds. Values above 1 add padding. */
      padding: number
      /** Fraction of the remaining zoom distance applied per rendered frame. */
      smoothing: number
    }
    bloom: {
      enabled: boolean
      intensity: number
      threshold: number
      radiusPx: number
      scatter: number
      levels: number
    }
    particleRadiusPx: number
    targetVisible: boolean
    whiteDotsAlpha: number
    activationAlpha: number
    neuralColorAlpha: number
    internalStateAlpha: number
    internalStateChannelStart: number
    /** Amount of wrapped memory channels +3/+4/+5 subtracted from particle RGB. */
    chemicalMemoryOpponentSubtraction: number
    growthAxisLengthPx: number
  }
  tools: {
    selected: Tool
    deform: DeformSettings
  }
  lab: {
    scenario: "boundary-tangent" | "vertical" | "repeated-top-row"
  }
}

/**
 * User-facing viewer startup defaults.
 *
 * These values initialize both the Training and Lab views. Controls remain
 * editable at runtime, but changes made in the browser are not persisted.
 */
export const VIEWER_DEFAULTS: ViewerDefaults = {
  playback: {
    loopAtTrainedSteps: false,
    paused: false,
    particleCap: 170_000,
    initialParticleCount: 100,
    particleDensityMultiplier: 2,
    physicsOverrides: {
      communicationSpeed: 3,
      internalStateSpeed: 3,
      neuralUpdatesPerMacro: 1,
      steeringStrength: 0,
    },
  },
  rendering: {
    fieldMode: "none",
    substrateChannelStart: 0,
    morphologyGradientVisible: true,
    morphologyDensityVisible: true,
    accent: 0,
    blur: 0,
    gradientExponent: 1,
    particleRenderMode: "dots-internal-state",
    boundaryGradientScale: 0.01,
    zoom: 1,
    autoZoom: {
      enabled: true,
      sampleEveryFrames: 12,
      maxSamples: 256,
      fitFraction: 0.5,
      padding: 1.2,
      smoothing: 0.01,
    },
    bloom: {
      enabled: true,
      intensity: 1.3,
      threshold: 0.22,
      radiusPx: 2.5,
      scatter: 0.5,
      levels: 4,
    },
    particleRadiusPx: 1,
    targetVisible: true,
    whiteDotsAlpha: 1,
    activationAlpha: 0.2,
    neuralColorAlpha: 1,
    internalStateAlpha: 0.4,
    internalStateChannelStart: 0,
    chemicalMemoryOpponentSubtraction: 0,
    growthAxisLengthPx: 28,
  },
  tools: {
    selected: "none",
    deform: {
      direction: "outward",
      strength: 1,
      radius: 0.08,
      mode: "velocity",
    },
  },
  lab: {
    scenario: "boundary-tangent",
  },
}

/** Applies viewer-only playback defaults without changing serialized runs or
 * backend/training settings. The merged result is the UI reset baseline. */
export function applyViewerPhysicsOverrides(
  physics: PhysicsSettings,
): PhysicsSettings {
  return { ...physics, ...VIEWER_DEFAULTS.playback.physicsOverrides }
}

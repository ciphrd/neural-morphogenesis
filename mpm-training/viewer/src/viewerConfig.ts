import type { FieldMode, ParticleColorMode, ParticleShape } from "./gpu/render"
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
    /** null follows the chemical-substrate resolution stored in the run. */
    substrateResolution: number | null
  }
  rendering: {
    fieldMode: FieldMode
    substrateChannelStart: number
    substrateZeroIsBlack: boolean
    boundaryGradientZeroIsBlack: boolean
    morphologyGradientVisible: boolean
    morphologyDensityVisible: boolean
    accent: number
    blur: number
    gradientExponent: number
    particleShape: ParticleShape
    particleColorMode: ParticleColorMode
    particleAlpha: number
    directionalLineVisible: boolean
    growthLineVisible: boolean
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
    mitosisSignalBoost: number
    internalStateChannelStart: number
    /** Amount of wrapped memory channels +3/+4/+5 subtracted from particle RGB. */
    chemicalMemoryOpponentSubtraction: number
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
    particleCap: null,
    initialParticleCount: null,
    particleDensityMultiplier: null,
    substrateResolution: null,
  },
  rendering: {
    fieldMode: "none",
    substrateChannelStart: 0,
    substrateZeroIsBlack: false,
    boundaryGradientZeroIsBlack: false,
    morphologyGradientVisible: true,
    morphologyDensityVisible: true,
    accent: 0,
    blur: 0,
    gradientExponent: 1,
    particleShape: "dot",
    particleColorMode: "neural-memory",
    particleAlpha: 0.4,
    directionalLineVisible: false,
    growthLineVisible: false,
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
    mitosisSignalBoost: 1,
    internalStateChannelStart: 0,
    chemicalMemoryOpponentSubtraction: 0,
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

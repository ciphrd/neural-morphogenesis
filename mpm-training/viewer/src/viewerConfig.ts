import type { FieldMode, ParticleColorMode, ParticleShape } from "./gpu/render"
import type { DeformSettings, Tool } from "./render/GridCanvas"

interface ViewerDefaults {
  policyExploration: { weightGain: number; mutationStrength: number }
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
    domainVisible: boolean
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
    growthMagnitudeBoost: number
    internalStateChannelStart: number
    /** Amount of wrapped memory channels +3/+4/+5 subtracted from particle RGB. */
    chemicalMemoryOpponentSubtraction: number
  }
  tools: {
    selected: Tool
    deform: DeformSettings
  }
  lab: {
    scenario: "boundary-tangent" | "vertical" | "repeated-top-row" | "radial-inward-circle"
  }
}

/**
 * User-facing viewer startup defaults.
 *
 * These values initialize both the Training and Lab views. Controls remain
 * editable at runtime, but changes made in the browser are not persisted.
 */
import config from "../../core/config.json"

export const VIEWER_DEFAULTS = config.viewer as ViewerDefaults

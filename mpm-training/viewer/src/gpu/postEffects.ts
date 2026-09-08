export const MAX_WAVE_SPEED = 1000
export const POST_EFFECT_DEFAULTS = {
  dofEnabled: false, dofFocus: 0.5, dofWidth: 0.3, dofTop: 12, dofFront: 18,
  rgbEnabled: false, rgbMix: 1, rgbAngle: 0,
  redFrequency: 3, greenFrequency: 5, blueFrequency: 7,
  redExponent: 1, greenExponent: 1.5, blueExponent: 2,
  redPeriod: 7, greenPeriod: 9, bluePeriod: 11,
  redSpeed: 1, greenSpeed: 1, blueSpeed: 1,
  strobeEnabled: false, strobeMix: 1,
  strobeRedFrequency: 1, strobeGreenFrequency: 1, strobeBlueFrequency: 1,
  strobeRedPeriod: 0.125, strobeGreenPeriod: 0.125, strobeBluePeriod: 0.125,
  strobeRedExponent: 16, strobeGreenExponent: 16, strobeBlueExponent: 16,
  strobeRedSpeed: 1, strobeGreenSpeed: 1, strobeBlueSpeed: 1,
}
export type PostEffectSettings = typeof POST_EFFECT_DEFAULTS
export function normalizePostEffects(value: Partial<PostEffectSettings>): PostEffectSettings {
  const result = { ...POST_EFFECT_DEFAULTS }
  for (const key of Object.keys(result) as (keyof PostEffectSettings)[]) {
    const v = value[key]
    if (typeof result[key] === "boolean") Object.assign(result, { [key]: v === true })
    else if (typeof v === "number" && Number.isFinite(v)) {
      const min = key.endsWith("Speed") ? -MAX_WAVE_SPEED : key.endsWith("Period") ? 0.01 : key.endsWith("Exponent") ? 0.1 : 0
      const max = key.endsWith("Speed") ? MAX_WAVE_SPEED : key.endsWith("Period") ? 60 : key.endsWith("Exponent") ? 64 : key.endsWith("Frequency") ? 40 : key === "rgbAngle" ? 360 : key === "dofTop" || key === "dofFront" ? 32 : 1
      Object.assign(result, { [key]: Math.min(max, Math.max(min, v)) })
    }
  }
  return result
}

/** Integrate phase so changing speed/period never jumps to another wave position. */
export function advanceRgbPhases(phases: readonly number[], elapsed: number, settings: PostEffectSettings): number[] {
  const dt = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0
  return (["red", "green", "blue"] as const).map((channel, index) => {
    const phase = phases[index] + dt * settings[`${channel}Speed`] / settings[`${channel}Period`]
    return ((phase % 1) + 1) % 1
  })
}

/** Spatially uniform oscillators; all channels start in sync for white flashes. */
export function advanceStrobePhases(phases: readonly number[], elapsed: number, settings: PostEffectSettings): number[] {
  const dt = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0
  return (["Red", "Green", "Blue"] as const).map((channel, index) => {
    const phase = phases[index] + dt * settings[`strobe${channel}Speed`] * settings[`strobe${channel}Frequency`] / settings[`strobe${channel}Period`]
    return ((phase % 1) + 1) % 1
  })
}

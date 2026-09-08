export interface AttractorSettings {
  enabled: boolean
  range: number
  /** Positive attracts, negative repels. */
  strength: number
  /** Domain widths per second. */
  speed: number
  amplitude: number
  /** Sine cycles per second, independent of horizontal speed. */
  frequency: number
  /** Sine phase in degrees. */
  phase: number
  centerY: number
}

export const DEFAULT_ATTRACTOR: AttractorSettings = {
  enabled: false, range: 0.15, strength: 1, speed: 0.1,
  amplitude: 0.25, frequency: 0.2, phase: 0, centerY: 0.5,
}

export const ATTRACTOR_CONTROLS = [
  { key: "range", label: "Range", min: 0, max: 0.5, step: 0.005 },
  { key: "strength", label: "Strength", min: -10, max: 10, step: 0.05 },
  { key: "speed", label: "Speed (widths/s)", min: 0, max: 2, step: 0.01 },
  { key: "amplitude", label: "Sine amplitude", min: 0, max: 0.5, step: 0.005 },
  { key: "frequency", label: "Sine frequency (Hz)", min: 0, max: 5, step: 0.01 },
  { key: "phase", label: "Sine phase (°)", min: -180, max: 180, step: 1 },
  { key: "centerY", label: "Vertical center", min: 0, max: 1, step: 0.005 },
] as const

/** Clamp audio mapping endpoints as well as manually edited settings. */
export function normalizeAttractor(value?: AttractorSettings): AttractorSettings {
  const result = { ...DEFAULT_ATTRACTOR, ...value }
  for (const { key, min, max } of ATTRACTOR_CONTROLS) {
    result[key] = Number.isFinite(result[key]) ? Math.max(min, Math.min(max, result[key])) : DEFAULT_ATTRACTOR[key]
  }
  result.enabled = result.enabled === true
  return result
}

export interface AttractorPosition { x: number; y: number }

/** Integrating rates keeps position continuous while audio modulates speed. */
export class AttractorMotion {
  private x = 0
  private cycle = 0
  advance(settings: AttractorSettings, seconds: number): AttractorPosition {
    const dt = settings.enabled ? Math.max(0, Math.min(0.1, seconds)) : 0
    this.x = (this.x + settings.speed * dt) % 1
    this.cycle = (this.cycle + settings.frequency * dt) % 1
    return {
      x: this.x,
      y: Math.max(0, Math.min(1, settings.centerY + settings.amplitude * Math.sin(this.cycle * 2 * Math.PI + settings.phase * Math.PI / 180))),
    }
  }
}

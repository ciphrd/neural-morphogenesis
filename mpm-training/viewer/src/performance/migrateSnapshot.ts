import { normalizeAttractor } from "./actuators"
import type { PerformanceSnapshot } from "./types"

/** Carry compatible values forward; newly introduced engine fields use defaults. */
export function compatibleValues<T extends object>(defaults: T, stored: unknown): T {
  const result = { ...defaults }
  if (!stored || typeof stored !== "object") return result
  for (const key of Object.keys(defaults) as Array<keyof T>) {
    const value = (stored as T)[key]
    if (typeof value === typeof defaults[key] &&
      (typeof value === "boolean" || typeof value === "string" ||
        (typeof value === "number" && Number.isFinite(value)))) result[key] = value
  }
  return result
}

export function migrateSnapshot(stored: PerformanceSnapshot, defaults: PerformanceSnapshot): PerformanceSnapshot {
  const render = compatibleValues(defaults.render, stored.render)
  const legacy = stored.render as unknown as Record<string, unknown>
  const colors = {
    "dots-white": "white", "dots-neural-color": "neural-color",
    "dots-internal-state": "neural-memory", "dots-chemical-levels": "chemical-memory",
    "dots-boundary-value": "boundary-value", "dots-activation": "neurons",
    "dots-activation-translucent": "neurons", "directional-arrows": "neural-color",
  } as const
  const mode = legacy?.particleRenderMode as keyof typeof colors
  if (Object.prototype.hasOwnProperty.call(colors, mode)) {
    render.particleColorMode = colors[mode]
    render.particleShape = mode === "directional-arrows" ? "triangle" : "dot"
    const alphaKey = mode === "dots-white" ? "whiteDotsAlpha"
      : mode === "dots-neural-color" ? "neuralColorAlpha"
      : mode === "dots-activation-translucent" ? "activationAlpha" : "internalStateAlpha"
    const alpha = legacy[alphaKey]
    if (typeof alpha === "number" && Number.isFinite(alpha)) render.particleAlpha = Math.max(0, Math.min(1, alpha))
  }
  render.centerDotSize = Number.isFinite(stored.render?.centerDotSize) ? Math.max(0.02, Math.min(0.5, stored.render.centerDotSize)) : 0.18
  render.autoZoom = compatibleValues(defaults.render.autoZoom, stored.render?.autoZoom)
  render.bloom = compatibleValues(defaults.render.bloom, stored.render?.bloom)
  return {
    ...compatibleValues(defaults, stored),
    physics: defaults.physics ? compatibleValues(defaults.physics, stored.physics) : null,
    render,
    attractor: normalizeAttractor(stored.attractor),
  }
}

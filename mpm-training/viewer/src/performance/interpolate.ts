import type { PhysicsSettings } from "../gpu/types"
import type { PerformanceRenderSettings, PerformanceSnapshot } from "./types"

function lerp(from: number, to: number, amount: number): number {
  return from + (to - from) * amount
}

function interpolateNumericObject<T extends object>(from: T, to: T, amount: number): T {
  const next = { ...(amount < 1 ? from : to) }
  for (const key of Object.keys(to) as Array<keyof T>) {
    const fromValue = from[key]
    const toValue = to[key]
    if (typeof fromValue === "number" && typeof toValue === "number") {
      next[key] = lerp(fromValue, toValue, amount) as T[keyof T]
    }
  }
  return next
}

export function interpolateSnapshot(
  from: PerformanceSnapshot,
  to: PerformanceSnapshot,
  amount: number,
): PerformanceSnapshot {
  const clamped = Math.max(0, Math.min(1, amount))
  const physics = from.physics && to.physics
    ? interpolateNumericObject<PhysicsSettings>(from.physics, to.physics, clamped)
    : (clamped < 1 ? from.physics : to.physics)
  return {
    ...(clamped < 1 ? from : to),
    physics,
    noiseDisplacementStrength: lerp(
      from.noiseDisplacementStrength ?? 0,
      to.noiseDisplacementStrength ?? 0,
      clamped,
    ),
    render: interpolateNumericObject<PerformanceRenderSettings>(
      from.render,
      to.render,
      clamped,
    ),
  }
}

export interface CutPoint { x: number; y: number }
export interface CutPush { from: CutPoint; to: CutPoint; radius: number; speed: number }

export function isCutPush(push: CutPush): boolean {
  return !!push && [push.from, push.to].every(p => !!p && Number.isFinite(p.x) && Number.isFinite(p.y)
    && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1)
    && Number.isFinite(push.radius) && push.radius > 0 && push.radius <= 0.1
    && Number.isFinite(push.speed) && push.speed >= 0 && push.speed <= 8
}

/** A narrow geometric blade. Speed remains in the message for compatibility, not force. */
export function makeCutPush(from: CutPoint, to: CutPoint, elapsedMs: number, zoom: number, width: number): CutPush | null {
  if (![elapsedMs, zoom, width, from.x, from.y, to.x, to.y].every(Number.isFinite)
    || elapsedMs <= 0 || zoom <= 0 || width <= 0) return null
  const distance = Math.hypot(to.x - from.x, to.y - from.y)
  if (distance < 1e-6) return null
  const world = (p: CutPoint): CutPoint => ({
    x: 0.5 + (p.x - 0.5) / zoom,
    y: 0.5 + (0.5 - p.y) / zoom,
  })
  const start = world(from), end = world(to)
  let low = 0, high = 1
  for (const axis of ["x", "y"] as const) {
    const delta = end[axis] - start[axis]
    if (Math.abs(delta) < 1e-12) {
      if (start[axis] < 0 || start[axis] > 1) return null
    } else {
      const a = -start[axis] / delta, b = (1 - start[axis]) / delta
      low = Math.max(low, Math.min(a, b)); high = Math.min(high, Math.max(a, b))
    }
  }
  if (low >= high) return null
  const at = (t: number): CutPoint => ({
    x: Math.max(0, Math.min(1, start.x + t * (end.x - start.x))),
    y: Math.max(0, Math.min(1, start.y + t * (end.y - start.y))),
  })
  const push = { from: at(low), to: at(high), radius: Math.min(0.1, 2 / width / zoom),
    speed: Math.min(8, distance * 1000 / Math.max(1, elapsedMs)) }
  return isCutPush(push) ? push : null
}


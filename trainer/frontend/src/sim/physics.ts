/**
 * TypeScript port of trainer/backend/physics.py — same algorithm, same
 * math, kept in sync by hand (there's no shared source between the two
 * languages). See physics.py's module docstring for the full rationale
 * of the two-pass Jacobi solver; this file only restates the mechanics
 * needed to reproduce it exactly.
 *
 * The constants themselves are *not* hand-copied, unlike the algorithm —
 * they arrive at runtime as a PhysicsConfig, broadcast by train_server.py
 * on every generation message (see net/trainingSocket.ts), sourced
 * directly from physics.py's own module constants. That's the single
 * source of truth this file used to duplicate as its own hardcoded
 * RADIUS/TENSION_STIFFNESS/etc. exports — tuning a constant backend-side
 * used to require a matching hand-edit here to avoid silent drift; now
 * there's nothing here to edit.
 *
 * The one structural difference from physics.py: relaxSteps() is a
 * generator that yields the in-progress positions after every single
 * correction (settle pass, then cleanup pass) instead of only returning
 * the converged result — driving this generator a few steps at a time
 * from requestAnimationFrame is what makes the settling motion animate
 * smoothly instead of jumping straight to rest.
 *
 * Performance: TENSION_RANGE is a small, fixed interaction radius, so in
 * any reasonably spread-out graph most pairs of nodes are never within
 * range of each other — the original implementation still checked every
 * one of them, every iteration, via a dense O(n^2) double loop. This
 * version buckets nodes into a uniform grid sized to the query radius
 * (mirroring physics.py's KD-tree query_pairs, minus the library — JS
 * has no builtin spatial tree) and only checks the cell a node is in
 * plus its neighbors, which is where essentially all the frame time was
 * going once a replay reached a couple hundred nodes.
 */

/**
 * Everything physics.py hardcodes as module constants, sent over the
 * wire instead of mirrored here — see this file's own docstring. Field
 * names match train_server.py's PHYSICS_CONFIG dict keys exactly (plain
 * hand-matched, same as every other server->client message shape in
 * this codebase — see e.g. ReplayConfig in runner.ts).
 */
export interface PhysicsConfig {
  radius: number
  contactDistance: number
  tensionRange: number
  tensionStiffness: number
  settleStiffness: number
  settleIterations: number
  cleanupIterations: number
  settleConvergenceTol: number
  cleanupConvergenceTol: number
  // False removes the hard "circles never overlap" guarantee entirely —
  // mirrors physics.py's ENABLE_COLLISION, see that constant's own
  // comment for the full rationale.
  collisionEnabled: boolean
}

export type Vec2 = [number, number]
type Pair = [number, number]

// The 5 directions that, applied to every occupied cell, cover the full
// 3x3 Moore neighborhood of cell-pairs exactly once each — no pair
// double-counted, none missed. (0,0) is the cell itself (self-pairs);
// the other 4 are the "forward" half of the 8 neighbors — whichever
// cell would be their "backward" counterpart picks them up when *it's*
// the reference cell. Verified against brute force before shipping.
const NEIGHBOR_OFFSETS: Pair[] = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
  [-1, 1],
]

function findPairs(pos: Vec2[], radius: number): Pair[] {
  const grid = new Map<string, number[]>()
  for (let i = 0; i < pos.length; i++) {
    const cx = Math.floor(pos[i][0] / radius)
    const cy = Math.floor(pos[i][1] / radius)
    const key = `${cx},${cy}`
    const bucket = grid.get(key)
    if (bucket) bucket.push(i)
    else grid.set(key, [i])
  }

  const radiusSq = radius * radius
  const pairs: Pair[] = []

  for (const [key, bucket] of grid) {
    const [cxStr, cyStr] = key.split(",")
    const cx = Number(cxStr)
    const cy = Number(cyStr)
    for (const [dx, dy] of NEIGHBOR_OFFSETS) {
      const neighborBucket = grid.get(`${cx + dx},${cy + dy}`)
      if (!neighborBucket) continue
      const sameCell = dx === 0 && dy === 0
      for (let a = 0; a < bucket.length; a++) {
        const startB = sameCell ? a + 1 : 0
        for (let b = startB; b < neighborBucket.length; b++) {
          const i = bucket[a]
          const j = neighborBucket[b]
          const ddx = pos[i][0] - pos[j][0]
          const ddy = pos[i][1] - pos[j][1]
          if (ddx * ddx + ddy * ddy <= radiusSq) pairs.push([i, j])
        }
      }
    }
  }
  return pairs
}

function tensionCompatibility(idVectors: number[][]): number[][] {
  const n = idVectors.length
  const norms = idVectors.map((v) => Math.hypot(...v))
  const unit = idVectors.map((v, i) => {
    const norm = norms[i] < 1e-9 ? 1.0 : norms[i]
    return v.map((c) => c / norm)
  })
  const compat: number[][] = Array.from({ length: n }, () =>
    new Array(n).fill(0)
  )
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let dot = 0
      for (let k = 0; k < unit[i].length; k++) dot += unit[i][k] * unit[j][k]
      compat[i][j] = Math.min(1, Math.max(0, dot))
    }
  }
  return compat
}

/**
 * Jacobi-averaged correction from a sparse set of candidate pairs (found
 * within some search radius — see findPairs). `stiffnessOf` returning
 * null means "not actually active" (e.g. a settle-phase candidate that
 * turned out to be past tensionRange at the queried radius's boundary,
 * or a cleanup-phase candidate that isn't really colliding) — such a
 * pair contributes nothing and isn't counted, exactly like the old dense
 * version's active_mask gating both magnitude and count together.
 */
function pairCorrections(
  n: number,
  pos: Vec2[],
  pairs: Pair[],
  target: number,
  stiffnessOf: (i: number, j: number, d: number) => number | null,
  freeMask: boolean[]
): Vec2[] {
  const sumX = new Array(n).fill(0)
  const sumY = new Array(n).fill(0)
  const count = new Array(n).fill(0)

  for (const [i, j] of pairs) {
    const dx = pos[j][0] - pos[i][0]
    const dy = pos[j][1] - pos[i][1]
    const d = Math.hypot(dx, dy)
    const stiffness = stiffnessOf(i, j, d)
    if (stiffness === null) continue

    const dSafe = d < 1e-9 ? 1e-9 : d
    const dirX = dx / dSafe
    const dirY = dy / dSafe
    const magnitude = (d - target) * stiffness
    const cx = dirX * magnitude
    const cy = dirY * magnitude

    sumX[i] += cx
    sumY[i] += cy
    count[i]++
    sumX[j] -= cx
    sumY[j] -= cy
    count[j]++
  }

  const correction: Vec2[] = Array.from({ length: n }, () => [0, 0] as Vec2)
  for (let i = 0; i < n; i++) {
    if (!freeMask[i] || count[i] === 0) continue
    correction[i][0] = sumX[i] / count[i]
    correction[i][1] = sumY[i] / count[i]
  }
  return correction
}

function maxNorm(correction: Vec2[]): number {
  let max = 0
  for (const [x, y] of correction) max = Math.max(max, Math.hypot(x, y))
  return max
}

function applyCorrection(pos: Vec2[], correction: Vec2[]): Vec2[] {
  return pos.map(
    ([x, y], i) => [x + correction[i][0], y + correction[i][1]] as Vec2
  )
}

/**
 * Yields the in-progress positions after every Jacobi correction (the
 * settle pass, then the collision-cleanup pass); the generator's return
 * value is the fully-converged result, same as physics.py's relax().
 */
export function* relaxSteps(
  positions: Vec2[],
  pinned: Set<number>,
  idVectors: number[][],
  config: PhysicsConfig
): Generator<Vec2[], Vec2[], void> {
  let pos: Vec2[] = positions.map(([x, y]) => [x, y])
  const n = pos.length
  if (n < 2) return pos

  const { contactDistance, tensionRange, tensionStiffness, settleStiffness, collisionEnabled } = config
  const freeMask = pos.map((_, i) => !pinned.has(i))
  const compatibility = tensionCompatibility(idVectors)

  for (let iter = 0; iter < config.settleIterations; iter++) {
    const pairs = findPairs(pos, tensionRange)
    const correction = pairCorrections(
      n,
      pos,
      pairs,
      contactDistance,
      (i, j, d) => {
        if (collisionEnabled && d < contactDistance) return settleStiffness
        if (d <= tensionRange) return tensionStiffness * compatibility[i][j]
        return null
      },
      freeMask
    )
    pos = applyCorrection(pos, correction)
    yield pos
    if (maxNorm(correction) < config.settleConvergenceTol) break
  }

  if (collisionEnabled) {
    for (let iter = 0; iter < config.cleanupIterations; iter++) {
      const pairs = findPairs(pos, contactDistance)
      const correction = pairCorrections(
        n,
        pos,
        pairs,
        contactDistance,
        (_i, _j, d) => {
          return d < contactDistance ? 1.0 : null
        },
        freeMask
      )
      pos = applyCorrection(pos, correction)
      yield pos
      if (maxNorm(correction) < config.cleanupConvergenceTol) break
    }
  }

  return pos
}

/** Drains relaxSteps() to convergence in one shot, for callers that don't need the animation. */
export function relax(
  positions: Vec2[],
  pinned: Set<number>,
  idVectors: number[][],
  config: PhysicsConfig
): Vec2[] {
  const gen = relaxSteps(positions, pinned, idVectors, config)
  let result = gen.next()
  while (!result.done) result = gen.next()
  return result.value
}

// Collision-cleanup budget for a single relaxTick() call — small and
// fixed rather than convergence-capped, since (unlike relaxSteps(),
// which runs once per discrete growth step and is expected to fully
// settle before the next one) this runs every animation frame forever;
// any overlap not fully resolved in one tick just keeps resolving over
// the next few frames instead, invisible at animation speed. Purely a
// client-side animation-pacing choice with no backend counterpart
// (unlike everything in PhysicsConfig), so it stays a local constant
// here rather than something sent over the wire. See
// sim/useRealtimeSimulation.ts.
const TICK_COLLISION_ITERATIONS = 4

/**
 * One frame's worth of physics: a single settle-pass correction
 * (tension + soft collision) followed by a small fixed number of hard
 * collision-cleanup corrections — never run to convergence the way
 * relax()/relaxSteps() are. Built for a caller that ticks this every
 * animation frame indefinitely (realtime growth) rather than one that
 * needs "fully settled" before moving on (the discrete step-based
 * replay). Reuses relaxSteps()'s exact same pair-finding/correction
 * math, just called once instead of looped to a tolerance.
 */
export function relaxTick(
  positions: Vec2[],
  pinned: Set<number>,
  idVectors: number[][],
  config: PhysicsConfig
): Vec2[] {
  let pos: Vec2[] = positions.map(([x, y]) => [x, y])
  const n = pos.length
  if (n < 2) return pos

  const { contactDistance, tensionRange, tensionStiffness, settleStiffness, collisionEnabled } = config
  const freeMask = pos.map((_, i) => !pinned.has(i))
  const compatibility = tensionCompatibility(idVectors)

  {
    const pairs = findPairs(pos, tensionRange)
    const correction = pairCorrections(
      n,
      pos,
      pairs,
      contactDistance,
      (i, j, d) => {
        if (collisionEnabled && d < contactDistance) return settleStiffness
        if (d <= tensionRange) return tensionStiffness * compatibility[i][j]
        return null
      },
      freeMask
    )
    pos = applyCorrection(pos, correction)
  }

  if (collisionEnabled) {
    for (let iter = 0; iter < TICK_COLLISION_ITERATIONS; iter++) {
      const pairs = findPairs(pos, contactDistance)
      const correction = pairCorrections(
        n,
        pos,
        pairs,
        contactDistance,
        (_i, _j, d) => {
          return d < contactDistance ? 1.0 : null
        },
        freeMask
      )
      pos = applyCorrection(pos, correction)
    }
  }

  return pos
}

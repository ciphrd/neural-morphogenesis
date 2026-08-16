/**
 * TypeScript port of trainer/backend/physics.py — same algorithm, same
 * constants, kept in sync by hand (there's no shared source between the
 * two languages). See physics.py's module docstring for the full
 * rationale of the two-pass Jacobi solver; this file only restates the
 * mechanics needed to reproduce it exactly.
 *
 * The one structural difference: relaxSteps() is a generator that
 * yields the in-progress positions after every single correction
 * (settle pass, then cleanup pass) instead of only returning the
 * converged result — driving this generator a few steps at a time from
 * requestAnimationFrame is what makes the settling motion animate
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

export const RADIUS = 0.5;
export const CONTACT_DISTANCE = 2 * RADIUS;
export const TENSION_RANGE = CONTACT_DISTANCE * 1.15;
export const TENSION_STIFFNESS = 0.3;
export const SETTLE_STIFFNESS = 0.5;
export const SETTLE_ITERATIONS = 100;
export const CLEANUP_ITERATIONS = 400;

// Two separate tolerances, not one shared CONVERGENCE_TOL — mirrors
// physics.py's own split; see that file's constants block for the full
// empirical justification (measured against a graph grown the normal
// way, one split + relax() per step). Short version: settle (tension) is
// a soft constraint that almost always burned its full iteration cap
// regardless of tolerance, so loosening it cut iterations substantially
// with imperceptible position differences; cleanup (hard collision) is
// the actual "circles never overlap" guarantee and stays tight, since
// loosening it directly increases real, visible interpenetration.
export const SETTLE_CONVERGENCE_TOL = 5e-3;
export const CLEANUP_CONVERGENCE_TOL = 1e-4;

export type Vec2 = [number, number];
type Pair = [number, number];

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
];

function findPairs(pos: Vec2[], radius: number): Pair[] {
  const grid = new Map<string, number[]>();
  for (let i = 0; i < pos.length; i++) {
    const cx = Math.floor(pos[i][0] / radius);
    const cy = Math.floor(pos[i][1] / radius);
    const key = `${cx},${cy}`;
    const bucket = grid.get(key);
    if (bucket) bucket.push(i);
    else grid.set(key, [i]);
  }

  const radiusSq = radius * radius;
  const pairs: Pair[] = [];

  for (const [key, bucket] of grid) {
    const [cxStr, cyStr] = key.split(",");
    const cx = Number(cxStr);
    const cy = Number(cyStr);
    for (const [dx, dy] of NEIGHBOR_OFFSETS) {
      const neighborBucket = grid.get(`${cx + dx},${cy + dy}`);
      if (!neighborBucket) continue;
      const sameCell = dx === 0 && dy === 0;
      for (let a = 0; a < bucket.length; a++) {
        const startB = sameCell ? a + 1 : 0;
        for (let b = startB; b < neighborBucket.length; b++) {
          const i = bucket[a];
          const j = neighborBucket[b];
          const ddx = pos[i][0] - pos[j][0];
          const ddy = pos[i][1] - pos[j][1];
          if (ddx * ddx + ddy * ddy <= radiusSq) pairs.push([i, j]);
        }
      }
    }
  }
  return pairs;
}

function tensionCompatibility(idVectors: number[][]): number[][] {
  const n = idVectors.length;
  const norms = idVectors.map((v) => Math.hypot(...v));
  const unit = idVectors.map((v, i) => {
    const norm = norms[i] < 1e-9 ? 1.0 : norms[i];
    return v.map((c) => c / norm);
  });
  const compat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < unit[i].length; k++) dot += unit[i][k] * unit[j][k];
      compat[i][j] = Math.min(1, Math.max(0, dot));
    }
  }
  return compat;
}

/**
 * Jacobi-averaged correction from a sparse set of candidate pairs (found
 * within some search radius — see findPairs). `stiffnessOf` returning
 * null means "not actually active" (e.g. a settle-phase candidate that
 * turned out to be past TENSION_RANGE at the queried radius's boundary,
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
  const sumX = new Array(n).fill(0);
  const sumY = new Array(n).fill(0);
  const count = new Array(n).fill(0);

  for (const [i, j] of pairs) {
    const dx = pos[j][0] - pos[i][0];
    const dy = pos[j][1] - pos[i][1];
    const d = Math.hypot(dx, dy);
    const stiffness = stiffnessOf(i, j, d);
    if (stiffness === null) continue;

    const dSafe = d < 1e-9 ? 1e-9 : d;
    const dirX = dx / dSafe;
    const dirY = dy / dSafe;
    const magnitude = (d - target) * stiffness;
    const cx = dirX * magnitude;
    const cy = dirY * magnitude;

    sumX[i] += cx;
    sumY[i] += cy;
    count[i]++;
    sumX[j] -= cx;
    sumY[j] -= cy;
    count[j]++;
  }

  const correction: Vec2[] = Array.from({ length: n }, () => [0, 0] as Vec2);
  for (let i = 0; i < n; i++) {
    if (!freeMask[i] || count[i] === 0) continue;
    correction[i][0] = sumX[i] / count[i];
    correction[i][1] = sumY[i] / count[i];
  }
  return correction;
}

function maxNorm(correction: Vec2[]): number {
  let max = 0;
  for (const [x, y] of correction) max = Math.max(max, Math.hypot(x, y));
  return max;
}

function applyCorrection(pos: Vec2[], correction: Vec2[]): Vec2[] {
  return pos.map(([x, y], i) => [x + correction[i][0], y + correction[i][1]] as Vec2);
}

/**
 * Yields the in-progress positions after every Jacobi correction (the
 * settle pass, then the collision-cleanup pass); the generator's return
 * value is the fully-converged result, same as physics.py's relax().
 */
export function* relaxSteps(positions: Vec2[], pinned: Set<number>, idVectors: number[][]): Generator<Vec2[], Vec2[], void> {
  let pos: Vec2[] = positions.map(([x, y]) => [x, y]);
  const n = pos.length;
  if (n < 2) return pos;

  const freeMask = pos.map((_, i) => !pinned.has(i));
  const compatibility = tensionCompatibility(idVectors);

  for (let iter = 0; iter < SETTLE_ITERATIONS; iter++) {
    const pairs = findPairs(pos, TENSION_RANGE);
    const correction = pairCorrections(n, pos, pairs, CONTACT_DISTANCE, (i, j, d) => {
      if (d < CONTACT_DISTANCE) return SETTLE_STIFFNESS;
      if (d <= TENSION_RANGE) return TENSION_STIFFNESS * compatibility[i][j];
      return null;
    }, freeMask);
    pos = applyCorrection(pos, correction);
    yield pos;
    if (maxNorm(correction) < SETTLE_CONVERGENCE_TOL) break;
  }

  for (let iter = 0; iter < CLEANUP_ITERATIONS; iter++) {
    const pairs = findPairs(pos, CONTACT_DISTANCE);
    const correction = pairCorrections(n, pos, pairs, CONTACT_DISTANCE, (_i, _j, d) => {
      return d < CONTACT_DISTANCE ? 1.0 : null;
    }, freeMask);
    pos = applyCorrection(pos, correction);
    yield pos;
    if (maxNorm(correction) < CLEANUP_CONVERGENCE_TOL) break;
  }

  return pos;
}

/** Drains relaxSteps() to convergence in one shot, for callers that don't need the animation. */
export function relax(positions: Vec2[], pinned: Set<number>, idVectors: number[][]): Vec2[] {
  const gen = relaxSteps(positions, pinned, idVectors);
  let result = gen.next();
  while (!result.done) result = gen.next();
  return result.value;
}

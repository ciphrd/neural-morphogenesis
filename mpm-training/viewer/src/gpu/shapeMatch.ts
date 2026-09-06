import config from "../../../core/config.json";
/** Material-area stopping metrics. Keep in parity with trainer/domain_fitness.py.
 * Exact triangle/cell intersections precede bilinear rotation alignment.
 */
/** Resolved PNG occupancy mask. Optional point fields read legacy runs. */
export interface ShapeTarget {
  mask?: number[];
  resolution?: number;
  center?: number[];
  points?: number[][];
  texelSize?: number;
}
export interface MatchMetrics { missing: number; spill: number; overlap: number }
export interface ShapeStopSettings {
  stableStop?: boolean;
  shapeCheckInterval?: number;
  shapeConfirmations?: number;
  shapeSettleSteps?: number;
  shapeMissingTolerance?: number;
  shapeSpillTolerance?: number;
  shapeOverlapTolerance?: number;
  shapeTarget?: ShapeTarget;
}
type Point = [number, number];
const mod = (x: number) => ((x % 1) + 1) % 1;
const area = (p: Point[]) => Math.abs(p.reduce((s, a, i) => {
  const b = p[(i + 1) % p.length];
  return s + a[0] * b[1] - a[1] * b[0];
}, 0)) / 2;

export function targetMask(target: ShapeTarget, n: number): Float64Array {
  if (target.mask) {
    if (target.resolution !== n || target.mask.length !== n*n) {
      throw new Error(`Target mask resolution ${target.resolution} does not match fitness resolution ${n}`);
    }
    return Float64Array.from(target.mask);
  }
  if (!target.points || target.texelSize === undefined) throw new Error("Invalid shape target");
  const mask = new Float64Array(n * n), half = target.texelSize / 2;
  const seen = new Set<string>();
  for (const [x, y] of target.points) {
    const key = `${x},${y}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const left = (x - half) * n, right = (x + half) * n;
    const top = (y - half) * n, bottom = (y + half) * n;
    for (let j = Math.max(0, Math.floor(top)); j < Math.min(n, Math.ceil(bottom)); j++) {
      for (let i = Math.max(0, Math.floor(left)); i < Math.min(n, Math.ceil(right)); i++) {
        mask[j*n+i] += Math.max(0, Math.min(i+1, right)-Math.max(i, left)) *
          Math.max(0, Math.min(j+1, bottom)-Math.max(j, top));
      }
    }
  }
  return mask.map(v => Math.min(1, v));
}

export function centeredTriangles(vertices: ArrayLike<number>, center: Point): { triangles: Point[][]; materialArea: number } {
  const triangles: Point[][] = [];
  if (!vertices.length || vertices.length % 6) throw new Error("Empty or invalid triangle geometry");
  for (let i = 0; i < vertices.length; i += 6) {
    const a: Point = [vertices[i], vertices[i+1]], p: Point[] = [a];
    for (let j = 2; j < 6; j += 2) p.push([
      a[0] + mod(vertices[i+j]-a[0]+.5)-.5,
      a[1] + mod(vertices[i+j+1]-a[1]+.5)-.5,
    ]);
    if (!p.flat().every(Number.isFinite)) throw new Error("Nonfinite triangle geometry");
    triangles.push(p);
  }
  for (let axis = 0; axis < 2; axis++) {
    const centers = triangles.map(t => t.reduce((s, p) => s+p[axis], 0)/3);
    const ordered = centers.map(mod).sort((a, b) => a-b);
    let largest = -1, start = 0;
    for (let i = 0; i < ordered.length; i++) {
      const gap = (i+1 < ordered.length ? ordered[i+1] : ordered[0]+1)-ordered[i];
      if (gap > largest) { largest = gap; start = ordered[(i+1)%ordered.length]; }
    }
    triangles.forEach((t, i) => {
      const shift = start+mod(mod(centers[i])-start)-centers[i];
      t.forEach(p => { p[axis] += shift; });
    });
  }
  let materialArea = 0;
  const centroid: Point = [0, 0];
  for (const t of triangles) {
    const a = area(t); materialArea += a;
    for (let axis = 0; axis < 2; axis++) centroid[axis] += a*t.reduce((s,p) => s+p[axis], 0)/3;
  }
  if (!(materialArea > 1e-16)) throw new Error("Degenerate material domains");
  triangles.forEach(t => t.forEach(p => {
    p[0] += center[0]-centroid[0]/materialArea;
    p[1] += center[1]-centroid[1]/materialArea;
  }));
  return { triangles, materialArea };
}

function clip(poly: Point[], axis: number, boundary: number, sign: number): Point[] {
  const out: Point[] = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i+1)%poly.length];
    const da = sign*(a[axis]-boundary), db = sign*(b[axis]-boundary);
    if ((da >= 0) !== (db >= 0)) {
      const f = da/(da-db);
      out.push([a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])]);
    }
    if (db >= 0) out.push(b);
  }
  return out;
}

export function rasterizeTriangles(triangles: Point[][], n: number): Float64Array {
  const density = new Float64Array(n*n);
  for (const triangle of triangles) {
    const t = triangle.map(p => [p[0]*n, p[1]*n] as Point);
    const x0 = Math.max(0, Math.floor(Math.min(...t.map(p => p[0]))));
    const x1 = Math.min(n, Math.ceil(Math.max(...t.map(p => p[0]))));
    const y0 = Math.max(0, Math.floor(Math.min(...t.map(p => p[1]))));
    const y1 = Math.min(n, Math.ceil(Math.max(...t.map(p => p[1]))));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const poly = clip(clip(clip(clip(t, 0, x, 1), 0, x+1, -1), 1, y, 1), 1, y+1, -1);
      if (poly.length >= 3) density[y*n+x] += area(poly);
    }
  }
  return density;
}

function rotated(density: Float64Array, n: number, angle: number, center: Point): Float64Array {
  if (angle === 0) return density;
  const out = new Float64Array(n*n), c = Math.cos(angle), s = Math.sin(angle);
  const ox = center[0]*n-.5, oy = center[1]*n-.5;
  const get = (x: number, y: number) => x < 0 || x >= n || y < 0 || y >= n ? 0 : density[y*n+x];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const sy = c*(y-oy)+s*(x-ox)+oy, sx = -s*(y-oy)+c*(x-ox)+ox;
    const ix = Math.floor(sx), iy = Math.floor(sy), fx = sx-ix, fy = sy-iy;
    out[y*n+x] = (1-fy)*((1-fx)*get(ix, iy)+fx*get(ix+1, iy)) +
      fy*((1-fx)*get(ix, iy+1)+fx*get(ix+1, iy+1));
  }
  return out;
}

export function matchDomains(vertices: ArrayLike<number>, target: ShapeTarget, mask: Float64Array): MatchMetrics {
  const failure = { missing: Infinity, spill: Infinity, overlap: Infinity };
  const n = Math.sqrt(mask.length);
  const center = (target.center ?? [0, 1].map(axis =>
    target.points!.reduce((s,p) => s+p[axis], 0)/target.points!.length)) as Point;
  let source: ReturnType<typeof centeredTriangles>;
  try { source = centeredTriangles(vertices, center); } catch { return failure; }
  const density = rasterizeTriangles(source.triangles, n);
  const mass = mask.reduce((s,v) => s+v, 0);
  if (!(mass > 0)) return failure;
  let best = failure, bestAngle = 0;
  const evaluate = (angle: number) => {
    const d = rotated(density, n, angle, center);
    let missing = 0, spill = 0, overlap = 0, sum = 0;
    for (let i = 0; i < d.length; i++) {
      const occupancy = Math.min(1, Math.max(0, d[i]));
      sum += d[i]; missing += Math.max(0, mask[i]-occupancy);
      spill += Math.max(0, occupancy-mask[i]); overlap += Math.max(0, d[i]-1);
    }
    spill += Math.max(0, source.materialArea*n*n-sum);
    const m = { missing: missing/mass, spill: spill/mass, overlap: overlap/mass };
    if (m.missing+m.spill+m.overlap < best.missing+best.spill+best.overlap-1e-10) { best = m; bestAngle = angle; }
  };
  for (let i = 0; i < 16; i++) evaluate(2*Math.PI*i/16);
  let step = 2*Math.PI/16;
  for (let i = 0; i < 2; i++) { step /= 3; const angle = bestAngle; evaluate(angle-step); evaluate(angle+step); }
  return best;
}

export class StableMatchStop {
  streak = 0;
  settlingSince: number | null = null;
  complete = false;
  match: MatchMetrics | null = null;
  constructor(readonly settings: ShapeStopSettings) {}
  get enabled() { return this.settings.stableStop === true && !!this.settings.shapeTarget; }
  get growthEnabled() { return this.settlingSince === null && !this.complete; }
  due(step: number) {
    return this.enabled && (step % (this.settings.shapeCheckInterval ?? config.run.shapeCheckInterval) === 0 ||
      (this.settlingSince !== null && step-this.settlingSince >= (this.settings.shapeSettleSteps ?? config.run.shapeSettleSteps)));
  }
  observe(step: number, match: MatchMetrics, blocked = false) {
    this.match = match;
    const good = !blocked && [match.missing, match.spill, match.overlap].every((v, i) =>
      Number.isFinite(v) && v <= [this.settings.shapeMissingTolerance ?? config.run.shapeMissingTolerance,
        this.settings.shapeSpillTolerance ?? config.run.shapeSpillTolerance, this.settings.shapeOverlapTolerance ?? config.run.shapeOverlapTolerance][i]);
    if (!good) { this.streak = 0; this.settlingSince = null; }
    else if (this.settlingSince !== null) this.complete = step-this.settlingSince >= (this.settings.shapeSettleSteps ?? config.run.shapeSettleSteps);
    else if (++this.streak >= (this.settings.shapeConfirmations ?? config.run.shapeConfirmations)) this.settlingSince = step;
    return this.complete;
  }
}

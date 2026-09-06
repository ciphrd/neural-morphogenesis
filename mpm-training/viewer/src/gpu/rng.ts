import densityModelConfig from "../../../core/config.json";
const densityModel = densityModelConfig.density;
import type { SceneData } from "./types";

// Bit-exact, portable integer hash (Chris Wellons' "lowbias32" — public
// domain), mirrored exactly by ../../../trainer/agents_gpu.py's own
// _hash_u32(). Only uses uint32 add/xor/shift/multiply-with-wraparound,
// so it's trivial to reproduce exactly in numpy — Math.imul (forced back
// to unsigned via >>> 0) wraps mod 2**32 identically to numpy's own
// uint32 dtype.
function hashU32(x: number): number {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

/** Fixed world-space random field shared by every sampling density. */

const SPAWN_HASH_DOMAIN = 0xc0ffee00;

export function spawnUniform01(seed: number, index: number): number {
  const combined = ((seed >>> 0) ^ hashU32((SPAWN_HASH_DOMAIN ^ index) >>> 0)) >>> 0;
  const hashed = hashU32(combined);
  return (hashed >>> 8) / 16777216;
}

export interface SeedBlobConfig {
  /** Material budget in seed-cell units; emits twice as many area-weighted triangles. */
  count: number;
  centerX: number;
  centerY: number;
  spacing: number;
  seed: number;
}

/** Diagonal partition of each cell, preserving its footprint and material.
 * Paired samples have indices 2*i and 2*i+1. Mirrors triangle_seed.py. */
export function triangulateSeedCells(scene: Omit<SceneData, "domainGeometry"> & { domain: Float32Array }): SceneData {
  const count = 2*scene.count;
  const positions = new Float32Array(count*2);
  const domain = new Float32Array(count*6);
  const quadratureWeights = new Float32Array(count);
  const velocities = new Float32Array(count*2);
  const F = new Float32Array(count*4);
  const C = new Float32Array(count*4);
  const Jp = new Float32Array(count);
  const [ax,bx,ay,by] = scene.domain.subarray(0,4);
  const det = 4*(ax*by-bx*ay);
  if (!(det>0)) throw new Error("Seed cells must have positive winding");
  const wrap = (v: number) => v-Math.floor(v);
  const delta = (v: number) => v-Math.floor(v+.5);
  const cornerCache = new Map<string, readonly number[]>();
  function corner(q: number, r: number): readonly number[] {
    const key = `${q},${r}`;
    let value = cornerCache.get(key);
    if (!value) {
      value = [wrap(Math.fround(wrap(scene.positions[0]+ax*q+bx*r))),
               wrap(Math.fround(wrap(scene.positions[1]+ay*q+by*r)))];
      cornerCache.set(key,value);
    }
    return value;
  }
  for (let i = 0; i < scene.count; i++) {
    const dx = delta(scene.positions[2*i]-scene.positions[0]);
    const dy = delta(scene.positions[2*i+1]-scene.positions[1]);
    const q = Math.round((2*by*dx-2*bx*dy)/det);
    const r = Math.round((-2*ay*dx+2*ax*dy)/det);
    const a=corner(2*q-1,2*r-1), b=corner(2*q+1,2*r-1);
    const c=corner(2*q-1,2*r+1), d=corner(2*q+1,2*r+1);
    const triangles = ax*bx+ay*by>0 ? [[a,b,c],[b,d,c]] : [[a,b,d],[a,d,c]];
    for (let j = 0; j < 2; j++) {
      const index = 2*i+j;
      const [va,vb,vc] = triangles[j];
      domain.set([...va,...vb,...vc],6*index);
      positions[2*index] = wrap(va[0]+(delta(vb[0]-va[0])+delta(vc[0]-va[0]))/3);
      positions[2*index+1] = wrap(va[1]+(delta(vb[1]-va[1])+delta(vc[1]-va[1]))/3);
      velocities.set(scene.velocities.subarray(2*i, 2*i+2), 2*index);
      F.set(scene.F.subarray(4*i, 4*i+4), 4*index);
      C.set(scene.C.subarray(4*i, 4*i+4), 4*index);
      Jp[index] = scene.Jp[i];
      quadratureWeights[index] = .5*(scene.quadratureWeights?.[i] ?? 1);
    }
  }
  return { count, positions, velocities, F, C, Jp, domain, quadratureWeights, domainGeometry: "triangle-vertices" };
}

/** Exact, axis-aligned rows for deterministic lab scenarios. Unlike seedBlob,
 * this layout has no packing scale or seed-derived rotation: adjacent cells
 * are separated by the target spacing. Cells are emitted bottom-to-top and
 * left-to-right, with two consecutive triangle samples per cell. */
export function seedRows(config: {
  rows: number;
  columns: number;
  centerX: number;
  centerY: number;
  spacing: number;
}) {
  const { rows, columns, centerX, centerY, spacing } = config;
  const count = rows * columns;
  const positions = new Float32Array(count * 2);
  const velocities = new Float32Array(count * 2);
  const F = new Float32Array(count * 4);
  const C = new Float32Array(count * 4);
  const Jp = new Float32Array(count).fill(1);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      positions[index * 2] = centerX + (column - (columns - 1) / 2) * spacing;
      positions[index * 2 + 1] = centerY + (row - (rows - 1) / 2) * spacing;
      F[index * 4] = 1;
      F[index * 4 + 3] = 1;
    }
  }
  const domain = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) { domain[i * 4] = spacing / 2; domain[i * 4 + 3] = spacing / 2; }
  return triangulateSeedCells({ count, positions, velocities, F, C, Jp, domain });
}

/** Concentric-ring disk with a regular circular boundary and exactly 2*count
 * triangles. Mirrors triangle_seed.py; weights track triangle area so material
 * density stays uniform. The two-triangle minimum is a square. */
export function seedBlob(config: SeedBlobConfig): SceneData {
  const { count: cells, centerX, centerY, spacing, seed } = config;
  if (!Number.isInteger(cells) || cells < 1 || !Number.isFinite(spacing) || spacing <= 0)
    throw new Error("Disk seeds require a positive integer count and spacing");
  const count = 2*cells;
  const rings = Math.max(1, Math.floor(Math.sqrt(count/6)+.5));
  const sizes: number[] = [];
  for (let k=1; k<rings; k++) sizes.push(Math.floor(count*k/(rings*rings)+.5));
  sizes.push(count-2*sizes.reduce((a,b)=>a+b,0));
  const theta = (spawnUniform01(seed, 2)*2-1)*Math.PI;
  const points: number[][] = [[0,0]];
  const faces: number[][] = [];
  let previous = [0];
  for (let k=1; k<=rings; k++) {
    const size = cells === 1 ? 4 : sizes[k-1];
    const current: number[] = [];
    for (let j=0; j<size; j++) {
      const angle = theta+2*Math.PI*j/size;
      current.push(points.length);
      points.push([k/rings*Math.cos(angle),k/rings*Math.sin(angle)]);
    }
    if (cells === 1) faces.push([1,2,3],[1,3,4]);
    else if (k === 1) {
      for (let j=0; j<size; j++) faces.push([0,current[j],current[(j+1)%size]]);
    } else {
      let i=0, j=0;
      const inner = previous.length;
      while (i<inner || j<size) {
        const a=previous[i%inner], b=current[j%size];
        if (i<inner && (j===size || (i+1)*size <= (j+1)*inner)) {
          faces.push([a,b,previous[(i+1)%inner]]); i++;
        } else {
          faces.push([a,b,current[(j+1)%size]]); j++;
        }
      }
    }
    previous = current;
  }
  const boundary = previous.length;
  const packedSpacing = spacing*densityModel.INITIAL_PACKING_SPACING_SCALE;
  const targetArea = cells*packedSpacing**2*Math.sqrt(3)/2;
  const radius = Math.sqrt(targetArea/(.5*boundary*Math.sin(2*Math.PI/boundary)));
  const wrap = (v: number) => v-Math.floor(v);
  const delta = (v: number) => v-Math.floor(v+.5);
  const vertices = points.map(([x,y])=>[
    wrap(Math.fround(wrap(centerX+radius*x))),wrap(Math.fround(wrap(centerY+radius*y)))]);
  const positions = new Float32Array(count*2), domain = new Float32Array(count*6);
  const quadratureWeights = new Float32Array(count), areas: number[] = [];
  const velocities = new Float32Array(count*2), F = new Float32Array(count*4);
  const C = new Float32Array(count*4), Jp = new Float32Array(count).fill(1);
  for (let i=0; i<count; i++) {
    const [a,b,c] = faces[i].map(j=>vertices[j]);
    const bx=delta(b[0]-a[0]), by=delta(b[1]-a[1]);
    const cx=delta(c[0]-a[0]), cy=delta(c[1]-a[1]);
    domain.set([...a,...b,...c],6*i);
    positions[2*i]=wrap(a[0]+(bx+cx)/3);
    positions[2*i+1]=wrap(a[1]+(by+cy)/3);
    areas.push(.5*(bx*cy-by*cx));
    F[4*i]=F[4*i+3]=1;
  }
  const area = areas.reduce((a,b)=>a+b,0);
  for (let i=0; i<count; i++) quadratureWeights[i]=cells*areas[i]/area;
  return {count,positions,velocities,F,C,Jp,domain,quadratureWeights,domainGeometry:"triangle-vertices"};
}

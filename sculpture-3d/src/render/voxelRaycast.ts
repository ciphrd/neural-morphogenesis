import * as THREE from "three";
import { VoxelGrid } from "../voxel/VoxelGrid";

export type BrushMode = "add" | "erase";

interface RayBoxHit {
  tEnter: number;
  tExit: number;
}

const AXES = ["x", "y", "z"] as const;

function rayBoxIntersect(origin: THREE.Vector3, dir: THREE.Vector3, boxMax: THREE.Vector3): RayBoxHit | null {
  let tMin = -Infinity;
  let tMax = Infinity;
  for (const axis of AXES) {
    const o = origin[axis];
    const d = dir[axis];
    const max = boxMax[axis];
    if (Math.abs(d) < 1e-12) {
      if (o < 0 || o > max) return null;
      continue;
    }
    let t1 = (0 - o) / d;
    let t2 = (max - o) / d;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return null;
  }
  if (tMax < 0) return null;
  return { tEnter: tMin, tExit: tMax };
}

/**
 * Amanatides & Woo fast voxel traversal: walks the ray cell-by-cell through
 * the grid.
 * - erase: returns the first filled voxel the ray touches.
 * - add: returns the empty voxel just before the first filled voxel (so you
 *   build onto an existing surface), or the ray's entry voxel if it never
 *   hits anything (placing into empty space).
 */
export function raycastVoxel(
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  grid: VoxelGrid,
  cellSize: number,
  mode: BrushMode
): [number, number, number] | null {
  const boxMax = new THREE.Vector3(grid.nx * cellSize, grid.ny * cellSize, grid.nz * cellSize);
  const hit = rayBoxIntersect(origin, dir, boxMax);
  if (!hit) return null;

  const t0 = Math.max(hit.tEnter, 0) + 1e-4;
  const start = origin.clone().addScaledVector(dir, t0);

  let x = Math.min(grid.nx - 1, Math.max(0, Math.floor(start.x / cellSize)));
  let y = Math.min(grid.ny - 1, Math.max(0, Math.floor(start.y / cellSize)));
  let z = Math.min(grid.nz - 1, Math.max(0, Math.floor(start.z / cellSize)));

  const stepX = dir.x > 0 ? 1 : dir.x < 0 ? -1 : 0;
  const stepY = dir.y > 0 ? 1 : dir.y < 0 ? -1 : 0;
  const stepZ = dir.z > 0 ? 1 : dir.z < 0 ? -1 : 0;

  const boundaryT = (pos: number, d: number, idx: number, step: number) => {
    if (step === 0) return Infinity;
    const boundary = (idx + (step > 0 ? 1 : 0)) * cellSize;
    return (boundary - pos) / d;
  };
  let tMaxX = boundaryT(origin.x, dir.x, x, stepX);
  let tMaxY = boundaryT(origin.y, dir.y, y, stepY);
  let tMaxZ = boundaryT(origin.z, dir.z, z, stepZ);

  const tDeltaX = stepX !== 0 ? Math.abs(cellSize / dir.x) : Infinity;
  const tDeltaY = stepY !== 0 ? Math.abs(cellSize / dir.y) : Infinity;
  const tDeltaZ = stepZ !== 0 ? Math.abs(cellSize / dir.z) : Infinity;

  let lastEmpty: [number, number, number] | null = null;
  const maxSteps = grid.nx + grid.ny + grid.nz + 3;

  for (let i = 0; i < maxSteps; i++) {
    if (x < 0 || y < 0 || z < 0 || x >= grid.nx || y >= grid.ny || z >= grid.nz) break;

    const filled = grid.get(x, y, z);
    if (mode === "erase") {
      if (filled) return [x, y, z];
    } else {
      if (filled) return lastEmpty;
      lastEmpty = [x, y, z];
    }

    if (tMaxX < tMaxY && tMaxX < tMaxZ) {
      x += stepX;
      tMaxX += tDeltaX;
    } else if (tMaxY < tMaxZ) {
      y += stepY;
      tMaxY += tDeltaY;
    } else {
      z += stepZ;
      tMaxZ += tDeltaZ;
    }
  }

  return mode === "add" ? lastEmpty : null;
}

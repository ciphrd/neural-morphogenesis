import { VoxelGrid } from "./VoxelGrid";

/** Sets every voxel within Euclidean `radius` of `center` to `fill`, in place. */
export function applyBrushStamp(
  grid: VoxelGrid,
  center: [number, number, number],
  radius: number,
  fill: boolean
): void {
  const [cx, cy, cz] = center;
  const rCeil = Math.ceil(radius);
  const r2 = radius * radius;
  for (let dz = -rCeil; dz <= rCeil; dz++) {
    const z = cz + dz;
    if (z < 0 || z >= grid.nz) continue;
    for (let dy = -rCeil; dy <= rCeil; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= grid.ny) continue;
      for (let dx = -rCeil; dx <= rCeil; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= grid.nx) continue;
        if (dx * dx + dy * dy + dz * dz <= r2 + 1e-6) grid.set(x, y, z, fill);
      }
    }
  }
}

import { VoxelGrid } from "./VoxelGrid";

export interface SphereParams {
  centerX: number;
  centerY: number;
  centerZ: number;
  radius: number;
}

/** Fills every voxel whose center lies within `radius` of the given center. */
export function populateSphere(grid: VoxelGrid, params: SphereParams): void {
  const { centerX, centerY, centerZ, radius } = params;
  const r2 = radius * radius;

  const minX = Math.max(0, Math.floor(centerX - radius));
  const maxX = Math.min(grid.nx - 1, Math.ceil(centerX + radius));
  const minY = Math.max(0, Math.floor(centerY - radius));
  const maxY = Math.min(grid.ny - 1, Math.ceil(centerY + radius));
  const minZ = Math.max(0, Math.floor(centerZ - radius));
  const maxZ = Math.min(grid.nz - 1, Math.ceil(centerZ + radius));

  for (let z = minZ; z <= maxZ; z++) {
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x + 0.5 - centerX;
        const dy = y + 0.5 - centerY;
        const dz = z + 0.5 - centerZ;
        if (dx * dx + dy * dy + dz * dz <= r2) {
          grid.set(x, y, z, true);
        }
      }
    }
  }
}

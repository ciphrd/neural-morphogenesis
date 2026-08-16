import { VoxelGrid } from "./VoxelGrid";

/**
 * Minimal, deliberately schema-loose export: each voxel is its own object so
 * later techniques can attach extra per-voxel fields (color, material,
 * density, ...) without changing this shape.
 */
export interface VoxelExport {
  nx: number;
  ny: number;
  nz: number;
  voxels: { x: number; y: number; z: number }[];
}

export function voxelGridToJSON(grid: VoxelGrid): VoxelExport {
  const voxels: { x: number; y: number; z: number }[] = [];
  for (const [x, y, z] of grid.filled()) voxels.push({ x, y, z });
  return { nx: grid.nx, ny: grid.ny, nz: grid.nz, voxels };
}

export function downloadVoxelGridJSON(grid: VoxelGrid, filename = "sculpture.json"): void {
  const data = voxelGridToJSON(grid);
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

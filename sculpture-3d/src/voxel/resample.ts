import { VoxelGrid } from "./VoxelGrid";

/**
 * Resamples the occupancy of `source` onto a new grid of the given
 * dimensions via nearest-neighbor point sampling. Both grids share the same
 * physical bounding cube (see WORLD_SIZE in render/VoxelRenderer), so this
 * keeps whatever is currently filled in roughly the same place in world
 * space across a resolution change — regardless of which technique (or
 * manual edit) produced it.
 */
export function resampleVoxelGrid(source: VoxelGrid, nx: number, ny: number, nz: number): VoxelGrid {
  const target = new VoxelGrid(nx, ny, nz);
  for (let z = 0; z < nz; z++) {
    const sz = Math.min(source.nz - 1, Math.floor(((z + 0.5) / nz) * source.nz));
    for (let y = 0; y < ny; y++) {
      const sy = Math.min(source.ny - 1, Math.floor(((y + 0.5) / ny) * source.ny));
      for (let x = 0; x < nx; x++) {
        const sx = Math.min(source.nx - 1, Math.floor(((x + 0.5) / nx) * source.nx));
        if (source.get(sx, sy, sz)) target.set(x, y, z, true);
      }
    }
  }
  return target;
}

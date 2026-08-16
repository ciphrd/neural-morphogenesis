import { PixelGrid } from "./PixelGrid";

/**
 * Minimal, deliberately schema-loose export: each pixel is its own object
 * so later techniques can attach extra per-pixel fields (color, density,
 * ...) without changing this shape. 2D analogue of sculpture-3d's
 * voxelGridToJSON.
 */
export interface PixelExport {
  nx: number;
  ny: number;
  pixels: { x: number; y: number }[];
}

export function pixelGridToJSON(grid: PixelGrid): PixelExport {
  const pixels: { x: number; y: number }[] = [];
  for (const [x, y] of grid.filled()) pixels.push({ x, y });
  return { nx: grid.nx, ny: grid.ny, pixels };
}

export function downloadPixelGridJSON(grid: PixelGrid, filename = "sculpture.json"): void {
  const data = pixelGridToJSON(grid);
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

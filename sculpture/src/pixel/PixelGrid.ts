/** A dense boolean pixel grid of size nx * ny, indexed [x + y*nx]. 2D
 * analogue of sculpture-3d's VoxelGrid. */
export class PixelGrid {
  readonly nx: number;
  readonly ny: number;
  readonly cells: Uint8Array;

  constructor(nx: number, ny: number) {
    this.nx = nx;
    this.ny = ny;
    this.cells = new Uint8Array(nx * ny);
  }

  index(x: number, y: number): number {
    return x + y * this.nx;
  }

  inBounds(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < this.nx && y < this.ny;
  }

  get(x: number, y: number): boolean {
    return this.inBounds(x, y) ? this.cells[this.index(x, y)] !== 0 : false;
  }

  set(x: number, y: number, value: boolean): void {
    if (!this.inBounds(x, y)) return;
    this.cells[this.index(x, y)] = value ? 1 : 0;
  }

  clear(): void {
    this.cells.fill(0);
  }

  count(): number {
    let n = 0;
    for (let i = 0; i < this.cells.length; i++) n += this.cells[i];
    return n;
  }

  *filled(): Generator<[number, number]> {
    for (let y = 0; y < this.ny; y++) {
      for (let x = 0; x < this.nx; x++) {
        if (this.cells[this.index(x, y)]) yield [x, y];
      }
    }
  }
}

/** Every cell a `size`x`size` square brush centered at (roughly) `(cx, cy)`
 * covers, for `size` >= 1 — odd sizes land exactly centered on `(cx, cy)`;
 * even sizes extend one extra cell right/down rather than split a cell in
 * half (the standard "even brush" convention most pixel editors use).
 * Not bounds-checked against any particular grid — callers that need that
 * (PixelCanvas's paint and preview both do) filter/clip themselves, since
 * PixelGrid.set() already no-ops out-of-bounds and the hover preview wants
 * to *see* a brush clipped at the edge, not have it silently shrink.
 *
 * The one and only place brush shape is defined — PixelCanvas's actual
 * paint operation and its hover preview both call this, so the preview is
 * guaranteed pixel-for-pixel identical to what painting will actually do,
 * not a lookalike approximation of it. */
export function brushCells(cx: number, cy: number, size: number): [number, number][] {
  const lo = -Math.floor((size - 1) / 2);
  const hi = size - 1 + lo;
  const cells: [number, number][] = [];
  for (let dy = lo; dy <= hi; dy++) {
    for (let dx = lo; dx <= hi; dx++) {
      cells.push([cx + dx, cy + dy]);
    }
  }
  return cells;
}

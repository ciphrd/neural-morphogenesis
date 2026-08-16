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

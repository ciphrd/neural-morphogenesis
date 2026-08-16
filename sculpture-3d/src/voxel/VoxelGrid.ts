/** A dense boolean voxel grid of size nx * ny * nz, indexed [x + y*nx + z*nx*ny]. */
export class VoxelGrid {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly cells: Uint8Array;

  constructor(nx: number, ny: number, nz: number) {
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    this.cells = new Uint8Array(nx * ny * nz);
  }

  index(x: number, y: number, z: number): number {
    return x + y * this.nx + z * this.nx * this.ny;
  }

  inBounds(x: number, y: number, z: number): boolean {
    return x >= 0 && y >= 0 && z >= 0 && x < this.nx && y < this.ny && z < this.nz;
  }

  get(x: number, y: number, z: number): boolean {
    return this.inBounds(x, y, z) ? this.cells[this.index(x, y, z)] !== 0 : false;
  }

  set(x: number, y: number, z: number, value: boolean): void {
    if (!this.inBounds(x, y, z)) return;
    this.cells[this.index(x, y, z)] = value ? 1 : 0;
  }

  clear(): void {
    this.cells.fill(0);
  }

  clone(): VoxelGrid {
    const copy = new VoxelGrid(this.nx, this.ny, this.nz);
    copy.cells.set(this.cells);
    return copy;
  }

  count(): number {
    let n = 0;
    for (let i = 0; i < this.cells.length; i++) n += this.cells[i];
    return n;
  }

  *filled(): Generator<[number, number, number]> {
    for (let z = 0; z < this.nz; z++) {
      for (let y = 0; y < this.ny; y++) {
        for (let x = 0; x < this.nx; x++) {
          if (this.cells[this.index(x, y, z)]) yield [x, y, z];
        }
      }
    }
  }
}

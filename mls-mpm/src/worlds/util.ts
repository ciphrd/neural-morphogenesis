export function hexToRgb(hex: number): [number, number, number] {
  return [((hex >> 16) & 0xff) / 255, ((hex >> 8) & 0xff) / 255, (hex & 0xff) / 255];
}

/** Allocates the 6 flat arrays every world's SceneData needs, sized to
 * `count` — factored out purely so each world file doesn't repeat the
 * same 6 `new Float32Array(...)` lines. */
export function allocateScene(count: number): {
  positions: Float32Array;
  velocities: Float32Array;
  F: Float32Array;
  C: Float32Array;
  Jp: Float32Array;
  colors: Float32Array;
} {
  return {
    positions: new Float32Array(count * 2),
    velocities: new Float32Array(count * 2),
    F: new Float32Array(count * 4),
    C: new Float32Array(count * 4),
    Jp: new Float32Array(count),
    colors: new Float32Array(count * 4),
  };
}

/** Writes particle `i`'s F (deformation gradient) as the identity matrix
 * and Jp (plastic Jacobian) as 1 — the at-rest state every world's
 * particles start from, mirroring mls-mpm88-explained.cpp's own
 * Particle constructor (F(1), Jp=1). */
export function setRestState(F: Float32Array, Jp: Float32Array, i: number): void {
  F[i * 4] = 1;
  F[i * 4 + 3] = 1;
  Jp[i] = 1;
}

export function setColor(colors: Float32Array, i: number, r: number, g: number, b: number): void {
  colors[i * 4] = r;
  colors[i * 4 + 1] = g;
  colors[i * 4 + 2] = b;
  colors[i * 4 + 3] = 1;
}

import type { SceneData, World } from "./types";
import { allocateScene, hexToRgb, setColor, setRestState } from "./util";

/** A grid of blobs, each jittered uniformly in a square around its own
 * center, mirroring mls-mpm88-explained.cpp's add_object() (same
 * per-blob half-width/color-cycling shape). "Way more points, filling
 * more space" is achieved by tiling *more of the reference's own blob*
 * across the domain, deliberately NOT by making each blob bigger or
 * denser: an earlier version tried exactly that (fewer, bigger/denser
 * blobs) and it never stopped visibly wobbling — this explicit,
 * undamped MPM integration was tuned around the reference's own
 * particles-per-cell ratio (and, it turns out, its own per-blob *mass*
 * too — see the paragraph below), and pushing either up reintroduces a
 * sustained, un-decaying jitter rather than a one-time settle. Verified
 * empirically (a plain-JS CPU port of this exact algorithm, tracking
 * mean particle speed over hundreds of frames): the reference's own
 * 1000-particles/±0.08-half-width blob reliably decays to near-zero
 * residual speed; the same total particle count crammed into fewer,
 * proportionally-bigger blobs (matched per-cell density, just more mass
 * each) does not, and neither does packing too many reference-sized
 * blobs too close together — gravity collapses everything toward the
 * floor regardless of starting spread, so what actually matters is the
 * *settled* packing density along that floor line, not just each blob's
 * own starting density. GRID_COLS x GRID_ROWS x 0.2 spacing is the
 * largest layout that stayed stable in that testing (at PARTICLES_PER_BLOB
 * =1000 — this file's own count has since been bumped well past that by
 * hand; if this world starts wobbling persistently again, that's why). */

interface Blob {
  center: readonly [number, number];
  colorHex: number;
}

// Reference's own 3 colors (mls-mpm88-explained.cpp's add_object() calls),
// cycled across every blob rather than one color per blob.
const COLORS = [0xed553b, 0xf2b134, 0x068587];

const GRID_COLS = [0.2, 0.4, 0.6, 0.8];
const GRID_ROWS = [0.35, 0.65];
const BLOBS: readonly Blob[] = GRID_ROWS.flatMap((y, rowIdx) =>
  GRID_COLS.map((x, colIdx) => ({
    center: [x, y] as const,
    colorHex: COLORS[(rowIdx * GRID_COLS.length + colIdx) % COLORS.length],
  }))
);

const BLOB_HALF_WIDTH = 0.08; // reference's own value
const PARTICLES_PER_BLOB = 5_000; // reference's own is 1000 — see this file's own docstring
const COUNT = BLOBS.length * PARTICLES_PER_BLOB;

function buildScene(): SceneData {
  const { positions, velocities, F, C, Jp, colors } = allocateScene(COUNT);

  let idx = 0;
  for (const blob of BLOBS) {
    const [r, g, b] = hexToRgb(blob.colorHex);
    for (let k = 0; k < PARTICLES_PER_BLOB; k++) {
      positions[idx * 2] = blob.center[0] + (Math.random() * 2 - 1) * BLOB_HALF_WIDTH;
      positions[idx * 2 + 1] = blob.center[1] + (Math.random() * 2 - 1) * BLOB_HALF_WIDTH;
      setRestState(F, Jp, idx);
      setColor(colors, idx, r, g, b);
      idx++;
    }
  }

  return { count: COUNT, positions, velocities, F, C, Jp, colors };
}

export const blocksWorld: World = {
  id: "blocks",
  label: "Blocks",
  buildScene,
  // No defaults override — gravity starts at 0 (gpu/mpm.ts's own
  // DEFAULT_GRAVITY) like every other world; this file's own docstring's
  // "gravity collapses everything toward the floor" testing was done
  // with the Gravity slider turned up by hand (200), not from a
  // world-level default.
};

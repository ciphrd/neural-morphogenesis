import type { SceneData, World } from "./types";
import { allocateScene, hexToRgb, setColor, setRestState } from "./util";

/** One solid filled disc of particles, centered in the domain — a
 * simple, single coherent "shape" rather than blocks.ts's several
 * independent piles, meant as a starting point for shape-coherency
 * experiments (does it hold together under a tool's interaction? does
 * it drift/deform/settle back?). Zero gravity by default (gpu/mpm.ts's
 * own DEFAULT_GRAVITY, shared by every world now) so it just floats in
 * place until something (a tool, or the Gravity slider turned up by
 * hand) actually acts on it, rather than immediately collapsing into a
 * puddle on the floor.
 *
 * Sampled via `r = RADIUS*sqrt(rand())` (not `RADIUS*rand()`), the
 * standard trick for a uniform-by-*area* disc fill — the naive version
 * without the sqrt biases samples toward the center (area scales with
 * r^2, so linearly-sampled r under-fills the outer rings). Particle
 * count is picked to roughly match blocks.ts's own per-area density
 * (that world's own blob: 5000 particles over a (2*0.08)^2 square), not
 * an independently-chosen number — an untested density here would be
 * exactly the kind of "reintroduces the wobble" mistake this project's
 * git history has already made more than once. */

const CENTER: readonly [number, number] = [0.5, 0.5];
const RADIUS = 0.15;
const COUNT = 14_000;

// Radial color gradient (center -> edge) rather than one flat color —
// makes shear/mixing under a tool's interaction visually legible in a
// way a solid fill wouldn't (a uniformly-colored disc that gets stirred
// still just looks like a disc).
const CENTER_COLOR = 0x7dd3fc;
const EDGE_COLOR = 0x1e6f99;

function buildScene(): SceneData {
  const { positions, velocities, F, C, Jp, colors } = allocateScene(COUNT);
  const [cr, cg, cb] = hexToRgb(CENTER_COLOR);
  const [er, eg, eb] = hexToRgb(EDGE_COLOR);

  for (let i = 0; i < COUNT; i++) {
    const r = RADIUS * Math.sqrt(Math.random());
    const theta = Math.random() * 2 * Math.PI;
    positions[i * 2] = CENTER[0] + r * Math.cos(theta);
    positions[i * 2 + 1] = CENTER[1] + r * Math.sin(theta);
    setRestState(F, Jp, i);
    const t = r / RADIUS;
    setColor(colors, i, cr + (er - cr) * t, cg + (eg - cg) * t, cb + (eb - cb) * t);
  }

  return { count: COUNT, positions, velocities, F, C, Jp, colors };
}

export const discWorld: World = {
  id: "disc",
  label: "Disc",
  buildScene,
  // No defaults override — gravity starts at 0 (gpu/mpm.ts's own
  // DEFAULT_GRAVITY) like every other world now, which happens to
  // already be what this world wants (see its own docstring above).
};

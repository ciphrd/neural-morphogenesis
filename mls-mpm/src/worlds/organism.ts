import type { SceneData, World } from "./types";
import { allocateScene, hexToRgb, setColor, setRestState } from "./util";

/** A single elongated, semi-rigid soft body — a capsule/"stadium" shape
 * (a straight segment with two rounded caps), meant to be grabbed from
 * one end with the Move tool (see tools/types.ts) and pulled: it should
 * stretch, then relax back toward its original shape once released,
 * rather than staying permanently deformed the way worlds/blocks.ts's
 * snow-like material does.
 *
 * That "springs back" behavior is NOT something this file configures on
 * its own — it comes entirely from `defaults.elasticity: 1` below (see
 * gpu/mpm.ts's own yieldBounds() docstring): the reference's own
 * plasticity clamp is snow-tight (a particle stretched more than ~0.75%
 * along any axis gets that excess baked in as *permanent* deformation,
 * never recovered by the elastic term), which is exactly wrong for an
 * "organism" that should deform elastically under ordinary manipulation.
 * Widening that clamp is what lets the corotated elastic stress term
 * (which is always pulling F back toward the identity, at a rate set by
 * `stiffness`) actually do its job instead of having its work
 * overwritten by the plasticity clamp on every single step.
 *
 * Particle density matches worlds/disc.ts's own (same reasoning as that
 * file: an independently-chosen density here would risk reintroducing
 * the persistent-jitter problem this project's git history has already
 * hit more than once from exactly that mistake) — not derived
 * automatically, just computed by hand from the same target ratio. */

const CENTER_Y = 0.5;
const SEGMENT_X0 = 0.3;
const SEGMENT_X1 = 0.7;
const RADIUS = 0.05;
const COUNT = 9_000;

const HEAD_COLOR = 0xf2b134;
const TAIL_COLOR = 0x068587;

function buildScene(): SceneData {
  const { positions, velocities, F, C, Jp, colors } = allocateScene(COUNT);
  const [hr, hg, hb] = hexToRgb(HEAD_COLOR);
  const [tr, tg, tb] = hexToRgb(TAIL_COLOR);

  const minX = SEGMENT_X0 - RADIUS;
  const maxX = SEGMENT_X1 + RADIUS;
  const minY = CENTER_Y - RADIUS;
  const maxY = CENTER_Y + RADIUS;
  const segLen = SEGMENT_X1 - SEGMENT_X0;

  let i = 0;
  while (i < COUNT) {
    const x = minX + Math.random() * (maxX - minX);
    const y = minY + Math.random() * (maxY - minY);
    // Closest point on the central segment (clamped projection), then a
    // plain circle-radius test against it — the standard capsule/
    // stadium SDF via rejection sampling.
    const t = Math.min(Math.max((x - SEGMENT_X0) / segLen, 0), 1);
    const nx = SEGMENT_X0 + t * segLen;
    const ny = CENTER_Y;
    const dist = Math.hypot(x - nx, y - ny);
    if (dist > RADIUS) continue;

    positions[i * 2] = x;
    positions[i * 2 + 1] = y;
    setRestState(F, Jp, i);
    // Head-to-tail gradient (by position along the segment, not `t`
    // above which is clamped into the caps too) — makes which end is
    // which unambiguous at a glance, and makes shear/stretch visually
    // legible once it's grabbed and pulled.
    const gradT = Math.min(Math.max((x - SEGMENT_X0) / segLen, 0), 1);
    setColor(colors, i, hr + (tr - hr) * gradT, hg + (tg - hg) * gradT, hb + (tb - hb) * gradT);
    i++;
  }

  return { count: COUNT, positions, velocities, F, C, Jp, colors };
}

export const organismWorld: World = {
  id: "organism",
  label: "Organism",
  buildScene,
  // NOT yet verified against a long CPU-side stability run the way
  // worlds/blocks.ts's own defaults were (see that file's own
  // docstring) — this project's git history has hit "stacked untested
  // parameter changes at once" as a real, immediate-explosion failure
  // mode before (see index.html's hardening-slider comment, gpu/mpm.ts's
  // DT comment), and a first pass at this world (stiffness:15000,
  // poisson:0.3, elasticity:1 all at once) reproduced exactly that —
  // instant scatter, not a slow drift. stiffness/poisson here are
  // deliberately at-or-below worlds/blocks.ts's own already-verified
  // values rather than above them; elasticity is 0.5, not 1.0 (a real
  // widening of the plasticity clamp from the snow-tight default, but
  // not the most extreme setting, since that clamp was quietly doing
  // double duty as a numerical stabilizer for the two worlds this was
  // actually tuned against — removing it entirely removes that margin
  // too). Treat these as a starting point for further tuning, not a
  // verified-stable endpoint.
  // Gravity not overridden — starts at 0 (gpu/mpm.ts's own
  // DEFAULT_GRAVITY), same as every other world.
  defaults: { stiffness: 8_000, poisson: 0.2, hardening: 0, elasticity: 0.5 },
};

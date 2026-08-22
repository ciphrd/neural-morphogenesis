import type { SceneData, World } from "./types";
import { allocateScene } from "./util";

/** Starts with zero particles — unlike blocks.ts/disc.ts (a pre-built
 * shape from the first frame), this world's whole premise is the "Add"
 * tool (tools/types.ts): click/drag to spawn particles at the cursor via
 * MpmSimulation.addParticles() and watch the material accumulate and
 * settle as it grows, rather than starting from a fixed initial
 * condition. Gravity starts at 0 like every other world (see this
 * world's own `defaults` below) — spawned particles stay exactly where
 * they're placed until the Gravity slider is turned up by hand. */
function buildScene(): SceneData {
  const { positions, velocities, F, C, Jp, colors } = allocateScene(0);
  return { count: 0, positions, velocities, F, C, Jp, colors };
}

export const growthWorld: World = {
  id: "growth",
  label: "Growth",
  buildScene,
  // Gravity NOT overridden — starts at 0 (gpu/mpm.ts's own
  // DEFAULT_GRAVITY), same as every other world, so newly added
  // particles stay exactly where they're placed rather than immediately
  // falling; turn the Gravity slider up by hand to watch a growing
  // structure settle instead. particleSize IS overridden, well past the
  // 1px default — particles here get placed one (or a short "string" of
  // a few) at a time via the "Add Particles" tool rather than in a dense
  // thousands-strong cloud, so the default 1px squares blocks.ts/disc.ts
  // rely on would be all but invisible. stiffness/elasticity/
  // repulsionStrength are hand-tuned live (values read off a screenshot
  // of a session that had settled on a good feel for this world, not
  // re-derived from anything) — a bit stiffer and considerably more
  // elastic than the global Material defaults, so grown structures hold
  // their own shape rather than behaving like snow, and repulsion much
  // gentler than the global default: this world's own particles get
  // placed one at a time right next to whatever's already there, so the
  // strength that suits an already-dense cloud drifting apart too fast
  // is far too strong for "keep freshly-placed particles from
  // overlapping." poisson/hardening/damping/field resolution/splat
  // radius matched the global defaults exactly in that same screenshot,
  // so they're left unset here rather than pinned to a value that'd
  // silently drift out of sync if those globals ever change.
  defaults: { particleSize: 6, stiffness: 11000, elasticity: 0.65, repulsionStrength: 0.005 },
};

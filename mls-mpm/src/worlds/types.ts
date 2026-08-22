/** A world's initial particle state — same fields scene.ts used to
 * export directly, now carrying its own `count` explicitly (worlds vary
 * in how many particles they seed) rather than relying on a single
 * module-level PARTICLE_COUNT constant. Every array is sized to `count`
 * exactly, not to gpu/mpm.ts's MAX_PARTICLES cap — MpmSimulation.reset()
 * uploads it into the head of its (fixed-capacity) buffers and sets its
 * own activeCount uniform from `count`, so a world never needs to know
 * about that cap. */
export interface SceneData {
  count: number;
  positions: Float32Array; // (count,2)
  velocities: Float32Array; // (count,2), all zero
  F: Float32Array; // (count,4) row-major (m00,m01,m10,m11), identity
  C: Float32Array; // (count,4), zero
  Jp: Float32Array; // (count,), ones
  colors: Float32Array; // (count,4), rgba
}

/** Suggested slider values a world switch applies on top of
 * MpmSimulation's own hard-coded defaults (gpu/mpm.ts's
 * DEFAULT_GRAVITY/DEFAULT_E/...) — e.g. a "drop" world wants gravity on,
 * a free-floating shape-coherency world might want it off. Every field
 * optional: main.ts only touches the sliders a world actually specifies
 * an opinion on, leaving the others at whatever the user last set. */
export interface WorldDefaults {
  gravity?: number;
  stiffness?: number;
  poisson?: number;
  hardening?: number;
  /** 0..1 — see gpu/mpm.ts's yieldBounds() for what this actually
   * controls (how wide a stretch/compression the material's plasticity
   * clamp allows before treating it as permanent rather than lettng the
   * elastic term spring it back). Omitted = DEFAULT_ELASTICITY (0, the
   * reference's own snow-tight bounds) — worlds/blocks.ts and disc.ts
   * both rely on that default rather than setting it explicitly;
   * worlds/organism.ts is the one that actually needs this wide open. */
  elasticity?: number;
  /** Fraction of velocity lost per rendered frame (0..~1) — see
   * gpu/mpm.ts's DEFAULT_DAMPING/perSubstepDamping() for the exact
   * semantics and gridUpdate.wgsl's own comment for the real tradeoff
   * this trades against (calms at-rest jitter, but measurably
   * destabilizes active manipulation — not a free win at any value).
   * Omitted = DEFAULT_DAMPING (~6%, reproducing the project's original
   * hardcoded per-substep constant exactly). */
  damping?: number;
  /** Device-pixel particle radius — see gpu/render.ts's own
   * DEFAULT_POINT_RADIUS_PX docstring. Omitted = that 1px default, right
   * for blocks.ts/disc.ts's dense clouds; worlds/growth.ts is the one
   * that actually needs this bigger (particles placed one at a time via
   * the "Add Particles" tool are otherwise too small to see). */
  particleSize?: number;
  /** Force-scale multiplier on repulsion.wgsl's own density-field
   * gradient — see gpu/mpm.ts's DEFAULT_REPULSION_STRENGTH docstring for
   * the mechanism and its own known tradeoff. Omitted = that global
   * default (0.1); worlds/growth.ts is the one that wants this much
   * gentler (0.005) — its own particles are placed one at a time right
   * next to whatever's already there, so a strength tuned for "keep an
   * already-dense cloud from drifting apart too fast" is far too strong
   * for "gently keep freshly-placed particles from overlapping." */
  repulsionStrength?: number;
}

/** One pluggable initial condition + interaction context — the unit
 * main.ts's World dropdown switches between. `buildScene()` is called
 * fresh on every select/Reset (not cached), so it can use randomness
 * (jittered particle placement) without every reset looking identical. */
export interface World {
  id: string;
  label: string;
  buildScene(): SceneData;
  defaults?: WorldDefaults;
}

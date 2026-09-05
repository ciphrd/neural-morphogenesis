import densityModel from "../../../core/density.json";

// Deterministic, bit-exact PRNG for a rollout's ENTIRE starting condition
// — spawn-position jitter (seedBlob() below), back-to-back theta
// (gpu/simulation.ts's own restartRollout()), and heading's own per-slot
// fill (Agents.resetHeading()) all derive from a portable integer hash,
// mirrored exactly by ../../../trainer/agents_gpu.py's own
// _spawn_uniform01()/_spawn_uniform01_batch() — no numpy Generator or
// mulberry32 stream involved on either side anymore. This used to be a
// deliberately-accepted, NOT-bit-exact gap (mulberry32 here vs numpy's
// own Generator/PCG64 on the Python trainer) — "a replay only needs to
// *look* like a plausible rollout from the same seed, not reproduce the
// Python trainer's exact float sequence" — but that gap turned out to
// matter more than expected: since MLS-MPM elastic material + repulsion
// + growth is a real, chaotic dynamical system, even a tiny difference
// in starting position/heading compounds over a rollout's own macro
// steps into a visibly different (if structurally similar) final shape,
// not just cosmetic noise. Fully closing it (this file, matching
// growthSeed() below's own already-bit-exact precedent) is what actually
// makes a frontend replay reproduce the exact same rollout a checkpoint
// was trained under.

// Bit-exact, portable integer hash (Chris Wellons' "lowbias32" — public
// domain), mirrored exactly by ../../../trainer/agents_gpu.py's own
// _hash_u32(). Only uses uint32 add/xor/shift/multiply-with-wraparound,
// so it's trivial to reproduce exactly in numpy — Math.imul (forced back
// to unsigned via >>> 0) wraps mod 2**32 identically to numpy's own
// uint32 dtype.
function hashU32(x: number): number {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

const SPATIAL_HEADING_DOMAIN = 0x48454144;

/** Fixed world-space random field shared by every sampling density. */
export function spatialUniform01(seed: number, x: number, y: number, domain = SPATIAL_HEADING_DOMAIN): number {
  const cells = densityModel.SPATIAL_RANDOM_CELLS;
  const cellX = Math.floor((((x % 1) + 1) % 1) * cells) >>> 0;
  const cellY = Math.floor((((y % 1) + 1) % 1) * cells) >>> 0;
  const combined = (
    (seed >>> 0)
    ^ hashU32((cellX + 0x9e3779b9) >>> 0)
    ^ hashU32((cellY + 0x85ebca6b) >>> 0)
    ^ (domain >>> 0)
  ) >>> 0;
  return (hashU32(combined) >>> 8) / 16777216;
}

/** particleMeta.rng's own initial per-particle seed — bit-exact with
 * agents_gpu.py's own _growth_seed(seed, count). `seed` is the
 * rollout's own raw seed (config.seed on this side, matching evolve.py's
 * own rollout(seed, ...) argument on the Python side). A DELIBERATELY
 * SEPARATE hash domain from spawnUniform01() below (no shared magic
 * constant) — see that function's own comment for why the two must
 * never correlate despite both being bit-exact now: growth is a near-
 * critical branching process (agentStep()'s own split-decision logic),
 * so even a merely-correlated seed stream risks a systematic bias in
 * which particles tend to split together. Nonzero always (xorshift32's
 * own fixed point at 0 — core/agents.wgsl's own comment). */
export function growthSeed(seed: number, index: number): number {
  const combined = ((seed >>> 0) ^ hashU32((index + 1) >>> 0)) >>> 0;
  return hashU32(combined) || 1;
}

// Magic domain-separator XOR'd into the index before hashing — keeps
// spawnUniform01() below's own output space disjoint from growthSeed()
// above even when both happen to be called with the same (seed, index)
// pair (seedBlob()'s/resetHeading()'s own indices are small integers,
// the same range growthSeed() iterates particle slots over) — mirrors
// ../../../trainer/agents_gpu.py's own _SPAWN_HASH_DOMAIN exactly (must
// match bit-for-bit). Arbitrary, just needs to be nonzero.
const SPAWN_HASH_DOMAIN = 0xc0ffee00;

/** One deterministic float in [0,1), bit-exact with
 * ../../../trainer/agents_gpu.py's own _spawn_uniform01(seed, index) —
 * the portable hash EVERY piece of a rollout's own starting-condition
 * randomness that ISN'T growth now goes through: seedBlob() below's own
 * spawn-position jitter, gpu/simulation.ts's own back-to-back theta, and
 * Agents.resetHeading()'s own per-slot heading fill. Domain-separated
 * from growthSeed() above via SPAWN_HASH_DOMAIN (see that constant's own
 * comment). Top 24 bits of the hash -> a uniform float, same "use every
 * bit of f32 mantissa precision" convention core/agents.wgsl's own
 * xorshift32-derived draw already uses
 * (`f32(rngNext >> 8u) * (1.0/16777216.0)`). */
export function spawnUniform01(seed: number, index: number): number {
  const combined = ((seed >>> 0) ^ hashU32((SPAWN_HASH_DOMAIN ^ index) >>> 0)) >>> 0;
  const hashed = hashU32(combined);
  return (hashed >>> 8) / 16777216;
}

export interface SeedBlobConfig {
  count: number;
  centerX: number;
  centerY: number;
  spacing: number;
  seed: number;
}

/** Exact, axis-aligned rows for deterministic lab scenarios. Unlike seedBlob,
 * this layout has no packing scale or seed-derived rotation: adjacent cells
 * are separated by the simulation's actual daughter split distance. Rows are
 * emitted bottom-to-top and columns left-to-right, making cell indices stable. */
export function seedRows(config: {
  rows: number;
  columns: number;
  centerX: number;
  centerY: number;
  spacing: number;
}) {
  const { rows, columns, centerX, centerY, spacing } = config;
  const count = rows * columns;
  const positions = new Float32Array(count * 2);
  const velocities = new Float32Array(count * 2);
  const F = new Float32Array(count * 4);
  const C = new Float32Array(count * 4);
  const Jp = new Float32Array(count).fill(1);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      positions[index * 2] = centerX + (column - (columns - 1) / 2) * spacing;
      positions[index * 2 + 1] = centerY + (row - (rows - 1) / 2) * spacing;
      F[index * 4] = 1;
      F[index * 4 + 3] = 1;
    }
  }
  const domain = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) { domain[i * 4] = spacing / 2; domain[i * 4 + 3] = spacing / 2; }
  return { count, positions, velocities, F, C, Jp, domain };
}

/** Circular clipping of a perfect hexagonal lattice, mirrored by
 * trainer/training_sim.py's seed_blob(). Sites fill in exact Euclidean-radius
 * shells (the axial metric q²+qr+r²), rather than hex-coordinate rings whose
 * outer contour is a hexagon. A partial final shell samples sites evenly around
 * its circumference. The whole disk receives one deterministic seed-derived
 * rotation. `spacing` remains the later daughter split distance; initial
 * nearest neighbors use the shared compact-packing scale from density.json. */
export function seedBlob(config: SeedBlobConfig) {
  const { count, centerX, centerY, spacing, seed } = config;
  const packedSpacing = spacing * densityModel.INITIAL_PACKING_SPACING_SCALE;

  const limit = Math.ceil(Math.sqrt(count)) + 2;
  const shells = new Map<number, Array<[number, number]>>();
  for (let q = -limit; q <= limit; q++) {
    for (let r = -limit; r <= limit; r++) {
      const radiusSquared = q * q + q * r + r * r;
      const shell = shells.get(radiusSquared) ?? [];
      shell.push([packedSpacing * (q + 0.5 * r), packedSpacing * (Math.sqrt(3) * 0.5 * r)]);
      shells.set(radiusSquared, shell);
    }
  }
  const offsets: Array<[number, number]> = [];
  for (const radiusSquared of Array.from(shells.keys()).sort((a, b) => a - b)) {
    if (offsets.length >= count) break;
    const shell = shells.get(radiusSquared)!;
    shell.sort((a, b) => Math.atan2(a[1], a[0]) - Math.atan2(b[1], b[0]));
    const take = Math.min(count - offsets.length, shell.length);
    if (take === shell.length) offsets.push(...shell);
    else {
      for (let j = 0; j < take; j++) {
        offsets.push(shell[Math.floor((j + 0.5) * shell.length / take)]);
      }
    }
  }
  const meanX = offsets.reduce((sum, p) => sum + p[0], 0) / count;
  const meanY = offsets.reduce((sum, p) => sum + p[1], 0) / count;
  // Keep world orientation identical when density changes `count`. Index 2 is
  // the seed-blob rotation domain; per-particle headings begin at index 5.
  const theta = (spawnUniform01(seed, 2) * 2 - 1) * Math.PI;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  const positions = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const x = offsets[i][0] - meanX;
    const y = offsets[i][1] - meanY;
    positions[i * 2] = ((centerX + x * cosTheta - y * sinTheta) % 1 + 1) % 1;
    positions[i * 2 + 1] = ((centerY + x * sinTheta + y * cosTheta) % 1 + 1) % 1;
  }
  const velocities = new Float32Array(count * 2); // zero
  const F = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    F[i * 4] = 1;
    F[i * 4 + 1] = 0;
    F[i * 4 + 2] = 0;
    F[i * 4 + 3] = 1;
  }
  const C = new Float32Array(count * 4); // zero
  const Jp = new Float32Array(count).fill(1);

  const domain = new Float32Array(count * 4);
  const a = packedSpacing / 2;
  const b = packedSpacing / 4;
  const d = packedSpacing * Math.sqrt(3) / 4;
  for (let i = 0; i < count; i++) {
    domain.set([cosTheta*a, cosTheta*b-sinTheta*d, sinTheta*a, sinTheta*b+cosTheta*d], i*4);
  }
  return { count, positions, velocities, F, C, Jp, domain };
}

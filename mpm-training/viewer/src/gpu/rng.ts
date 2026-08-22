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
  halfWidth: number;
  seed: number;
}

/** Mirrors trainer/training_sim.py's own seed_blob(): `count` particles
 * jittered uniformly within `halfWidth` of (centerX,centerY), zero
 * velocity, identity F, zero C, Jp=1 — the standard MpmCore.loadScene()
 * scene shape every rollout (training or replay) starts from. Jittered
 * via spawnUniform01(seed, 2*i)/2*i+1 for particle i's own x/y — bit-
 * exact with the Python trainer's own seed_blob(), see that function's
 * own docstring. Indices 0..2*count-1 are reserved for this function's
 * own draws — gpu/simulation.ts's own theta draw (the back-to-back
 * placement) starts at index 2*count to never collide, regardless of
 * `count`. */
export function seedBlob(config: SeedBlobConfig) {
  const { count, centerX, centerY, halfWidth, seed } = config;

  const positions = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    positions[i * 2] = centerX + (spawnUniform01(seed, 2 * i) * 2 - 1) * halfWidth;
    positions[i * 2 + 1] = centerY + (spawnUniform01(seed, 2 * i + 1) * 2 - 1) * halfWidth;
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

  return { count, positions, velocities, F, C, Jp };
}

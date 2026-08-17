/** Browser-side agent-jitter seeding — a *deterministic* PRNG (not
 * torch's), seeded with the real `seed` train_server.py now reports
 * (evolve.py's run_generation() returns the winner's actual seed — see
 * that function's docstring).
 *
 * Known gap, not silently absorbed: agent_state.py::seed() draws its
 * jitter via `torch.rand(..., generator=torch.Generator().manual_seed(
 * seed))` — PyTorch's CPU Mersenne-Twister generator plus its own
 * uint->uniform-float conversion. mulberry32 below is a different
 * algorithm; seeding it with the same integer produces a *different*
 * bit sequence, not torch's actual jitter. True bit-exact reproduction
 * would mean porting PyTorch's exact CPU RNG to JS/WGSL, which this does
 * not attempt. What this does buy: the real seed (not a synthetic
 * generation-derived stand-in) drives a stable, reproducible-per-
 * generation jitter in the browser — replaying the same generation twice
 * looks identical, and it's seeded by the actual value training used,
 * just not bit-identical to what training actually drew. Given
 * spread is small (a few pixels) and this is the only source of
 * randomness in an otherwise fully deterministic rollout, the visual
 * difference should be negligible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Frontend-only viewer setting — doesn't exist server-side, since
 * training's own rollout() always uses agent_state.py's actual jitter
 * (this project's "default" below). These other two are for visually
 * exploring how a trained update rule behaves from a different initial
 * spread than it was ever trained on, not for reproducing training. */
export type SpawnDistribution = "default" | "square" | "quarterSquare";

/** `"default"` mirrors agent_state.py::AgentState.seed()'s jitter formula
 * exactly (`offsets = (rand(n,2) - 0.5) * 2 * spread`, centered on the
 * grid center) — just with the RNG-algorithm caveat above. The other two
 * distributions are viewer-only alternatives, uniform over a square
 * region instead of a small jittered point:
 * - `"square"`: uniform across the *entire* grid extent.
 * - `"quarterSquare"`: uniform across a centered square spanning half the
 *   grid's width/height on each axis — a quarter of the grid's total
 *   area, not a quarter of its side length.
 *
 * Returns a flat (n*2) Float32Array, [x0,y0,x1,y1,...], ready to upload
 * to the `positions` storage buffer. */
export function seedAgentPositions(
  n: number,
  gridWidth: number,
  gridHeight: number,
  spread: number,
  seed: number,
  distribution: SpawnDistribution = "default"
): Float32Array {
  const rand = mulberry32(seed);
  const positions = new Float32Array(n * 2);
  const centerX = gridWidth / 2;
  const centerY = gridHeight / 2;

  for (let i = 0; i < n; i++) {
    let x: number;
    let y: number;
    if (distribution === "square") {
      x = rand() * gridWidth;
      y = rand() * gridHeight;
    } else if (distribution === "quarterSquare") {
      const halfExtentX = gridWidth / 4;
      const halfExtentY = gridHeight / 4;
      x = centerX + (rand() - 0.5) * 2.0 * halfExtentX;
      y = centerY + (rand() - 0.5) * 2.0 * halfExtentY;
    } else {
      x = centerX + (rand() - 0.5) * 2.0 * spread;
      y = centerY + (rand() - 0.5) * 2.0 * spread;
    }
    positions[i * 2] = x;
    positions[i * 2 + 1] = y;
  }
  return positions;
}

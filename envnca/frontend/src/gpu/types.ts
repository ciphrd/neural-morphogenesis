/** JSON-ready weights matching update_rule.py's export_weights() — see
 * that method's docstring for the (out_features, in_features) row-major
 * orientation convention (`y = x @ W.T + b`). */
export interface UpdateRuleWeights {
  fc1w: number[][]; // (hiddenDim, 3*channels)
  fc1b: number[]; // (hiddenDim,)
  fc2w: number[][]; // (channels+4, hiddenDim)
  fc2b: number[]; // (channels+4,)
}

/** Everything one WebGPU rollout needs to reproduce a generation's
 * winner, mirroring train_server.py's per-generation broadcast message
 * one field at a time so nothing here duplicates a constant the backend
 * already owns (grid size, physics-ish constants, weights). */
export interface SimulationConfig {
  weights: UpdateRuleWeights;
  gridWidth: number;
  gridHeight: number;
  channels: number;
  agentCount: number;
  spawnSpread: number;
  steps: number;
  decay: number;
  maxSpeed: number;
  maxAccel: number;
  maxStrafe: number;
  edgeMargin: number;
  hiddenDim: number;
  // The real seed torch used to jitter this generation's winning
  // rollout (evolve.py's run_generation() now reports it) — the
  // frontend's own jitter PRNG is seeded with this value for a
  // reproducible-per-generation (but not bit-exact, see gpu/rng.ts)
  // replay.
  seed: number;
}

/** How gpu/render.ts's colorize pass fills the canvas background — see
 * colorize.wgsl's own comment for the mode->color mapping. */
export type BackgroundMode = "gray" | "black" | "substrate";

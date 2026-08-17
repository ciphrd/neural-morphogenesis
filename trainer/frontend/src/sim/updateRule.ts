/**
 * TypeScript port of trainer/backend/update_rule.py's UpdateRule
 * forward pass — hand-rolled rather than via onnxruntime-web/tfjs since
 * the network is just two small matmuls with a tanh in between
 * (36->128->20). Weights come
 * from the backend's UpdateRule.export_weights(), which documents the
 * (out_features, in_features) orientation this expects: `y = x @ Wᵀ + b`,
 * i.e. `y[o] = sum_i(W[o][i] * x[i]) + b[o]`.
 */

import { ID_DIM, MOTION_DIM, NUM_CHEMICAL_CHANNELS, SPAWN_DIR_DIM } from "./cellState";

export interface UpdateRuleWeights {
  fc1w: number[][];
  fc1b: number[];
  fc2w: number[][];
  fc2b: number[];
}

export interface UpdateRuleOutput {
  splitProb: number;
  chemicalDelta: number[];
  idDelta: number[];
  /** Raw (not normalized) — mirrors update_rule.py's step_numpy: which
   * way this node would place a child *if* it split this step, always
   * computed regardless of whether the energy-gated split draw actually
   * lands. Normalizing (and falling back to a random angle when this is
   * too close to zero to have a direction) is the caller's job — see
   * graph.ts's addChild and runner.ts's simStep. */
  spawnDirection: number[];
  /** Raw (not tanh-squashed/scaled) — mirrors update_rule.py's
   * step_numpy: a 2D acceleration in the node's own *local* frame
   * ([0] = forward, [1] = lateral), integrated onto the node's running
   * velocity after being rotated into world coordinates — not a direct
   * velocity or heading. Squashing each component
   * (`tanh(v) * maxAccel`), rotating into world space, and clamping the
   * resulting velocity's *magnitude* to maxSpeed are all the caller's
   * job — see runner.ts's simStep. */
  localAccel: [number, number];
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function linear(x: number[], w: number[][], b: number[]): number[] {
  const out = new Array(w.length);
  for (let o = 0; o < w.length; o++) {
    let sum = b[o];
    const row = w[o];
    for (let i = 0; i < row.length; i++) sum += row[i] * x[i];
    out[o] = sum;
  }
  return out;
}

/**
 * Input order must match update_rule.py's forward():
 * [chemicals, gradForward, gradLateral]. No energy — see
 * update_rule.py's "Energy" docstring section: the network is
 * deliberately blind to its own energy level. `gradForward`/
 * `gradLateral` must already be rotated into the node's own local
 * frame (forward = heading direction, lateral = 90° left of it) by the
 * caller — this function itself is frame-agnostic, it just consumes
 * whatever two gradient components it's handed, same as the Python
 * side's forward().
 */
export function forward(
  weights: UpdateRuleWeights,
  chemicals: number[],
  gradForward: number[],
  gradLateral: number[]
): UpdateRuleOutput {
  const x = [...chemicals, ...gradForward, ...gradLateral];
  const hidden = linear(x, weights.fc1w, weights.fc1b).map(Math.tanh);
  const out = linear(hidden, weights.fc2w, weights.fc2b);

  const idEnd = 1 + NUM_CHEMICAL_CHANNELS + ID_DIM;
  const spawnDirEnd = idEnd + SPAWN_DIR_DIM;
  const motionEnd = spawnDirEnd + MOTION_DIM;
  return {
    splitProb: sigmoid(out[0]),
    chemicalDelta: out.slice(1, 1 + NUM_CHEMICAL_CHANNELS),
    idDelta: out.slice(1 + NUM_CHEMICAL_CHANNELS, idEnd),
    spawnDirection: out.slice(idEnd, spawnDirEnd),
    // Same (2-wide block) treatment as spawnDirection just above, not
    // two separate scalar reads — see MOTION_DIM's own comment.
    localAccel: out.slice(spawnDirEnd, motionEnd) as [number, number],
  };
}

/**
 * TypeScript port of the single-bandwidth case of
 * trainer/backend/substrate.py's weighted_field_and_gradient — the only
 * case the update rule actually uses (sensing the gradient of each
 * chemical channel's locally-diffused value, not the multi-scale
 * density field that's only used for the backend's /substrate
 * inspection endpoint).
 */

import type { Vec2 } from "./physics";

/** Returns gradients[queryIndex][channel] = [dValue/dx, dValue/dy]. */
export function weightedFieldGradient(positions: Vec2[], weights: number[][], sigma: number): Vec2[][] {
  const n = positions.length;
  const channels = weights.length > 0 ? weights[0].length : 0;
  const sigma2 = sigma * sigma;

  const gradients: Vec2[][] = Array.from({ length: n }, () =>
    Array.from({ length: channels }, () => [0, 0] as Vec2)
  );

  for (let i = 0; i < n; i++) {
    for (let s = 0; s < n; s++) {
      const dx = positions[s][0] - positions[i][0];
      const dy = positions[s][1] - positions[i][1];
      const sqDist = dx * dx + dy * dy;
      const kernel = Math.exp(-sqDist / (2 * sigma2));
      const w = kernel / sigma2;
      for (let k = 0; k < channels; k++) {
        const weighted = w * weights[s][k];
        gradients[i][k][0] += -weighted * dx;
        gradients[i][k][1] += -weighted * dy;
      }
    }
  }

  return gradients;
}

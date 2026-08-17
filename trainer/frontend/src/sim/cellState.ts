/** TypeScript port of trainer/backend/cell_state.py — keep in sync by hand. */

export const ID_DIM = 3;
export const NUM_CHEMICAL_CHANNELS = 12;
export const SPAWN_DIR_DIM = 2;
// Not part of the persistent state vector, same footing as
// SPAWN_DIR_DIM — a 2D local-frame acceleration the network emits every
// step, rotated into world coordinates and integrated onto
// SimGraph.velocities rather than read back as raw network output next
// step. See update_rule.py's "Velocity & heading" docstring section.
export const MOTION_DIM = 2;

function randomUniform(dim: number): number[] {
  return Array.from({ length: dim }, () => Math.random() * 2 - 1);
}

export function randomId(): number[] {
  return randomUniform(ID_DIM);
}

export function randomChemical(): number[] {
  return randomUniform(NUM_CHEMICAL_CHANNELS);
}

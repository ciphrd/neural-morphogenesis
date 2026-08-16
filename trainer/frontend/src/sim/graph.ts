/** TypeScript port of the parts of trainer/backend/graph.py needed for a client-side replay. */

import { CONTACT_DISTANCE, type Vec2 } from "./physics";
import { randomChemical, randomId, SPAWN_DIR_DIM } from "./cellState";

export interface SimGraph {
  positions: Vec2[];
  idVectors: number[][];
  chemicals: number[][];
  energy: number[];
  // Latest spawn-direction reading per node (unit 2D vector, or [0, 0]
  // for a node that hasn't run through the update rule yet) — see
  // update_rule.py's module docstring for why this isn't accumulated
  // state the way id/chemicals are.
  spawnDirections: number[][];
  // Latest split-probability reading per node (raw sigmoid(split_logit),
  // before the energy gate scales it down) — 0 for a node that hasn't
  // run through the update rule yet. Same "fresh reading every step"
  // treatment as spawnDirections, for the same reason.
  splitProbs: number[];
  pinned: Set<number>;
}

export function seedGraph(initialEnergy: number): SimGraph {
  return {
    positions: [[0, 0]],
    idVectors: [randomId()],
    chemicals: [randomChemical()],
    energy: [initialEnergy],
    spawnDirections: [new Array(SPAWN_DIR_DIM).fill(0)],
    splitProbs: [0],
    pinned: new Set(),
  };
}

/**
 * Spawns a child touching `parentId`, mirroring Graph.add_child.
 * `direction` (raw, not assumed normalized — the update rule's own
 * learned spawn-direction output) steers where the child lands; a
 * random angle is used instead when omitted or too close to zero to
 * normalize (the network expressed no preference this step).
 */
export function addChild(
  graph: SimGraph,
  parentId: number,
  idVector: number[],
  chemicals: number[],
  energy: number,
  direction?: number[]
): number {
  const origin = graph.positions[parentId];
  let unit: Vec2 | null = null;
  if (direction) {
    const norm = Math.hypot(...direction);
    if (norm >= 1e-9) unit = [direction[0] / norm, direction[1] / norm];
  }
  if (!unit) {
    const angle = Math.random() * Math.PI * 2;
    unit = [Math.cos(angle), Math.sin(angle)];
  }
  const newId = graph.positions.length;
  graph.positions.push([origin[0] + unit[0] * CONTACT_DISTANCE, origin[1] + unit[1] * CONTACT_DISTANCE]);
  graph.idVectors.push(idVector);
  graph.chemicals.push(chemicals);
  graph.energy.push(energy);
  graph.spawnDirections.push(new Array(SPAWN_DIR_DIM).fill(0));
  graph.splitProbs.push(0);
  return newId;
}

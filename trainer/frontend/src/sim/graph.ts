/** TypeScript port of the parts of trainer/backend/graph.py needed for a client-side replay. */

import type { Vec2 } from "./physics";
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
  // Persistent, accumulated state — same footing as idVectors/chemicals,
  // not a fresh-every-step reading like spawnDirections/splitProbs. Each
  // entry is [vx, vy] in world coordinates — the *only* motion state
  // stored; there is no separate heading field. "Which way is this node
  // facing" is always derived on demand as atan2(vy, vx), never read
  // from here directly. See update_rule.py's "Velocity & heading"
  // docstring section.
  velocities: Vec2[];
  // Latest world-frame acceleration reading per node — the network's
  // raw per-step output (after tanh-squash/maxAccel/local-to-world
  // rotation, see simStep()), *before* it's added to velocity. Same
  // "fresh reading every step, not accumulated state" treatment as
  // spawnDirections/splitProbs — exists purely for display (see
  // GraphRenderer.tsx's always-on velocity/accel ticks).
  accels: Vec2[];
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
    velocities: [[0, 0]],
    accels: [[0, 0]],
    pinned: new Set(),
  };
}

/**
 * Spawns a child touching `parentId`, mirroring Graph.add_child.
 * `direction` (raw, not assumed normalized — the update rule's own
 * learned spawn-direction output) steers where the child lands; a
 * random angle is used instead when omitted or too close to zero to
 * normalize (the network expressed no preference this step).
 * `contactDistance` comes from the server (see physics.ts's
 * PhysicsConfig) rather than a hardcoded local constant, same reason as
 * everywhere else this codebase used to duplicate it.
 *
 * The child starts at rest ([0, 0] velocity) regardless of which way
 * `unit` pointed it — heading being purely derived from velocity means
 * there's no "facing" to inherit separately from motion; it'll pick up
 * its own heading the moment it actually starts accelerating, same as
 * every other node.
 */
export function addChild(
  graph: SimGraph,
  parentId: number,
  idVector: number[],
  chemicals: number[],
  energy: number,
  contactDistance: number,
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
  graph.positions.push([
    origin[0] + unit[0] * contactDistance,
    origin[1] + unit[1] * contactDistance,
  ]);
  graph.idVectors.push(idVector);
  graph.chemicals.push(chemicals);
  graph.energy.push(energy);
  graph.spawnDirections.push(new Array(SPAWN_DIR_DIM).fill(0));
  graph.splitProbs.push(0);
  graph.velocities.push([0, 0]);
  graph.accels.push([0, 0]);
  return newId;
}

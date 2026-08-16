/**
 * Orchestrates one full client-side replay of a trained UpdateRule:
 * seed -> for `steps` iterations, sense -> decide -> act -> physics
 * settle -> repeat. Mirrors update_rule.py's step() semantics exactly
 * (one consistent pre-step snapshot, post-delta state copied to
 * children, only the original snapshot's nodes get a chance to split
 * this step, energy gates the network's own split probability rather
 * than replacing it) plus physics.py's relax(), but animated: physics
 * iterations are streamed a few at a time via requestAnimationFrame so
 * the settling motion is visible, not just the converged result.
 */

import type { GraphNode } from "../net/socket";
import { relaxSteps, type Vec2 } from "./physics";
import { weightedFieldGradient } from "./substrate";
import { forward, type UpdateRuleWeights } from "./updateRule";
import { addChild, seedGraph, type SimGraph } from "./graph";

// Pacing constants for the replay animation — purely visual, no physics
// or training significance.
const PHYSICS_ITERATIONS_PER_FRAME = 2;
const STEP_DELAY_MS = 150;

// Bounds chemicals' per-step additive growth — see the numerical-safety
// comment in simStep() below. Must match update_rule.py's CHEMICAL_CLIP.
const CHEMICAL_CLIP = 10.0;

/** Everything one replay() call needs to reproduce a generation's winner — see update_rule.py for the energy fields. */
export interface ReplayConfig {
  weights: UpdateRuleWeights;
  steps: number;
  maxNodes: number;
  sensingSigma: number;
  initialEnergy: number;
  minSplitEnergy: number;
  maxEnergy: number;
  energyInjection: number;
  energyInjectionNoise: number;
}

/**
 * Live position override for whichever node the viewer is currently
 * dragging in the browser — a plain mutable ref rather than a parameter,
 * since dragging happens on its own timeline (mouse events) completely
 * independent of replay()'s step/frame loop; mutating a shared ref is
 * how GraphRenderer already handles this kind of thing internally (see
 * its own hover/pan refs) rather than routing every mouse move through
 * React state and a re-render.
 */
export interface DragRef {
  current: { nodeId: number; position: [number, number] } | null;
}

// chemicals/idVectors/spawnDirections/splitProbs/energy don't change
// during a physics settle (only positions do), so every frame yielded
// mid-relax still carries whatever the graph's current values are —
// always correct, not stale.
function toGraphNodes(graph: SimGraph): GraphNode[] {
  return graph.positions.map((position, id) => ({
    id,
    position,
    chemicals: graph.chemicals[id],
    idVector: graph.idVectors[id],
    spawnDirection: graph.spawnDirections[id],
    splitProb: graph.splitProbs[id],
    energy: graph.energy[id],
  }));
}

// Forces whichever node is currently being dragged to the cursor's world
// position, overriding whatever physics computed for it. Returns a new
// array (never mutates `positions` in place) since callers hand this
// straight to relaxSteps()/graph.positions, which expect their own copy.
function withDragOverride(positions: Vec2[], dragRef?: DragRef): Vec2[] {
  const drag = dragRef?.current;
  if (!drag || drag.nodeId >= positions.length) return positions;
  const next = positions.slice();
  next[drag.nodeId] = drag.position;
  return next;
}

// The dragged node (if any) is treated as pinned for the duration of one
// relaxSteps() call, same mechanism the graph's permanently-pinned seed
// vertex uses — physics simply never moves it, rather than fighting a
// force against the drag every iteration.
function pinnedIncludingDrag(graph: SimGraph, dragRef?: DragRef): Set<number> {
  const drag = dragRef?.current;
  if (!drag) return graph.pinned;
  const pinned = new Set(graph.pinned);
  pinned.add(drag.nodeId);
  return pinned;
}

function waitForAnimationFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomUniform(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

/**
 * One autonomous simulation step: mutates `graph` in place, does not
 * relax physics. Returns whether any node actually split — callers use
 * this to skip a pointless physics settle when nothing moved.
 */
function simStep(graph: SimGraph, config: ReplayConfig): boolean {
  const n = graph.positions.length;
  if (n === 0) return false;

  const { weights, sensingSigma, maxNodes, minSplitEnergy, maxEnergy, energyInjection, energyInjectionNoise } =
    config;

  // Energy regenerates before this step's decision is made, so a node
  // senses (and can act on) its own post-injection level, not last
  // step's stale one.
  const injectedEnergy = graph.energy.map((e) =>
    Math.min(maxEnergy, Math.max(0, e + energyInjection + randomUniform(-energyInjectionNoise, energyInjectionNoise)))
  );
  const normalizedEnergy = injectedEnergy.map((e) => (e / maxEnergy) * 2 - 1);

  const gradients = weightedFieldGradient(graph.positions, graph.chemicals, sensingSigma);

  const shouldSplit: boolean[] = new Array(n);
  const newChemicals: number[][] = new Array(n);
  const newId: number[][] = new Array(n);
  const rawSpawnDirection: number[][] = new Array(n);
  const rawSplitProb: number[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const gradX = gradients[i].map(([gx]) => gx);
    const gradY = gradients[i].map(([, gy]) => gy);
    const { splitProb, chemicalDelta, idDelta, spawnDirection } = forward(
      weights,
      graph.chemicals[i],
      gradX,
      gradY,
      normalizedEnergy[i]
    );
    rawSpawnDirection[i] = spawnDirection;
    rawSplitProb[i] = splitProb;

    // The network's own probability is a ceiling, not the final word:
    // energyWeight is 0 below minSplitEnergy (hard gate) and ramps
    // linearly to 1 at maxEnergy, so a low-energy node can never split
    // regardless of how confident the network is.
    const energyWeight = Math.min(1, Math.max(0, (injectedEnergy[i] - minSplitEnergy) / (maxEnergy - minSplitEnergy)));
    shouldSplit[i] = Math.random() < splitProb * energyWeight;

    // Numerical safety, mirroring update_rule.py exactly: an untrained
    // network's per-step deltas are unbounded, and both chemicals and id
    // feed additively into next step's own input — an uncapped feedback
    // loop that reliably diverges given enough steps and eventually
    // overflows to Infinity, corrupting positions via
    // tensionCompatibility's norm-based division (Infinity/Infinity =
    // NaN). chemicals are clipped to a bounded range; id is renormalized
    // to unit length instead, since every consumer only ever reads it via
    // cosine similarity (direction, never magnitude — see physics.ts's
    // tensionCompatibility).
    newChemicals[i] = graph.chemicals[i].map((c, k) =>
      Math.min(CHEMICAL_CLIP, Math.max(-CHEMICAL_CLIP, c + chemicalDelta[k]))
    );
    const rawId = graph.idVectors[i].map((v, k) => v + idDelta[k]);
    const idNorm = Math.hypot(...rawId) || 1;
    newId[i] = rawId.map((v) => v / idNorm);
  }

  for (let i = 0; i < n; i++) {
    graph.chemicals[i] = newChemicals[i];
    graph.idVectors[i] = newId[i];
    graph.energy[i] = injectedEnergy[i];
    // Unlike id, spawnDirection isn't added to a running state — it's a
    // fresh reading every step, so just this step's raw output
    // normalized for storage/display (a near-zero raw output has no
    // direction; store [0, 0] rather than dividing by ~0 — addChild
    // below does its own norm+fallback on the raw value for actually
    // placing a child, same convention as update_rule.py's step()).
    const dirNorm = Math.hypot(...rawSpawnDirection[i]);
    graph.spawnDirections[i] = dirNorm < 1e-9 ? [0, 0] : rawSpawnDirection[i].map((v) => v / dirNorm);
    graph.splitProbs[i] = rawSplitProb[i];
  }

  // Splitting uses post-delta/post-injection state and only iterates
  // over the original snapshot's node count, so children spawned this
  // step don't themselves get a chance to split again until next step.
  let didSplit = false;
  for (let i = 0; i < n; i++) {
    if (shouldSplit[i] && graph.positions.length < maxNodes) {
      const splitEnergy = graph.energy[i] / 2;
      graph.energy[i] = splitEnergy;
      addChild(graph, i, [...newId[i]], [...newChemicals[i]], splitEnergy, rawSpawnDirection[i]);
      didSplit = true;
    }
  }
  return didSplit;
}

/** One animation frame of a replay: the node snapshot plus which simulation
 * step it belongs to (0 = the initial seed, before any step has run). */
export interface ReplayFrame {
  nodes: GraphNode[];
  step: number;
}

/** Runs a full seed-to-`steps` replay, yielding node snapshots as it animates. */
export async function* replay(config: ReplayConfig, dragRef?: DragRef): AsyncGenerator<ReplayFrame> {
  const graph = seedGraph(config.initialEnergy);
  graph.positions = withDragOverride(graph.positions, dragRef);
  yield { nodes: toGraphNodes(graph), step: 0 };
  await delay(STEP_DELAY_MS);

  for (let step = 0; step < config.steps; step++) {
    const didSplit = simStep(graph, config);
    const currentStep = step + 1;

    // Nothing to animate if the node set didn't change this step —
    // positions were already at equilibrium from the previous relax.
    // Still yield once so chemicals/id/energy updates (visible via the
    // color-mode dropdown) don't go stale even on a no-growth step.
    if (!didSplit) {
      yield { nodes: toGraphNodes(graph), step: currentStep };
      await delay(STEP_DELAY_MS);
      continue;
    }

    const pinned = pinnedIncludingDrag(graph, dragRef);
    const gen = relaxSteps(graph.positions, pinned, graph.idVectors);
    let result = gen.next();
    let frame = 0;
    while (!result.done) {
      if (frame % PHYSICS_ITERATIONS_PER_FRAME === 0) {
        graph.positions = withDragOverride(result.value, dragRef);
        yield { nodes: toGraphNodes(graph), step: currentStep };
        await waitForAnimationFrame();
      }
      frame++;
      result = gen.next();
    }
    graph.positions = withDragOverride(result.value, dragRef);
    yield { nodes: toGraphNodes(graph), step: currentStep };

    await delay(STEP_DELAY_MS);
  }
}

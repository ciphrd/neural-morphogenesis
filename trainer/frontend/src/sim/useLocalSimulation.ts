import { useEffect, useRef, useState } from "react";
import type { GraphNode } from "../net/socket";
import type { LatestGeneration } from "../net/trainingSocket";
import { replay, type DragRef } from "./runner";

// Only ever shown while `nodes` is still empty (before any generation
// has arrived) — the real radius comes from `latest.physics.radius`
// once it does, so this never needs to track physics.py; nothing
// renders using it.
const PRE_CONNECT_RADIUS = 0.5;

/**
 * Drives the client-side replay (sim/runner.ts) for whatever generation
 * the training server has most recently sent weights for. Restarting
 * the effect whenever `latest` changes reference cancels any in-flight
 * replay and starts a new one — in practice this doesn't visibly cut a
 * replay short, since a full population evaluation server-side (many
 * rollouts of the same complexity as one client replay) reliably takes
 * longer than a single client-side replay finishes animating.
 *
 * Also owns the live drag override: dragNode()/releaseNode() let the
 * viewer grab a node mid-replay and watch physics react in real time.
 * The ref is reset (not carried over) whenever a new replay starts,
 * since node IDs from the previous generation's seed-to-steps run are
 * meaningless once a fresh graph is seeded.
 *
 * `enabled` (default true) skips starting the replay loop entirely —
 * for a caller (TrainingView's realtime-growth checkbox) that wants
 * this hook mounted but idle while a *different* driver
 * (useRealtimeSimulation) is the one actually animating, so the two
 * don't both run a full replay/tick loop off the same `latest` at once
 * for no reason.
 */
export function useLocalSimulation(latest: LatestGeneration | null, enabled = true) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [step, setStep] = useState(0);
  const dragRef = useRef<DragRef["current"]>(null);

  useEffect(() => {
    dragRef.current = null;
    setStep(0);
    if (!latest || !enabled) return;
    let cancelled = false;

    (async () => {
      for await (const frame of replay(latest, dragRef)) {
        if (cancelled) break;
        setNodes(frame.nodes);
        setStep(frame.step);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [latest, enabled]);

  const dragNode = (nodeId: number, position: [number, number]) => {
    dragRef.current = { nodeId, position };
  };

  const releaseNode = () => {
    dragRef.current = null;
  };

  return { nodes, radius: latest?.physics.radius ?? PRE_CONNECT_RADIUS, dragNode, releaseNode, step };
}

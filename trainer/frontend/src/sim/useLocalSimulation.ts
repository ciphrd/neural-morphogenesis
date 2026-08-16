import { useEffect, useRef, useState } from "react";
import type { GraphNode } from "../net/socket";
import type { LatestGeneration } from "../net/trainingSocket";
import { RADIUS } from "./physics";
import { replay, type DragRef } from "./runner";

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
 */
export function useLocalSimulation(latest: LatestGeneration | null) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [step, setStep] = useState(0);
  const dragRef = useRef<DragRef["current"]>(null);

  useEffect(() => {
    dragRef.current = null;
    setStep(0);
    if (!latest) return;
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
  }, [latest]);

  const dragNode = (nodeId: number, position: [number, number]) => {
    dragRef.current = { nodeId, position };
  };

  const releaseNode = () => {
    dragRef.current = null;
  };

  return { nodes, radius: RADIUS, dragNode, releaseNode, step };
}

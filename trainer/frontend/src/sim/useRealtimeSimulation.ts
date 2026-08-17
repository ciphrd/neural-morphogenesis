/**
 * "Realtime growth": a deliberately different client-side driver from
 * runner.ts's replay() — that one is the *authoritative* replay
 * (mirrors update_rule.py's step() semantics exactly, and always lets
 * physics settle to convergence before the next network step, same as
 * the real backend). This one is a loose, always-live visualization
 * that intentionally diverges from that:
 *
 * - Runs continuously off requestAnimationFrame rather than a fixed
 *   `steps` count with STEP_DELAY_MS pauses — there's no "done," it
 *   just keeps ticking (and keeps strafing/jittering even once growth
 *   plateaus at maxNodes) for as long as the caller leaves it running.
 * - Every tick does one full simStep() (sense/decide/act — the same
 *   O(n) MLP forward pass + O(n²) gradient sensing runner.ts's replay()
 *   uses) *and* one physics tick, both sized to comfortably fit inside
 *   a single animation frame's budget rather than running either to
 *   convergence — see physics.ts's relaxTick() for why that's a
 *   separate function from relaxSteps()/relax(). Doesn't have to be
 *   accurate to what a "real" rollout would produce, just has to look
 *   alive.
 */

import { useEffect, useRef, useState } from "react";
import type { GraphNode } from "../net/socket";
import { relaxTick } from "./physics";
import { seedGraph, type SimGraph } from "./graph";
import {
  simStep,
  toGraphNodes,
  withDragOverride,
  pinnedIncludingDrag,
  type ReplayConfig,
  type DragRef,
} from "./runner";

// Only ever shown while `nodes` is still empty (before any generation
// has arrived) — the real radius comes from `config.physics.radius`
// once it does, so this never needs to track physics.py; nothing
// renders using it.
const PRE_CONNECT_RADIUS = 0.5;

export function useRealtimeSimulation(
  config: ReplayConfig | null,
  running: boolean,
  ticksPerFrame: number
) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [tick, setTick] = useState(0);
  const graphRef = useRef<SimGraph | null>(null);
  const dragRef = useRef<DragRef["current"]>(null);

  // Fresh organism whenever the weights identity changes (a new
  // generation arrives, or this is the first config we've ever seen) —
  // same reset trigger useLocalSimulation uses for the batch replay, so
  // both views stay in sync about what a "new generation" means.
  useEffect(() => {
    dragRef.current = null;
    setTick(0);
    if (!config) {
      graphRef.current = null;
      setNodes([]);
      return;
    }
    graphRef.current = seedGraph(config.initialEnergy);
    setNodes(toGraphNodes(graphRef.current));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config?.weights]);

  useEffect(() => {
    if (!running || !config) return;
    let raf = 0;
    let cancelled = false;

    const frame = () => {
      const graph = graphRef.current;
      if (graph) {
        for (let i = 0; i < ticksPerFrame; i++) {
          simStep(graph, config);
          const pinned = pinnedIncludingDrag(graph, dragRef);
          graph.positions = withDragOverride(
            relaxTick(graph.positions, pinned, graph.idVectors, config.physics),
            dragRef
          );
        }
        setNodes(toGraphNodes(graph));
        setTick((t) => t + ticksPerFrame);
      }
      if (!cancelled) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [running, config, ticksPerFrame]);

  const dragNode = (nodeId: number, position: [number, number]) => {
    dragRef.current = { nodeId, position };
  };

  const releaseNode = () => {
    dragRef.current = null;
  };

  return { nodes, radius: config?.physics.radius ?? PRE_CONNECT_RADIUS, dragNode, releaseNode, tick };
}

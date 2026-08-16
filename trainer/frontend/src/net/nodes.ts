import { useEffect, useState } from "react";
import { API_URL } from "./targets";
import type { GraphNode } from "./socket";

export interface NodeState {
  id: number;
  position: [number, number];
  idVector: number[];
  chemicals: number[];
  energy: number;
  spawnDirection: number[];
}

// Refetches whenever the selected node id changes, or the graph itself
// changes — unlike before, id/chemicals are no longer assigned once and
// left alone: the autonomous update rule mutates them every step, so a
// selected node's panel needs to stay live while it's being watched.
export function useNodeInspector(nodeId: number | null, nodes: GraphNode[]) {
  const [nodeState, setNodeState] = useState<NodeState | null>(null);

  useEffect(() => {
    if (nodeId === null) {
      setNodeState(null);
      return;
    }

    let cancelled = false;
    fetch(`${API_URL}/nodes/${nodeId}`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) setNodeState(data);
      })
      .catch((err) => console.error("[trainer] failed to fetch node state", err));

    return () => {
      cancelled = true;
    };
  }, [nodeId, nodes]);

  return nodeState;
}

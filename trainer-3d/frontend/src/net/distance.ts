import { useEffect, useState } from "react";
import { API_URL } from "./targets";
import type { GraphNode } from "./socket";

export interface DistanceMetrics {
  points_to_target: number;
  target_to_points: number;
  chamfer: number;
}

// Refetches whenever the selected target or the graph's node positions
// change — cheap (a KD-tree query over a few thousand points) so simplest
// to just recompute on every graph update rather than debounce.
export function useTargetDistance(targetName: string | null, nodes: GraphNode[]) {
  const [distance, setDistance] = useState<DistanceMetrics | null>(null);

  useEffect(() => {
    if (!targetName) {
      setDistance(null);
      return;
    }

    let cancelled = false;
    fetch(`${API_URL}/targets/${encodeURIComponent(targetName)}/distance`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) setDistance(data);
      })
      .catch((err) => console.error("[trainer] failed to fetch distance", err));

    return () => {
      cancelled = true;
    };
  }, [targetName, nodes]);

  return distance;
}

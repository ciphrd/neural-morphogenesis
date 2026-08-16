import { useEffect, useState } from "react";
import { API_URL } from "./targets";
import type { GraphNode } from "./socket";

export interface DistanceMetrics {
  points_to_target: number;
  target_to_points: number;
  chamfer: number;
  // best-fit rotation (radians) and translation found by the alignment
  // search — the distance above is measured at this orientation, not the
  // structure's actual as-grown orientation
  rotation: number;
  translation: [number, number];
}

// Refetches whenever the selected target or the graph's node positions
// change. This now runs a rotation/translation alignment search
// (~100-250ms), not just a KD-tree query, so it's noticeably slower than
// it looks — still simplest to just recompute on every graph update
// rather than debounce, the "Computing…" state covers the latency.
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

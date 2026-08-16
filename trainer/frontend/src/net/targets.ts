import { useCallback, useEffect, useState } from "react";

export const API_URL = "http://localhost:8000";

export interface TargetSummary {
  name: string;
  points: number;
  resolution: [number, number];
  preview: [number, number][];
}

export function useTargets() {
  const [targets, setTargets] = useState<TargetSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [targetPoints, setTargetPoints] = useState<[number, number][] | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/targets`)
      .then((res) => res.json())
      .then((data) => setTargets(data.targets))
      .catch((err) => console.error("[trainer] failed to list targets", err));
  }, []);

  // Selecting the already-active target deselects it, clearing the
  // overlay and the backend's loaded target together.
  const select = useCallback(
    async (name: string) => {
      if (selected === name) {
        await fetch(`${API_URL}/target/clear`, { method: "POST" });
        setSelected(null);
        setTargetPoints(null);
        return;
      }

      await fetch(`${API_URL}/targets/${encodeURIComponent(name)}/load`, { method: "POST" });
      const res = await fetch(`${API_URL}/target/points`);
      const data = await res.json();
      setSelected(name);
      setTargetPoints(data.points);
    },
    [selected]
  );

  return { targets, selected, targetPoints, select };
}

import { useEffect, useState } from "react";
import type { GenerationStat } from "../charts/FitnessChart";
import type { ReplayConfig } from "../sim/runner";

/** Everything the client-side replay (sim/) needs to reproduce this generation's winner. */
export interface LatestGeneration extends ReplayConfig {
  generation: number;
}

export interface TrainingSocketState {
  history: GenerationStat[];
  latest: LatestGeneration | null;
  // Every retained generation's full replay data, not just the latest —
  // train_server.py only ever sends a generation's weights once, so this
  // is the only way to navigate back to an earlier generation's winner
  // later. Evicted in lockstep with `history` (same cap, same oldest-first
  // order), so any generation number visible in `history` is guaranteed to
  // still have its weights here.
  weightsByGeneration: Map<number, LatestGeneration>;
}

const EMPTY_STATE: TrainingSocketState = { history: [], latest: null, weightsByGeneration: new Map() };

// History is capped so a long run doesn't grow the chart's render cost (and
// the array) without bound — the most recent generations are what's actually
// useful to watch live; older ones are still in train_server.py's own stdout
// and checkpoints/history.jsonl if needed. This same cap now also bounds
// weightsByGeneration (see above) — a generation's weights are ~7k floats as
// JSON, so 500 retained generations is on the order of tens of MB, comfortably
// fine for a browser tab.
const MAX_HISTORY = 500;

interface RawGenerationMessage {
  generation: number;
  best: number;
  mean: number;
  worst: number;
  allTimeBest: number;
  weights: ReplayConfig["weights"];
  steps: number;
  maxNodes: number;
  sensingSigma: number;
  initialEnergy: number;
  minSplitEnergy: number;
  maxEnergy: number;
  energyInjection: number;
  energyInjectionNoise: number;
  chemicalClip: number;
  maxAccel: number;
  maxSpeed: number;
  physics: ReplayConfig["physics"];
}

// Both the REST backfill (on mount, from train_server.py's saved
// checkpoints/history.jsonl) and the live websocket funnel through this,
// so there's exactly one place that builds history/weightsByGeneration —
// and so the two can safely race (the backfill fetch and the first live
// message have no guaranteed order relative to each other) without
// corrupting state: keyed by generation number and re-sorted on every
// update rather than assumed to already arrive in order.
function applyGeneration(prev: TrainingSocketState, message: RawGenerationMessage): TrainingSocketState {
  const entry: LatestGeneration = {
    generation: message.generation,
    weights: message.weights,
    steps: message.steps,
    maxNodes: message.maxNodes,
    sensingSigma: message.sensingSigma,
    initialEnergy: message.initialEnergy,
    minSplitEnergy: message.minSplitEnergy,
    maxEnergy: message.maxEnergy,
    energyInjection: message.energyInjection,
    energyInjectionNoise: message.energyInjectionNoise,
    chemicalClip: message.chemicalClip,
    maxAccel: message.maxAccel,
    maxSpeed: message.maxSpeed,
    physics: message.physics,
  };

  const weightsByGeneration = new Map(prev.weightsByGeneration);
  weightsByGeneration.set(message.generation, entry);

  const statByGeneration = new Map(prev.history.map((h) => [h.generation, h]));
  statByGeneration.set(message.generation, {
    generation: message.generation,
    best: message.best,
    mean: message.mean,
    worst: message.worst,
    allTimeBest: message.allTimeBest,
  });

  const sortedGenerations = Array.from(statByGeneration.keys()).sort((a, b) => a - b);
  const kept = new Set(sortedGenerations.slice(-MAX_HISTORY));

  for (const gen of statByGeneration.keys()) {
    if (!kept.has(gen)) statByGeneration.delete(gen);
  }
  for (const gen of weightsByGeneration.keys()) {
    if (!kept.has(gen)) weightsByGeneration.delete(gen);
  }

  const history = Array.from(kept)
    .sort((a, b) => a - b)
    .map((gen) => statByGeneration.get(gen)!);

  const latest =
    prev.latest === null || message.generation >= prev.latest.generation ? entry : prev.latest;

  return { history, latest, weightsByGeneration };
}

export function useTrainingSocket(wsUrl: string, apiUrl: string) {
  const [state, setState] = useState<TrainingSocketState>(EMPTY_STATE);

  // Backfill: train_server.py saves every generation to disk incrementally
  // specifically so a fresh or reloaded tab isn't starting blind — pull in
  // the whole run-so-far once on mount. Safe to race against the live
  // websocket below since applyGeneration tolerates out-of-order arrival.
  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/history`)
      .then((res) => res.json())
      .then((data: { generations: RawGenerationMessage[] }) => {
        if (cancelled) return;
        setState((prev) => data.generations.reduce(applyGeneration, prev));
      })
      .catch((err) => console.error("[trainer] failed to fetch training history", err));
    return () => {
      cancelled = true;
    };
  }, [apiUrl]);

  useEffect(() => {
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type !== "generation") return;
      setState((prev) => applyGeneration(prev, message));
    };

    // Same StrictMode double-mount consideration as useGraphSocket: don't
    // close a socket that's still mid-handshake.
    return () => {
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.addEventListener("open", () => socket.close());
      } else {
        socket.close();
      }
    };
  }, [wsUrl]);

  return state;
}

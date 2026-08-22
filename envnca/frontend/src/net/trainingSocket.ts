import { useEffect, useState } from "react";
import type { GenerationStat } from "../charts/FitnessChart";
import type { SimulationConfig } from "../gpu/types";

/** Everything the client-side WebGPU replay (gpu/) needs to reproduce
 * this generation's winner, plus `target` (see RawGenerationMessage's
 * own comment on it — not itself replay data, but net/runs.ts's picker
 * needs to know which target's points to fetch for whatever run/
 * generation is currently active). */
export interface LatestGeneration extends SimulationConfig {
  generation: number;
  target: string;
}

export interface TrainingSocketState {
  history: GenerationStat[];
  latest: LatestGeneration | null;
  // Every retained generation's full replay data, not just the latest —
  // train_server.py only ever sends a generation's weights once, so this
  // is the only way to navigate back to an earlier generation's winner
  // later. Evicted in lockstep with `history` (same cap, same oldest-
  // first order), so any generation number visible in `history` is
  // guaranteed to still have its weights here.
  configByGeneration: Map<number, LatestGeneration>;
}

export const EMPTY_STATE: TrainingSocketState = { history: [], latest: null, configByGeneration: new Map() };

// History is capped so a long run doesn't grow the chart's render cost
// (and the array) without bound — the most recent generations are what's
// actually useful to watch live; older ones are still in train_server.py's
// own stdout and checkpoints/history.jsonl if needed. This same cap also
// bounds configByGeneration.
const MAX_HISTORY = 500;

// Exported: net/runs.ts's archived-run loading builds the exact same
// TrainingSocketState shape from a REST fetch instead of a live socket
// (see buildRunState()), so the rest of the app (FitnessChart,
// GridCanvas) doesn't need to know or care whether it's looking at the
// live run or a past one.
export interface RawGenerationMessage extends SimulationConfig {
  generation: number;
  best: number;
  mean: number;
  worst: number;
  allTimeBest: number;
  // Not part of SimulationConfig (a WebGPU replay doesn't need to know
  // its own target's *name*, just its points, fetched separately) — but
  // net/runs.ts's run picker needs it to fetch the *right* target's
  // points when browsing an archived run trained on a different target
  // than whatever this server's own --target currently is.
  target: string;
}

// Both the REST backfill (on mount, from train_server.py's saved
// checkpoints/history.jsonl) and the live websocket funnel through this,
// so there's exactly one place that builds history/configByGeneration —
// and so the two can safely race (the backfill fetch and the first live
// message have no guaranteed order relative to each other) without
// corrupting state: keyed by generation number and re-sorted on every
// update rather than assumed to already arrive in order.
export function applyGeneration(prev: TrainingSocketState, message: RawGenerationMessage): TrainingSocketState {
  const entry: LatestGeneration = {
    generation: message.generation,
    target: message.target,
    weights: message.weights,
    gridWidth: message.gridWidth,
    gridHeight: message.gridHeight,
    channels: message.channels,
    agentCount: message.agentCount,
    spawnSpread: message.spawnSpread,
    steps: message.steps,
    decay: message.decay,
    maxSpeed: message.maxSpeed,
    maxAccel: message.maxAccel,
    maxStrafe: message.maxStrafe,
    maxEnvWrite: message.maxEnvWrite,
    repulsionSigma: message.repulsionSigma,
    repulsionStrength: message.repulsionStrength,
    hiddenDim: message.hiddenDim,
    seed: message.seed,
  };

  const configByGeneration = new Map(prev.configByGeneration);
  configByGeneration.set(message.generation, entry);

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
  for (const gen of configByGeneration.keys()) {
    if (!kept.has(gen)) configByGeneration.delete(gen);
  }

  const history = Array.from(kept)
    .sort((a, b) => a - b)
    .map((gen) => statByGeneration.get(gen)!);

  const latest = prev.latest === null || message.generation >= prev.latest.generation ? entry : prev.latest;

  return { history, latest, configByGeneration };
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
      .catch((err) => console.error("[envnca] failed to fetch training history", err));
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

    // Same StrictMode double-mount consideration as trainer/frontend's
    // own sockets: don't close a socket that's still mid-handshake.
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

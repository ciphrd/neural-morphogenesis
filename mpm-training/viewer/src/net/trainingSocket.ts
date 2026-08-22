// Live training state — the core state-management hook, mirroring
// envnca/frontend/src/net/trainingSocket.ts's own shape almost exactly
// (same reducer-over-both-REST-and-WS design, same MAX_HISTORY eviction),
// PLUS a settings/generation split train_server.py's own SETTINGS_PATH
// global explains the "why" of: a run's SETTINGS (target/particles/
// channels/decay/population/...) are fetched ONCE via GET /settings,
// separately from each generation's own record (best/mean/worst/
// allTimeBest/seed/weights) — so applySettings()/applyGeneration() below
// are two independent, mechanical reducers over a raw Accumulator, and
// deriveState() is the ONE place that merges the two into the
// TrainingSocketState shape GridCanvas/TrainingView actually consume
// (unchanged from before the split — see gpu/types.ts's own
// SimulationConfig docstring). Reused by net/runs.ts's own
// fetchRunState() for archived-run history, same as before.

import { useEffect, useMemo, useState } from "react";
import { randomWeights } from "../gpu/agents";
import type { GenerationRecord, RunSettings, SimulationConfig } from "../gpu/types";

export interface GenerationStat {
  generation: number;
  best: number;
  mean: number;
  worst: number;
  allTimeBest: number;
}

export interface TrainingSocketState {
  history: GenerationStat[];
  latest: SimulationConfig | null;
  configByGeneration: Map<number, SimulationConfig>;
}

export const EMPTY_STATE: TrainingSocketState = { history: [], latest: null, configByGeneration: new Map() };
const MAX_HISTORY = 500;

// Never a real generation number (train_server.py's own counter starts
// at 0) — see deriveState()'s own comment for what this placeholder is
// for and why it's never added to history/configByGeneration.
const PLACEHOLDER_GENERATION = -1;

function placeholderRecord(settings: RunSettings): GenerationRecord {
  return {
    generation: PLACEHOLDER_GENERATION,
    best: NaN,
    mean: NaN,
    worst: NaN,
    allTimeBest: NaN,
    seed: 0,
    weights: randomWeights(settings.channels, settings.hiddenDim),
  };
}

// Raw accumulator both useTrainingSocket() (live) and net/runs.ts's own
// fetchRunState() (archived) build up before deriving the public
// TrainingSocketState shape above. `records` carries a full weights
// export per generation, so it's capped to MAX_HISTORY in lockstep with
// `history` used to be — same unbounded-memory reasoning, just moved
// here now that settings live separately.
export interface Accumulator {
  settings: RunSettings | null;
  records: Map<number, GenerationRecord>;
}
export const EMPTY_ACCUMULATOR: Accumulator = { settings: null, records: new Map() };

export function applySettings(prev: Accumulator, settings: RunSettings): Accumulator {
  return { ...prev, settings };
}

export function applyGeneration(prev: Accumulator, message: GenerationRecord): Accumulator {
  const records = new Map(prev.records);
  records.set(message.generation, message);
  if (records.size > MAX_HISTORY) {
    // Only ever one over at a time in practice (applyGeneration is
    // called once per new message, live or during backfill) — dropping
    // the single oldest key is enough, no need for envnca's own
    // multi-drop slice dance.
    records.delete(Math.min(...records.keys()));
  }
  return { ...prev, records };
}

/** The one place settings + every known generation record get merged
 * into what GridCanvas/TrainingView actually consume — recomputed
 * (cheap, at most MAX_HISTORY entries) whenever the accumulator changes
 * rather than maintained incrementally, so applySettings()/
 * applyGeneration() above can stay plain, mechanical reducers. */
export function deriveState(acc: Accumulator): TrainingSocketState {
  const history: GenerationStat[] = Array.from(acc.records.values())
    .map((r) => ({ generation: r.generation, best: r.best, mean: r.mean, worst: r.worst, allTimeBest: r.allTimeBest }))
    .sort((a, b) => a.generation - b.generation);

  if (!acc.settings) return { history, latest: null, configByGeneration: new Map() };

  const configByGeneration = new Map<number, SimulationConfig>();
  for (const record of acc.records.values()) {
    configByGeneration.set(record.generation, { ...acc.settings, ...record });
  }

  const latest: SimulationConfig =
    history.length > 0
      ? (configByGeneration.get(history[history.length - 1].generation) as SimulationConfig)
      : // Settings exist but generation 0 hasn't finished evaluating yet
        // (population x workers can take real time) — render a LIVE
        // rollout under freshly random-initialized weights (same
        // generator "Randomize weights" uses) instead of a blank canvas;
        // the whole reason settings/generation records were split apart.
        // Deliberately not added to history/configByGeneration —
        // PLACEHOLDER_GENERATION has no real generation number to key it
        // by, and the fitness chart would show a fake, meaningless data
        // point for it.
        { ...acc.settings, ...placeholderRecord(acc.settings) };

  return { history, latest, configByGeneration };
}

export function useTrainingSocket(wsUrl: string, apiUrl: string): TrainingSocketState {
  const [acc, setAcc] = useState<Accumulator>(EMPTY_ACCUMULATOR);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const attempt = () => {
      fetch(`${apiUrl}/settings`)
        .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`GET /settings -> ${res.status}`))))
        .then((data: RunSettings) => {
          if (!cancelled) setAcc((prev) => applySettings(prev, data));
        })
        .catch(() => {
          // 503 while _training_loop_body() hasn't reached its own
          // settings assignment yet (see train_server.py's own /settings
          // docstring) — retry rather than give up; this is the ONLY
          // source of `settings`, and deriveState() can't build anything
          // without it. In practice this resolves within one or two
          // attempts (settings are written near the very top of the
          // training loop, well before generation 0 finishes).
          if (!cancelled) timer = setTimeout(attempt, 500);
        });
    };
    attempt();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [apiUrl]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/history`)
      .then((res) => res.json())
      .then((data: { generations: GenerationRecord[] }) => {
        if (cancelled) return;
        setAcc((prev) => data.generations.reduce((a, message) => applyGeneration(a, message), prev));
      })
      .catch((err) => console.error("[trainingSocket] history backfill failed:", err));
    return () => {
      cancelled = true;
    };
  }, [apiUrl]);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event: MessageEvent<string>) => {
      let message: GenerationRecord & { type?: string };
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type !== "generation") return;
      setAcc((prev) => applyGeneration(prev, message));
    };
    ws.onerror = (err) => console.error("[trainingSocket] websocket error:", err);
    return () => {
      // StrictMode double-mount safety: closing a still-CONNECTING socket
      // immediately can throw/warn on some browsers — wait for open first.
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.addEventListener("open", () => ws.close());
      } else {
        ws.close();
      }
    };
  }, [wsUrl]);

  return useMemo(() => deriveState(acc), [acc]);
}

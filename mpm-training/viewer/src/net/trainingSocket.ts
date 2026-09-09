// Compact chart history plus a bounded cache of full replay policies.
// Live and archived runs share the same bulk history merge.

import { useEffect, useMemo, useState } from "react";
import { randomWeights } from "../gpu/agents";
import type { TimingEntry, GenerationTiming, GenerationRecord, RunSettings, SimulationConfig } from "../gpu/types";
import { DEFAULT_RUN_SETTINGS, loadInitialRunSettings } from "./settingsStorage";

export interface GenerationStat {
  optimizerState?: GenerationRecord["optimizerState"];
  timing?: GenerationTiming;
  generation: number;
  best: number;
  mean: number;
  worst: number;
  allTimeBest: number;
}

export interface TrainingSocketState {
  history: GenerationStat[];
  timingHistory: TimingEntry[];
  latest: SimulationConfig | null;
  configByGeneration: Map<number, SimulationConfig>;
}

export interface LiveTrainingSocketState extends TrainingSocketState {
  /** True only while the viewer has an open connection to train_server.py. */
  serverConnected: boolean;
}

export const EMPTY_STATE: TrainingSocketState = { history: [], timingHistory: [], latest: null, configByGeneration: new Map() };
const MAX_WEIGHT_SNAPSHOTS = 8;
export type HistoryRecord = Omit<GenerationRecord, "weights"> & { weights?: GenerationRecord["weights"] };
export interface HistoryPayload {
  generations: HistoryRecord[];
  timings?: TimingEntry[];
  latestGeneration?: GenerationRecord | null;
}

function trimWeights(records: Map<number, HistoryRecord>): void {
  const full = [...records.values()].filter(r => r.weights).sort((a, b) => b.generation - a.generation);
  for (const r of full.slice(MAX_WEIGHT_SNAPSHOTS)) {
    records.set(r.generation, { generation: r.generation, seed: r.seed, best: r.best,
      mean: r.mean, worst: r.worst, allTimeBest: r.allTimeBest, optimizerState: r.optimizerState });
  }
}

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
    weights: randomWeights(settings.channels, settings.hiddenDim, settings.policyArchitecture),
  };
}

// Chart summaries stay available; only the latest eight policies retain weights.
export interface Accumulator {
  settings: RunSettings | null;
  records: Map<number, HistoryRecord>;
  timings?: Map<number, GenerationTiming>;
}
export const EMPTY_ACCUMULATOR: Accumulator = { settings: null, records: new Map() };

export function applySettings(prev: Accumulator, settings: RunSettings): Accumulator {
  if (settings.growthModelVersion !== DEFAULT_RUN_SETTINGS.growthModelVersion ||
      settings.densityModelVersion !== DEFAULT_RUN_SETTINGS.densityModelVersion ||
      settings.domainGeometry !== DEFAULT_RUN_SETTINGS.domainGeometry) {
    throw new Error("Run schema does not match the current simulation; start a new run.");
  }
  for (const key of Object.keys(DEFAULT_RUN_SETTINGS)) {
    // Optimizer and temporal-scoring metadata may be absent in older
    // archives and do not affect simulation playback.
    if (key === "optimizer" || key === "cmaCovariance" || key === "fitnessTemporalAggregation") continue;
    if (!(key in settings)) throw new Error(`Run settings missing required field: ${key}`);
  }
  return { ...prev, settings };
}

export function applyGeneration(prev: Accumulator, message: GenerationRecord): Accumulator {
  const records = new Map(prev.records);
  records.set(message.generation, message);
  trimWeights(records);
  const timings = new Map(prev.timings);
  if (message.timing) timings.set(message.generation, message.timing);
  return { ...prev, records, timings };
}

export function applyHistory(prev: Accumulator, data: HistoryPayload): Accumulator {
  // One copy per collection, not one copy per generation during backfill.
  const records = new Map(prev.records);
  const timings = new Map(prev.timings);
  for (const entry of data.timings ?? []) {
    if (!timings.has(entry.generation)) timings.set(entry.generation, entry.timing);
  }
  for (const record of data.generations) {
    if (!records.get(record.generation)?.weights) records.set(record.generation, record);
    if (record.timing && !timings.has(record.generation)) timings.set(record.generation, record.timing);
  }
  if (data.latestGeneration) records.set(data.latestGeneration.generation, data.latestGeneration);
  trimWeights(records);
  return { ...prev, records, timings };
}

/** Merge settings with cached policies, without materializing configs for summaries. */
export function deriveState(acc: Accumulator): TrainingSocketState {
  const history: GenerationStat[] = Array.from(acc.records.values())
    .map((r) => ({ generation: r.generation, best: r.best, mean: r.mean, worst: r.worst, allTimeBest: r.allTimeBest, timing: r.timing, optimizerState: r.optimizerState }))
    .sort((a, b) => a.generation - b.generation);

  const timingHistory = Array.from(acc.timings ?? []).map(([generation, timing]) => ({ generation, timing })).sort((a, b) => a.generation - b.generation);
  if (!acc.settings) return { history, timingHistory, latest: null, configByGeneration: new Map() };

  const configByGeneration = new Map<number, SimulationConfig>();
  for (const record of acc.records.values()) {
    if (record.weights) configByGeneration.set(record.generation, { ...acc.settings, ...record, weights: record.weights });
  }

  const latest: SimulationConfig =
    configByGeneration.size > 0
      ? configByGeneration.get(Math.max(...configByGeneration.keys()))!
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

  return { history, timingHistory, latest, configByGeneration };
}

export function useTrainingSocket(wsUrl: string, apiUrl: string): LiveTrainingSocketState {
  // Seed a randomized placeholder immediately from either the last settings
  // supplied by a backend or the shared trainer/viewer defaults.
  const [acc, setAcc] = useState<Accumulator>(() => ({
    settings: loadInitialRunSettings(),
    records: new Map(),
  }));
  const [serverConnected, setServerConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const attempt = () => {
      fetch(`${apiUrl}/settings`)
        .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`GET /settings -> ${res.status}`))))
        .then((data: RunSettings) => {
          if (!cancelled) {
            // Validate before scheduling React's updater so failures reach
            // the fetch catch/retry below rather than throwing during render.
            const { settings } = applySettings(EMPTY_ACCUMULATOR, data);
            setAcc((prev) => ({ ...prev, settings }));
          }
        })
        .catch(() => {
          // 503 while _training_loop_body() hasn't reached its own
          // settings assignment yet (see train_server.py's own /settings
          // docstring) — retry rather than give up; this is the ONLY
          // authoritative source of live-run settings. The shared fallback
          // settings keep the viewer usable while this retries; in practice
          // a running backend resolves within one or two attempts (settings
          // are written near the top of the training loop, well before
          // generation 0 finishes).
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
    const controller = new AbortController();
    fetch(`${apiUrl}/history?compact=true`, { signal: controller.signal })
      .then((res) => res.json())
      .then((data: HistoryPayload) => {
        if (cancelled) return;
        setAcc((prev) => applyHistory(prev, data));
      })
      .catch((err) => { if (!cancelled) console.error("[trainingSocket] history backfill failed:", err); });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [apiUrl]);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let hadConnection = false;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        if (!cancelled) {
          setServerConnected(true);
          if (hadConnection) {
            fetch(`${apiUrl}/history?compact=true`).then(res => res.json()).then(data => {
              if (!cancelled) setAcc(prev => applyHistory(prev, data));
            }).catch(err => console.error("[trainingSocket] timing/history reconnect failed:", err));
          }
          hadConnection = true;
        }
      };
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
      ws.onerror = () => {
        if (!cancelled) setServerConnected(false);
      };
      ws.onclose = () => {
        if (cancelled) return;
        setServerConnected(false);
        reconnectTimer = setTimeout(connect, 1000);
      };
    };
    connect();

    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      // StrictMode double-mount safety: closing a still-CONNECTING socket
      // immediately can throw/warn on some browsers — wait for open first.
      const socket = ws;
      if (!socket) return;
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.addEventListener("open", () => socket.close());
      } else {
        socket.close();
      }
    };
  }, [wsUrl]);

  return useMemo(
    () => ({ ...deriveState(acc), serverConnected }),
    [acc, serverConnected]
  );
}

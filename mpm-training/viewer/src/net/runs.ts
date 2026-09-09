// REST calls against train_server.py's own /runs endpoints — mirrors
// envnca/frontend/src/net/runs.ts.

import type { GenerationRecord, RunSettings } from "../gpu/types";
import {
  applyHistory,
  applySettings,
  deriveState,
  EMPTY_ACCUMULATOR,
  type Accumulator,
  type HistoryPayload,
  type TrainingSocketState,
} from "./trainingSocket";

export interface RunSummary {
  id: string;
  isLive: boolean;
  label: string;
  target: string | null;
  generation: number | null;
  bestFitness: number | null;
  /** Best-result thumbnail — the winning rollout's own best-rotation raster. */
  previewUrl: string;
  /** The SAME generation's own target raster (train_server.py's own
   * run_target_preview() derives both from one shared generation
   * prefix — see that route's own docstring) — a genuinely comparable
   * pair, not two independently-"latest" images. */
  targetPreviewUrl: string;
}

export async function fetchRuns(apiUrl: string): Promise<RunSummary[]> {
  const res = await fetch(`${apiUrl}/runs`);
  const data: { runs: RunSummary[] } = await res.json();
  return data.runs;
}

export async function fetchRunState(apiUrl: string, runId: string, signal?: AbortSignal): Promise<TrainingSocketState> {
  const [settingsRes, historyRes] = await Promise.all([
    fetch(`${apiUrl}/runs/${encodeURIComponent(runId)}/settings`, { signal }),
    fetch(`${apiUrl}/runs/${encodeURIComponent(runId)}/history?compact=true`, { signal }),
  ]);
  if (!settingsRes.ok || !historyRes.ok) throw new Error("Run settings or history could not be loaded");
  const settings: RunSettings = await settingsRes.json();
  const data: HistoryPayload = await historyRes.json();
  let acc: Accumulator = applySettings(EMPTY_ACCUMULATOR, settings);
  acc = applyHistory(acc, data);
  return deriveState(acc);
}

export async function fetchGeneration(apiUrl: string, runId: string, generation: number, signal?: AbortSignal): Promise<GenerationRecord> {
  const response = await fetch(`${apiUrl}/runs/${encodeURIComponent(runId)}/generations/${generation}`, { signal });
  if (!response.ok) throw new Error(`Generation ${generation} could not be loaded (${response.status})`);
  return response.json();
}

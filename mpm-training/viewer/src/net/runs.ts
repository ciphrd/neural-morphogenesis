// REST calls against train_server.py's own /runs endpoints — mirrors
// envnca/frontend/src/net/runs.ts.

import type { GenerationRecord, RunSettings } from "../gpu/types";
import {
  applyGeneration,
  applySettings,
  deriveState,
  EMPTY_ACCUMULATOR,
  type Accumulator,
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

export async function fetchRunState(apiUrl: string, runId: string): Promise<TrainingSocketState> {
  const [settingsRes, historyRes] = await Promise.all([
    fetch(`${apiUrl}/runs/${encodeURIComponent(runId)}/settings`),
    fetch(`${apiUrl}/runs/${encodeURIComponent(runId)}/history`),
  ]);
  // A run archived before GET /runs/{id}/settings existed has no
  // settings.json of its own (train_server.py's own run_settings() 404s
  // on that, not the transient 503 the LIVE /settings can return) —
  // degrades to history-only (configByGeneration/latest stay empty,
  // same as deriveState()'s own "no settings yet" case), not a crash.
  const settings: RunSettings | null = settingsRes.ok ? await settingsRes.json() : null;
  const data: { generations: GenerationRecord[] } = await historyRes.json();

  let acc: Accumulator = settings ? applySettings(EMPTY_ACCUMULATOR, settings) : EMPTY_ACCUMULATOR;
  acc = data.generations.reduce((a, message) => applyGeneration(a, message), acc);
  return deriveState(acc);
}

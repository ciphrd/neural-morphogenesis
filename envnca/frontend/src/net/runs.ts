import { applyGeneration, EMPTY_STATE } from "./trainingSocket";
import type { RawGenerationMessage, TrainingSocketState } from "./trainingSocket";

/** One entry in the "Load run" picker — train_server.py's GET /runs.
 * Enough to render a list item (label, target, generation, fitness,
 * preview thumbnail) without fetching that run's full history, which
 * can be hundreds of generations of weights. */
export interface RunSummary {
  id: string;
  isLive: boolean;
  label: string;
  target: string | null;
  generation: number | null;
  bestFitness: number | null;
  previewUrl: string;
}

export async function fetchRuns(apiUrl: string): Promise<RunSummary[]> {
  const res = await fetch(`${apiUrl}/runs`);
  if (!res.ok) throw new Error(`GET /runs failed: ${res.status}`);
  const data: { runs: RunSummary[] } = await res.json();
  return data.runs;
}

/** Builds the exact same TrainingSocketState shape useTrainingSocket's
 * live websocket does (via the same applyGeneration reducer), just from
 * one REST fetch instead of a live, ongoing stream — an archived run
 * never changes once train_server.py has moved it under checkpoints/
 * runs/, so there's nothing to subscribe to, only something to load
 * once. `runId === "current"` reads the live run's own history (same
 * data /history itself serves), which is what backs the "Current run"
 * entry in the picker. */
export async function fetchRunState(apiUrl: string, runId: string): Promise<TrainingSocketState> {
  const res = await fetch(`${apiUrl}/runs/${encodeURIComponent(runId)}/history`);
  if (!res.ok) throw new Error(`GET /runs/${runId}/history failed: ${res.status}`);
  const data: { generations: RawGenerationMessage[] } = await res.json();
  return data.generations.reduce(applyGeneration, EMPTY_STATE);
}

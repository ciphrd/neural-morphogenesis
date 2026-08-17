/** URLs for the end-of-generation debug PNGs train_server.py saves to
 * checkpoints/generation_images/ (or, for an archived run, that run's
 * own copy under checkpoints/runs/{id}/generation_images/) and serves
 * back out via GET /runs/{runId}/images/{filename} — see that file's
 * _save_generation_images() for what each one is. The frontend never
 * recomputes this rasterization itself: the server's copy is the exact
 * one training actually scored the winner against (same weights + seed),
 * whereas the browser's own WebGPU replay uses its own (not bit-exact —
 * see gpu/rng.ts) jitter, so re-deriving a raster client-side would
 * drift from what training really saw.
 *
 * `runId` — "current" for the live, in-progress run, or an archived
 * run's own id (see net/runs.ts's RunSummary) — must match whichever
 * run's history is actually being displayed, not always "current": an
 * archived run's generation N image lives in a completely different
 * directory than the live run's own generation N, if the live run even
 * has that many generations at all. */
export type GenerationImageKind = "target" | "agents" | "raster";

export function generationImageUrl(
  apiUrl: string,
  runId: string,
  generation: number,
  kind: GenerationImageKind
): string {
  const padded = String(generation).padStart(5, "0");
  return `${apiUrl}/runs/${encodeURIComponent(runId)}/images/gen_${padded}_${kind}.png`;
}

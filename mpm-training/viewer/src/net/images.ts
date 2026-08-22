// Builds URLs for train_server.py's own GET /runs/{run_id}/images/{filename}
// — mirrors envnca/frontend/src/net/images.ts's own generationImageUrl().
// This project's own image kinds (see debug_images.py's own module
// docstring for what each shows): "grown" (raw, un-aligned positions),
// "target" (the target's own raster, fixed for the whole run), "agents"
// (the winner's own positions, rasterized at whichever rotation
// raster.training_raster_distance() actually scored it under — meant to
// sit right next to "target" for a direct visual comparison). Named
// differently from envnca's own "target"/"agents"/"raster" trio — there,
// "agents" is the raw scatter and "raster" is the aligned one; here,
// "grown" is the raw scatter and "agents" is the aligned raster instead
// — same three concepts, just not a 1:1 name mapping across projects.

export type GenerationImageKind = "grown" | "target" | "agents";

export function generationImageUrl(apiUrl: string, runId: string, generation: number, kind: GenerationImageKind): string {
  return `${apiUrl}/runs/${encodeURIComponent(runId)}/images/gen_${String(generation).padStart(5, "0")}_${kind}.png`;
}

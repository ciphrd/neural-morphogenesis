/** URLs for the end-of-generation debug PNGs train_server.py saves to
 * checkpoints/generation_images/ and serves back out under /images —
 * see that file's _save_generation_images() for what each one is. The
 * frontend never recomputes this rasterization itself: the server's
 * copy is the exact one training actually scored the winner against
 * (same weights + seed), whereas the browser's own WebGPU replay uses
 * its own (not bit-exact — see gpu/rng.ts) jitter, so re-deriving a
 * raster client-side would drift from what training really saw. */

export type GenerationImageKind = "target" | "agents" | "raster";

export function generationImageUrl(apiUrl: string, generation: number, kind: GenerationImageKind): string {
  const padded = String(generation).padStart(5, "0");
  return `${apiUrl}/images/gen_${padded}_${kind}.png`;
}

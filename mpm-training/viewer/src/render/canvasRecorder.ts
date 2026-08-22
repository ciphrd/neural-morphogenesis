// Records a <canvas> element's own rendered output into a downloadable
// video file — canvas.captureStream() + MediaRecorder, the standard
// browser-native canvas-recording path (no server round-trip, no extra
// dependencies). Prefers a real .mp4 container when this browser's own
// MediaRecorder can actually encode one — that support is inconsistent
// across browsers (solid in Safari, unreliable/version-dependent in
// Chrome/Firefox) — and falls back to .webm (broadly supported)
// otherwise, saving with THAT container's own real file extension rather
// than mislabeling a webm file as .mp4 (per this feature's own explicit
// request: no bundled transcoder like ffmpeg.wasm to force real .mp4
// everywhere, just record whatever this browser can natively produce and
// name the file honestly).

const CAPTURE_FPS = 30;

interface RecordingFormat {
  mimeType: string;
  ext: string;
}

// Ordered most- to least-preferred — real MP4 first (checked with an
// explicit H.264 codec string first, since some browsers report support
// for the bare "video/mp4" container without actually being able to
// encode into it), then WebM variants as the universal fallback.
const MIME_CANDIDATES: RecordingFormat[] = [
  { mimeType: "video/mp4;codecs=avc1.42E01E", ext: "mp4" },
  { mimeType: "video/mp4", ext: "mp4" },
  { mimeType: "video/webm;codecs=vp9", ext: "webm" },
  { mimeType: "video/webm;codecs=vp8", ext: "webm" },
  { mimeType: "video/webm", ext: "webm" },
];

/** Picks the best capture format this browser's own MediaRecorder
 * actually supports, checked once via MediaRecorder.isTypeSupported() —
 * real MP4 first, WebM as the fallback. Returns null if MediaRecorder
 * itself doesn't exist at all (very old browsers) so callers can
 * disable the record button entirely rather than fail when clicked. */
export function pickRecordingFormat(): RecordingFormat | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const candidate of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(candidate.mimeType)) return candidate;
  }
  // No candidate matched this browser's own supported list — still try
  // recording with no mimeType hint (the browser picks its own default
  // container) rather than refusing outright; ext defaults to webm,
  // the overwhelmingly likely real result of an unhinted request.
  return { mimeType: "", ext: "webm" };
}

/** Owns one recording session's worth of MediaRecorder/MediaStream state
 * — construct once per GridCanvas instance (see that component's own
 * ref), reused across repeated start()/stop() cycles rather than
 * recreated each time, since pickRecordingFormat()'s own result never
 * changes for a given browser session. */
export class CanvasRecorder {
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: Blob[] = [];

  readonly format: RecordingFormat | null = pickRecordingFormat();

  get isSupported(): boolean {
    return this.format !== null;
  }

  get isRecording(): boolean {
    return this.recorder !== null && this.recorder.state === "recording";
  }

  /** Starts capturing `canvas`'s own rendered output at CAPTURE_FPS. A
   * no-op if already recording (repeated clicks on an already-"rec"
   * button shouldn't restart the capture). Throws if this browser has
   * no MediaRecorder at all — check isSupported first (TrainingView
   * disables the button instead of ever calling this then). */
  start(canvas: HTMLCanvasElement): void {
    if (!this.format) throw new Error("MediaRecorder is not supported in this browser");
    if (this.isRecording) return;
    this.stream = canvas.captureStream(CAPTURE_FPS);
    this.chunks = [];
    this.recorder = this.format.mimeType
      ? new MediaRecorder(this.stream, { mimeType: this.format.mimeType })
      : new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.recorder.start();
  }

  /** Stops recording and triggers a browser download of the captured
   * video — a plain <a download> click on a blob: URL, the standard
   * client-side "save this file" pattern (this is a normal page running
   * in the user's own browser tab, not a sandboxed context that would
   * block a real download the way a published Artifact would).
   * `filenamePrefix` gets a timestamp and the real container's own
   * extension appended (e.g. "mpm-training" ->
   * "mpm-training-2026-08-21T12-00-00.mp4"). Resolves once the file's
   * been handed to the browser's own download flow; a no-op (resolves
   * immediately) if not currently recording. */
  stop(filenamePrefix: string): Promise<void> {
    return new Promise((resolve) => {
      const recorder = this.recorder;
      if (!recorder || recorder.state !== "recording") {
        resolve();
        return;
      }
      recorder.onstop = () => {
        const ext = this.format?.ext ?? "webm";
        const blob = new Blob(this.chunks, { type: this.format?.mimeType || `video/${ext}` });
        const url = URL.createObjectURL(blob);
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const a = document.createElement("a");
        a.href = url;
        a.download = `${filenamePrefix}-${timestamp}.${ext}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        this.stream?.getTracks().forEach((track) => track.stop());
        this.stream = null;
        this.recorder = null;
        this.chunks = [];
        resolve();
      };
      recorder.stop();
    });
  }
}

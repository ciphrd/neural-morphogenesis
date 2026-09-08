import { isCutPush, type CutPush } from "../gpu/cutPush"
import { useCallback, useEffect, useRef } from "react"

export const OUTPUT_PREVIEW_CHANNEL = "mpm-output-preview-v1"
export const OUTPUT_PREVIEW_INTERVAL_MS = 100
export const OUTPUT_PREVIEW_TIMEOUT_MS = 2000
const PREVIEW_SIZE = 320

export type OutputPreviewMessage =
  | { type: "request"; id: string }
  | { type: "frame"; id: string; image: Blob | null; zoom: number }
  | { type: "cut-push"; push: CutPush }

/** Copy after rendering, before the WebGPU canvas is presented and discarded. */
export function useOutputPreviewSender( onCut: (push: CutPush) => void) {
  const channelRef = useRef<BroadcastChannel | null>(null)
  const pendingRef = useRef<string | null>(null)
  const busyRef = useRef(false)
  const thumbnailRef = useRef<HTMLCanvasElement | null>(null)
  const cutRef = useRef(onCut)
  cutRef.current = onCut

  useEffect(() => {
    const channel = new BroadcastChannel(OUTPUT_PREVIEW_CHANNEL)
    channelRef.current = channel
    channel.onmessage = (event: MessageEvent<OutputPreviewMessage>) => {
      if (event.data.type === "cut-push" && isCutPush(event.data.push)) cutRef.current(event.data.push)
      if (event.data.type === "request" && !busyRef.current) pendingRef.current = event.data.id
    }
    return () => {
      channel.close()
      channelRef.current = null
      pendingRef.current = null
      thumbnailRef.current = null
    }
  }, [])

  return useCallback((source: HTMLCanvasElement, zoom: number) => {
    const id = pendingRef.current
    const channel = channelRef.current
    if (!id || !channel || busyRef.current) return
    pendingRef.current = null
    busyRef.current = true
    const finish = (image: Blob | null) => {
      busyRef.current = false
      if (channelRef.current === channel) {
        channel.postMessage({ type: "frame", id, image, zoom } satisfies OutputPreviewMessage)
      }
    }
    try {
      const canvas = thumbnailRef.current ?? document.createElement("canvas")
      if (!thumbnailRef.current) {
        canvas.width = PREVIEW_SIZE
        canvas.height = PREVIEW_SIZE
        thumbnailRef.current = canvas
      }
      const context = canvas.getContext("2d", { alpha: false })
      if (!context) { finish(null); return }
      context.fillStyle = "#000"
      context.fillRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE)
      context.drawImage(source, 0, 0, PREVIEW_SIZE, PREVIEW_SIZE)
      canvas.toBlob(finish, "image/jpeg", 0.8)
    } catch {
      finish(null)
    }
  }, [])
}

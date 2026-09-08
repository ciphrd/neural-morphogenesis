import { makeCutPush, type CutPoint } from "../gpu/cutPush"
import { useEffect, useRef, useState, type PointerEvent } from "react"
import {
  OUTPUT_PREVIEW_CHANNEL,
  OUTPUT_PREVIEW_INTERVAL_MS,
  OUTPUT_PREVIEW_TIMEOUT_MS,
  type OutputPreviewMessage,
} from "../performance/outputPreview"
import { ToggleButton } from "./ToggleButton"

export function OutputPreview({ connected, enabled, setEnabled, cutEnabled, setCutEnabled }: {
  connected: boolean
  enabled: boolean
  setEnabled: (value: boolean) => void
  cutEnabled: boolean
  setCutEnabled: (value: boolean) => void
}) {
  const [image, setImage] = useState<Blob | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [status, setStatus] = useState("Waiting for preview…")
  const [stroke, setStroke] = useState<CutPoint[]>([])
  const [zoom, setZoom] = useState(1)
  const channelRef = useRef<BroadcastChannel | null>(null)
  const dragRef = useRef<{ pointer: number; point: CutPoint; time: number; trail: CutPoint[] } | null>(null)


  useEffect(() => {
    setImage(null)
    setStatus("Waiting for preview…")
    if (!enabled || !connected) return
    const channel = new BroadcastChannel(OUTPUT_PREVIEW_CHANNEL)
    channelRef.current = channel
    let pending: { id: string; time: number } | null = null
    let lastFrameAt = performance.now()
    channel.onmessage = (event: MessageEvent<OutputPreviewMessage>) => {
      const message = event.data
      if (message.type !== "frame" || message.id !== pending?.id) return
      pending = null
      if (message.image) {
        setZoom(message.zoom || 1)
        lastFrameAt = performance.now()
        setImage(message.image)
        setStatus("")
      } else {
        setImage(null)
        setStatus("Preview unavailable")
      }
    }
    const request = () => {
      const now = performance.now()
      if (now - lastFrameAt > OUTPUT_PREVIEW_TIMEOUT_MS) {
        setImage(null)
        setStatus("Waiting for preview…")
      }
      if (document.hidden || (pending && now - pending.time < OUTPUT_PREVIEW_TIMEOUT_MS)) return
      pending = { id: crypto.randomUUID(), time: now }
      channel.postMessage({ type: "request", id: pending.id } satisfies OutputPreviewMessage)
    }
    request()
    const timer = window.setInterval(request, OUTPUT_PREVIEW_INTERVAL_MS)

    return () => {
      window.clearInterval(timer)
      channel.close()
      channelRef.current = null
      dragRef.current = null
      setStroke([])
    }
  }, [connected, enabled])

  useEffect(() => {
    if (!image) { setUrl(null); return }
    const nextUrl = URL.createObjectURL(image)
    setUrl(nextUrl)
    return () => URL.revokeObjectURL(nextUrl)
  }, [image])

  const pointAt = (event: PointerEvent<HTMLDivElement>): CutPoint => {
    const rect = event.currentTarget.getBoundingClientRect()
    return { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) }
  }
  const beginCut = (event: PointerEvent<HTMLDivElement>) => {
    if (!cutEnabled || !connected || !url || status || event.button !== 0 || dragRef.current) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const point = pointAt(event)
    dragRef.current = { pointer: event.pointerId, point, time: event.timeStamp, trail: [point] }
    setStroke([point])
  }
  const extendCut = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointer !== event.pointerId || !channelRef.current) return
    const point = pointAt(event)
    const push = makeCutPush(drag.point, point, event.timeStamp - drag.time, zoom, event.currentTarget.clientWidth)
    if (push && connected && url && !status) {
      channelRef.current.postMessage({ type: "cut-push", push } satisfies OutputPreviewMessage)
    }
    drag.point = point
    drag.time = event.timeStamp
    drag.trail = [...drag.trail.slice(-7), point]
    setStroke(drag.trail)
  }
  const finishCut = (event: PointerEvent<HTMLDivElement>, cancel = false) => {
    const drag = dragRef.current
    if (!drag || drag.pointer !== event.pointerId) return
    if (!cancel) extendCut(event)
    dragRef.current = null
    setStroke([])
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }


  return (
    <div className="output-preview">
      <div className="output-preview-header">
        <ToggleButton label="Preview" checked={enabled} onChange={setEnabled} />
        <div className="output-preview-tools">
          <span>10 FPS</span>
          <button type="button" className="select output-cut-tool" aria-pressed={cutEnabled}
            disabled={!enabled || !connected} title="Drag to cut a narrow path through the material." onClick={() => setCutEnabled(!cutEnabled)}>Cut</button>
        </div>
      </div>
      {enabled && <div className={"output-preview-frame" + (cutEnabled ? " is-cutting" : "")}
        onPointerDown={beginCut} onPointerMove={extendCut} onPointerUp={event => finishCut(event)}
        onPointerCancel={event => finishCut(event, true)} onLostPointerCapture={event => finishCut(event, true)}
        onContextMenu={event => { if (cutEnabled) event.preventDefault() }}>
        {connected && url && !status
          ? <img src={url} alt="Live rendered output preview" draggable={false} />
          : <span>{connected ? status || "Waiting for preview…" : "Offline"}</span>}
        {stroke.length > 0 && <svg className="output-cut-stroke" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
          <polyline points={stroke.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#f87171" strokeWidth="4" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />
        </svg>}
      </div>}
      {enabled && cutEnabled && <p className="output-cut-hint" role="status">Drag to cut material along the stroke.</p>}
    </div>
  )
}

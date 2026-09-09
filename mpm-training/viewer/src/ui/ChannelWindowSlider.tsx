import { useRef, type PointerEvent, type KeyboardEvent } from "react"

interface ChannelWindowSliderProps {
  channels: number
  value: number
  size: number
  onChange: (value: number, size: number) => void
  channelKind?: string
}

/** Native range input for moving the window, with independently resizable edges. */
export function ChannelWindowSlider({ channels, value, size, onChange, channelKind = "substrate" }: ChannelWindowSliderProps) {
  const channelCount = Math.max(1, Math.floor(channels))
  const windowSize = Math.max(1, Math.min(3, channelCount, Math.floor(size)))
  const maxStart = channelCount - windowSize
  const start = Math.min(maxStart, Math.max(0, Math.floor(value)))
  const end = start + windowSize - 1
  const trackRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ x: number; start: number; end: number; pitch: number } | null>(null)

  function resize(edge: "left" | "right", next: number, first = start, last = end) {
    if (edge === "left") {
      const nextStart = Math.max(0, last - 2, Math.min(last, next))
      onChange(nextStart, last - nextStart + 1)
    } else {
      const nextEnd = Math.min(channelCount - 1, first + 2, Math.max(first, next))
      onChange(first, nextEnd - first + 1)
    }
  }

  function beginResize(event: PointerEvent<HTMLButtonElement>) {
    if (event.button !== 0 || !trackRef.current) return
    event.preventDefault()
    event.currentTarget.focus()
    event.currentTarget.setPointerCapture(event.pointerId)
    const width = trackRef.current.getBoundingClientRect().width
    const gap = parseFloat(getComputedStyle(trackRef.current).columnGap) || 0
    dragRef.current = { x: event.clientX, start, end, pitch: (width + gap) / channelCount }
  }

  function moveResize(event: PointerEvent<HTMLButtonElement>, edge: "left" | "right") {
    const drag = dragRef.current
    if (!drag || !event.currentTarget.hasPointerCapture(event.pointerId)) return
    const delta = Math.round((event.clientX - drag.x) / drag.pitch)
    resize(edge, (edge === "left" ? drag.start : drag.end) + delta, drag.start, drag.end)
  }

  function keyResize(event: KeyboardEvent<HTMLButtonElement>, edge: "left" | "right") {
    const current = edge === "left" ? start : end
    const next = event.key === "ArrowLeft" || event.key === "ArrowDown" ? current - 1
      : event.key === "ArrowRight" || event.key === "ArrowUp" ? current + 1
      : event.key === "Home" ? (edge === "left" ? Math.max(0, end - 2) : start)
      : event.key === "End" ? (edge === "left" ? end : Math.min(channelCount - 1, start + 2))
      : null
    if (next === null) return
    event.preventDefault()
    resize(edge, next)
  }

  return (
    <div ref={trackRef} className="channel-window-slider" style={{ gridTemplateColumns: `repeat(${channelCount}, minmax(0, 1fr))` }}>
      {Array.from({ length: channelCount }, (_, channel) => {
        const offset = channel - start
        const selected = offset >= 0 && offset < windowSize
        return (
          <span key={channel}
            className={`channel-window-cell${selected ? ` is-selected is-${["red", "green", "blue"][offset]}` : ""}`}
            style={{ gridColumn: channel + 1 }}
            title={selected ? `Channel ${channel} → ${"RGB"[offset]}` : `Channel ${channel}`}>
            {channel}
          </span>
        )
      })}
      <span className="channel-window-selection" style={{ gridColumn: `${start + 1} / span ${windowSize}` }} aria-hidden="true" />
      <input className="channel-window-input" type="range" min={0} max={maxStart} step={1} value={start}
        aria-label={`First ${channelKind} channel displayed as RGB`}
        aria-valuetext={`Channels ${start} through ${end}`}
        onChange={(event) => onChange(Number(event.currentTarget.value), windowSize)} />
      {(["left", "right"] as const).map((edge) => (
        <button key={edge} type="button" role="slider"
          className={`channel-window-handle is-${edge}`}
          style={{ gridColumn: edge === "left" ? start + 1 : end + 1 }}
          aria-label={`${edge === "left" ? "First" : "Last"} ${channelKind} channel — resize selection`}
          aria-valuemin={edge === "left" ? Math.max(0, end - 2) : start}
          aria-valuemax={edge === "left" ? end : Math.min(channelCount - 1, start + 2)}
          aria-valuenow={edge === "left" ? start : end}
          aria-valuetext={`Channels ${start} through ${end}, ${windowSize} selected`}
          title="Drag to resize selection (1–3 channels)"
          onPointerDown={beginResize}
          onPointerMove={(event) => moveResize(event, edge)}
          onPointerUp={(event) => { event.currentTarget.releasePointerCapture(event.pointerId); dragRef.current = null }}
          onLostPointerCapture={() => { dragRef.current = null }}
          onKeyDown={(event) => keyResize(event, edge)} />
      ))}
    </div>
  )
}

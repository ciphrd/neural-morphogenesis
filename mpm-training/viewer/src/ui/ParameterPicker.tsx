import { useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import type { AudioTarget, AudioTargetSpec } from "../audio/audioReactivity"

/** Uses explicit parameter identities, never translated labels or displayed values. */
export function ParameterPicker({ targets, onPick }: {
  targets: readonly AudioTargetSpec[]
  onPick: (target: AudioTarget) => void
}) {
  const [active, setActive] = useState(false)
  const button = useRef<HTMLButtonElement>(null)
  const line = useRef<SVGLineElement>(null)
  const latest = useRef({ targets, onPick })
  latest.current = { targets, onPick }

  useEffect(() => {
    if (!active) return
    let hovered: Element | null = null
    let frame = 0
    const origin = button.current!.getBoundingClientRect()
    let cursor = { x: origin.x + origin.width / 2, y: origin.y + origin.height / 2 }
    const targetAt = (element: EventTarget | null) => {
      if (!(element instanceof Element)) return null
      if (element.closest("button, select, input[type=checkbox]")) return null
      const row = element.closest("[data-audio-target]")
        ?? element.closest("label, .slider-row")?.querySelector("[data-audio-target]")
      const key = row?.getAttribute("data-audio-target")
      return key && latest.current.targets.some(target => target.key === key) ? row! : null
    }
    const highlight = (element: Element | null) => {
      hovered?.classList.remove("is-audio-pick-hover")
      hovered = element
      hovered?.classList.add("is-audio-pick-hover")
    }
    const draw = () => {
      const rect = button.current?.getBoundingClientRect()
      if (!rect || !button.current?.isConnected) { setActive(false); return }
      line.current?.setAttribute("x1", String(rect.x + rect.width / 2))
      line.current?.setAttribute("y1", String(rect.y + rect.height / 2))
      line.current?.setAttribute("x2", String(cursor.x))
      line.current?.setAttribute("y2", String(cursor.y))
      frame = requestAnimationFrame(draw)
    }
    const move = (event: PointerEvent) => {
      cursor = { x: event.clientX, y: event.clientY }
      highlight(targetAt(event.target))
    }
    // Keep the mode alive until click so the underlying slider/button never
    // receives any part of the selecting gesture, including its default action.
    const blockPointer = (event: PointerEvent) => {
      event.preventDefault()
      event.stopImmediatePropagation()
    }
    const click = (event: MouseEvent) => {
      event.preventDefault()
      event.stopImmediatePropagation()
      const row = targetAt(event.target)
      if (row) latest.current.onPick(row.getAttribute("data-audio-target") as AudioTarget)
      setActive(false)
    }
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault(); event.stopImmediatePropagation(); setActive(false)
      }
    }
    const cancel = () => setActive(false)
    document.body.classList.add("is-audio-picking")
    window.addEventListener("pointermove", move, true)
    window.addEventListener("pointerdown", blockPointer, true)
    window.addEventListener("pointerup", blockPointer, true)
    window.addEventListener("click", click, true)
    window.addEventListener("keydown", key, true)
    window.addEventListener("blur", cancel)
    frame = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(frame)
      highlight(null)
      document.body.classList.remove("is-audio-picking")
      window.removeEventListener("pointermove", move, true)
      window.removeEventListener("pointerdown", blockPointer, true)
      window.removeEventListener("pointerup", blockPointer, true)
      window.removeEventListener("click", click, true)
      window.removeEventListener("keydown", key, true)
      window.removeEventListener("blur", cancel)
    }
  }, [active])

  return <>
    <button ref={button} type="button" className="icon-button audio-parameter-picker"
      aria-label="Pick parameter from interface" aria-pressed={active}
      title={active ? "Cancel parameter picking (Escape)" : "Pick a parameter from the interface"}
      onClick={() => setActive(value => !value)}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <circle cx="12" cy="12" r="6" /><path d="M12 2v6m0 8v6M2 12h6m8 0h6" />
      </svg>
    </button>
    {active && createPortal(<>
      <svg className="audio-picker-overlay" aria-hidden="true"><line ref={line} /></svg>
      <div className="audio-picker-hint" role="status">Click a parameter to map it · Escape to cancel</div>
    </>, document.body)}
  </>
}

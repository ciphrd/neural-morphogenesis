import { useEffect, useState } from "react"
import { createPortal } from "react-dom"

function ActionIcon({ kind }: { kind: "restart" | "randomize" | "randomize-restart" | "kill" | "keep-circle" }) {
  if (kind === "randomize-restart") {
    return <span className="performance-icon-pair" aria-hidden="true">
      <ActionIcon kind="randomize" /><ActionIcon kind="restart" />
    </span>
  }
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
    strokeLinecap="square" strokeLinejoin="miter" aria-hidden="true">
    {kind === "restart" ? <path d="M4 10a8 8 0 1 1 1.1 6M4 4v6h6" />
      : kind === "keep-circle" ? <><circle cx="12" cy="12" r="5" /><path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5" /></>
      : kind === "kill" ? <path d="m6 6 12 12M18 6 6 18" />
      : <path d="M3 6h4l10 12h4M17 14l4 4-4 4M3 18h4l3-3.6M14 9.6 17 6h4M17 2l4 4-4 4" />}
  </svg>
}

export function PerformanceToolbar({ onRestart, onRandomize, onKillTwenty, onKillEighty, onKeepCircle, autoPruneCircle, onAutoPruneCircleChange }: {
  onRestart: () => void
  onRandomize: (restart: boolean) => void
  onKillTwenty: () => void
  onKillEighty: () => void
  onKeepCircle: () => void
  autoPruneCircle: boolean
  onAutoPruneCircleChange: (enabled: boolean) => void
}) {
  const [host, setHost] = useState<HTMLElement | null>(null)
  const [focused, setFocused] = useState(0)
  useEffect(() => { setHost(document.getElementById("performance-toolbar-host")) }, [])
  const actions = [
    { label: "Restart", kind: "restart" as const, run: onRestart },
    { label: "Randomize", kind: "randomize" as const, run: () => onRandomize(false) },
    { label: "Randomize and restart", kind: "randomize-restart" as const, run: () => onRandomize(true) },
    { label: "Kill 20%", kind: "kill" as const, badge: "20%", run: onKillTwenty },
    { label: "Kill 80%", kind: "kill" as const, badge: "80%", run: onKillEighty },
    { label: "Kill outside centered circle (diameter: 40% of world width)", kind: "keep-circle" as const, badge: "40%", run: onKeepCircle },
    { label: "Auto-prune outside centered circle (diameter: 90% of world width)", kind: "keep-circle" as const, badge: "90%", pressed: autoPruneCircle, run: () => onAutoPruneCircleChange(!autoPruneCircle) },
  ]
  if (!host) return null
  return createPortal(<div className="performance-toolbar" role="toolbar" aria-label="Simulation actions" onKeyDown={event => {
    let next: number
    if (event.key === "ArrowRight") next = (focused + 1) % actions.length
    else if (event.key === "ArrowLeft") next = (focused + actions.length - 1) % actions.length
    else if (event.key === "Home") next = 0
    else if (event.key === "End") next = actions.length - 1
    else return
    event.preventDefault()
    event.currentTarget.querySelectorAll("button")[next]?.focus()
  }}>
    {actions.map((action, index) => <button key={action.label} type="button"
      className={"performance-tool" + ((action.kind === "kill" || action.kind === "keep-circle") ? " performance-tool-kill" : "")}
      aria-pressed={"pressed" in action ? action.pressed : undefined}
      title={action.label} aria-label={action.label} tabIndex={focused === index ? 0 : -1}
      onFocus={() => setFocused(index)} onClick={action.run}>
      <ActionIcon kind={action.kind} />
      {action.badge && <span className="performance-tool-badge" aria-hidden="true">{action.badge}</span>}
    </button>)}
  </div>, host)
}

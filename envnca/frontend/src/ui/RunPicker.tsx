import { useEffect, useState } from "react"
import type { RunSummary } from "../net/runs"
import { fetchRuns } from "../net/runs"

interface RunPickerProps {
  apiUrl: string
  /** null = following the live/current run. */
  activeRunId: string | null
  onSelectRun: (runId: string | null) => void
}

/** Button + dropdown panel listing every archived run
 * (checkpoints/runs/*, train_server.py's GET /runs) plus the current one
 * (if training has produced at least one generation), each with a
 * preview thumbnail. Fetched fresh every time the panel opens rather
 * than once on mount or kept live — the list only changes when a run
 * finishes and gets archived, and this is a rarely-opened panel, not
 * something that needs to track that in real time. */
export function RunPicker({
  apiUrl,
  activeRunId,
  onSelectRun,
}: RunPickerProps) {
  const [open, setOpen] = useState(false)
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setRuns(null)
    setError(null)
    fetchRuns(apiUrl)
      .then((r) => {
        if (!cancelled) setRuns(r)
      })
      .catch((err) => {
        if (!cancelled) setError(String(err))
      })
    return () => {
      cancelled = true
    }
  }, [open, apiUrl])

  const activeLabel = (() => {
    if (activeRunId === null)
      return <span className="run-picker-live">● Live</span>
    return runs?.find((r) => r.id === activeRunId)?.label ?? activeRunId
  })()

  return (
    <div className="run-picker">
      <button className="playback-button" onClick={() => setOpen((o) => !o)}>
        {open ? "Close" : <>Run: {activeLabel}</>}
      </button>
      {open && (
        <div className="run-picker-panel">
          {error && <p className="hint">Failed to load runs: {error}</p>}
          {!error && runs === null && <p className="hint">Loading…</p>}
          {runs && runs.length === 0 && <p className="hint">No runs yet.</p>}
          {runs?.map((run) => {
            const isActive = run.isLive
              ? activeRunId === null
              : activeRunId === run.id
            return (
              <button
                key={run.id}
                className={"run-picker-item" + (isActive ? " is-active" : "")}
                onClick={() => {
                  onSelectRun(run.isLive ? null : run.id)
                  setOpen(false)
                }}
              >
                <img
                  className="run-picker-preview"
                  src={`${apiUrl}${run.previewUrl}`}
                  alt=""
                  onError={(e) => {
                    e.currentTarget.style.visibility = "hidden"
                  }}
                />
                <span className="run-picker-info">
                  <span className="run-picker-title">
                    {run.target ?? "unknown target"}
                    {run.isLive && (
                      <span className="run-picker-live">● Live</span>
                    )}
                  </span>
                  <span className="run-picker-meta">
                    gen {run.generation ?? "—"} · best{" "}
                    {run.bestFitness != null ? run.bestFitness.toFixed(3) : "—"}
                  </span>
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

import { useState } from "react"
import {
  type PerformanceSequence,
  SEQUENCE_ACTIONS,
  type SequenceAction,
} from "../performance/sequences"
import { ToggleButton } from "./ToggleButton"

interface SequencesPanelProps {
  sequences: PerformanceSequence[]
  onChange: (sequences: PerformanceSequence[]) => void
}

export function SequencesPanel({ sequences, onChange }: SequencesPanelProps) {
  const [open, setOpen] = useState(true)
  const update = (id: string, patch: Partial<PerformanceSequence>) =>
    onChange(
      sequences.map((sequence) =>
        sequence.id === id ? { ...sequence, ...patch } : sequence
      )
    )
  return (
    <section className="sequences-panel">
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen(!open)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>
            ▸
          </span>
          <h2>Sequences</h2>
        </button>
        <button
          className="icon-button"
          aria-label="Add sequence"
          title="Add sequence"
          onClick={() => {
            setOpen(true)
            onChange([
              ...sequences,
              {
                id: crypto.randomUUID(),
                action: "randomize",
                intervalSeconds: 20,
                enabled: false,
              },
            ])
          }}
        >
          +
        </button>
      </div>
      {open && (
        <div className="physics-panel-body audio-panel-body">
          {sequences.map((sequence, index) => (
            <div className="audio-mapping" key={sequence.id}>
              <div className="audio-mapping-top-row">
                <ToggleButton
                  label={`Sequence ${index + 1}`}
                  hideLabel
                  checked={sequence.enabled}
                  onChange={(enabled) => update(sequence.id, { enabled })}
                />
                <select
                  className="select"
                  aria-label={`Sequence ${index + 1} action`}
                  value={sequence.action}
                  onChange={(event) =>
                    update(sequence.id, {
                      action: event.target.value as SequenceAction,
                    })
                  }
                >
                  {SEQUENCE_ACTIONS.map(({ value, label }) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
              <label className="sequence-interval">
                <span>Every</span>
                <input
                  className="number-input"
                  type="number"
                  min="0.1"
                  max="86400"
                  step="0.1"
                  aria-label={`Sequence ${index + 1} interval in seconds`}
                  defaultValue={sequence.intervalSeconds}
                  onBlur={(event) => {
                    const value = Number(event.target.value)
                    const intervalSeconds =
                      Number.isFinite(value) && value >= 0.1
                        ? Math.min(86400, value)
                        : sequence.intervalSeconds
                    event.target.value = String(intervalSeconds)
                    update(sequence.id, { intervalSeconds })
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur()
                  }}
                />
                <span>seconds</span>
              </label>
              <button
                className="audio-remove"
                aria-label={`Remove sequence ${index + 1}`}
                onClick={() =>
                  onChange(sequences.filter(({ id }) => id !== sequence.id))
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

export const SEQUENCE_ACTIONS = [
  { value: "restart", label: "Restart" },
  { value: "randomize", label: "Randomize" },
  { value: "randomize-and-restart", label: "Randomize + restart" },
  { value: "kill-20-percent", label: "Kill 20%" },
  { value: "kill-80-percent", label: "Kill 80%" },
  { value: "prune", label: "Prune 99.5%" },
  { value: "toggle-auto-prune", label: "Toggle auto prune" },
  { value: "blackout", label: "Blackout / restore" },
] as const

export type SequenceAction = typeof SEQUENCE_ACTIONS[number]["value"]
export interface PerformanceSequence {
  id: string
  action: SequenceAction
  intervalSeconds: number
  enabled: boolean
}
export const SEQUENCES_STORAGE_KEY = "mpm-training-performance-sequences-v1"

export function loadSequences(): PerformanceSequence[] {
  try {
    const raw = localStorage.getItem(SEQUENCES_STORAGE_KEY)
    if (raw !== null) {
      const parsed: unknown = JSON.parse(raw)
      if (!Array.isArray(parsed)) return []
      const ids = new Set<string>()
      return parsed.filter((item): item is PerformanceSequence => {
        if (!item || typeof item.id !== "string" || ids.has(item.id)
          || !SEQUENCE_ACTIONS.some(({ value }) => value === item.action)
          || typeof item.enabled !== "boolean" || !Number.isFinite(item.intervalSeconds)
          || item.intervalSeconds < 0.1 || item.intervalSeconds > 86400) return false
        ids.add(item.id)
        return true
      })
    }
    return (["randomize", "reset"] as const).flatMap((legacy) => {
      const prefix = `mpm-training-performance-auto-${legacy}`
      if (localStorage.getItem(`${prefix}-v1`) !== "true") return []
      const duration = Number(localStorage.getItem(`${prefix}-duration-v1`) ?? 20)
      return [{ id: `legacy-${legacy}`, action: legacy === "reset" ? "restart" : "randomize",
        intervalSeconds: Number.isFinite(duration) ? Math.max(1, Math.min(120, duration)) : 20,
        enabled: true }]
    })
  } catch {
    return []
  }
}

/** Independent deadlines; editing one sequence leaves all other clocks alone. */
export class SequenceScheduler {
  private deadlines = new Map<string, { signature: string; at: number }>()

  tick(sequences: PerformanceSequence[], now: number): SequenceAction[] {
    const active = new Set(sequences.filter((sequence) => sequence.enabled).map(({ id }) => id))
    for (const id of this.deadlines.keys()) if (!active.has(id)) this.deadlines.delete(id)
    const due: SequenceAction[] = []
    for (const sequence of sequences) {
      if (!sequence.enabled) continue
      const signature = `${sequence.action}:${sequence.intervalSeconds}`
      const existing = this.deadlines.get(sequence.id)
      if (!existing || existing.signature !== signature) {
        this.deadlines.set(sequence.id, { signature, at: now + sequence.intervalSeconds * 1000 })
      } else if (now >= existing.at) {
        due.push(sequence.action)
        // Skip missed repetitions after a suspended/backgrounded tab.
        existing.at = now + sequence.intervalSeconds * 1000
      }
    }
    return due
  }
}

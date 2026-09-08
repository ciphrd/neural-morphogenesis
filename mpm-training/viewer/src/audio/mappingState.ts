import type { AudioMapping, AudioTargetSpec } from "./audioReactivity"

/** Repair IDs left by an older hot-reloaded counter without changing values. */
export function uniqueMappingIds(mappings: AudioMapping[]): AudioMapping[] {
  let nextId = mappings.reduce((max, mapping) => Math.max(max, mapping.id), 0) + 1
  const seen = new Set<number>()
  let changed = false
  const result = mappings.map(mapping => {
    if (seen.has(mapping.id)) {
      changed = true
      return { ...mapping, id: nextId++ }
    }
    seen.add(mapping.id)
    return mapping
  })
  return changed ? result : mappings
}

/** Pure state update: safe to replay, batch, or hot-reload. */
export function appendAudioMapping(mappings: AudioMapping[], targets: readonly AudioTargetSpec[]): AudioMapping[] {
  const current = uniqueMappingIds(mappings)
  const spec = targets.find(({ key }) => !current.some(mapping => mapping.target === key)) ?? targets[0]
  if (!spec) return current
  const id = current.reduce((max, mapping) => Math.max(max, mapping.id), 0) + 1
  return [...current, { id, enabled: true, target: spec.key, min: spec.min, max: spec.max }]
}

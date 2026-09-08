import test from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import ts from "typescript"

const source = readFileSync(new URL("../src/performance/sequences.ts", import.meta.url), "utf8")
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { SequenceScheduler, loadSequences, SEQUENCES_STORAGE_KEY } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`)
const sequence = (id, intervalSeconds, action = "restart") => ({ id, intervalSeconds, action, enabled: true })

test("independent intervals fire together when due and skip missed repetitions", () => {
  const scheduler = new SequenceScheduler()
  const sequences = [sequence("a", 1), sequence("b", 2, "randomize")]
  assert.deepEqual(scheduler.tick(sequences, 0), [])
  assert.deepEqual(scheduler.tick(sequences, 999), [])
  assert.deepEqual(scheduler.tick(sequences, 1000), ["restart"])
  assert.deepEqual(scheduler.tick(sequences, 2000), ["restart", "randomize"])
  assert.deepEqual(scheduler.tick(sequences, 90000), ["restart", "randomize"])
  assert.deepEqual(scheduler.tick(sequences, 90001), [])
})

test("edits reset only the edited clock; disabling and removal cancel pending actions", () => {
  const scheduler = new SequenceScheduler()
  const a = sequence("a", 1), b = sequence("b", 2, "randomize")
  scheduler.tick([a, b], 0)
  const edited = { ...a, action: "kill-20-percent", intervalSeconds: 3 }
  assert.deepEqual(scheduler.tick([edited, b], 500), [])
  assert.deepEqual(scheduler.tick([edited, b], 2000), ["randomize"])
  assert.deepEqual(scheduler.tick([{ ...edited, enabled: false }], 3400), [])
  assert.deepEqual(scheduler.tick([edited], 3500), [])
  assert.deepEqual(scheduler.tick([edited], 6499), [])
  assert.deepEqual(scheduler.tick([edited], 6500), ["kill-20-percent"])
  assert.deepEqual(scheduler.tick([], 10000), [])
})

test("storage migrates legacy settings and rejects malformed sequences", () => {
  const storage = new Map()
  globalThis.localStorage = { getItem: (key) => storage.get(key) ?? null }
  storage.set("mpm-training-performance-auto-reset-v1", "true")
  storage.set("mpm-training-performance-auto-reset-duration-v1", "12")
  assert.deepEqual(loadSequences(), [sequence("legacy-reset", 12)])
  storage.set(SEQUENCES_STORAGE_KEY, "[]")
  assert.deepEqual(loadSequences(), [])
  storage.set(SEQUENCES_STORAGE_KEY, JSON.stringify([sequence("valid", 0.5), sequence("valid", 2), sequence("bad", -1), sequence("unknown", 1, "invalid"), sequence("removed-blackout", 1, "blackout"), sequence("removed-pruning", 1, "toggle-auto-prune"), null]))
  assert.deepEqual(loadSequences(), [sequence("valid", 0.5)])
  storage.set(SEQUENCES_STORAGE_KEY, "broken JSON")
  assert.deepEqual(loadSequences(), [])
})

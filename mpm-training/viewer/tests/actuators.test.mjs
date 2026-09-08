import test from "node:test"
import assert from "node:assert/strict"
import { createServer } from "vite"

test("attractor motion, audio mappings and legacy scene compatibility", async () => {
  const server = await createServer({ configFile: false, server: { middlewareMode: true }, appType: "custom" })
  try {
    const { AttractorMotion, DEFAULT_ATTRACTOR, normalizeAttractor } = await server.ssrLoadModule("/src/performance/actuators.ts")
    const { applyAudioMappings, audioTargetSpecsFor } = await server.ssrLoadModule("/src/audio/audioReactivity.ts")
    const { migrateSnapshot } = await server.ssrLoadModule("/src/performance/migrateSnapshot.ts")
    const motion = new AttractorMotion()
    assert.deepEqual(motion.advance(DEFAULT_ATTRACTOR, 0.1), { x: 0, y: 0.5 })
    const settings = { ...DEFAULT_ATTRACTOR, enabled: true, speed: 1, frequency: 1 }
    const first = motion.advance(settings, 0.1)
    assert.equal(first.x, 0.1)
    assert.ok(first.y > 0.5)
    assert.deepEqual(motion.advance(settings, 0), first) // paused
    assert.deepEqual(motion.advance({ ...settings, enabled: false }, 0.1), first)
    const changed = motion.advance({ ...settings, speed: 2 }, 0.1)
    assert.ok(Math.abs(changed.x - 0.3) < 1e-10)
    for (let i = 0; i < 100; i++) {
      const point = motion.advance({ ...settings, centerY: 0.9 }, 0.1)
      assert.ok(point.x >= 0 && point.x < 1 && point.y >= 0 && point.y <= 1)
    }
    const base = { physics: null, render: { autoZoom: {}, bloom: {} }, noiseDisplacementStrength: 0, attractor: settings }
    const targets = audioTargetSpecsFor(base).filter(target => target.group === "Actuators")
    assert.equal(targets.length, 7)
    const mappings = targets.map((target, id) => ({ id, target: target.key, min: target.min, max: target.max }))
    const low = applyAudioMappings(base, mappings, 0, true)
    const high = applyAudioMappings(base, mappings, 1, true)
    assert.equal(low.attractor.strength, -10)
    assert.equal(high.attractor.strength, 10)
    assert.equal(base.attractor.strength, 1) // mapping must not mutate manual settings
    assert.equal(applyAudioMappings(base, mappings, 1, false), base)
    assert.equal(applyAudioMappings(base, mappings.map(m => ({ ...m, enabled: false })), 1, true), base)
    assert.equal(normalizeAttractor({ ...settings, range: -1 }).range, 0)
    assert.equal(normalizeAttractor({ ...settings, frequency: NaN }).frequency, DEFAULT_ATTRACTOR.frequency)
    const legacy = { ...base }; delete legacy.attractor
    assert.deepEqual(migrateSnapshot(legacy, base).attractor, DEFAULT_ATTRACTOR)
  } finally { await server.close() }
})

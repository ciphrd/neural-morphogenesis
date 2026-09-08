import test from "node:test"
import assert from "node:assert/strict"
import { createServer } from "vite"

test("performance starts with fresh compatible policies without browser storage or a backend", async () => {
  const server = await createServer({ configFile: false, server: { middlewareMode: true }, appType: "custom" })
  try {
    const { createPerformanceConfig } = await server.ssrLoadModule("/src/performance/config.ts")
    const { policyWeightsShapeError } = await server.ssrLoadModule("/src/gpu/policyEval.ts")
    const first = createPerformanceConfig(42)
    const second = createPerformanceConfig(43)
    assert.equal(policyWeightsShapeError(first.weights, first.channels, first.hiddenDim, first.policyArchitecture), null)
    assert.notDeepEqual(first.weights, second.weights)
    assert.equal(first.generation, -1)
    assert.equal(first.stableStop, false)
    assert.equal(first.growthSteps, null)
    const { migrateSnapshot } = await server.ssrLoadModule("/src/performance/migrateSnapshot.ts")
    const { VIEWER_DEFAULTS } = await server.ssrLoadModule("/src/viewerConfig.ts")
    const { physicsSettingsFromConfig } = await server.ssrLoadModule("/src/gpu/types.ts")
    const defaults = { physics: physicsSettingsFromConfig(first), render: VIEWER_DEFAULTS.rendering,
      particleCap: 4000, initialParticleCount: 20, noiseDisplacementStrength: 0,
      paused: false, loopAtTrainedSteps: false, blackout: false }
    const migrated = migrateSnapshot({ ...defaults, physics: { gravity: 12, steeringStrength: 1 },
      render: { particleRenderMode: "dots-neural-color", neuralColorAlpha: 0.6, zoom: 2 } }, defaults)
    assert.equal(migrated.render.particleColorMode, "neural-color")
    assert.equal(migrated.render.particleAlpha, 0.6)
    assert.equal(migrated.render.zoom, 2)
    assert.equal(migrated.physics.gravity, 12)
    assert.equal(migrated.physics.growthSpeedMultiplier, defaults.physics.growthSpeedMultiplier)
    assert.equal("steeringStrength" in migrated.physics, false)
  } finally {
    await server.close()
  }
})

import { randomWeights } from "../gpu/agents"
import { policyArchitectureForCellMemory, type SimulationConfig } from "../gpu/types"
import { DEFAULT_RUN_SETTINGS } from "../net/settingsStorage"

/** Performance sessions never consume cached settings or checkpoint weights. */
export function createPerformanceConfig(seed: number, fastAccumulation = true): SimulationConfig {
  const cellMemory = DEFAULT_RUN_SETTINGS.cellMemory ?? "recurrent"
  const policyArchitecture = policyArchitectureForCellMemory(cellMemory)
  return {
    ...DEFAULT_RUN_SETTINGS,
    hiddenLayers: [DEFAULT_RUN_SETTINGS.hiddenDim],
    cellMemory,
    policyArchitecture,
    fastAccumulation,
    stableStop: false,
    shapeTarget: undefined,
    growthSteps: null,
    generation: -1,
    best: NaN,
    mean: NaN,
    worst: NaN,
    allTimeBest: NaN,
    seed,
    weights: randomWeights(DEFAULT_RUN_SETTINGS.channels, DEFAULT_RUN_SETTINGS.hiddenDim, policyArchitecture, seed),
  }
}

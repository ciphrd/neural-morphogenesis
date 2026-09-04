import policyParameters from "../../../core/policy_parameters.json"
import {
  policyHasRecurrence,
  type PolicyArchitecture,
  type UpdateRuleWeights,
} from "./types"

type HeadName = keyof typeof policyParameters.heads

function gaussian(random: () => number): () => number {
  let spare: number | null = null
  return () => {
    if (spare !== null) {
      const value = spare
      spare = null
      return value
    }
    const u = Math.max(Number.MIN_VALUE, random())
    const v = random()
    const radius = Math.sqrt(-2 * Math.log(u))
    const angle = 2 * Math.PI * v
    spare = radius * Math.sin(angle)
    return radius * Math.cos(angle)
  }
}

function repeatedScale(name: HeadName, count: number): number[] {
  return Array.from(
    { length: count },
    () => policyParameters.heads[name].mutationScale
  )
}

/** Browser-side equivalent of trainer/evolve.py's mutate(): independent
 * Gaussian noise for every weight and bias, with the same per-head scale
 * buckets from core/policy_parameters.json. */
export function mutatePolicyWeights(
  weights: UpdateRuleWeights,
  channels: number,
  architecture: PolicyArchitecture,
  sigma: number,
  random: () => number = Math.random
): UpdateRuleWeights {
  if (!Number.isFinite(sigma) || sigma < 0) {
    throw new Error("mutation sigma must be finite and non-negative")
  }

  const headScales = [
    ...repeatedScale("chemical", channels),
    ...repeatedScale("anisotropy", 1),
    ...repeatedScale("division", 1),
    ...repeatedScale("growthDirection", 2),
    ...repeatedScale("divisionDrive", 1),
    ...(policyHasRecurrence(architecture)
      ? [
          ...repeatedScale("stateDelta", 8),
          ...repeatedScale("stateGate", 8),
        ]
      : repeatedScale("color", 3)),
  ]
  if (
    headScales.length !== weights.fc2w.length ||
    headScales.length !== weights.fc2b.length
  ) {
    throw new Error("policy output shape does not match its mutation buckets")
  }

  const normal = gaussian(random)
  const mutate = (value: number, scale: number) =>
    value + normal() * sigma * scale
  const trunkScale = policyParameters.trunk.mutationScale
  return {
    fc1w: weights.fc1w.map((row) =>
      row.map((value) => mutate(value, trunkScale))
    ),
    fc1b: weights.fc1b.map((value) => mutate(value, trunkScale)),
    fc2w: weights.fc2w.map((row, output) =>
      row.map((value) => mutate(value, headScales[output]))
    ),
    fc2b: weights.fc2b.map((value, output) =>
      mutate(value, headScales[output])
    ),
  }
}

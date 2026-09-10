import coreConfig from "../../../core/config.json";

import { policyHasRecurrence, type PolicyArchitecture, type UpdateRuleWeights } from "./types";

export interface PolicyOutput {
  /** One signed chemical delta rate per channel. */
  envWrite: Float32Array;
  growthVector: [number, number];
  color: [number, number, number] | null;
  stateDelta: Float32Array;
  stateGate: Float32Array;
}

/** Validates serialized weights at the untyped network boundary. A training
 * process can remain alive across a viewer hot-reload and keep broadcasting
 * weights from the former output architecture. */
export function policyWeightsShapeError(
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number,
  architecture: PolicyArchitecture = "stateless-128",
): string | null {
  const stateful = policyHasRecurrence(architecture);
  const inDim = channels * 3 + 6 + (stateful ? 8 : 0);
  const outDim = channels + (stateful ? 18 : 2) + (coreConfig.coloring.source === "neural" ? 3 : 0);
  const fc1w = weights?.fc1w;
  const fc1b = weights?.fc1b;
  const fc2w = weights?.fc2w;
  const fc2b = weights?.fc2b;
  const valid =
    Array.isArray(fc1w) &&
    fc1w.length === hiddenDim &&
    fc1w.every((row) => Array.isArray(row) && row.length === inDim) &&
    Array.isArray(fc1b) &&
    fc1b.length === hiddenDim &&
    Array.isArray(fc2w) &&
    fc2w.length === outDim &&
    fc2w.every((row) => Array.isArray(row) && row.length === hiddenDim) &&
    Array.isArray(fc2b) &&
    fc2b.length === outDim;
  if (valid) return null;

  const receivedIn = Array.isArray(fc1w) && Array.isArray(fc1w[0]) ? fc1w[0].length : "missing";
  const receivedOut = Array.isArray(fc2w) ? fc2w.length : "missing";
  return (
    `Incompatible policy weights: expected ${inDim} inputs and ${outDim} outputs ` +
    `(chemical deltas plus a 2-D growth vector and state/RGB outputs), but received ${receivedIn} inputs and ${receivedOut} output rows. ` +
    "The configured coloring source determines whether RGB outputs exist; use matching core/config.json settings and restart the backend for incompatible checkpoints."
  );
}

function safeSigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

export function evalPolicy(
  input: Float32Array,
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number,
  maxEnvWrite: number,
  architecture: PolicyArchitecture = "stateless-128",
): PolicyOutput {
  const shapeError = policyWeightsShapeError(weights, channels, hiddenDim, architecture);
  if (shapeError) throw new Error(shapeError);
  const hidden = new Float32Array(hiddenDim);
  for (let j = 0; j < hiddenDim; j++) {
    let acc = weights.fc1b[j];
    const row = weights.fc1w[j];
    for (let i = 0; i < row.length; i++) acc += (input[i] ?? 0) * row[i];
    hidden[j] = Math.max(acc, 0);
  }

  const stateful = policyHasRecurrence(architecture);
  const outDim = channels + (stateful ? 18 : 2) + (coreConfig.coloring.source === "neural" ? 3 : 0);
  const outVec = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let acc = weights.fc2b[j];
    const row = weights.fc2w[j];
    for (let i = 0; i < hiddenDim; i++) acc += hidden[i] * row[i];
    outVec[j] = acc;
  }

  const envWriteDim = channels;
  const envWrite = new Float32Array(envWriteDim);
  for (let k = 0; k < envWriteDim; k++) envWrite[k] = outVec[k] * maxEnvWrite;
  const growthVector: [number, number] = [
    outVec[envWriteDim],
    outVec[envWriteDim + 1],
  ];
  const stateDelta = new Float32Array(8);
  const stateGate = new Float32Array(8);
  let color: [number, number, number] | null;
  if (stateful) {
    for (let i = 0; i < 8; i++) {
      stateDelta[i] = outVec[envWriteDim + 2 + i];
      stateGate[i] = safeSigmoid(outVec[envWriteDim + 10 + i]);
    }
  }
  const colorOffset = envWriteDim + (stateful ? 18 : 2);
  // Substrate color requires raw spatial field samples, unavailable to this NN-only evaluator.
  color = coreConfig.coloring.source === "substrate" ? null : [
    safeSigmoid(outVec[colorOffset]),
    safeSigmoid(outVec[colorOffset + 1]),
    safeSigmoid(outVec[colorOffset + 2]),
  ];
  return { envWrite, growthVector, color, stateDelta, stateGate };
}

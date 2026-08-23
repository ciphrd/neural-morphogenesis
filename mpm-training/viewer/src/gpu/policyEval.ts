// Pure-JS reimplementation of core/agents.wgsl's own evalPolicy() —
// mirrors that function's exact math (Dense(hiddenDim) -> sin ->
// Dense(outDim), tanh-squashed output with per-channel scaling) so
// ui/NetworkPanel.tsx can visualize the CURRENT generation's policy as
// a response surface (see that component's own module docstring) using
// activeConfig.weights directly (already plain JS number[][]/number[] —
// see gpu/types.ts's own UpdateRuleWeights — no GPU round-trip needed).
// This does NOT replace the real GPU forward pass agentStep() runs
// during an actual rollout, and deliberately excludes CHIRALITY's own
// mirror-averaging (see agents.wgsl's own module docstring for what
// that does). This inspector deliberately shows the raw weight matrix's
// response to the manually supplied vector; live simulation additionally
// enforces chirality by evaluating the mirrored lateral gradient and
// combining the two responses.

import type { UpdateRuleWeights } from "./types";

export interface PolicyOutput {
  /** channels*4, spot-major: front, left, back, right. */
  envWrite: Float32Array;
  angularAccel: number;
  anisotropy: number;
  divisionBias: number;
  direction: [number, number];
  color: [number, number, number];
}

/** Validates serialized weights at the untyped network boundary. A training
 * process can remain alive across a viewer hot-reload and keep broadcasting
 * weights from the former output architecture. */
export function policyWeightsShapeError(
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number
): string | null {
  const inDim = channels * 3 + 6;
  const outDim = channels * 4 + 8;
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
    `(four directional chemical writes plus growth/RGB outputs), but received ${receivedIn} inputs and ${receivedOut} output rows. ` +
    "The current first layer includes three elastic Hencky-strain inputs; restart the training backend and retrain older checkpoints."
  );
}

// Same overflow guard as agents.wgsl's own safeTanh() — naive tanh via
// (e^2x-1)/(e^2x+1) isn't at risk in JS's own Math.tanh the way it was
// on that WGSL backend, but clamping first costs nothing and keeps this
// a faithful line-for-line port rather than a "close enough" one.
function safeTanh(x: number): number {
  return Math.tanh(Math.max(-20, Math.min(20, x)));
}

function safeSigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

/** One Dense(hiddenDim) -> sin -> Dense(channels*4+8) forward pass,
 * squashed exactly like agents.wgsl's own evalPolicy() — see that
 * function's own comment for the exact math this mirrors, and this
 * file's own module docstring for why CHIRALITY's mirror-averaging is
 * NOT applied here. `input` must be exactly channels*3+6 long
 * ([chemical value/forward/lateral, morphology occupancy/forward/lateral,
 * elastic volume/axial/shear strain] — see
 * agents.wgsl's own IN_DIM). */
export function evalPolicy(
  input: Float32Array,
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number,
  maxEnvWrite: number,
  maxAngularAccel: number,
  _maxStrafe: number
): PolicyOutput {
  const shapeError = policyWeightsShapeError(weights, channels, hiddenDim);
  if (shapeError) throw new Error(shapeError);
  const hidden = new Float32Array(hiddenDim);
  for (let j = 0; j < hiddenDim; j++) {
    let acc = weights.fc1b[j];
    const row = weights.fc1w[j];
    for (let i = 0; i < input.length; i++) acc += input[i] * row[i];
    hidden[j] = Math.sin(acc);
  }

  const outDim = channels * 4 + 8;
  const outVec = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let acc = weights.fc2b[j];
    const row = weights.fc2w[j];
    for (let i = 0; i < hiddenDim; i++) acc += hidden[i] * row[i];
    outVec[j] = acc;
  }

  const envWriteDim = channels * 4;
  const envWrite = new Float32Array(envWriteDim);
  for (let k = 0; k < envWriteDim; k++) envWrite[k] = safeTanh(outVec[k]) * maxEnvWrite;
  const angularAccel = safeTanh(outVec[envWriteDim]) * maxAngularAccel;
  // The two former acceleration outputs are independent bounded growth
  // controls; the former strafe pair supplies direction only.
  const anisotropy = safeSigmoid(outVec[envWriteDim + 1]);
  const divisionBias = safeSigmoid(outVec[envWriteDim + 2]);
  const rawX = safeTanh(outVec[envWriteDim + 3]);
  const rawY = safeTanh(outVec[envWriteDim + 4]);
  const magnitude = Math.hypot(rawX, rawY);
  const direction: [number, number] = magnitude > 1e-6
    ? [rawX / magnitude, rawY / magnitude]
    : [0, 0];
  const color: [number, number, number] = [
    safeSigmoid(outVec[envWriteDim + 5]),
    safeSigmoid(outVec[envWriteDim + 6]),
    safeSigmoid(outVec[envWriteDim + 7]),
  ];
  return { envWrite, angularAccel, anisotropy, divisionBias, direction, color };
}

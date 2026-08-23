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
  /** One under-particle chemical write per channel. */
  envWrite: Float32Array;
  angularAccel: number;
  strafe: [number, number];
}

// Same overflow guard as agents.wgsl's own safeTanh() — naive tanh via
// (e^2x-1)/(e^2x+1) isn't at risk in JS's own Math.tanh the way it was
// on that WGSL backend, but clamping first costs nothing and keeps this
// a faithful line-for-line port rather than a "close enough" one.
function safeTanh(x: number): number {
  return Math.tanh(Math.max(-20, Math.min(20, x)));
}

/** One Dense(hiddenDim) -> sin -> Dense(channels+5) forward pass,
 * squashed exactly like agents.wgsl's own evalPolicy() — see that
 * function's own comment for the exact math this mirrors, and this
 * file's own module docstring for why CHIRALITY's mirror-averaging is
 * NOT applied here. `input` must be exactly channels*3+2 long
 * ([value×channels, gradForward×channels, gradLateral×channels, dx,
 * dy] — see agents.wgsl's own IN_DIM). */
export function evalPolicy(
  input: Float32Array,
  weights: UpdateRuleWeights,
  channels: number,
  hiddenDim: number,
  maxEnvWrite: number,
  maxAngularAccel: number,
  _maxStrafe: number
): PolicyOutput {
  const hidden = new Float32Array(hiddenDim);
  for (let j = 0; j < hiddenDim; j++) {
    let acc = weights.fc1b[j];
    const row = weights.fc1w[j];
    for (let i = 0; i < input.length; i++) acc += input[i] * row[i];
    hidden[j] = Math.sin(acc);
  }

  const outDim = channels + 5;
  const outVec = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let acc = weights.fc2b[j];
    const row = weights.fc2w[j];
    for (let i = 0; i < hiddenDim; i++) acc += hidden[i] * row[i];
    outVec[j] = acc;
  }

  const envWriteDim = channels;
  const envWrite = new Float32Array(envWriteDim);
  for (let k = 0; k < envWriteDim; k++) envWrite[k] = safeTanh(outVec[k]) * maxEnvWrite;
  // envWriteDim+1/+2 are the network's own unused "accel" output — see
  // agents.wgsl's own PolicyOutput comment, intentionally skipped here too.
  const angularAccel = safeTanh(outVec[envWriteDim]) * maxAngularAccel;
  const strafe: [number, number] = [
    safeTanh(outVec[envWriteDim + 3]),
    safeTanh(outVec[envWriteDim + 4]),
  ];
  return { envWrite, angularAccel, strafe };
}

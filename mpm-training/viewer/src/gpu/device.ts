// GPU device acquisition — same shape as mls-mpm/src/gpu/device.ts and
// envnca/frontend/src/gpu/device.ts (both identical); reused verbatim
// (just the log prefix changed).

export type GpuAcquireResult = { ok: true; device: GPUDevice } | { ok: false; reason: string };

export async function acquireGpuDevice(): Promise<GpuAcquireResult> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    return { ok: false, reason: "This browser has no navigator.gpu — WebGPU isn't available at all." };
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (err) {
    return { ok: false, reason: `requestAdapter() failed: ${String(err)}` };
  }
  if (!adapter) return { ok: false, reason: "No WebGPU adapter available on this system." };
  let device: GPUDevice;
  try {
    // WebGPU's own spec-default maxStorageBuffersPerShaderStage is only
    // 8 — core/agents.wgsl's own bind group (weights/positions/
    // gridCurrent/gradient/depositScratch/heading/angularVelocity/
    // growthCount/rngState) is at 9 as of growth (see that file's own
    // module docstring), so requestDevice() needs an explicit
    // requiredLimits bump or Chrome/Dawn's own CreateComputePipeline
    // rejects the bind group layout outright. Requesting the adapter's
    // OWN reported max (not a hardcoded higher number) is what the
    // WebGPU spec itself recommends here — some adapters may not have
    // headroom past 8 at all, in which case this just requests exactly
    // what's already the default and changes nothing. The Python trainer
    // (trainer/device.py's own pick_device()) never hit this: wgpu-native's
    // own default device limits are already higher out of the box on
    // this project's own dev machine, unlike Dawn's spec-minimum default.
    device = await adapter.requestDevice({
      requiredLimits: { maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage },
    });
  } catch (err) {
    return { ok: false, reason: `requestDevice() failed: ${String(err)}` };
  }
  return { ok: true, device };
}

export function watchDeviceLoss(device: GPUDevice, onLost: (message: string) => void): void {
  device.lost.then((info) => {
    const message = `WebGPU device lost (${info.reason}): ${info.message}`;
    console.error(`[mpm-training] ${message}`);
    onLost(message);
  });
}

/** Surfaces validation/out-of-memory errors the device didn't attribute
 * to any specific promise (createShaderModule/createBindGroup/etc all
 * fail "successfully," logging async instead of throwing) — without
 * this, a bad pipeline/bind-group setup shows up only as a downstream
 * "invalid texture"/"device lost" cascade with no indication of the
 * actual root cause. */
export function watchUncapturedErrors(device: GPUDevice): void {
  device.addEventListener("uncapturederror", (event) => {
    const gpuError = (event as GPUUncapturedErrorEvent).error;
    console.error(`[mpm-training] WebGPU uncaptured error: ${gpuError.message}`);
  });
}

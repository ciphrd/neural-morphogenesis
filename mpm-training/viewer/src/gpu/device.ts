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
    const requiredFeatures: GPUFeatureName[] = adapter.features.has("float32-filterable")
      ? ["float32-filterable"]
      : [];
    device = await adapter.requestDevice({
      requiredFeatures,
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

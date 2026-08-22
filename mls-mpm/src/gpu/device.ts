/** Adapter/device acquisition, with an explicit "unsupported" result
 * checked against actual adapter presence (not just `navigator.gpu`
 * existing) — a present-but-failed-to-acquire adapter is a worse failure
 * mode (cryptic pipeline-creation errors downstream) than a clean
 * unsupported banner, so this is the one place that decides. */

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
  if (!adapter) {
    return { ok: false, reason: "No WebGPU adapter available on this system." };
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice();
  } catch (err) {
    return { ok: false, reason: `requestDevice() failed: ${String(err)}` };
  }

  return { ok: true, device };
}

export function watchDeviceLoss(device: GPUDevice, onLost: (message: string) => void): void {
  device.lost.then((info) => {
    const message = `WebGPU device lost (${info.reason}): ${info.message}`;
    console.error(`[mls-mpm] ${message}`);
    onLost(message);
  });
}

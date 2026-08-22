// Small shared helpers — same as mls-mpm/src/gpu/gpuUtil.ts and
// envnca/frontend/src/gpu/gpuUtil.ts (both identical).

// The cast works around a real @webgpu/types vs TS 5.7+ lib.dom.d.ts
// TypedArray generic mismatch, not a data bug.
export function writeFloat32(device: GPUDevice, buffer: GPUBuffer, bufferOffset: number, data: Float32Array | Int32Array | Uint32Array): void {
  device.queue.writeBuffer(buffer, bufferOffset, data as unknown as ArrayBuffer);
}

export function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

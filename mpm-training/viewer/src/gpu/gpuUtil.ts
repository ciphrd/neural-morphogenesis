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

/** Maps a flat compute workload onto one or two dispatch dimensions while
 * respecting WebGPU's per-dimension workgroup limit. The shader must flatten
 * gid.x/gid.y using num_workgroups.x and its workgroup size. */
export function flatDispatch2D(
  totalThreads: number,
  workgroupSize: number,
  maxDimension: number,
): [number, number] {
  const groups = ceilDiv(totalThreads, workgroupSize);
  if (groups <= maxDimension) return [groups, 1];
  const x = Math.min(maxDimension, Math.ceil(Math.sqrt(groups)));
  const y = ceilDiv(groups, x);
  if (y > maxDimension) {
    throw new Error(`Workload requires ${groups} workgroups, exceeding the device's 2D dispatch capacity`);
  }
  return [x, y];
}

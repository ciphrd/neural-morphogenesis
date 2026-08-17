/** TS 5.7+'s lib.dom.d.ts made TypedArray constructors generic over
 * ArrayBufferLike (which includes SharedArrayBuffer), while
 * @webgpu/types' GPUQueue.writeBuffer still expects a plain
 * ArrayBufferView<ArrayBuffer> — a real type mismatch between the two
 * packages' declarations, not a bug in the data being passed (every
 * caller here always constructs a plain, non-shared Float32Array). One
 * cast, centralized here, instead of scattered at every call site. */
export function writeFloat32(device: GPUDevice, buffer: GPUBuffer, bufferOffset: number, data: Float32Array): void {
  device.queue.writeBuffer(buffer, bufferOffset, data as unknown as ArrayBuffer);
}

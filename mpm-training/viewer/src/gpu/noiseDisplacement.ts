import noiseDisplacementSrc from "./noiseDisplacement.wgsl?raw"
import { ceilDiv, writeFloat32 } from "./gpuUtil"
import type { MpmCore } from "./mpmCore"

const WORKGROUP_SIZE = 64

/** Viewer-only coherent position displacement, kept outside core/ so the
 * training implementation and its rollout semantics remain unchanged. */
export class NoiseDisplacement {
  private readonly device: GPUDevice
  private readonly mpmCore: MpmCore
  private readonly paramsUniform: GPUBuffer
  private readonly pipeline: GPUComputePipeline
  private readonly bindGroup: GPUBindGroup

  constructor(device: GPUDevice, mpmCore: MpmCore) {
    this.device = device
    this.mpmCore = mpmCore
    this.paramsUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const module = device.createShaderModule({ code: noiseDisplacementSrc })
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "displaceWithNoise" },
    })
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: mpmCore.activeCountUniform } },
        { binding: 2, resource: { buffer: this.paramsUniform } },
      ],
    })
  }

  apply(strength: number, timeSeconds: number): void {
    if (strength <= 0 || this.mpmCore.activeCount === 0) return
    writeFloat32(
      this.device,
      this.paramsUniform,
      0,
      new Float32Array([strength, timeSeconds, 4, 0]),
    )
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(ceilDiv(this.mpmCore.activeCount, WORKGROUP_SIZE))
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  destroy(): void {
    this.paramsUniform.destroy()
  }
}

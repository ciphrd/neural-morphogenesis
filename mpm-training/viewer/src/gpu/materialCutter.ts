import source from "./materialCut.wgsl?raw"
import { isCutPush, type CutPush } from "./cutPush"
import { MAX_PARTICLES, type MpmCore } from "./mpmCore"

const MAX_STROKES = 128

/** Classify first, then let the simulation compact complete surviving records. */
export class MaterialCutter {
  private readonly blades: GPUBuffer
  private readonly victims: GPUBuffer
  private readonly counts: GPUBuffer
  private readonly pipeline: GPUComputePipeline
  private readonly bindings: GPUBindGroup
  private busy = false

  constructor(private readonly device: GPUDevice, private readonly core: Pick<MpmCore, "positions" | "rest" | "activeCount">) {
    this.blades=device.createBuffer({size:MAX_STROKES*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})
    this.victims=device.createBuffer({size:MAX_PARTICLES*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})
    this.counts=device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})
    this.pipeline=device.createComputePipeline({layout:"auto",compute:{module:device.createShaderModule({code:source}),entryPoint:"classifyCut"}})
    this.bindings=device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:core.positions}}, {binding:1,resource:{buffer:core.rest}},
      {binding:2,resource:{buffer:this.blades}}, {binding:3,resource:{buffer:this.victims}},
      {binding:4,resource:{buffer:this.counts}},
    ]})
  }

  async classify(strokes: readonly CutPush[]): Promise<Uint8Array> {
    if (this.busy) throw Error("A material cut is already in progress")
    const count=this.core.activeCount
    const valid=strokes.filter(isCutPush)
    const mask=new Uint8Array(count)
    if (!count || !valid.length) return mask
    this.busy=true
    const staging=this.device.createBuffer({size:count*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})
    try {
      // Union batches without dropping fast or coalesced pointer segments.
      for(let offset=0;offset<valid.length;offset+=MAX_STROKES) {
        const batch=valid.slice(offset,offset+MAX_STROKES)
        const data=new Float32Array(batch.length*8)
        batch.forEach((stroke,i)=>data.set([stroke.from.x,stroke.from.y,stroke.to.x,stroke.to.y,stroke.radius,0,0,0],i*8))
        this.device.queue.writeBuffer(this.blades,0,data)
        this.device.queue.writeBuffer(this.counts,0,new Uint32Array([count,batch.length]))
        const encoder=this.device.createCommandEncoder()
        const pass=encoder.beginComputePass()
        pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindings)
        pass.dispatchWorkgroups(Math.ceil(count/64));pass.end()
        encoder.copyBufferToBuffer(this.victims,0,staging,0,count*4)
        this.device.queue.submit([encoder.finish()])
        await staging.mapAsync(GPUMapMode.READ)
        const flags=new Uint32Array(staging.getMappedRange())
        for(let i=0;i<count;i++) if(flags[i]) mask[i]=1
        staging.unmap()
      }
      return mask
    } finally { staging.destroy();this.busy=false }
  }

  destroy(): void { this.blades.destroy();this.victims.destroy();this.counts.destroy() }
}

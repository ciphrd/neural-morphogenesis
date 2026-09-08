import { VIEWER_DEFAULTS } from "../viewerConfig";
import { advanceStrobePhases, advanceRgbPhases, normalizePostEffects, type PostEffectSettings } from "./postEffects"
import bloomSrc from "./bloom.wgsl?raw"
import { writeFloat32 } from "./gpuUtil"

export const BLOOM_SCENE_FORMAT: GPUTextureFormat = "rgba16float"

export interface BloomSettings extends Partial<PostEffectSettings> {
  enabled: boolean
  intensity: number
  threshold: number
  radiusPx: number
  scatter: number
  levels: number
}

interface Level {
  width: number
  height: number
  down: GPUTexture
  up: GPUTexture
}

export class BloomPostProcess {
  private readonly device: GPUDevice
  private readonly downsamplePipeline: GPURenderPipeline
  private readonly upsamplePipeline: GPURenderPipeline
  private readonly compositePipeline: GPURenderPipeline
  private readonly sampler: GPUSampler
  private readonly compositeUniform: GPUBuffer
  private sceneTexture: GPUTexture
  private levels: Level[] = []
  private downsampleUniforms: GPUBuffer[] = []
  private upsampleUniforms: GPUBuffer[] = []
  private downsampleBindGroups: GPUBindGroup[] = []
  private upsampleBindGroups: GPUBindGroup[] = []
  private compositeBindGroup: GPUBindGroup
  private width = 1
  private height = 1
  private strobePhases = [0, 0, 0]
  private rgbPhases = [0, 1/3, 2/3]
  private rgbLastTime = performance.now()
  private settings: BloomSettings = { ...VIEWER_DEFAULTS.rendering.bloom }

  constructor(device: GPUDevice, outputFormat: GPUTextureFormat) {
    this.device = device
    const module = device.createShaderModule({ code: bloomSrc })
    this.downsamplePipeline = this.createPipeline(module, "downsampleFragment", BLOOM_SCENE_FORMAT)
    this.upsamplePipeline = this.createPipeline(module, "upsampleFragment", BLOOM_SCENE_FORMAT)
    this.compositePipeline = this.createPipeline(module, "compositeFragment", outputFormat)
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    })
    this.compositeUniform = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.sceneTexture = this.createTexture(1, 1)
    this.compositeBindGroup = this.createCompositeBindGroup(this.sceneTexture)
    this.rebuildPyramid()
  }

  get sceneView(): GPUTextureView {
    return this.sceneTexture.createView()
  }

  setSettings(settings: BloomSettings): void {
    this.advanceRgbAnimation()
    const levels = Math.min(10, Math.max(2, Math.floor(settings.levels)))
    const levelsChanged = levels !== this.settings.levels
    this.settings = {
      ...normalizePostEffects(settings),
      enabled: settings.enabled,
      intensity: Math.max(0, settings.intensity),
      threshold: Math.max(0, settings.threshold),
      radiusPx: Math.max(0.25, settings.radiusPx),
      scatter: Math.min(1, Math.max(0, settings.scatter)),
      levels,
    }
    if (levelsChanged) this.rebuildPyramid()
    else this.writeUniforms()
  }

  resize(width: number, height: number): void {
    const nextWidth = Math.max(1, Math.floor(width))
    const nextHeight = Math.max(1, Math.floor(height))
    if (nextWidth === this.width && nextHeight === this.height) return
    this.width = nextWidth
    this.height = nextHeight
    this.sceneTexture.destroy()
    this.sceneTexture = this.createTexture(nextWidth, nextHeight)
    this.rebuildPyramid()
  }

  encode(encoder: GPUCommandEncoder, destination: GPUTextureView): void {
    if (this.settings.enabled) {
      for (let index = 0; index < this.levels.length; index += 1) {
        this.draw(encoder, this.levels[index].down.createView(), this.downsamplePipeline, 0, this.downsampleBindGroups[index])
      }
      for (let index = this.levels.length - 2; index >= 0; index -= 1) {
        this.draw(encoder, this.levels[index].up.createView(), this.upsamplePipeline, 1, this.upsampleBindGroups[index])
      }
    }
    this.writeCompositeUniform()
    this.draw(encoder, destination, this.compositePipeline, 2, this.compositeBindGroup)
  }

  destroy(): void {
    this.sceneTexture.destroy()
    this.destroyPyramid()
    this.compositeUniform.destroy()
  }

  private rebuildPyramid(): void {
    this.destroyPyramid()
    let width = Math.max(1, Math.floor(this.width / 2))
    let height = Math.max(1, Math.floor(this.height / 2))
    const available = 1 + Math.floor(Math.log2(Math.max(1, Math.min(width, height))))
    const levelCount = Math.min(this.settings.levels, available)
    for (let index = 0; index < levelCount; index += 1) {
      this.levels.push({ width, height, down: this.createTexture(width, height), up: this.createTexture(width, height) })
      this.downsampleUniforms.push(this.createUniform())
      this.upsampleUniforms.push(this.createUniform())
      width = Math.max(1, Math.floor(width / 2))
      height = Math.max(1, Math.floor(height / 2))
    }

    for (let index = 0; index < this.levels.length; index += 1) {
      const source = index === 0 ? this.sceneTexture : this.levels[index - 1].down
      this.downsampleBindGroups.push(this.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: source.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.downsampleUniforms[index] } },
        ],
      }))
    }
    for (let index = 0; index < this.levels.length - 1; index += 1) {
      const coarse = index === this.levels.length - 2 ? this.levels[index + 1].down : this.levels[index + 1].up
      this.upsampleBindGroups.push(this.device.createBindGroup({
        layout: this.upsamplePipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: this.levels[index].down.createView() },
          { binding: 1, resource: coarse.createView() },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.upsampleUniforms[index] } },
        ],
      }))
    }
    this.compositeBindGroup = this.createCompositeBindGroup(this.levels[0]?.up ?? this.sceneTexture)
    this.writeUniforms()
  }

  private writeUniforms(): void {
    for (let index = 0; index < this.levels.length; index += 1) {
      const sourceWidth = index === 0 ? this.width : this.levels[index - 1].width
      const sourceHeight = index === 0 ? this.height : this.levels[index - 1].height
      writeFloat32(this.device, this.downsampleUniforms[index], 0, new Float32Array([
        1 / sourceWidth, 1 / sourceHeight, this.settings.threshold, index === 0 ? 1 : 0,
      ]))
      if (index < this.levels.length - 1) {
        writeFloat32(this.device, this.upsampleUniforms[index], 0, new Float32Array([
          1 / this.levels[index + 1].width, 1 / this.levels[index + 1].height, this.settings.radiusPx, this.settings.scatter,
        ]))
      }
    }
    this.writeCompositeUniform()
  }

  private advanceRgbAnimation(): void {
    const now = performance.now()
    this.rgbPhases = advanceRgbPhases(this.rgbPhases, (now - this.rgbLastTime) / 1000, normalizePostEffects(this.settings))
    this.strobePhases = advanceStrobePhases(this.strobePhases, (now - this.rgbLastTime) / 1000, normalizePostEffects(this.settings))
    this.rgbLastTime = now
  }

  private writeCompositeUniform(): void {
    this.advanceRgbAnimation()
    const s = normalizePostEffects(this.settings)
    writeFloat32(this.device, this.compositeUniform, 0, new Float32Array([
      this.settings.intensity, this.settings.enabled ? 1 : 0, 0, 0,
      s.dofFocus, s.dofWidth, s.dofTop, s.dofFront,
      s.dofEnabled ? 1 : 0, s.rgbEnabled ? 1 : 0, s.rgbMix, s.rgbAngle * Math.PI / 180,
      s.redFrequency, s.greenFrequency, s.blueFrequency, 0,
      s.redExponent, s.greenExponent, s.blueExponent, 0,
      ...this.rgbPhases, 0,
      1 / this.width, 1 / this.height, 0, 0,
      s.strobeEnabled ? 1 : 0, s.strobeMix, 0, 0,
      ...this.strobePhases, 0,
      s.strobeRedExponent, s.strobeGreenExponent, s.strobeBlueExponent, 0,
    ]))
  }

  private createCompositeBindGroup(bloom: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.compositePipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: this.sceneTexture.createView() },
        { binding: 1, resource: bloom.createView() },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.compositeUniform } },
      ],
    })
  }

  private draw(encoder: GPUCommandEncoder, target: GPUTextureView, pipeline: GPURenderPipeline, groupIndex: number, bindGroup: GPUBindGroup): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(groupIndex, bindGroup)
    pass.draw(3)
    pass.end()
  }

  private createPipeline(module: GPUShaderModule, fragment: string, format: GPUTextureFormat): GPURenderPipeline {
    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "fullscreenVertex" },
      fragment: { module, entryPoint: fragment, targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    })
  }

  private createTexture(width: number, height: number): GPUTexture {
    return this.device.createTexture({
      size: [width, height], format: BLOOM_SCENE_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
  }

  private createUniform(): GPUBuffer {
    return this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  }

  private destroyPyramid(): void {
    for (const level of this.levels) { level.down.destroy(); level.up.destroy() }
    for (const uniform of this.downsampleUniforms) uniform.destroy()
    for (const uniform of this.upsampleUniforms) uniform.destroy()
    this.levels = []
    this.downsampleUniforms = []
    this.upsampleUniforms = []
    this.downsampleBindGroups = []
    this.upsampleBindGroups = []
  }
}

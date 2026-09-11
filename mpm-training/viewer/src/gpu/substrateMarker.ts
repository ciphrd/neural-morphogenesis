import environmentSrc from "../../../core/environment.wgsl?raw";
import { packChemicalChannelLayout } from "./chemicalChannels";
import { GRID_N } from "./mpmCore";
import { templateShader } from "./shaderTemplate";
import { writeFloat32 } from "./gpuUtil";

export const SUBSTRATE_MARKER_N = 512;

/** A transported radial texture coordinate. Render rings from this coordinate
 * so fine black/white bands survive without repeatedly blurring their colors.
 * Uses the production substrate advection, never particle deformation or G. */
export class SubstrateMarker {
  readonly buffers: [GPUBuffer, GPUBuffer];
  parity = 0;
  enabled = false;
  private readonly physics: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly groups: [GPUBindGroup, GPUBindGroup];

  constructor(private readonly device: GPUDevice, velocity: GPUBuffer) {
    this.buffers = [0, 1].map(() => device.createBuffer({
      size: SUBSTRATE_MARKER_N ** 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })) as [GPUBuffer, GPUBuffer];
    this.physics = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const layout = packChemicalChannelLayout(SUBSTRATE_MARKER_N, SUBSTRATE_MARKER_N, [{
      scale: "local", resolutionScale: 1, relaxationTime: 1,
      fieldResponseTime: 1, decayExponent: 1, diffusionMultiplier: 0,
    }]);
    const module = device.createShaderModule({ code: templateShader(environmentSrc, {
      CHANNELS: 1, GRID_N, ...layout.shaderConstants,
    }) });
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "advectOnly" } });
    this.groups = [0, 1].map(p => device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.buffers[p] } },
        { binding: 3, resource: { buffer: this.buffers[1 - p] } },
        { binding: 4, resource: { buffer: this.physics } },
        { binding: 5, resource: { buffer: velocity } },
      ],
    })) as [GPUBindGroup, GPUBindGroup];
  }

  reset(centerX: number, centerY: number): void {
    const radial = new Float32Array(SUBSTRATE_MARKER_N ** 2);
    for (let y = 0; y < SUBSTRATE_MARKER_N; y++) {
      for (let x = 0; x < SUBSTRATE_MARKER_N; x++) {
        let dx = (x + 0.5) / SUBSTRATE_MARKER_N - centerX;
        let dy = (y + 0.5) / SUBSTRATE_MARKER_N - centerY;
        dx -= Math.floor(dx + 0.5); dy -= Math.floor(dy + 0.5);
        radial[y * SUBSTRATE_MARKER_N + x] = Math.hypot(dx, dy);
      }
    }
    for (const buffer of this.buffers) writeFloat32(this.device, buffer, 0, radial);
    this.parity = 0;
  }

  encodeTransport(encoder: GPUCommandEncoder, dt: number): void {
    if (!this.enabled || dt <= 0) return;
    writeFloat32(this.device, this.physics, 16, new Float32Array([dt]));
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.groups[this.parity]);
    pass.dispatchWorkgroups(SUBSTRATE_MARKER_N / 16, SUBSTRATE_MARKER_N / 16);
    pass.end();
    this.parity = 1 - this.parity;
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
    this.physics.destroy();
  }
}

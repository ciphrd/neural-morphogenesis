import developmentalConfig from "../../../core/developmental_fields.json";
import developmentalSrc from "../../../core/developmentalFields.wgsl?raw";
import { GRID_N, type MpmCore } from "./mpmCore";
import { templateShader } from "./shaderTemplate";
import { ceilDiv, writeFloat32 } from "./gpuUtil";

export interface DevelopmentalSettings {
  enabled: boolean;
  timeScale: number;
  seedOffset: number;
  seedSigma: number;
  activatorDiffusion: number;
  inhibitorDiffusion: number;
  sourceProduction: number;
  activatorDecay: number;
  inhibitorProduction: number;
  inhibitorDecay: number;
  inhibitorSuppression: number;
  occupancyHalfSaturation: number;
  occupancyHillExponent: number;
}

export const DEFAULT_DEVELOPMENTAL_SETTINGS: DevelopmentalSettings = {
  ...developmentalConfig.defaults,
};

export class DevelopmentalFields {
  readonly size: number;
  readonly buffers: [GPUBuffer, GPUBuffer];
  readonly gradient: GPUBuffer;
  private readonly device: GPUDevice;
  private readonly params: GPUBuffer;
  private readonly organizers: GPUBuffer;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly seedBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly evolvePipeline: GPUComputePipeline;
  private readonly evolveBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly advectOrganizersPipeline: GPUComputePipeline;
  private readonly advectOrganizersBindGroup: GPUBindGroup;
  private readonly gradientPipeline: GPUComputePipeline;
  private readonly gradientBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly dispatch: [number, number];
  private settings: DevelopmentalSettings = { ...DEFAULT_DEVELOPMENTAL_SETTINGS };
  private seedCenter: [number, number] = [0.5, 0.5];
  private seedHeading = 0;
  private advectionDt = 0;
  private _parity = 0;

  get parity(): number { return this._parity; }

  constructor(device: GPUDevice, mpmCore: MpmCore, baseFieldN: number) {
    this.device = device;
    this.size = Math.max(8, Math.min(
      developmentalConfig.maximumResolution,
      Math.round(baseFieldN * developmentalConfig.resolutionScale),
    ));
    const values = 3 * this.size * this.size;
    this.buffers = [0, 1].map(() => device.createBuffer({
      size: values * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })) as [GPUBuffer, GPUBuffer];
    this.gradient = device.createBuffer({ size: values * 2 * 4, usage: GPUBufferUsage.STORAGE });
    this.params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.organizers = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const module = device.createShaderModule({
      code: templateShader(developmentalSrc, { FIELD_N: this.size, MPM_GRID_N: GRID_N }),
    });
    this.seedPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "seed" } });
    this.evolvePipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "evolve" } });
    this.advectOrganizersPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "advectOrganizers" } });
    this.gradientPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "computeGradient" } });

    this.seedBindGroups = [0, 1].map((p) => device.createBindGroup({
      layout: this.seedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.buffers[p] } },
        { binding: 3, resource: { buffer: this.params } },
        { binding: 6, resource: { buffer: this.organizers } },
      ],
    })) as [GPUBindGroup, GPUBindGroup];
    this.evolveBindGroups = [0, 1].map((p) => device.createBindGroup({
      layout: this.evolvePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers[p] } },
        { binding: 1, resource: { buffer: this.buffers[1 - p] } },
        { binding: 3, resource: { buffer: this.params } },
        { binding: 4, resource: { buffer: mpmCore.gridVel } },
        { binding: 5, resource: mpmCore.morphologyTexture.createView() },
        { binding: 6, resource: { buffer: this.organizers } },
      ],
    })) as [GPUBindGroup, GPUBindGroup];
    this.advectOrganizersBindGroup = device.createBindGroup({
      layout: this.advectOrganizersPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: this.params } },
        { binding: 4, resource: { buffer: mpmCore.gridVel } },
        { binding: 6, resource: { buffer: this.organizers } },
      ],
    });
    this.gradientBindGroups = [0, 1].map((p) => device.createBindGroup({
      layout: this.gradientPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers[p] } },
        { binding: 2, resource: { buffer: this.gradient } },
      ],
    })) as [GPUBindGroup, GPUBindGroup];
    this.dispatch = [ceilDiv(this.size, 8), ceilDiv(this.size, 8)];
    this.writeParams(0, 0);
  }

  setSettings(settings: DevelopmentalSettings): void {
    this.settings = { ...settings };
    this.writeParams(0, 0);
  }

  setAdvectionTimestep(dt: number): void {
    this.advectionDt = Math.max(0, dt);
  }

  reset(center: readonly [number, number], heading: number): void {
    this.seedCenter = [center[0], center[1]];
    this.seedHeading = heading;
    const headingX = Math.cos(heading);
    const headingY = Math.sin(heading);
    const wrap = (value: number) => ((value % 1) + 1) % 1;
    writeFloat32(this.device, this.organizers, 0, new Float32Array([
      wrap(center[0] + headingX * this.settings.seedOffset),
      wrap(center[1] + headingY * this.settings.seedOffset),
      wrap(center[0] - headingX * this.settings.seedOffset),
      wrap(center[1] - headingY * this.settings.seedOffset),
    ]));
    const zeros = new Float32Array(3 * this.size * this.size);
    writeFloat32(this.device, this.buffers[0], 0, zeros);
    writeFloat32(this.device, this.buffers[1], 0, zeros);
    this._parity = 0;
    this.writeParams(0, 0);
    if (!this.settings.enabled) return;
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.seedPipeline);
    pass.setBindGroup(0, this.seedBindGroups[0]);
    pass.dispatchWorkgroups(...this.dispatch);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  reseed(): void { this.reset(this.seedCenter, this.seedHeading); }

  encodeStep(encoder: GPUCommandEncoder): void {
    if (!this.settings.enabled) return;
    const totalDt = developmentalConfig.baseTimeStep * Math.max(0, this.settings.timeScale);
    const maxDiffusion = Math.max(this.settings.activatorDiffusion, this.settings.inhibitorDiffusion);
    const diffusionSubsteps = Math.ceil(maxDiffusion * totalDt * this.size * this.size / 0.24);
    const maxReactionRate = this.settings.sourceProduction + this.settings.activatorDecay
      + this.settings.inhibitorSuppression
      + 2 * this.settings.inhibitorProduction + this.settings.inhibitorDecay;
    const reactionSubsteps = Math.ceil(maxReactionRate * totalDt / 0.25);
    const substeps = Math.max(1, developmentalConfig.minimumIntegrationSubsteps, diffusionSubsteps, reactionSubsteps);
    const dt = totalDt / substeps;
    const advectionDt = this.advectionDt / substeps;
    this.writeParams(dt, advectionDt);
    for (let i = 0; i < substeps; i++) {
      const organizerPass = encoder.beginComputePass();
      organizerPass.setPipeline(this.advectOrganizersPipeline);
      organizerPass.setBindGroup(0, this.advectOrganizersBindGroup);
      organizerPass.dispatchWorkgroups(1);
      organizerPass.end();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.evolvePipeline);
      pass.setBindGroup(0, this.evolveBindGroups[this._parity]);
      pass.dispatchWorkgroups(...this.dispatch);
      pass.end();
      this._parity = 1 - this._parity;
    }
    const gradientPass = encoder.beginComputePass();
    gradientPass.setPipeline(this.gradientPipeline);
    gradientPass.setBindGroup(0, this.gradientBindGroups[this._parity]);
    gradientPass.dispatchWorkgroups(this.dispatch[0], this.dispatch[1], 3);
    gradientPass.end();
  }

  private writeParams(dt: number, advectionDt: number): void {
    const s = this.settings;
    writeFloat32(this.device, this.params, 0, new Float32Array([
      dt, advectionDt, s.seedSigma, s.enabled ? 1 : 0,
      s.activatorDiffusion, s.inhibitorDiffusion, s.sourceProduction, s.activatorDecay,
      s.inhibitorProduction, s.inhibitorDecay, s.inhibitorSuppression, s.occupancyHalfSaturation,
      s.occupancyHillExponent, 0, 0, 0,
    ]));
  }

  destroy(): void {
    this.buffers[0].destroy();
    this.buffers[1].destroy();
    this.gradient.destroy();
    this.params.destroy();
    this.organizers.destroy();
  }
}

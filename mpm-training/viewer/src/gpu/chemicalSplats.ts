import type { PackedChemicalLayout } from "./chemicalChannels";

/** Replace contended deposition with one private record per particle. */
export function chemicalSplatShader(source: string): string {
  const start = source.indexOf("fn addDepositFloat(");
  const end = source.indexOf("@compute", start);
  if (start < 0 || end < 0) throw new Error("Missing chemical deposition kernel");
  return (source.slice(0, start) + `
fn depositMaterialSample(pi: u32, envWrite: array<f32, ENV_WRITE_DIM>, pos: vec2<f32>, rest: ParticleRest) {
  let originalArea = select(physics.sampleSpacing * physics.sampleSpacing * max(rest.quadratureWeight, 0.0),
    rest.originalArea, rest.originalArea > 0.0);
  let base = pi * (CHANNELS + 3u);
  depositScratch[base] = fract(pos.x);
  depositScratch[base + 1u] = fract(pos.y);
  depositScratch[base + 2u] = originalArea * max(matDet(rest.growthF), 0.0);
  for (var c = 0u; c < CHANNELS; c++) {
    depositScratch[base + 3u + c] = envWrite[c];
  }
}
` + source.slice(end))
    .replace("depositScratch: array<atomic<i32>>", "depositScratch: array<f32>")
    .replace("depositMaterialSample(levels,", "depositMaterialSample(pi, levels,")
    .replace("depositMaterialSample(result.envWrite,", "depositMaterialSample(pi, result.envWrite,");
}

/** Pack up to three same-resolution chemical numerators beside one area sum. */
export function chemicalSplatGroups(layout: PackedChemicalLayout): number[][] {
  const groups: number[][] = [];
  for (let c = 0; c < layout.widths.length; c++) {
    const group = groups.find(g => g.length < 3 && layout.widths[g[0]] === layout.widths[c] && layout.heights[g[0]] === layout.heights[c]);
    if (group) group.push(c); else groups.push([c]);
  }
  return groups;
}

// Render all channels before copying: the shared scratch buffer holds particle
// records during rendering and packed numerator/area grids during resolution.
export class ChemicalSplats {
  private groups: number[][];
  private textures: GPUTexture[] = [];
  private uniforms: GPUBuffer[] = [];
  private renderGroups: GPUBindGroup[] = [];
  private resolveGroups: GPUBindGroup[] = [];
  private renderPipeline: GPURenderPipeline;
  private resolvePipeline: GPUComputePipeline;

  constructor(device: GPUDevice, private layout: PackedChemicalLayout, scratch: GPUBuffer) {
    const channels = layout.widths.length;
    this.groups = chemicalSplatGroups(layout);
    const module = device.createShaderModule({label: "Chemical instanced splats", code: `
struct Params { width: u32, height: u32, count: u32, images: u32, channels: vec4<u32>, offsets: vec4<u32> }
@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) center: vec2<f32>,
  @location(1) @interpolate(flat) amounts: vec4<f32>,
}
@vertex fn vertexMain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
  let base = (instance / params.images) * ${channels + 3}u;
  let image = instance % params.images;
  let size = vec2<f32>(f32(params.width), f32(params.height));
  let position = vec2<f32>(samples[base], samples[base + 1u]);
  var shift = vec2<f32>(f32(image % 3u) - 1.0, f32(image / 3u) - 1.0) * size;
  if (params.images == 4u) {
    shift = vec2<f32>(f32(image % 2u), f32(image / 2u)) * select(-size, size, position < vec2<f32>(0.5));
  }
  let center = position * size + shift;
  let low = floor(center - vec2<f32>(1.0));
  let corners = array<vec2<f32>, 6>(vec2<f32>(0,0), vec2<f32>(1,0), vec2<f32>(0,1),
    vec2<f32>(0,1), vec2<f32>(1,0), vec2<f32>(1,1));
  let pixel = low + corners[vertex] * 3.0;
  var out: VertexOutput;
  out.position = vec4<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0, 0.0, 1.0);
  out.center = center;
  let area = samples[base + 2u];
  out.amounts = vec4<f32>(0.0, 0.0, 0.0, area);
  for (var c = 0u; c < params.count; c++) { out.amounts[c] = samples[base + 3u + params.channels[c]] * area; }
  return out;
}
fn spline(distance: f32) -> f32 {
  let d = abs(distance);
  if (d < 0.5) { return 0.75 - d*d; }
  return 0.5 * pow(max(1.5 - d, 0.0), 2.0);
}
@fragment fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let d = in.position.xy - in.center;
  return in.amounts * spline(d.x) * spline(d.y);
}
`});
    const blend: GPUBlendComponent = {srcFactor: "one", dstFactor: "one", operation: "add"};
    this.renderPipeline = device.createRenderPipeline({label: "Grouped chemical splats", layout: "auto",
      vertex: {module, entryPoint: "vertexMain"},
      fragment: {module, entryPoint: "fragmentMain", targets: [{format: "rgba32float", blend: {color: blend, alpha: blend}}]},
      primitive: {topology: "triangle-list"},
    });
    const resolveModule = device.createShaderModule({code: `
struct Params { width: u32, height: u32, count: u32, images: u32, channels: vec4<u32>, offsets: vec4<u32> }
@group(0) @binding(0) var deposit: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@compute @workgroup_size(8,8) fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.width || id.y >= params.height) { return; }
  let value = textureLoad(deposit, vec2<i32>(id.xy), 0);
  for (var c = 0u; c < params.count; c++) {
    let index = params.offsets[c] + id.y * params.width + id.x;
    scratch[index] = value[c];
    scratch[${layout.total}u + index] = value.w;
  }
}
`});
    // Unfilterable float textures do not require float32-filterable support.
    const resolveLayout = device.createBindGroupLayout({entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {sampleType: "unfilterable-float"}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
    ]});
    this.resolvePipeline = device.createComputePipeline({layout: device.createPipelineLayout({bindGroupLayouts: [resolveLayout]}),
      compute: {module: resolveModule, entryPoint: "resolve"}});
    for (const group of this.groups) {
      const c = group[0];
      const texture = device.createTexture({size: [layout.widths[c], layout.heights[c]], format: "rgba32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING});
      const uniform = device.createBuffer({size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(uniform, 0, new Uint32Array([layout.widths[c], layout.heights[c], group.length, this.imageCount(c), ...[0,1,2,3].map(i => group[i] ?? 0), ...[0,1,2,3].map(i => layout.offsets[group[i] ?? 0])]));
      this.textures.push(texture);
      this.uniforms.push(uniform);
      this.renderGroups.push(device.createBindGroup({layout: this.renderPipeline.getBindGroupLayout(0), entries: [
        {binding: 0, resource: {buffer: scratch}}, {binding: 1, resource: {buffer: uniform}},
      ]}));
      this.resolveGroups.push(device.createBindGroup({layout: resolveLayout, entries: [
        {binding: 0, resource: texture.createView()}, {binding: 1, resource: {buffer: uniform}},
        {binding: 2, resource: {buffer: scratch}},
      ]}));
    }
  }

  private imageCount(channel: number): number {
    return Math.min(this.layout.widths[channel], this.layout.heights[channel]) >= 3 ? 4 : 9;
  }

  encode(encoder: GPUCommandEncoder, particles: number): void {
    for (let c = 0; c < this.textures.length; c++) {
      const pass = encoder.beginRenderPass({colorAttachments: [{view: this.textures[c].createView(),
        clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store"}]});
      pass.setPipeline(this.renderPipeline);
      pass.setBindGroup(0, this.renderGroups[c]);
      pass.draw(6, particles * this.imageCount(this.groups[c][0]));
      pass.end();
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.resolvePipeline);
    for (let c = 0; c < this.textures.length; c++) {
      pass.setBindGroup(0, this.resolveGroups[c]);
      const channel = this.groups[c][0];
      pass.dispatchWorkgroups(Math.ceil(this.layout.widths[channel]/8), Math.ceil(this.layout.heights[channel]/8));
    }
    pass.end();
  }

  destroy(): void {
    for (const texture of this.textures) texture.destroy();
    for (const uniform of this.uniforms) uniform.destroy();
  }
}

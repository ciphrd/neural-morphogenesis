import attractSrc from "./attract.wgsl?raw"
import fieldSrc from "./field.wgsl?raw"
import { ceilDiv, writeBuffer } from "./gpuUtil"
import type { MpmSimulation } from "./mpm"
import { GRID_N } from "./mpm"
import renderSrc from "./render.wgsl?raw"
import repulsionSrc from "./repulsion.wgsl?raw"
import { templateShader } from "./shaderTemplate"

// Ring marker's own radius, as a multiple of the particle point radius
// (writePointRadius() below) — big enough to clearly ring a particle
// without looking like a giant unrelated blob.
const MARKER_RADIUS_SCALE = 2.5

// Physical-*device*-pixel radius (not CSS pixel; setCanvasSizePx is fed
// canvas.width, already DPR-scaled — see main.ts's resizeCanvas()),
// reconverted to an NDC half-width whenever either it or the canvas size
// changes, so particles stay a fixed real screen-pixel width regardless
// of window size or display density. render.wgsl's quadOffsets span
// [-1,1] (a full-width-2 quad in this half-width's own units), so the
// *radius* passed to the shader must be half the desired on-screen pixel
// width. 1px is the original fixed value (right for blocks.ts/disc.ts's
// dense clouds, where particles overlap regardless of exact size);
// main.ts's Particle Size slider is what lets a sparse world (e.g.
// worlds/growth.ts, added one dot at a time) turn this up so individual
// particles are actually visible.
export const DEFAULT_POINT_RADIUS_PX = 1

const NODES = GRID_N + 1
const FIELD_WORKGROUP = 16

/** Which of the grid's own fields (see field.wgsl) the background shows
 * — "none" is the original flat clear color, skipping the colorize pass
 * and background quad draw entirely rather than just rendering a no-op
 * field. Must match field.wgsl's own MODE_NONE/MODE_DENSITY/MODE_SPEED/
 * MODE_DEFORMATION/MODE_PRESSURE/MODE_SHEAR. "repulsion" is NOT one of
 * field.wgsl's own modes — it draws from a wholly separate texture (see
 * repulsion.wgsl), at its own independently-configurable resolution, so
 * it's drawn via its own render pipeline (repulsionFieldPipeline below)
 * rather than field.wgsl's shared fieldPipeline/fieldTexture. */
export type FieldMode = "none" | "density" | "speed" | "deformation" | "pressure" | "shear" | "repulsion"
// "repulsion" has no entry — it never reaches field.wgsl's own mode
// uniform at all (see setFieldMode()/render() below, which special-case
// it before this map is ever consulted).
const FIELD_MODE_CODE: Record<Exclude<FieldMode, "repulsion">, number> = {
  none: 0,
  density: 1,
  speed: 2,
  deformation: 3,
  pressure: 4,
  shear: 5,
}

/** Presentation: an optional field-visualize background (gpu/mpm.ts's
 * own gridMass/gridVel, colorized — see field.wgsl) behind particles
 * (colored circles, a per-pixel discard against each quad's own offset —
 * see render.wgsl's own particleFragment) and the static boundary box
 * outline — the
 * WebGPU render-pass half of this project; see gpu/mpm.ts for the
 * compute half. Draw order: field background, then boundary, then
 * particles on top of both. */
export class MpmRenderer {
  private readonly device: GPUDevice
  // Kept for its own activeCount getter — see render()'s draw call.
  // Particle count now varies per world/at runtime (worlds/types.ts,
  // gpu/mpm.ts's MAX_PARTICLES) rather than being a fixed module
  // constant, so this is the only source of truth for "how many to draw."
  private readonly sim: MpmSimulation
  private readonly particlePipeline: GPURenderPipeline
  private readonly particleBindGroup: GPUBindGroup
  private readonly boundaryPipeline: GPURenderPipeline
  private readonly pointRadiusUniform: GPUBuffer
  private canvasSizePx = 0
  private pointRadiusPx = DEFAULT_POINT_RADIUS_PX

  private readonly fieldTexture: GPUTexture
  private readonly fieldModeUniform: GPUBuffer
  private readonly colorizeFieldPipeline: GPUComputePipeline
  private readonly colorizeFieldBindGroup: GPUBindGroup
  private readonly fieldPipeline: GPURenderPipeline
  private readonly fieldBindGroup: GPUBindGroup
  private readonly fieldDispatch: readonly [number, number]
  private fieldMode: FieldMode = "none"

  // Distance-field display (main.ts's "Field" dropdown, repulsion.wgsl's
  // own repulsionFieldVertex/repulsionFieldFragment) — a separate render
  // pipeline from fieldPipeline above since it samples a wholly separate,
  // independently-sized texture (sim.densityTexture, not field.wgsl's
  // shared GRID_N+1-resolution fieldTexture). Unlike fieldPipeline, BOTH
  // the pipeline and its bind group get rebuilt by
  // rebuildRepulsionDisplay() whenever sim.densityTexture's own identity
  // changes (main.ts's Field Resolution control): repulsion.wgsl's own
  // FIELD_N is baked into the shader module at compile time (same as
  // p2g.wgsl's GRID_N), and this display shader actually reads it (to
  // convert a sampled UV into a texel index — see repulsionFieldFragment's
  // own textureLoad), so a stale FIELD_N would misalign the display, not
  // just risk pointing at a destroyed texture the way a stale bind group
  // alone would.
  private repulsionFieldPipeline!: GPURenderPipeline
  private repulsionFieldBindGroup!: GPUBindGroup
  private canvasFormat!: GPUTextureFormat

  // Attract-to-point tool's own marker ring (attract.wgsl's
  // attractorMarkerVertex/Fragment) — drawn in its own pass, strictly
  // AFTER the main particle draw (see render()'s own ordering comment),
  // so a targeted particle stays visibly marked no matter which particle
  // index last overdrew its pixels (this project's particle pass has no
  // depth buffer — see that pass's own header for why that made the
  // highlight color alone unreliable).
  private readonly attractorMarkerPipeline: GPURenderPipeline
  private readonly attractorMarkerBindGroup: GPUBindGroup
  private readonly markerRadiusUniform: GPUBuffer

  constructor(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
    sim: MpmSimulation
  ) {
    this.device = device
    this.sim = sim
    const module = device.createShaderModule({ code: renderSrc })

    this.pointRadiusUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.particlePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "particleVertex" },
      fragment: {
        module,
        entryPoint: "particleFragment",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    })
    this.particleBindGroup = device.createBindGroup({
      layout: this.particlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sim.positions } },
        { binding: 1, resource: { buffer: sim.colors } },
        { binding: 2, resource: { buffer: this.pointRadiusUniform } },
      ],
    })

    // No bind group needed: boundaryVertex/Fragment reference no
    // resources at all (the rect's 5 points are a module-scope constant
    // array — see render.wgsl).
    this.boundaryPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "boundaryVertex" },
      fragment: {
        module,
        entryPoint: "boundaryFragment",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "line-strip" },
    })

    // --- field visualize setup ---
    const fieldModule = device.createShaderModule({
      code: templateShader(fieldSrc, { GRID_N }),
    })

    this.fieldTexture = device.createTexture({
      size: [NODES, NODES],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const fieldTextureView = this.fieldTexture.createView()

    this.fieldModeUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.setFieldMode("none")

    this.colorizeFieldPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: fieldModule, entryPoint: "colorizeField" },
    })
    this.colorizeFieldBindGroup = device.createBindGroup({
      layout: this.colorizeFieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sim.gridAccum } },
        { binding: 1, resource: { buffer: sim.gridVel } },
        { binding: 2, resource: { buffer: this.fieldModeUniform } },
        { binding: 3, resource: fieldTextureView },
      ],
    })
    this.fieldDispatch = [
      ceilDiv(NODES, FIELD_WORKGROUP),
      ceilDiv(NODES, FIELD_WORKGROUP),
    ]

    const fieldSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    })
    this.fieldPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "fieldVertex" },
      fragment: {
        module: fieldModule,
        entryPoint: "fieldFragment",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    })
    this.fieldBindGroup = device.createBindGroup({
      layout: this.fieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: fieldTextureView },
        { binding: 1, resource: fieldSampler },
      ],
    })

    // --- distance-field (repulsion) display setup ---
    this.canvasFormat = canvasFormat
    this.rebuildRepulsionDisplay()

    // --- attract-to-point marker overlay setup ---
    // DT is unused by attractorMarkerVertex/Fragment themselves, but
    // attract.wgsl's module-scope `const DT: f32 = __DT__;` still needs
    // *some* value substituted to compile this module at all — 0 here,
    // same placeholder convention rebuildRepulsionDisplay() already uses
    // above for repulsion.wgsl's own unused-here DT.
    const attractModule = device.createShaderModule({
      code: templateShader(attractSrc, { DT: 0 }),
    })
    this.markerRadiusUniform = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.attractorMarkerPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: attractModule, entryPoint: "attractorMarkerVertex" },
      fragment: {
        module: attractModule,
        entryPoint: "attractorMarkerFragment",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    })
    this.attractorMarkerBindGroup = device.createBindGroup({
      layout: this.attractorMarkerPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sim.attractorsBuffer } },
        { binding: 1, resource: { buffer: sim.positions } },
        { binding: 2, resource: { buffer: this.markerRadiusUniform } },
      ],
    })
    this.writePointRadius()
  }

  /** (Re)builds BOTH the repulsion display's shader module/pipeline AND
   * its bind group against `sim`'s CURRENT densityTexture — called once
   * from the constructor and again by main.ts every time it calls
   * MpmSimulation.setFieldResolution(). Rebuilds the pipeline too, not
   * just the bind group (unlike field.wgsl's own fieldPipeline, which
   * never changes): repulsion.wgsl's FIELD_N is baked into the shader
   * module at compile time and is actually READ by repulsionFieldFragment
   * (to convert a sampled UV into an integer texel index — see that
   * function's own textureLoad comment), so a stale FIELD_N would
   * misalign the whole display, not just risk a dangling texture
   * reference. `sim.densityTexture.width` is the source of truth for the
   * current resolution (not a separately-tracked number here) — always
   * in sync with whatever buildRepulsionResources() last created. */
  rebuildRepulsionDisplay(): void {
    const device = this.device
    const repulsionModule = device.createShaderModule({
      code: templateShader(repulsionSrc, { FIELD_N: this.sim.densityTexture.width, DT: 0 }),
    })
    this.repulsionFieldPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: repulsionModule, entryPoint: "repulsionFieldVertex" },
      fragment: {
        module: repulsionModule,
        entryPoint: "repulsionFieldFragment",
        targets: [{ format: this.canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    })
    this.repulsionFieldBindGroup = device.createBindGroup({
      layout: this.repulsionFieldPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.sim.densityTexture.createView() }],
    })
  }

  setCanvasSizePx(sizePx: number): void {
    if (sizePx === this.canvasSizePx) return
    this.canvasSizePx = sizePx
    this.writePointRadius()
  }

  /** Live-updates render.wgsl's per-particle quad half-width (see
   * DEFAULT_POINT_RADIUS_PX's own docstring) — a plain buffer write, no
   * pipeline recreation, safe to call on every tick of main.ts's Particle
   * Size slider. */
  setPointRadiusPx(px: number): void {
    this.pointRadiusPx = px
    this.writePointRadius()
  }

  private writePointRadius(): void {
    const ndcRadius = (this.pointRadiusPx * 2) / Math.max(this.canvasSizePx, 1)
    writeBuffer(
      this.device,
      this.pointRadiusUniform,
      0,
      new Float32Array([ndcRadius])
    )
    writeBuffer(
      this.device,
      this.markerRadiusUniform,
      0,
      new Float32Array([ndcRadius * MARKER_RADIUS_SCALE])
    )
  }

  setFieldMode(mode: FieldMode): void {
    this.fieldMode = mode
    // "repulsion" never touches field.wgsl's own mode uniform — its
    // colorize pass is skipped entirely below (render.ts's repulsion
    // display samples sim.densityTexture directly, already kept current
    // by sim.step() every substep, no separate colorize step needed).
    if (mode === "repulsion") return
    writeBuffer(
      this.device,
      this.fieldModeUniform,
      0,
      new Uint32Array([FIELD_MODE_CODE[mode]])
    )
  }

  render(context: GPUCanvasContext): void {
    const encoder = this.device.createCommandEncoder()

    // Colorize this frame's grid state into fieldTexture — skipped
    // entirely at mode "none"/"repulsion" rather than dispatched to
    // write a no-op flat color every frame for no reason ("repulsion"
    // draws from a wholly separate texture, sim.densityTexture, kept
    // live by sim.step() itself every substep — see repulsion.wgsl's own
    // header).
    if (this.fieldMode !== "none" && this.fieldMode !== "repulsion") {
      const colorizePass = encoder.beginComputePass()
      colorizePass.setPipeline(this.colorizeFieldPipeline)
      colorizePass.setBindGroup(0, this.colorizeFieldBindGroup)
      colorizePass.dispatchWorkgroups(
        this.fieldDispatch[0],
        this.fieldDispatch[1]
      )
      colorizePass.end()
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          // Reference's canvas.clear(0x112F41) — field.wgsl's own BG
          // constant matches this exactly, so mode "none" (this clear,
          // left untouched) and every field mode's own empty-cell color
          // are visually the same background.
          clearValue: { r: 0x11 / 255, g: 0x2f / 255, b: 0x41 / 255, a: 1 },
        },
      ],
    })

    if (this.fieldMode === "repulsion") {
      pass.setPipeline(this.repulsionFieldPipeline)
      pass.setBindGroup(0, this.repulsionFieldBindGroup)
      pass.draw(6, 1)
    } else if (this.fieldMode !== "none") {
      pass.setPipeline(this.fieldPipeline)
      pass.setBindGroup(0, this.fieldBindGroup)
      pass.draw(6, 1)
    }

    pass.setPipeline(this.boundaryPipeline)
    pass.draw(5, 1)

    pass.setPipeline(this.particlePipeline)
    pass.setBindGroup(0, this.particleBindGroup)
    pass.draw(6, this.sim.activeCount)

    // Marker overlay, AFTER particles (see attractorMarkerPipeline's own
    // class comment for why draw order matters here) — skipped entirely
    // rather than issuing a draw(6, 0), which WebGPU's own validation
    // flags as "unusual" every frame in the common zero-attractor case.
    if (this.sim.attractorCount > 0) {
      pass.setPipeline(this.attractorMarkerPipeline)
      pass.setBindGroup(0, this.attractorMarkerBindGroup)
      pass.draw(6, this.sim.attractorCount)
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  destroy(): void {
    this.pointRadiusUniform.destroy()
    this.fieldTexture.destroy()
    this.fieldModeUniform.destroy()
    this.markerRadiusUniform.destroy()
  }
}

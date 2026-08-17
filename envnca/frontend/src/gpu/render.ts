import reduceSrc from "./reduce.wgsl?raw";
import colorizeSrc from "./colorize.wgsl?raw";
import presentSrc from "./present.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import { writeFloat32 } from "./gpuUtil";
import type { GpuEnvironment } from "./environment";
import type { GpuAgents } from "./agents";
import type { BackgroundMode } from "./types";

// Must match colorize.wgsl's MODE_GRAY/MODE_BLACK/MODE_SUBSTRATE.
const BACKGROUND_MODE_CODE: Record<BackgroundMode, number> = { gray: 0, black: 1, substrate: 2 };

export interface RenderConfig {
  width: number;
  height: number;
  channels: number;
}

const REDUCE_WORKGROUP = 256;
const COLORIZE_WORKGROUP = 16;

// Roughly matches the deleted render.py's 3x3 white agent blocks / small
// red target blocks.
const AGENT_HALF_SIZE_PX = 1.5;
const TARGET_HALF_SIZE_PX = 1.5;

// render.py's DisplayScale alpha — how quickly the auto-scale adapts to
// the field's actual long-run magnitude.
const EMA_ALPHA = 0.05;

// Default contrast multiplier — see setIntensity()'s own comment. >1
// because the EMA-tracked scale is a genuine max-abs over the whole
// grid, so most pixels sit well inside it; the substrate reads as
// visually flat/washed-out at intensity=1 (colorize.wgsl's symmetric
// clamp maps the true max to pure white/black, but almost nothing
// actually reaches that far).
export const DEFAULT_INTENSITY = 2.5;

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

/** Grid colorization (compute) + presentation (render pass): the
 * WebGPU replacement for the deleted render.py's render_frame(). Split
 * into 3 stages — see this project's design notes for why:
 * 1. A two-pass GPU max-abs reduction over channels 0-2, feeding a
 *    JS-side EMA (`DisplayScale.update()`'s exact alpha/logic, ported).
 * 2. A colorize compute pass turning the raw float grid into an
 *    rgba8unorm texture using that EMA'd scale.
 * 3. A render pass: full-screen background quad sampling that texture,
 *    then two instanced-quad draws — agents first, target points last
 *    (on top), so the target outline stays visible as a reference even
 *    where agents currently sit on top of it. */
export class GpuRender {
  private readonly device: GPUDevice;

  // --- reduction ---
  private readonly reducePartialPipeline: GPUComputePipeline;
  private readonly reduceFinalPipeline: GPUComputePipeline;
  private readonly reducePartialBindGroups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly reduceFinalBindGroup: GPUBindGroup;
  private readonly partialsBuffer: GPUBuffer;
  private readonly finalBuffer: GPUBuffer;
  private readonly readbackBuffer: GPUBuffer;
  private readonly numReduceWorkgroups: number;
  private pendingReadback = false;
  private emaScale: [number, number, number] | null = null;
  private backgroundMode: BackgroundMode = "substrate";
  private intensity = DEFAULT_INTENSITY;

  // --- colorize ---
  private readonly colorizePipeline: GPUComputePipeline;
  private readonly colorizeBindGroups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly scaleUniformBuffer: GPUBuffer;
  private readonly outputTexture: GPUTexture;
  private readonly outputTextureView: GPUTextureView;
  private readonly colorize2DGroups: readonly [number, number];

  // --- present ---
  private readonly backgroundPipeline: GPURenderPipeline;
  private readonly backgroundBindGroup: GPUBindGroup;
  private readonly markerPipeline: GPURenderPipeline;
  private readonly agentMarkerBindGroup: GPUBindGroup;
  private readonly agentMarkerUniform: GPUBuffer;
  private readonly targetMarkerUniform: GPUBuffer;
  private targetMarkerBindGroup: GPUBindGroup | null = null;
  private targetPositionsBuffer: GPUBuffer | null = null;
  private targetPointCount = 0;

  // Letterbox rect — the grid is always square, but the canvas usually
  // isn't. A *real* GPU viewport (gpu/render.ts's render() calls
  // setViewport with this rect right before the background/marker
  // draws), not a shader-side NDC scale — see present.wgsl's module
  // docstring for why the shader-side version had a real bug (a visible
  // diagonal seam through the letterbox bar) that this sidesteps
  // entirely by construction.
  private readonly gridWidth: number;
  private readonly gridHeight: number;
  private lastCanvasWidth = -1;
  private lastCanvasHeight = -1;
  private viewportRect: { x: number; y: number; width: number; height: number } = {
    x: 0,
    y: 0,
    width: 1,
    height: 1,
  };

  constructor(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
    environment: GpuEnvironment,
    agents: GpuAgents,
    config: RenderConfig
  ) {
    this.device = device;
    const { width, height, channels } = config;
    this.gridWidth = width;
    this.gridHeight = height;

    // --- reduction setup ---
    const pixelCount = width * height;
    this.numReduceWorkgroups = ceilDiv(pixelCount, REDUCE_WORKGROUP);
    this.partialsBuffer = device.createBuffer({
      size: this.numReduceWorkgroups * 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.finalBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.readbackBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const reduceModule = device.createShaderModule({
      code: templateShader(reduceSrc, { CHANNELS: channels, WIDTH: width, HEIGHT: height }),
    });
    this.reducePartialPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: reduceModule, entryPoint: "reducePartial" },
    });
    this.reduceFinalPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: reduceModule, entryPoint: "reduceFinal" },
    });
    this.reducePartialBindGroups = [
      device.createBindGroup({
        layout: this.reducePartialPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: environment.gridA } },
          { binding: 1, resource: { buffer: this.partialsBuffer } },
        ],
      }),
      device.createBindGroup({
        layout: this.reducePartialPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: environment.gridB } },
          { binding: 1, resource: { buffer: this.partialsBuffer } },
        ],
      }),
    ];
    this.reduceFinalBindGroup = device.createBindGroup({
      layout: this.reduceFinalPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: this.partialsBuffer } },
        { binding: 3, resource: { buffer: this.finalBuffer } },
      ],
    });

    // --- colorize setup ---
    this.scaleUniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Harmless scale default until the first reduction readback lands —
    // the grid starts all-zero anyway (see GpuEnvironment.reset()), so
    // the colorized frame is flat gray regardless of scale for however
    // many steps it takes deposits to actually appear.
    this.writeScaleUniform();

    this.outputTexture = device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.outputTextureView = this.outputTexture.createView();

    const colorizeModule = device.createShaderModule({
      code: templateShader(colorizeSrc, { CHANNELS: channels, WIDTH: width, HEIGHT: height }),
    });
    this.colorizePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: colorizeModule, entryPoint: "colorize" },
    });
    this.colorizeBindGroups = [
      device.createBindGroup({
        layout: this.colorizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: environment.gridA } },
          { binding: 1, resource: { buffer: this.scaleUniformBuffer } },
          { binding: 2, resource: this.outputTextureView },
        ],
      }),
      device.createBindGroup({
        layout: this.colorizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: environment.gridB } },
          { binding: 1, resource: { buffer: this.scaleUniformBuffer } },
          { binding: 2, resource: this.outputTextureView },
        ],
      }),
    ];
    this.colorize2DGroups = [ceilDiv(width, COLORIZE_WORKGROUP), ceilDiv(height, COLORIZE_WORKGROUP)];

    // --- present setup ---
    const presentModule = device.createShaderModule({ code: presentSrc });

    const backgroundSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.backgroundPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "backgroundVertex" },
      fragment: { module: presentModule, entryPoint: "backgroundFragment", targets: [{ format: canvasFormat }] },
      primitive: { topology: "triangle-list" },
    });
    this.backgroundBindGroup = device.createBindGroup({
      layout: this.backgroundPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.outputTextureView },
        { binding: 1, resource: backgroundSampler },
      ],
    });

    this.markerPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "markerVertex" },
      fragment: { module: presentModule, entryPoint: "markerFragment", targets: [{ format: canvasFormat }] },
      primitive: { topology: "triangle-list" },
    });

    // MarkerUniforms: vec4 color + f32 halfSizePixels + f32 gridWidth + f32 gridHeight + f32 _pad = 32 bytes.
    this.agentMarkerUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeFloat32(
      device,
      this.agentMarkerUniform,
      0,
      new Float32Array([1, 1, 1, 1, AGENT_HALF_SIZE_PX, width, height, 0])
    );
    this.agentMarkerBindGroup = device.createBindGroup({
      layout: this.markerPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: agents.positions } },
        { binding: 1, resource: { buffer: this.agentMarkerUniform } },
      ],
    });

    this.targetMarkerUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // render.py's TARGET_COLOR = (220, 60, 60).
    writeFloat32(
      device,
      this.targetMarkerUniform,
      0,
      new Float32Array([220 / 255, 60 / 255, 60 / 255, 1, TARGET_HALF_SIZE_PX, width, height, 0])
    );
  }

  /** Writes the full scale uniform (EMA'd per-channel scale + the
   * current background mode packed into its w component — see
   * colorize.wgsl). Called after every EMA update and whenever the mode
   * or intensity changes, so either takes effect on the very next frame
   * instead of waiting for the next (up to a few hundred ms away)
   * reduction readback.
   *
   * `intensity` divides the scale actually uploaded — colorize.wgsl's
   * clamp treats whatever value it's given as "maps to pure
   * white/black," so shrinking it (dividing by intensity > 1) makes
   * pixels reach full saturation at a *smaller* raw value than the
   * genuine EMA-tracked max. That's the whole effect: more visual
   * contrast at the cost of clipping the (rare) most extreme values to
   * flat white/black instead of resolving them. Applied here, not
   * baked into colorize.wgsl as a shader constant, so it's adjustable
   * live from the UI without a pipeline rebuild. */
  private writeScaleUniform(): void {
    const [r, g, b] = this.emaScale ?? [1, 1, 1];
    const divisor = Math.max(this.intensity, 1e-3);
    writeFloat32(
      this.device,
      this.scaleUniformBuffer,
      0,
      new Float32Array([r / divisor, g / divisor, b / divisor, BACKGROUND_MODE_CODE[this.backgroundMode]])
    );
  }

  setBackgroundMode(mode: BackgroundMode): void {
    if (mode === this.backgroundMode) return;
    this.backgroundMode = mode;
    this.writeScaleUniform();
  }

  setIntensity(intensity: number): void {
    if (intensity === this.intensity) return;
    this.intensity = intensity;
    this.writeScaleUniform();
  }

  /** Uploads a target shape's points (grid pixel coords) — called once
   * per session (the target is fixed for train_server.py's whole run).
   * Recreates the buffer/bind group so a different point count is
   * handled safely, even though in practice this is only ever called
   * once. */
  uploadTargetPoints(points: readonly (readonly [number, number])[]): void {
    this.targetPositionsBuffer?.destroy();
    this.targetPointCount = points.length;
    if (points.length === 0) {
      this.targetPositionsBuffer = null;
      this.targetMarkerBindGroup = null;
      return;
    }
    const data = new Float32Array(points.length * 2);
    points.forEach(([x, y], i) => {
      data[i * 2] = x;
      data[i * 2 + 1] = y;
    });
    this.targetPositionsBuffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    writeFloat32(this.device, this.targetPositionsBuffer, 0, data);
    this.targetMarkerBindGroup = this.device.createBindGroup({
      layout: this.markerPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.targetPositionsBuffer } },
        { binding: 1, resource: { buffer: this.targetMarkerUniform } },
      ],
    });
  }

  /** Encodes one frame: colorize (using last known EMA scale) + render
   * pass, plus — unless a previous reduction readback is still pending —
   * the reduction passes that feed *next* frame's scale. `gridParity`
   * is whichever grid buffer holds the current, just-stepped state (see
   * gpu/simulation.ts). */
  render(context: GPUCanvasContext, gridParity: number, agentDispatchCount: number, canvasWidth: number, canvasHeight: number): void {
    if (canvasWidth !== this.lastCanvasWidth || canvasHeight !== this.lastCanvasHeight) {
      this.lastCanvasWidth = canvasWidth;
      this.lastCanvasHeight = canvasHeight;
      const gridAspect = this.gridWidth / this.gridHeight;
      const canvasAspect = canvasWidth / Math.max(canvasHeight, 1);
      // Exactly one axis stays full-size (the "fit" axis); the other
      // shrinks so the square grid content is fully visible, centered,
      // and undistorted. A real GPU viewport rect, not a shader-side NDC
      // scale — see present.wgsl's module docstring for why the latter
      // had a real bug (a visible diagonal seam through the letterbox
      // bar on non-square canvases).
      const scaleX = canvasAspect > gridAspect ? gridAspect / canvasAspect : 1;
      const scaleY = canvasAspect > gridAspect ? 1 : canvasAspect / gridAspect;
      const width = Math.max(1, Math.round(canvasWidth * scaleX));
      const height = Math.max(1, Math.round(canvasHeight * scaleY));
      this.viewportRect = {
        width,
        height,
        x: Math.round((canvasWidth - width) / 2),
        y: Math.round((canvasHeight - height) / 2),
      };
    }

    const encoder = this.device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.colorizePipeline);
    computePass.setBindGroup(0, this.colorizeBindGroups[gridParity]);
    computePass.dispatchWorkgroups(this.colorize2DGroups[0], this.colorize2DGroups[1]);

    if (!this.pendingReadback) {
      computePass.setPipeline(this.reducePartialPipeline);
      computePass.setBindGroup(0, this.reducePartialBindGroups[gridParity]);
      computePass.dispatchWorkgroups(this.numReduceWorkgroups);

      computePass.setPipeline(this.reduceFinalPipeline);
      computePass.setBindGroup(0, this.reduceFinalBindGroup);
      computePass.dispatchWorkgroups(1);
    }
    computePass.end();

    if (!this.pendingReadback) {
      encoder.copyBufferToBuffer(this.finalBuffer, 0, this.readbackBuffer, 0, 16);
    }

    const canvasView = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{ view: canvasView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });

    // Confines every draw below to the letterboxed rect — background and
    // both marker layers all emit plain [-1,1]^2 NDC now (see
    // present.wgsl), so this single viewport is what maps that shared
    // space onto the actual centered sub-rect, keeping all three layers
    // in registration. Pixels outside it are simply never touched by
    // this pass, left at the clear color above.
    const vp = this.viewportRect;
    renderPass.setViewport(vp.x, vp.y, vp.width, vp.height, 0, 1);

    renderPass.setPipeline(this.backgroundPipeline);
    renderPass.setBindGroup(0, this.backgroundBindGroup);
    renderPass.draw(3, 1);

    // Agents drawn before target points, not after — target dots stay
    // visible as a reference outline even where agents currently
    // overlap them, rather than getting covered up.
    renderPass.setPipeline(this.markerPipeline);
    renderPass.setBindGroup(0, this.agentMarkerBindGroup);
    renderPass.draw(6, agentDispatchCount);

    if (this.targetMarkerBindGroup) {
      renderPass.setPipeline(this.markerPipeline);
      renderPass.setBindGroup(0, this.targetMarkerBindGroup);
      renderPass.draw(6, this.targetPointCount);
    }

    renderPass.end();

    this.device.queue.submit([encoder.finish()]);

    if (!this.pendingReadback) {
      this.pendingReadback = true;
      this.readbackBuffer
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const values = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
          this.readbackBuffer.unmap();
          const newMax: [number, number, number] = [values[0], values[1], values[2]];
          // render.py's DisplayScale.update(): first reading sets the
          // scale directly (no EMA yet), every reading after that blends.
          if (this.emaScale === null) {
            this.emaScale = newMax;
          } else {
            this.emaScale = [
              this.emaScale[0] + EMA_ALPHA * (newMax[0] - this.emaScale[0]),
              this.emaScale[1] + EMA_ALPHA * (newMax[1] - this.emaScale[1]),
              this.emaScale[2] + EMA_ALPHA * (newMax[2] - this.emaScale[2]),
            ];
          }
          this.writeScaleUniform();
          this.pendingReadback = false;
        })
        .catch((err) => {
          console.error("[envnca] EMA scale readback failed", err);
          this.pendingReadback = false;
        });
    }
  }

  destroy(): void {
    this.partialsBuffer.destroy();
    this.finalBuffer.destroy();
    this.readbackBuffer.destroy();
    this.scaleUniformBuffer.destroy();
    this.outputTexture.destroy();
    this.agentMarkerUniform.destroy();
    this.targetMarkerUniform.destroy();
    this.targetPositionsBuffer?.destroy();
  }
}

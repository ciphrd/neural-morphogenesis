// TS wrapper around interact.wgsl — the "Move Particles" tool's own
// radius-grab-then-drag mechanism (see that file's own module docstring
// for why this lives outside ../core/). Owns its own tiny grab-offset/
// uniform buffers and the two pipelines; every call here does its own
// encode + submit rather than folding into MpmCore.encodeSteps()'s own
// per-macro-step encoder, since these are event/frame-driven (pointer
// input), not part of the physics substep cadence at all.

import interactSrc from "./interact.wgsl?raw";
import { ceilDiv, writeFloat32 } from "./gpuUtil";
import { MAX_PARTICLES, type MpmCore } from "./mpmCore";

const WORKGROUP = 64;

export class Interact {
  private readonly device: GPUDevice;
  private readonly mpmCore: MpmCore;

  // Per-particle offset from the click point, captured once at grab time
  // — sized to MAX_PARTICLES, not whatever activeCount happened to be at
  // construction, for the same reason gpu/agents.ts's own heading/
  // angularVelocity buffers are: the "Add Particle" tool can grow
  // MpmCore's own activeCount past that at runtime, with no rebuild.
  private readonly grabOffsetBuffer: GPUBuffer;
  private readonly pickPosUniform: GPUBuffer;
  private readonly dragTargetUniform: GPUBuffer;

  private readonly beginGrabPipeline: GPUComputePipeline;
  private readonly beginGrabBindGroup: GPUBindGroup;
  private readonly applyDragPipeline: GPUComputePipeline;
  private readonly applyDragBindGroup: GPUBindGroup;

  constructor(device: GPUDevice, mpmCore: MpmCore) {
    this.device = device;
    this.mpmCore = mpmCore;

    this.grabOffsetBuffer = device.createBuffer({ size: MAX_PARTICLES * 2 * 4, usage: GPUBufferUsage.STORAGE });
    this.pickPosUniform = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dragTargetUniform = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const module = device.createShaderModule({ code: interactSrc });

    this.beginGrabPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "beginGrab" } });
    this.beginGrabBindGroup = device.createBindGroup({
      layout: this.beginGrabPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: mpmCore.activeCountUniform } },
        { binding: 2, resource: { buffer: this.pickPosUniform } },
        { binding: 3, resource: { buffer: this.grabOffsetBuffer } },
      ],
    });

    this.applyDragPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "applyDrag" } });
    this.applyDragBindGroup = device.createBindGroup({
      layout: this.applyDragPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: mpmCore.activeCountUniform } },
        { binding: 3, resource: { buffer: this.grabOffsetBuffer } },
        { binding: 4, resource: { buffer: mpmCore.velocities } },
        { binding: 5, resource: { buffer: this.dragTargetUniform } },
      ],
    });
  }

  /** Call once, on pointerdown — resolves EVERY particle within
   * interact.wgsl's own GRAB_RADIUS of `(x, y)` (MpmCore's [0,1]^2
   * domain coords) and freezes each one's own offset from that point,
   * entirely GPU-side (see beginGrab's own comment for why this also
   * correctly un-grabs anything a previous gesture left grabbed). A
   * click that found nothing within range just makes every subsequent
   * dragTo() this gesture a no-op (see applyDrag's own UNGRABBED
   * check). */
  beginGrab(x: number, y: number): void {
    writeFloat32(this.device, this.pickPosUniform, 0, new Float32Array([x, y]));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.beginGrabPipeline);
    pass.setBindGroup(0, this.beginGrabBindGroup);
    pass.dispatchWorkgroups(ceilDiv(this.mpmCore.activeCount, WORKGROUP));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Call every animation frame a drag gesture is active (not just on
   * pointermove) — re-pins every particle beginGrab() found (each at its
   * own frozen offset, so the clump translates rigidly rather than
   * collapsing onto the cursor) to `(x, y)` every frame, overriding
   * whatever this frame's own physics substeps just did to them, so the
   * clump stays glued to the cursor smoothly even while play is running,
   * not just at the instant of each discrete pointer event. */
  dragTo(x: number, y: number): void {
    writeFloat32(this.device, this.dragTargetUniform, 0, new Float32Array([x, y]));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.applyDragPipeline);
    pass.setBindGroup(0, this.applyDragBindGroup);
    pass.dispatchWorkgroups(ceilDiv(this.mpmCore.activeCount, WORKGROUP));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Call on pointerup/pointerleave. A no-op today — the next beginGrab()
   * already re-resolves every particle's grabbed/ungrabbed state from
   * scratch (see that function's own comment), so nothing needs
   * resetting in between gestures. Kept as a real method (not just
   * dropped from call sites) so render/GridCanvas.tsx's own pointerup
   * handling doesn't need to know that's true — cheap insurance against
   * this becoming false again if beginGrab's own strategy ever changes. */
  endDrag(): void {}

  destroy(): void {
    this.grabOffsetBuffer.destroy();
    this.pickPosUniform.destroy();
    this.dragTargetUniform.destroy();
  }
}

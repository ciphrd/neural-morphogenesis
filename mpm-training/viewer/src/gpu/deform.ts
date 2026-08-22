// TS wrapper around deform.wgsl — the "Deform" tool's own radial
// push/pull injection mechanism (see that file's own module docstring
// for why this lives outside ../core/, and for the exact velocity-
// impulse vs deformation-gradient-edit math, and for the per-particle
// radial direction this drives — NOT a single uniform vector). Owns its
// own tiny params uniform and one pipeline; every call here does its own
// encode + submit rather than folding into MpmCore.encodeSteps()'s own
// per-macro-step encoder, since this is event-driven (a held pointer),
// not part of the physics substep cadence at all — same reasoning
// gpu/interact.ts already documents for the same non-training-critical
// reason. Called once per RENDERED FRAME while the tool is held (see
// render/GridCanvas.tsx's own RAF loop), not once per click — same
// cadence Interact.dragTo() already runs at, for the same reason: a
// single-shot impulse read like a poke, not a force you can lean into.

import deformSrc from "./deform.wgsl?raw";
import { ceilDiv, writeFloat32 } from "./gpuUtil";
import type { MpmCore } from "./mpmCore";

const WORKGROUP = 64;

export type DeformMode = "velocity" | "deformation";

/** "outward": pushes/stretches particles AWAY from the click point (an
 * "explosion"). "inward": pulls/compresses them TOWARD it (an
 * "implosion"). See deform.wgsl's own module docstring for exactly how
 * each mode reads this. */
export type DeformDirection = "outward" | "inward";

export class Deform {
  private readonly device: GPUDevice;
  private readonly mpmCore: MpmCore;

  private readonly clickPosUniform: GPUBuffer;
  private readonly paramsUniform: GPUBuffer;

  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;

  constructor(device: GPUDevice, mpmCore: MpmCore) {
    this.device = device;
    this.mpmCore = mpmCore;

    this.clickPosUniform = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // strength (f32, 4) + radius (f32, 4) + outward (f32, 4) + mode
    // (f32, 4) = 16 bytes — must match deform.wgsl's own DeformParams
    // struct exactly, same field order.
    this.paramsUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const module = device.createShaderModule({ code: deformSrc });
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "injectDeform" } });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mpmCore.positions } },
        { binding: 1, resource: { buffer: mpmCore.activeCountUniform } },
        { binding: 2, resource: { buffer: this.clickPosUniform } },
        { binding: 3, resource: { buffer: mpmCore.velocities } },
        { binding: 4, resource: { buffer: mpmCore.F } },
        { binding: 5, resource: { buffer: this.paramsUniform } },
      ],
    });
  }

  /** Injects a radial push (`direction`) at domain position (x,y)
   * (MpmCore's own [0,1]^2 domain coords), affecting every particle
   * within `radius` (same domain units) with a smoothstep falloff to
   * EXACTLY 0 at that radius — matching GridCanvas.tsx's own preview
   * circle exactly, so what's drawn is what's actually affected. Each
   * particle's own push direction is computed straight toward/away from
   * (x,y) individually — see deform.wgsl's own module docstring, this
   * isn't a single uniform vector. `strength` sets the effect's own
   * magnitude (raw scalar — see deform.wgsl's own VELOCITY_SCALE/
   * DEFORMATION_SCALE for the per-mode scaling this gets multiplied
   * through). Call once per RENDERED FRAME while the tool is held (see
   * render/GridCanvas.tsx's own RAF loop) — same cadence Interact.dragTo()
   * already runs at; each call is a full, independent injection (velocity
   * ADDS, deformation gradient left-multiplies — see deform.wgsl's own
   * injectDeform()), so holding for N frames compounds N times, same as
   * holding a real force/tool down would. */
  inject(x: number, y: number, direction: DeformDirection, strength: number, radius: number, mode: DeformMode): void {
    writeFloat32(this.device, this.clickPosUniform, 0, new Float32Array([x, y]));
    writeFloat32(
      this.device,
      this.paramsUniform,
      0,
      new Float32Array([strength, radius, direction === "outward" ? 1 : 0, mode === "velocity" ? 0 : 1])
    );
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(ceilDiv(this.mpmCore.activeCount, WORKGROUP));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.clickPosUniform.destroy();
    this.paramsUniform.destroy();
  }
}

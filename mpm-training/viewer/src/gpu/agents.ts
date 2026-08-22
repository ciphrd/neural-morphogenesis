// TS wrapper around agents.wgsl — owns the flattened NN weights buffer,
// the AgentPhysics uniform (maxAccel/maxStrafe/maxEnvWrite/
// maxAngularAccel/angularDamping/maxAngularVelocity/depositDistance/
// splitDisplacement/divisionCooldown/friction), and the persistent
// per-particle heading/angularVelocity/growthState state buffers (owned
// here, not MpmCore, not Environment — see agents.wgsl's own module
// docstring for why). Builds
// the two parity-indexed bind group variants agentStep needs
// (its gridCurrent/gradient bindings must track environment.ts's own
// ping-pong parity exactly, since both classes read/write the same
// physical buffers each macro step — see simulation.ts for how the two
// stay in lockstep).
//
// Also owns growth's own atomic "next free slot" counter (growthCountBuffer)
// and the tiny mappable staging buffer readGrownCount() reads it back
// through — see that method's own docstring, and gpu/simulation.ts's own
// module docstring, for why WebGPU's own async-only buffer readback
// (unlike the Python trainer's synchronous wgpu-py equivalent) makes
// step() itself async now.
//
// Strafe drives MpmCore's own `velocities` buffer directly again (an
// acceleration, damped by `friction`) — see agents.wgsl's own module
// docstring for the full history (this has flipped between velocity and
// a direct position nudge twice now). growthState packs growth's own
// per-particle rng+cooldown into ONE buffer (not two), purely to keep
// this shader's own storage buffer count under the 10-per-stage hardware
// ceiling this project's own dev adapter reported — see agents.wgsl's
// own module docstring for the confirmed uncaptured-error history behind
// that constraint.

import agentsSrc from "../../../core/agents.wgsl?raw";
import { templateShader } from "./shaderTemplate";
import { ceilDiv, writeFloat32 } from "./gpuUtil";
import type { Environment } from "./environment";
import { MAX_PARTICLES, type MpmCore } from "./mpmCore";
import { growthSeed, spawnUniform01 } from "./rng";
import type { UpdateRuleWeights } from "./types";

const WORKGROUP = 64;

export interface AgentsConfig {
  channels: number;
  hiddenDim: number;
  maxAccel: number;
  maxStrafe: number;
  maxEnvWrite: number;
  maxAngularAccel: number;
  angularDamping: number;
  maxAngularVelocity: number;
  chirality: boolean;
  depositDistance: number;
  depositSigma: number;
  splitDisplacement: number;
  divisionCooldown: number;
  friction: number;
  maxActiveParticles: number;
  spawnX: number;
  spawnY: number;
}

function weightLayout(channels: number, hiddenDim: number) {
  // +2 == core/agents.wgsl's own IN_DIM (value+grad_forward+grad_lateral
  // per channel, +2 for the agent's own spawn-center-relative (x,y)
  // position, appended after the per-channel triples — see that
  // constant's own comment) — hardcoded rather than imported, same
  // convention outDim's own "+5" below already follows.
  const inDim = channels * 3 + 2;
  // 4 == core/agents.wgsl's own SPOTS / trainer/simulation_settings.py's
  // own DEPOSIT_SPOTS (env_write, one per channel per deposit spot) + 5
  // == ANGULAR_DIM(1) + ACCEL_DIM(2) + STRAFE_DIM(2) — hardcoded rather
  // than imported (this file doesn't read simulation_settings.py at
  // all), matching how the "+5" tail was already hardcoded here before
  // DEPOSIT_SPOTS existed.
  const outDim = channels * 4 + 5;
  const fc1wOffset = 0;
  const fc1bOffset = fc1wOffset + hiddenDim * inDim;
  const fc2wOffset = fc1bOffset + hiddenDim;
  const fc2bOffset = fc2wOffset + outDim * hiddenDim;
  const totalFloats = fc2bOffset + outDim;
  return { inDim, outDim, fc1wOffset, fc1bOffset, fc2wOffset, fc2bOffset, totalFloats };
}

/** Flattens {fc1w,fc1b,fc2w,fc2b} (nn.Linear's own (out,in) orientation —
 * UpdateRule.export_weights()'s exact shape) into one Float32Array in
 * the fc1w/fc1b/fc2w/fc2b order agents.wgsl's own FC1W_OFFSET/etc.
 * consts expect. */
function flattenWeights(weights: UpdateRuleWeights, channels: number, hiddenDim: number): Float32Array {
  const { totalFloats } = weightLayout(channels, hiddenDim);
  const out = new Float32Array(totalFloats);
  let i = 0;
  for (const row of weights.fc1w) for (const v of row) out[i++] = v;
  for (const v of weights.fc1b) out[i++] = v;
  for (const row of weights.fc2w) for (const v of row) out[i++] = v;
  for (const v of weights.fc2b) out[i++] = v;
  return out;
}

/** PyTorch's own nn.Linear default init — kaiming_uniform_(a=sqrt(5)),
 * which for a plain Linear works out to uniform(-1/sqrt(fanIn), 1/sqrt(fanIn))
 * for both weight and bias (see torch's own Linear.reset_parameters()) —
 * matches what a FRESH, untrained update_rule.py UpdateRule instance
 * looks like, so the "Randomize weights" button produces the same kind
 * of network training itself starts from, not arbitrary-scale noise. */
function randomLinear(outDim: number, inDim: number): { w: number[][]; b: number[] } {
  const bound = 1 / Math.sqrt(inDim);
  const w = Array.from({ length: outDim }, () => Array.from({ length: inDim }, () => (Math.random() * 2 - 1) * bound));
  const b = Array.from({ length: outDim }, () => (Math.random() * 2 - 1) * bound);
  return { w, b };
}

/** Exported for net/trainingSocket.ts's own placeholder GenerationRecord
 * (random weights to render SOMETHING while generation 0 is still being
 * evaluated — see that hook's own comment) — the exact same generator
 * Agents.randomizeWeights() below uses for the "Randomize weights"
 * button, just also reachable without an Agents instance to call it on. */
export function randomWeights(channels: number, hiddenDim: number): UpdateRuleWeights {
  const { inDim, outDim } = weightLayout(channels, hiddenDim);
  const fc1 = randomLinear(hiddenDim, inDim);
  const fc2 = randomLinear(outDim, hiddenDim);
  return { fc1w: fc1.w, fc1b: fc1.b, fc2w: fc2.w, fc2b: fc2.b };
}

export class Agents {
  private readonly device: GPUDevice;
  private readonly channels: number;
  private readonly hiddenDim: number;

  private readonly weightsBuffer: GPUBuffer;
  private readonly physicsUniform: GPUBuffer;
  private readonly headingBuffer: GPUBuffer;
  private readonly angularVelocityBuffer: GPUBuffer;
  private readonly growthStateBuffer: GPUBuffer;
  private readonly growthCountBuffer: GPUBuffer;
  private readonly grownCountStaging: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroups: [GPUBindGroup, GPUBindGroup];
  // Assigned via setActiveCount() in the constructor (also growth's own
  // baseline write), not directly — `!` tells TS's definite-assignment
  // check that's fine, it just can't see through the method call itself.
  private dispatch!: number;

  constructor(device: GPUDevice, mpmCore: MpmCore, environment: Environment, config: AgentsConfig) {
    this.device = device;
    this.channels = config.channels;
    this.hiddenDim = config.hiddenDim;

    const { totalFloats } = weightLayout(config.channels, config.hiddenDim);
    this.weightsBuffer = device.createBuffer({ size: totalFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // 52 bytes — core/agents.wgsl's own AgentPhysics struct is 13 f32
    // fields (see that struct's own definition for the exact order
    // setPhysics() below must match) — the last 2 (spawnX/spawnY) are
    // NOT written by setPhysics() below; see setSpawnCenter()'s own
    // docstring for why those get a separate setter into this same
    // buffer instead.
    this.physicsUniform = device.createBuffer({ size: 52, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setPhysics({
      maxAccel: config.maxAccel,
      maxStrafe: config.maxStrafe,
      maxEnvWrite: config.maxEnvWrite,
      maxAngularAccel: config.maxAngularAccel,
      angularDamping: config.angularDamping,
      maxAngularVelocity: config.maxAngularVelocity,
      depositDistance: config.depositDistance,
      splitDisplacement: config.splitDisplacement,
      divisionCooldown: config.divisionCooldown,
      friction: config.friction,
      depositSigma: config.depositSigma,
    });
    this.setSpawnCenter(config.spawnX, config.spawnY);

    // Persistent per-particle heading/angularVelocity/growthState state —
    // owned here (not MpmCore, not Environment), zeroed at creation and
    // whenever resetHeading() is called (simulation.ts's own
    // restartRollout()). Sized to MAX_PARTICLES up front, like every one
    // of MpmCore's own per-particle buffers, NOT to config.particles (the
    // growth cap this run actually uses) — both the "Add Particle" tool
    // (gpu/interact.wgsl's own module docstring) AND growth (core/agents.wgsl's
    // own agentStep() — see that file's own module docstring) can grow
    // MpmCore's own activeCount past whatever this class was originally
    // constructed with, at runtime, with no rebuild; undersizing these
    // buffers would leave agentStep's own dispatch (sized off the SAME,
    // now-larger activeCount — see setActiveCount() below) reading/
    // writing past the end of them for every newly-added particle.
    const stateSize = MAX_PARTICLES * 4;
    this.headingBuffer = device.createBuffer({ size: stateSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.angularVelocityBuffer = device.createBuffer({ size: stateSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Growth's own rng (u32) + cooldown (f32) — packed into ONE 8-bytes-
    // per-particle buffer (core/agents.wgsl's own GrowthState struct:
    // {rng: u32, cooldown: f32}), not two separate buffers, purely to
    // stay under this shader's own 10-storage-buffer ceiling now that
    // `velocities` is bound too — see that file's own module docstring
    // for why. resetHeading() writes both fields interleaved via a
    // DataView matching this exact layout.
    this.growthStateBuffer = device.createBuffer({ size: MAX_PARTICLES * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Growth's own atomic "next free slot" counter — a single u32, kept
    // in sync with the "official" activeCount by setActiveCount() below
    // (not just at construction) — see core/agents.wgsl's own module
    // docstring for the full design. COPY_SRC so readGrownCount() can
    // copy it into grownCountStaging.
    this.growthCountBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // A separate, MAP_READ-capable buffer growthCountBuffer itself can't
    // be (STORAGE and MAP_READ are mutually exclusive usages in WebGPU)
    // — encodeReadGrownCount() copies into this every macro step,
    // readGrownCount() maps/reads/unmaps it asynchronously afterward.
    this.grownCountStaging = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // Every rollout always starts with exactly ONE particle and grows
    // via splitting from there (simulation.ts's own restartRollout()
    // calls setActiveCount(1) again itself, every rollout — this is just
    // a construction-time placeholder so there's a sane dispatch size
    // before the first rollout ever starts) — see types.ts's own
    // SimulationConfig.particles docstring for why that's now a CAP, not
    // a fixed starting count.
    this.setActiveCount(1);

    const module = device.createShaderModule({
      code: templateShader(agentsSrc, {
        CHANNELS: config.channels,
        HIDDEN_DIM: config.hiddenDim,
        FIELD_WIDTH: environment.width,
        FIELD_HEIGHT: environment.height,
        MAX_ACTIVE_PARTICLES: config.maxActiveParticles,
        // WGSL wants lowercase `true`/`false` — a raw JS boolean would
        // template-substitute as "true"/"false" too via String(), so
        // this one actually works either way, but spelled out for
        // parity with agents_gpu.py's own version of this same
        // gotcha (Python's str(bool) gives "True"/"False", invalid
        // WGSL, so that side needs the explicit conversion).
        CHIRALITY: config.chirality ? "true" : "false",
      }),
    });
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "agentStep" } });

    this.bindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.weightsBuffer } },
          { binding: 1, resource: { buffer: mpmCore.positions } },
          { binding: 2, resource: { buffer: mpmCore.activeCountUniform } },
          { binding: 3, resource: { buffer: environment.buffers[p] } },
          { binding: 4, resource: { buffer: environment.gradient } },
          { binding: 5, resource: { buffer: environment.depositScratch } },
          { binding: 6, resource: { buffer: this.physicsUniform } },
          { binding: 7, resource: { buffer: this.headingBuffer } },
          { binding: 8, resource: { buffer: this.angularVelocityBuffer } },
          { binding: 9, resource: { buffer: this.growthCountBuffer } },
          { binding: 10, resource: { buffer: this.growthStateBuffer } },
          { binding: 11, resource: { buffer: mpmCore.velocities } },
        ],
      })
    ) as [GPUBindGroup, GPUBindGroup];
  }

  /** Exposed so Renderer's own triangle-shape pipeline can point each
   * particle along its real heading state rather than (mis-)deriving one
   * from velocity — see render.wgsl's own triangleVertex. */
  get headingState(): GPUBuffer {
    return this.headingBuffer;
  }

  /** Exposed so gpu/nnProbe.ts can bind the exact same live weights
   * buffer agentStep() itself reads — a probe always reflects whatever's
   * currently loaded, loadWeights()/randomizeWeights() included, no
   * extra plumbing needed on either side. */
  get weightsState(): GPUBuffer {
    return this.weightsBuffer;
  }

  /** Same reasoning as weightsState above — gpu/nnProbe.ts needs
   * maxEnvWrite/maxAngularAccel/maxStrafe to squash/scale its own
   * forward pass identically to agentStep()'s own evalPolicy(), and this
   * is the one buffer that already holds them (AgentPhysics's full
   * struct — nnProbe.wgsl declares the identical struct layout so the
   * byte offsets line up without needing a second, narrower uniform). */
  get physicsState(): GPUBuffer {
    return this.physicsUniform;
  }

  loadWeights(weights: UpdateRuleWeights): void {
    writeFloat32(this.device, this.weightsBuffer, 0, flattenWeights(weights, this.channels, this.hiddenDim));
  }

  /** Overwrites the weights buffer with a fresh random init (see
   * randomWeights()'s own docstring) — same live buffer write as
   * loadWeights(), no rebuild. Callers generally want to follow this
   * with restartRollout() too (see GpuSimulation.randomizeWeights()) —
   * particles already grown under the old weights don't retroactively
   * un-grow just because the policy driving them changed. */
  randomizeWeights(): void {
    this.loadWeights(randomWeights(this.channels, this.hiddenDim));
  }

  setPhysics(settings: {
    maxAccel: number;
    maxStrafe: number;
    maxEnvWrite: number;
    maxAngularAccel: number;
    angularDamping: number;
    maxAngularVelocity: number;
    depositDistance: number;
    splitDisplacement: number;
    divisionCooldown: number;
    friction: number;
    depositSigma: number;
  }): void {
    writeFloat32(
      this.device,
      this.physicsUniform,
      0,
      new Float32Array([
        settings.maxAccel,
        settings.maxStrafe,
        settings.maxEnvWrite,
        settings.maxAngularAccel,
        settings.angularDamping,
        settings.maxAngularVelocity,
        settings.depositDistance,
        settings.splitDisplacement,
        settings.divisionCooldown,
        settings.friction,
        settings.depositSigma,
      ])
    );
  }

  /** Writes AgentPhysics.spawnX/spawnY — offset 44 (bytes), the last 2 of
   * the struct's own 13 f32 fields, past everything setPhysics() above
   * writes. A separate setter (not folded into setPhysics()) since spawn
   * center is fixed for a whole rollout, not something PhysicsPanel-style
   * live tuning ever touches — call once per rollout (simulation.ts's own
   * restartRollout()), not on every physics-slider tick the way
   * setPhysics() is. See core/agents.wgsl's own AgentPhysics.spawnX/
   * spawnY field comment for what this drives (the NN's own position
   * input, agentStep()'s own inputVec population — relative to spawn
   * center, not the domain's own fixed (0.5,0.5)). */
  setSpawnCenter(spawnX: number, spawnY: number): void {
    writeFloat32(this.device, this.physicsUniform, 44, new Float32Array([spawnX, spawnY]));
  }

  /** Updates this class's own agentStep() dispatch size AND growth's own
   * atomic "next free slot" counter (core/agents.wgsl's own module
   * docstring), which always needs to start from the current
   * activeCount — called once per rollout (simulation.ts's own
   * restartRollout()) and again every macro step growth actually changes
   * the count (that module's own step(), after awaiting
   * readGrownCount()). Deliberately does NOT touch
   * mpmCore.activeCountUniform itself — MpmCore.setActiveCount() (a
   * distinct method, on a distinct object) owns that, since it's shared
   * with p2g/gridUpdate-adjacent/g2p/repulsion too, not just this
   * class's own dispatch. */
  setActiveCount(activeCount: number): void {
    this.dispatch = ceilDiv(activeCount, WORKGROUP);
    writeFloat32(this.device, this.growthCountBuffer, 0, new Uint32Array([activeCount]));
  }

  /** Encodes the copy of growth's own atomic counter into a mappable
   * staging buffer — must be encoded AFTER encodeStep() in the SAME
   * submit (see gpu/simulation.ts's own step()), so the copied value
   * reflects whatever this macro step's own agentStep() pass just
   * claimed. Does not submit. */
  encodeReadGrownCount(encoder: GPUCommandEncoder): void {
    encoder.copyBufferToBuffer(this.growthCountBuffer, 0, this.grownCountStaging, 0, 4);
  }

  /** Reads back growth's own atomic counter, via encodeReadGrownCount()'s
   * own staging copy from THIS macro step's submit — a real, deliberate
   * host round-trip, once per macro step (gpu/simulation.ts's own step()
   * is the only caller), needed because dispatch sizing for EVERY pass
   * (P2G/gridUpdate/G2P/repulsion, and this class's own next
   * agentStep()) is decided on the CPU/JS side, from a cached count
   * nothing else updates automatically when growth happens purely on the
   * GPU. Async — unlike the Python trainer's synchronous wgpu-py
   * read_buffer(), WebGPU's own buffer readback (mapAsync) has no
   * synchronous equivalent, which is why gpu/simulation.ts's own step()
   * is async too (see that module's own module docstring). */
  async readGrownCount(): Promise<number> {
    await this.grownCountStaging.mapAsync(GPUMapMode.READ);
    const value = new Uint32Array(this.grownCountStaging.getMappedRange())[0];
    this.grownCountStaging.unmap();
    return value;
  }

  /** Randomizes persistent heading state (uniform over [-pi, pi], one
   * independent draw per particle slot, via rng.ts's own
   * spawnUniform01(seed, 5+i) — index 5+, not 0: seedBlob()'s own 2
   * particles' x/y jitter claims 0-3, simulation.ts's own theta draw
   * claims 4, this is the next range over, bit-exact with
   * agents_gpu.py's own reset_heading()), zeroes angularVelocity, and
   * reseeds growth's own persistent per-particle growthState.rng
   * (nonzero — see core/agents.wgsl's own xorshift32() for why, via
   * rng.ts's own growthSeed() — a DELIBERATELY SEPARATE hash domain from
   * heading's own spawnUniform01() above despite both being bit-exact
   * now, see growthSeed()'s own comment for why) while zeroing
   * growthState.cooldown ("not on cooldown," so a fresh rollout's own
   * starting particle can split immediately, same as before cooldown
   * existed) — called whenever a rollout restarts (simulation.ts's own
   * restartRollout()). Bundled into this same method (despite the name)
   * rather than a separate one since every caller already calls this
   * once per rollout, at exactly the right time; matches this method's
   * own existing "resetHeading also resets angularVelocity" precedent
   * for outgrowing its own name slightly. Heading is randomized (not
   * zeroed) so every particle doesn't start out facing an identical,
   * seed-independent direction — see agents_gpu.py's own reset_heading()
   * for the fuller reasoning, including why this was worth making bit-
   * exact (not just plausible) even though every slot's own value here
   * gets immediately overwritten either way (setHeadings() below for
   * slots 0/1, or growth copying from its own parent's live heading the
   * moment any other slot is actually claimed) — a standing "doesn't
   * matter today" caveat on an un-reproducible PRNG stream was fragile,
   * not a real savings. angularVelocity stays zeroed regardless — a
   * random *turn rate* would just be an initial spin, not a meaningfully
   * different starting condition the way a random facing direction is.
   *
   * Single `seed` param (the raw rollout seed) — heading's own
   * spawnUniform01() domain and growth's own growthSeed() domain are
   * both bit-exact and mutually uncorrelated by construction (distinct
   * hash domains, not distinct seed VALUES), so there's no more need for
   * the "offset the seed to decorrelate two draws off one shared
   * mulberry32 stream" trick this used to need (see rng.ts's own
   * spawnUniform01()/growthSeed() comments). */
  resetHeading(seed: number): void {
    const count = this.headingBuffer.size / 4;
    const headings = new Float32Array(count);
    for (let i = 0; i < count; i++) headings[i] = (spawnUniform01(seed, 5 + i) * 2 - 1) * Math.PI;
    const zeros = new Float32Array(count);
    this.device.queue.writeBuffer(this.headingBuffer, 0, headings);
    this.device.queue.writeBuffer(this.angularVelocityBuffer, 0, zeros);

    // Interleaved rng(u32)/cooldown(f32) pairs, matching
    // core/agents.wgsl's own GrowthState struct exactly (see this
    // class's own constructor comment for why they're packed together)
    // — a plain Float32Array/Uint32Array can't represent that mixed
    // layout, so this writes through a DataView instead, 8 bytes at a
    // time. cooldown is left at 0 (the buffer's own zero-initialized
    // default) — "not on cooldown."
    const growthState = new ArrayBuffer(count * 8);
    const view = new DataView(growthState);
    for (let i = 0; i < count; i++) {
      view.setUint32(i * 8, growthSeed(seed, i), true);
      view.setFloat32(i * 8 + 4, 0, true);
    }
    this.device.queue.writeBuffer(this.growthStateBuffer, 0, growthState);
  }

  /** Overwrites the FIRST headings.length heading slots directly, a
   * small follow-up write on top of whatever resetHeading() above just
   * wrote there (every slot, independently randomized) — for callers
   * that need a handful of slots' own heading coordinated with each
   * other instead of independent (currently: simulation.ts's own
   * restartRollout(), hardcoded 2-particle "back to back" start).
   * Bit-exact with agents_gpu.py's own set_headings() (not just in
   * spirit — resetHeading() above no longer has an accepted
   * reproducibility gap for callers of this to inherit), not folded into
   * resetHeading() itself, which stays a general, per-slot-independent
   * utility. */
  setHeadings(headings: Float32Array): void {
    writeFloat32(this.device, this.headingBuffer, 0, headings);
  }

  /** Encodes the NN forward pass — reads the environment's current
   * parity buffer (must match `parity`, see simulation.ts), writes a
   * strafe-driven acceleration straight into MpmCore's own velocities
   * buffer and env_write into the environment's deposit scratch (see
   * agents.wgsl's own module docstring for the full strafe/velocity
   * history). Does not submit. */
  encodeStep(encoder: GPUCommandEncoder, parity: number): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroups[parity]);
    pass.dispatchWorkgroups(this.dispatch);
    pass.end();
  }

  destroy(): void {
    this.weightsBuffer.destroy();
    this.physicsUniform.destroy();
    this.headingBuffer.destroy();
    this.angularVelocityBuffer.destroy();
    this.growthStateBuffer.destroy();
    this.growthCountBuffer.destroy();
    this.grownCountStaging.destroy();
  }
}

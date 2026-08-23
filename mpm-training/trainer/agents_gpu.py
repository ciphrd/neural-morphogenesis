"""The evolved per-particle policy, GPU-resident — Python (wgpu-py) port
of ../viewer/src/gpu/agents.ts, wrapping the exact same shared shader
module (../core/agents.wgsl) mpm_core.py already loads its own
core/*.wgsl passes from. Replaces update_rule.py's own UpdateRule.forward()
in the hot per-candidate training path — see training_sim.py's own
module docstring for why (eliminating the torch/MPS<->wgpu/Metal host
round-trip that dominated the old per-macro-step cost). UpdateRule/torch
itself isn't gone — evolve.py still uses it for random weight
initialization and checkpoint JSON export, both one-off/off-hot-path
uses that don't need a live forward pass.

One instance is built ONCE per training run (like MpmCore/EnvironmentGPU
— see evolve.py's own module docstring on why rebuilding wgpu pipelines
per candidate is real, avoidable overhead) and load_weights()/
reset_heading() are called per candidate/rollout instead, mirroring
agents.ts's own instance lifetime (rebuilt only when particle/channel/
field/hidden-dim shape changes, which never happens mid-run here since
those are fixed CLI args)."""
from __future__ import annotations

import numpy as np
import wgpu

from simulation_settings import (
    CHEMICAL_GRADIENT_INPUT_SCALE,
    CHEMICAL_VALUE_INPUT_SCALE,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    ELASTIC_STRAIN_SCALE,
    MORPHOLOGY_GRADIENT_INPUT_SCALE,
)

from environment_gpu import EnvironmentGPU, ceil_div
from mpm_core import MpmCore, REPULSION_FIELD_N
from shader_template import load_core_shader

WORKGROUP = 64
PARTICLE_META_BUFFER_OFFSET = 256


def _hash_u32(x: np.ndarray) -> np.ndarray:
    """Bit-exact, portable integer hash (Chris Wellons' "lowbias32" —
    public domain), mirrored exactly by ../viewer/src/gpu/rng.ts's own
    hashU32() — see that function's own comment for why growth's own
    seed needs this instead of drawing from `rng` (numpy's own
    Generator/PCG64 has no TS-side equivalent, unlike this: only uint32
    add/xor/shift/multiply-with-wraparound, which numpy's own uint32
    dtype and JS's Math.imul/>>> 0 both wrap mod 2**32 identically).
    Vectorized over an array of u32 rather than called per-scalar,
    matching every other per-particle draw in reset_heading() below."""
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def _growth_seed(seed: int, count: int) -> np.ndarray:
    """particleMeta.rng's own initial per-particle seed — bit-exact with
    ../viewer/src/gpu/rng.ts's own growthSeed(seed, index), see that
    function's own comment. `seed` is the rollout's own raw seed
    (evolve.py's own rollout(seed, ...) argument / render_rollout.py's
    own meta["seed"]) — the ENTIRE rollout's starting condition (this,
    reset_heading()'s own heading fill below, and training_sim.py's own
    seed_blob()/back-to-back theta) is now a pure function of this one
    integer, no numpy Generator/mulberry32 involved anywhere in the
    seeding path — see _spawn_uniform01()'s own comment for why growth
    keeps its own SEPARATE hash domain from that function rather than
    sharing it (near-critical branching process, chaotically sensitive
    to its own seed stream — the two are never meant to correlate
    regardless). Low=1 for any hash that comes out 0 — xorshift32's own
    fixed point, same guarantee reset_heading() used to get from
    rng.integers(1, ...) before this was hash-based."""
    index = np.arange(count, dtype=np.uint32)
    combined = np.uint32(seed) ^ _hash_u32(index + np.uint32(1))
    hashed = _hash_u32(combined)
    hashed[hashed == 0] = np.uint32(1)
    return hashed


# Magic domain-separator XOR'd into the index before hashing — keeps
# _spawn_uniform01() below's own output space disjoint from
# _growth_seed() above even when both happen to be called with the same
# (seed, index) pair (training_sim.py's own spawn-jitter/heading indices
# are small integers, the same range _growth_seed() iterates particle
# slots over) — same "don't reuse one stream for two kinds of
# randomness" reasoning this project already applies elsewhere (e.g.
# growth's own child-reseeding, core/agents.wgsl's own agentStep()
# comment). Arbitrary, just needs to be nonzero.
_SPAWN_HASH_DOMAIN = np.uint32(0xC0FFEE00)


def _spawn_uniform01(seed: int, index: int) -> float:
    """One deterministic float in [0,1), bit-exact with
    ../viewer/src/gpu/rng.ts's own spawnUniform01(seed, index) — the
    portable hash EVERY piece of a rollout's own starting-condition
    randomness that ISN'T growth now goes through: training_sim.py's own
    seed_blob() (spawn-position jitter) and back-to-back theta, and
    reset_heading() below's own per-slot heading fill. Domain-separated
    from _growth_seed() above via _SPAWN_HASH_DOMAIN (see that constant's
    own comment). Top 24 bits of the hash -> a uniform float, same "use
    every bit of f32 mantissa precision" convention core/agents.wgsl's
    own xorshift32-derived draw already uses
    (`f32(rngNext >> 8u) * (1.0/16777216.0)`)."""
    combined = np.uint32(seed) ^ _hash_u32(np.array([_SPAWN_HASH_DOMAIN ^ np.uint32(index)]))[0]
    hashed = _hash_u32(np.array([combined]))[0]
    return float(hashed >> np.uint32(8)) / 16777216.0


def _spawn_uniform01_batch(seed: int, indices: np.ndarray) -> np.ndarray:
    """Vectorized _spawn_uniform01() — same formula, called once over an
    array of indices rather than per-scalar, for reset_heading() below's
    own per-slot fill (up to max_active_particles draws every rollout)."""
    indices = indices.astype(np.uint32)
    combined = np.uint32(seed) ^ _hash_u32(_SPAWN_HASH_DOMAIN ^ indices)
    hashed = _hash_u32(combined)
    return (hashed >> np.uint32(8)).astype(np.float64) / 16777216.0


def weight_layout(channels: int, hidden_dim: int) -> dict[str, int]:
    # core/agents.wgsl's own IN_DIM: value + heading-forward gradient +
    # lateral gradient per channel, with no positional inputs.
    in_dim = channels * 3 + 6
    # One centered env_write per channel + ANGULAR_DIM(1) + ACCEL_DIM(2)
    # + STRAFE_DIM(2) + RGB_DIM(3).
    out_dim = channels + 8
    fc1w_offset = 0
    fc1b_offset = fc1w_offset + hidden_dim * in_dim
    fc2w_offset = fc1b_offset + hidden_dim
    fc2b_offset = fc2w_offset + out_dim * hidden_dim
    total_floats = fc2b_offset + out_dim
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "fc1w_offset": fc1w_offset,
        "fc1b_offset": fc1b_offset,
        "fc2w_offset": fc2w_offset,
        "fc2b_offset": fc2b_offset,
        "total_floats": total_floats,
    }


class AgentsGPU:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        core: MpmCore,
        environment: EnvironmentGPU,
        channels: int,
        hidden_dim: int,
        max_accel: float,
        max_strafe: float,
        max_env_write: float,
        max_angular_accel: float,
        angular_damping: float,
        max_angular_velocity: float,
        chirality: bool,
        deposit_distance: float,
        max_active_particles: int,
        split_displacement: float,
        division_cooldown: float,
        friction: float,
        deposit_sigma: float,
        growth_enabled: float,
        spawn_x: float,
        spawn_y: float,
        elastic_strain_scale: float = ELASTIC_STRAIN_SCALE,
        elastic_strain_inputs_enabled: bool = ELASTIC_STRAIN_INPUTS_ENABLED,
    ) -> None:
        self.device = device
        self.channels = channels
        self.hidden_dim = hidden_dim
        self._max_active_particles = max_active_particles
        # Public rollout geometry setting: TrainingRollout uses the same
        # displacement configured on this agent instance for its coordinated
        # two-particle seed, keeping diagnostic/replay overrides consistent.
        self.split_displacement = float(split_displacement)

        layout = weight_layout(channels, hidden_dim)
        self._total_floats = layout["total_floats"]
        self._weights_buffer = device.create_buffer(
            size=layout["total_floats"] * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        # 64 bytes — core/agents.wgsl's AgentPhysics ends with a u32 growth
        # cap and f32 elastic-strain normalization.
        # NOT written by set_physics() below; see set_spawn_center()'s own
        # docstring for why those get a separate setter into this same
        # buffer instead.
        self._physics_uniform = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_physics(
            max_accel,
            max_strafe,
            max_env_write,
            max_angular_accel,
            angular_damping,
            max_angular_velocity,
            deposit_distance,
            split_displacement,
            division_cooldown,
            friction,
            deposit_sigma,
            growth_enabled,
        )
        self.set_spawn_center(spawn_x, spawn_y)
        self.set_max_active_particles(max_active_particles)
        self.set_elastic_strain_scale(elastic_strain_scale)

        # Persistent per-particle state — owned here (not MpmCore, not
        # EnvironmentGPU), zeroed at creation (randomized/reseeded
        # properly once reset_heading() is called with a real rng — once
        # per rollout, see training_sim.py's own TrainingRollout.__init__)
        # and whenever reset_heading() is called again after that. Sized
        # to max_active_particles, NOT the single particle every rollout
        # actually starts with — growth (core/agents.wgsl's own
        # agentStep()) can write particleMeta[newIndex] for any newIndex
        # up to max_active_particles, at runtime, with no rebuild (mirrors
        # agents.ts's own reasoning for sizing THOSE buffers to
        # MAX_PARTICLES, for the "Add Particle" tool's own runtime growth
        # — same underlying need, smaller ceiling since this class has no
        # such interactive tool of its own).
        #
        # rng/cooldown/heading/angularVelocity plus aligned neural RGBA —
        # packed into ONE 48-bytes-per-particle buffer
        # (core/agents.wgsl's own ParticleMeta struct), not four separate
        # buffers: this shader hit a REAL, confirmed CreateComputePipeline
        # validation error the first time particleF/particleC/particleJp
        # tried to add 3 more bindings on top of heading/angularVelocity/
        # growthState each having their own — Chrome's own Dawn backend
        # reports a hard 10-storage-buffer-per-stage ceiling on real
        # browser adapters (NOT the much higher number wgpu-native/Metal
        # reports headlessly on this side, which is why this constructor
        # never hit the problem itself) — see core/agents.wgsl's own
        # module docstring for the full account. rng+cooldown were
        # already packed together once before, for the exact same reason,
        # when `velocities` was added; heading/angularVelocity joined them
        # here to free the 2 slots particleF/particleJp needed (particleC
        # was dropped instead of freeing a 3rd — see core/agents.wgsl's
        # own comment on why that one's safe to skip).
        self._particle_meta_dtype = np.dtype([
            ("rng", "<u4"),
            ("cooldown", "<f4"),
            ("heading", "<f4"),
            ("angularVelocity", "<f4"),
            ("color", "<f4", (4,)),
            ("divisionHazard", "<f4"),
            ("divisionThreshold", "<f4"),
            ("_padding", "<f4", (2,)),
        ])
        # AgentState packs the growth counter at byte 0 and ParticleMeta at
        # byte 256. Besides satisfying storage-offset alignment for the
        # viewer's render binding, this frees one agent shader binding for C.
        self._agent_state_buffer = device.create_buffer(
            size=PARTICLE_META_BUFFER_OFFSET + max(max_active_particles, 1) * self._particle_meta_dtype.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        # Rollouts start with their configured initial particle count and grow
        # via splitting from there (training_sim.py's own
        # TrainingRollout.__init__ sets that count every rollout — this is
        # just a construction-time placeholder so
        # there's a sane dispatch size before the first rollout ever
        # starts) — see evolve.py's own module docstring for why
        # --particles is now a CAP, not a fixed starting count.
        self.set_active_count(1)

        module = device.create_shader_module(
            code=load_core_shader(
                "agents.wgsl",
                {
                    "CHANNELS": channels,
                    "HIDDEN_DIM": hidden_dim,
                    "FIELD_WIDTH": environment.width,
                    "FIELD_HEIGHT": environment.height,
                    "MORPHOLOGY_FIELD_N": REPULSION_FIELD_N,
                    "CHEMICAL_VALUE_INPUT_SCALE": repr(CHEMICAL_VALUE_INPUT_SCALE),
                    "CHEMICAL_GRADIENT_INPUT_SCALE": repr(CHEMICAL_GRADIENT_INPUT_SCALE),
                    "MORPHOLOGY_GRADIENT_INPUT_SCALE": repr(MORPHOLOGY_GRADIENT_INPUT_SCALE),
                    # WGSL wants lowercase `true`/`false` — Python's own
                    # str(bool) gives "True"/"False", invalid WGSL syntax,
                    # so this can't just be passed through as-is.
                    "CHIRALITY": "true" if chirality else "false",
                    "ELASTIC_STRAIN_INPUTS_ENABLED": "true" if elastic_strain_inputs_enabled else "false",
                },
            )
        )
        self._pipeline = device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "agentStep"})

        self._step_mode_uniforms = [
            device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
        ]
        self.set_communication_timestep(1.0)

        def bind_group(p: int, commit_lifecycle: int):
            env_buf = environment.buffers[p]
            return device.create_bind_group(
                layout=self._pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self._weights_buffer, "offset": 0, "size": self._weights_buffer.size}},
                    {"binding": 1, "resource": {"buffer": core.positions, "offset": 0, "size": core.positions.size}},
                    {"binding": 2, "resource": {"buffer": core.active_count_uniform, "offset": 0, "size": core.active_count_uniform.size}},
                    {"binding": 3, "resource": {"buffer": env_buf, "offset": 0, "size": env_buf.size}},
                    {"binding": 4, "resource": {"buffer": environment.gradient, "offset": 0, "size": environment.gradient.size}},
                    {"binding": 5, "resource": {"buffer": environment.deposit_scratch, "offset": 0, "size": environment.deposit_scratch.size}},
                    {"binding": 6, "resource": {"buffer": self._physics_uniform, "offset": 0, "size": self._physics_uniform.size}},
                    {"binding": 7, "resource": {"buffer": self._agent_state_buffer, "offset": 0, "size": self._agent_state_buffer.size}},
                    {"binding": 8, "resource": {"buffer": core.C, "offset": 0, "size": core.C.size}},
                    {"binding": 9, "resource": {"buffer": core.velocities, "offset": 0, "size": core.velocities.size}},
                    # core.F/core.rest — same buffers core/p2g.wgsl and
                    # core/g2p.wgsl already read/write every physics substep
                    # — see core/agents.wgsl's own module docstring for why
                    # agentStep() now needs them too (a freshly-claimed
                    # particle inherits its parent's CURRENT deformation
                    # state at split time, not a fresh identity/zero rest
                    # state). core.C is binding 8, completing APIC-state
                    # inheritance across division.
                    {"binding": 10, "resource": {"buffer": core.F, "offset": 0, "size": core.F.size}},
                    {"binding": 11, "resource": {"buffer": core.rest, "offset": 0, "size": core.rest.size}},
                    {"binding": 12, "resource": core.morphology_texture.create_view()},
                    {"binding": 13, "resource": {"buffer": self._step_mode_uniforms[commit_lifecycle]}},
                ],
            )

        self._communication_bind_groups = [bind_group(0, 0), bind_group(1, 0)]
        self._commit_bind_groups = [bind_group(0, 1), bind_group(1, 1)]

    @property
    def max_active_particles(self) -> int:
        """The growth cap this instance was constructed with (--particles
        — see evolve.py's own module docstring) — training_sim.py's own
        TrainingRollout reads this rather than taking a redundant
        constructor parameter of its own, so there's exactly one source
        of truth for "how many slots does this instance's own
        particleMeta buffer actually have"."""
        return self._max_active_particles

    def load_weights(self, flat_weights: np.ndarray) -> None:
        """`flat_weights` is a flat (total_floats,) float array already
        laid out fc1w/fc1b/fc2w/fc2b, row-major within each — exactly
        torch.nn.utils.parameters_to_vector(UpdateRule(...).parameters())'s
        own output shape: nn.Linear registers `weight` before `bias`,
        nn.Sequential visits fc1 before fc2, and parameters_to_vector
        concatenates each parameter tensor's own row-major .view(-1) — the
        same fc1w/fc1b/fc2w/fc2b order and row-major layout agents.wgsl's
        own FC1W_OFFSET/FC1B_OFFSET/FC2W_OFFSET/FC2B_OFFSET indexing
        expects. evolve.py's own get_weights()/mutate() already produce
        exactly this representation (that's what parameters_to_vector
        gives them), so this is a straight write_buffer, no restructuring
        — unlike agents.ts's own flattenWeights(), which has to convert
        *from* UpdateRuleWeights' nested JSON shape (export_weights()'s
        own format), a shape this hot path never produces or needs."""
        assert flat_weights.shape[0] == self._total_floats, (
            f"expected {self._total_floats} floats, got {flat_weights.shape[0]}"
        )
        self.device.queue.write_buffer(self._weights_buffer, 0, flat_weights.astype(np.float32))

    def set_physics(
        self,
        max_accel: float,
        max_strafe: float,
        max_env_write: float,
        max_angular_accel: float,
        angular_damping: float,
        max_angular_velocity: float,
        deposit_distance: float,
        split_displacement: float,
        division_cooldown: float,
        friction: float,
        deposit_sigma: float,
        growth_enabled: float,
    ) -> None:
        self.device.queue.write_buffer(
            self._physics_uniform,
            0,
            np.array(
                [
                    max_accel,
                    max_strafe,
                    max_env_write,
                    max_angular_accel,
                    angular_damping,
                    max_angular_velocity,
                    deposit_distance,
                    split_displacement,
                    division_cooldown,
                    friction,
                    deposit_sigma,
                    growth_enabled,
                ],
                dtype=np.float32,
            ),
        )

    def set_growth_enabled(self, enabled: bool) -> None:
        """Controls whether agentStep may start new cell cycles.

        Byte offset 44 is AgentPhysics.growthEnabled, the twelfth f32.
        Existing cycles continue to g=2 and divide after this is false,
        which makes the rollout tail a settling window rather than an
        abrupt cancellation of already accumulated growth.
        """
        self.device.queue.write_buffer(
            self._physics_uniform,
            44,
            np.array([1.0 if enabled else 0.0], dtype=np.float32),
        )

    def set_communication_timestep(self, dt: float) -> None:
        """Set the neural substep clock while retaining two lifecycle modes."""
        for commit, buffer in enumerate(self._step_mode_uniforms):
            data = np.zeros(4, dtype=np.uint32)
            data[0] = commit
            data.view(np.float32)[1] = max(0.0, float(dt))
            self.device.queue.write_buffer(buffer, 0, data)

    def set_spawn_center(self, spawn_x: float, spawn_y: float) -> None:
        """Writes the legacy AgentPhysics spawn slots at byte offset 48.

        Spawn coordinates still configure rollout initialization, but the
        policy no longer reads them. The slots remain to preserve the uniform
        ABI while old and new frontend/backend processes overlap.
        """
        self.device.queue.write_buffer(self._physics_uniform, 48, np.array([spawn_x, spawn_y], dtype=np.float32))

    def set_max_active_particles(self, max_active_particles: int) -> None:
        """Writes AgentPhysics.maxActiveParticles at byte offset 56."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            56,
            np.array([max(1, int(max_active_particles))], dtype=np.uint32),
        )

    def set_elastic_strain_scale(self, scale: float) -> None:
        """Writes AgentPhysics.elasticStrainScale at byte offset 60."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            60,
            np.array([max(float(scale), 1e-6)], dtype=np.float32),
        )

    def set_active_count(self, active_count: int) -> None:
        """Updates this class's own agentStep() dispatch size AND
        growth's own atomic "next free slot" counter (core/agents.wgsl's
        own module docstring), which always needs to start from the
        current active_count — called once per rollout (training_sim.py's
        own TrainingRollout.__init__) and again every macro step growth
        actually changes the count (that module's own macro_step(), after
        reading read_grown_count() back). Deliberately does NOT touch
        core.active_count_uniform itself — MpmCore.set_active_count() (a
        distinct method, on a distinct object) owns that, since it's
        shared with p2g/gridUpdate-adjacent/g2p/repulsion too, not just
        this class's own dispatch."""
        self._dispatch = ceil_div(active_count, WORKGROUP)
        self.device.queue.write_buffer(self._agent_state_buffer, 0, np.array([active_count], dtype=np.uint32))

    def read_grown_count(self) -> int:
        """Reads back growth's own atomic counter (core/agents.wgsl's own
        module docstring) — a real, deliberate 4-byte host round-trip,
        once per macro step (training_sim.py's own macro_step() is the
        only caller), needed because dispatch sizing for EVERY pass
        (P2G/gridUpdate/G2P/repulsion, and this class's own next
        agentStep()) is decided on the CPU, from a CPU-cached count nothing
        else updates automatically when growth happens purely on the GPU.
        Blocks until every previously-submitted command (including the
        agentStep() pass that may have grown this count) has actually
        run — same "reading anything back necessarily waits for the
        queue's own timeline to catch up" property mpm_core.py's own
        step() already relies on for its per-chunk sync."""
        raw = self.device.queue.read_buffer(self._agent_state_buffer, 0, 4)
        return int(np.frombuffer(raw, dtype=np.uint32)[0])

    def reset_heading(self, seed: int) -> None:
        """Randomizes persistent heading state (uniform over [-pi, pi],
        one independent draw per particle slot) and zeroes
        angularVelocity/cooldown, EVERY slot up to max_active_particles
        (not just the currently-active ones — growth can claim any of
        them later in this same rollout, and agentStep() overwrites
        whatever a claimed slot's own particleMeta already held anyway,
        so pre-resetting the full range costs nothing extra and needs no
        separate "which slots are real yet" bookkeeping here). Also
        reseeds growth's own persistent per-particle rng (nonzero — see
        core/agents.wgsl's own xorshift32() for why) — bundled into this
        same method (despite the name) rather than a separate one since
        every caller already calls this once per rollout, with a real
        seed, at exactly the right time; matches this method's own
        existing "resetHeading also resets angularVelocity" precedent for
        outgrowing its own name slightly. Call at the start of every
        rollout, same as agents.ts's own resetHeading() (simulation.ts's
        own restartRollout() calls it every time a rollout restarts, for
        the same reason: fresh policy-side state, not carried over from
        whatever the previous rollout left it at). Heading is randomized
        (not zeroed) so every particle in a rollout doesn't start out
        facing an identical, seed-independent direction — via
        _spawn_uniform01_batch(seed, 5 + slot_index) (index 5+, not 0 —
        see training_sim.py's own seed_blob()/theta for what already
        claims indices 0-4 off this same `seed`; see
        _spawn_uniform01()'s own comment for why this stays a DIFFERENT
        hash domain from the growth rng below despite both iterating the
        same slot-index range), bit-exact with agents.ts's own
        resetHeading() — not just a *plausible* replay the way this used
        to be (numpy Generator vs TS mulberry32, an accepted gap this
        project carried for a while: for THIS specific field it never
        actually mattered in practice, since every slot's own pre-filled
        heading here gets immediately overwritten either by
        training_sim.py's own set_headings() call right after (slots 0/1)
        or by growth itself copying from its own parent's live heading
        the moment a slot is actually claimed (core/agents.wgsl's own
        agentStep()) — but leaving that as a standing "doesn't matter
        today" caveat was fragile, so it's bit-exact now too, same as
        everything else this rollout's starting condition depends on).
        angularVelocity stays zeroed regardless — a random *turn rate*
        would just be an initial spin, not a meaningfully different
        starting condition the way a random facing direction is. cooldown
        is zeroed too — "not on cooldown," so a fresh rollout's own
        starting particle can split as soon as its own hazard threshold
        allow, same as before cooldown existed.

        rng is seeded via `seed` through _growth_seed() instead (a
        deliberately SEPARATE hash domain from heading's own
        _spawn_uniform01_batch() above, despite both being bit-exact now
        — see _growth_seed()'s own comment for why the two are never
        meant to correlate): growth is a near-critical branching process
        (agentStep()'s own split-decision logic), so even a merely-
        correlated seed stream (as opposed to a genuinely independent
        one) risks a systematic bias in which particles tend to split
        together."""
        count = (self._agent_state_buffer.size - PARTICLE_META_BUFFER_OFFSET) // self._particle_meta_dtype.itemsize
        # One combined structured array, matching core/agents.wgsl's own
        # ParticleMeta struct exactly (see this class's own __init__
        # comment for why the state fields are packed into one buffer).
        particle_meta = np.zeros(count, dtype=self._particle_meta_dtype)
        particle_meta["rng"] = _growth_seed(seed, count)
        particle_meta["heading"] = (
            _spawn_uniform01_batch(seed, np.arange(count, dtype=np.uint32) + np.uint32(5)) * (2.0 * np.pi) - np.pi
        ).astype(np.float32)
        # cooldown/angularVelocity are already 0.0 from np.zeros — "not on
        # cooldown," "no spin."
        self.device.queue.write_buffer(self._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, particle_meta.tobytes())

    def set_headings(self, headings: np.ndarray) -> None:
        """Overwrites the FIRST len(headings) heading fields directly, a
        small follow-up write on top of whatever reset_heading() above
        just wrote there (every slot, independently randomized) — for
        callers that need a handful of slots' own heading coordinated
        with each other instead of independent (currently: training_sim.py's
        own TrainingRollout, hardcoded 2-particle "back to back" start).
        Not folded into reset_heading() itself, which stays a general,
        per-slot-independent utility. Written as individual per-index
        strided byte writes (heading is one f32 field inside
        ParticleMeta's own 48-byte stride, not a standalone tightly-
        packed array anymore) rather than one bulk write, to touch ONLY
        the heading field — leaving rng/cooldown/angularVelocity exactly
        as reset_heading() just set them, not overwritten with zeros.
        Only ever called with a couple of headings in practice, so the
        extra per-index write_buffer() calls cost nothing that matters."""
        heading_offset = self._particle_meta_dtype.fields["heading"][1]
        stride = self._particle_meta_dtype.itemsize
        headings32 = headings.astype(np.float32)
        for i, h in enumerate(headings32):
            self.device.queue.write_buffer(
                self._agent_state_buffer,
                PARTICLE_META_BUFFER_OFFSET + i * stride + heading_offset,
                np.array([h], dtype=np.float32),
            )

    def encode_step(self, encoder: wgpu.GPUCommandEncoder, parity: int, *, commit_lifecycle: bool = True) -> None:
        """Encodes the NN forward pass — reads environment's current
        parity buffer (must match `parity`), writes the policy's growth
        direction into MpmCore's particle-rest buffer, optionally applies
        that signal to velocity through maxStrafe, and writes env_write
        into the environment's deposit scratch. Does not submit."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._pipeline)
        groups = self._commit_bind_groups if commit_lifecycle else self._communication_bind_groups
        p.set_bind_group(0, groups[parity])
        p.dispatch_workgroups(self._dispatch)
        p.end()

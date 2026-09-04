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
    BOUNDARY_TANGENT_MIN_GRADIENT,
    CHEMICAL_GRADIENT_INPUT_SCALE,
    CHEMICAL_VALUE_INPUT_MULTIPLIER,
    DIRECTION_CONFIDENCE_SCALE,
    DIVISION_DRIVE_BOOST,
    DIVISION_DIRECTIONALITY,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    ELASTIC_STRAIN_SCALE,
    GROWTH_ANISOTROPY_RESPONSE_RATE,
    GROWTH_COMPRESSION_FEEDBACK,
    GROWTH_COMPRESSION_START,
    GROWTH_COMPRESSION_STOP,
    GROWTH_DIRECTION_RESPONSE_RATE,
    INTERNAL_STATE_SPEED,
    MORPHOLOGY_GRADIENT_INPUT_SCALE,
)

from environment_gpu import EnvironmentGPU, ceil_div
from density import SPATIAL_RANDOM_CELLS
from mpm_core import MpmCore, REPULSION_FIELD_N
from shader_template import load_core_shader
from policy_parameters import (
    CELL_OWNED_PROJECTION_ARCHITECTURE,
    PRIVATE_STATE_DIM,
    STATEFUL_ARCHITECTURE,
    STATELESS_ARCHITECTURE,
    normalize_architecture,
    normalize_chemical_communication_architecture,
    policy_heads,
    policy_has_recurrence,
    policy_input_dim,
)

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
_SPATIAL_HEADING_DOMAIN = np.uint32(0x48454144)


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


def _spatial_uniform01_batch(seed: int, positions: np.ndarray, domain: np.uint32) -> np.ndarray:
    """Sample a fixed world-space random field, independent of slot and q."""
    wrapped = np.mod(np.asarray(positions, dtype=np.float64), 1.0)
    cells = np.floor(wrapped * SPATIAL_RANDOM_CELLS).astype(np.uint32)
    combined = (
        np.uint32(seed)
        ^ _hash_u32(cells[:, 0] + np.uint32(0x9E3779B9))
        ^ _hash_u32(cells[:, 1] + np.uint32(0x85EBCA6B))
        ^ domain
    )
    hashed = _hash_u32(combined)
    return (hashed >> np.uint32(8)).astype(np.float64) / 16777216.0


def weight_layout(
    channels: int, hidden_dim: int, architecture: str = STATELESS_ARCHITECTURE
) -> dict[str, int]:
    # core/agents.wgsl's own IN_DIM: value + heading-forward gradient +
    # lateral gradient per channel, with no positional inputs.
    architecture = normalize_architecture(architecture)
    in_dim = policy_input_dim(channels, architecture)
    # Chemical expression targets + heading(2), anisotropy/division bias(2),
    # growth direction(2), division drive(1), then architecture-specific tail.
    out_dim = sum(head.size for head in policy_heads(channels, architecture))
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
        policy_architecture: str = STATELESS_ARCHITECTURE,
        internal_state_speed: float = INTERNAL_STATE_SPEED,
        division_directionality: float = DIVISION_DIRECTIONALITY,
        chemical_communication_architecture: str = CELL_OWNED_PROJECTION_ARCHITECTURE,
        growth_compression_start: float = GROWTH_COMPRESSION_START,
        growth_compression_stop: float = GROWTH_COMPRESSION_STOP,
        growth_compression_feedback: float = GROWTH_COMPRESSION_FEEDBACK,
        division_drive_boost: float = DIVISION_DRIVE_BOOST,
    ) -> None:
        self.device = device
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.policy_architecture = normalize_architecture(policy_architecture)
        self.chemical_communication_architecture = normalize_chemical_communication_architecture(
            chemical_communication_architecture
        )
        self._internal_state_speed = max(0.0, float(internal_state_speed))
        self._division_directionality = max(0.0, min(1.0, float(division_directionality)))
        self._particle_capacity = max(1, int(max_active_particles))
        self._max_active_particles = self._particle_capacity
        # Public rollout geometry setting: TrainingRollout uses the same
        # displacement configured on this agent instance for its coordinated
        # two-particle seed, keeping diagnostic/replay overrides consistent.
        self.split_displacement = float(split_displacement)

        layout = weight_layout(channels, hidden_dim, self.policy_architecture)
        self._total_floats = layout["total_floats"]
        self._weights_buffer = device.create_buffer(
            size=layout["total_floats"] * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        # 128 bytes — the original layout plus runtime neural-input controls
        # and trailing uniform-alignment padding.
        # NOT written by set_physics() below; see set_spawn_center()'s own
        # docstring for why those get a separate setter into this same
        # buffer instead.
        self._physics_uniform = device.create_buffer(size=128, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
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
        self.set_chemical_gradient_input_scale(CHEMICAL_GRADIENT_INPUT_SCALE)
        self.set_chemical_value_input_multiplier(CHEMICAL_VALUE_INPUT_MULTIPLIER)
        self.set_division_drive_boost(division_drive_boost)
        self.set_chemical_projection_weight(1.0)
        self.set_rollout_seed(0)
        self.set_boundary_tangent_min_gradient(BOUNDARY_TANGENT_MIN_GRADIENT)
        self.set_growth_compression_feedback(
            growth_compression_start,
            growth_compression_stop,
            growth_compression_feedback,
        )
        # Lab-only override is disabled during training. This trailing ABI
        # slot is shared with the browser's scheduled-scenario support.
        self.device.queue.write_buffer(
            self._physics_uniform, 80, np.array([0xFFFFFFFF], dtype=np.uint32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 84, np.array([0], dtype=np.uint32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 88, np.array([1.0, 0.0], dtype=np.float32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 96, np.array([0xFFFFFFFF], dtype=np.uint32)
        )

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
        # packed into one aligned per-particle buffer (112 bytes at C=8)
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
        chemical_padding_floats = (-(76 + channels * 4)) % 16 // 4
        self._particle_meta_dtype = np.dtype([
            ("rng", "<u4"),
            ("cooldown", "<f4"),
            ("heading", "<f4"),
            ("angularVelocity", "<f4"),
            ("color", "<f4", (4,)),
            ("divisionHazard", "<f4"),
            ("divisionThreshold", "<f4"),
            ("mitosisPropensity", "<f4"),
            ("privateState", "<f4", (PRIVATE_STATE_DIM,)),
            ("chemicalState", "<f4", (channels,)),
            ("_padding", "<f4", (chemical_padding_floats,)),
        ])
        # AgentState packs the growth counter at byte 0 and ParticleMeta at
        # byte 256. Besides satisfying storage-offset alignment for the
        # viewer's render binding, this frees one agent shader binding for C.
        self._agent_state_buffer = device.create_buffer(
            size=PARTICLE_META_BUFFER_OFFSET + self._particle_capacity * self._particle_meta_dtype.itemsize,
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

        filterable_morphology = wgpu.FeatureName.float32_filterable in device.features
        morphology_sampler = (
            device.create_sampler(
                address_mode_u=wgpu.AddressMode.repeat,
                address_mode_v=wgpu.AddressMode.repeat,
                min_filter=wgpu.FilterMode.linear,
                mag_filter=wgpu.FilterMode.linear,
            )
            if filterable_morphology else None
        )
        module = device.create_shader_module(
            code=load_core_shader(
                "agents.wgsl",
                {
                    "CHANNELS": channels,
                    "HIDDEN_DIM": hidden_dim,
                    "IN_DIM": layout["in_dim"],
                    "OUT_DIM": layout["out_dim"],
                    **environment.shader_constants,
                    "MORPHOLOGY_FIELD_N": REPULSION_FIELD_N,
                    "SPATIAL_RANDOM_CELLS": SPATIAL_RANDOM_CELLS,
                    "MORPHOLOGY_GRADIENT_INPUT_SCALE": repr(MORPHOLOGY_GRADIENT_INPUT_SCALE),
                    "GROWTH_DIRECTION_RESPONSE_RATE": repr(GROWTH_DIRECTION_RESPONSE_RATE),
                    "GROWTH_ANISOTROPY_RESPONSE_RATE": repr(GROWTH_ANISOTROPY_RESPONSE_RATE),
                    "DIRECTION_CONFIDENCE_SCALE": repr(DIRECTION_CONFIDENCE_SCALE),
                    # WGSL wants lowercase `true`/`false` — Python's own
                    # str(bool) gives "True"/"False", invalid WGSL syntax,
                    # so this can't just be passed through as-is.
                    "CHIRALITY": "true" if chirality else "false",
                    "STATEFUL": "true" if policy_has_recurrence(self.policy_architecture) else "false",
                    "CELL_OWNED_CHEMISTRY": (
                        "true" if self.chemical_communication_architecture == CELL_OWNED_PROJECTION_ARCHITECTURE else "false"
                    ),
                    "PRIVATE_STATE_INPUTS": (
                        "for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {\n"
                        "    inputVec[3u * CHANNELS + 6u + s] = tanh(agentState.particleMeta[pi].privateState[s]);\n"
                        "  }"
                        if policy_has_recurrence(self.policy_architecture) else ""
                    ),
                    "POLICY_TAIL_DECODE": (
                        "out.color = vec3<f32>(0.5);\n"
                        "  for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {\n"
                        "    out.stateDelta[s] = safeTanh(outVec[ENV_WRITE_DIM + 7u + s]);\n"
                        "    out.stateGate[s] = safeSigmoid(outVec[ENV_WRITE_DIM + 7u + PRIVATE_STATE_DIM + s]);\n"
                        "  }"
                        if policy_has_recurrence(self.policy_architecture) else
                        "out.color = vec3<f32>(\n"
                        "    safeSigmoid(outVec[ENV_WRITE_DIM + 7u]),\n"
                        "    safeSigmoid(outVec[ENV_WRITE_DIM + 8u]),\n"
                        "    safeSigmoid(outVec[ENV_WRITE_DIM + 9u])\n"
                        "  );\n"
                        "  for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {\n"
                        "    out.stateDelta[s] = 0.0; out.stateGate[s] = 0.0;\n"
                        "  }"
                    ),
                    "ELASTIC_STRAIN_INPUTS_ENABLED": "true" if elastic_strain_inputs_enabled else "false",
                    "MORPHOLOGY_SAMPLER_DECLARATION": (
                        "@group(0) @binding(14) var morphologySampler: sampler;"
                        if filterable_morphology else ""
                    ),
                    "MORPHOLOGY_SAMPLE_BODY": (
                        "return textureSampleLevel(morphologyTexture, morphologySampler, "
                        "(p + vec2<f32>(0.5)) / f32(MORPHOLOGY_FIELD_N), 0.0).x;"
                        if filterable_morphology else
                        "let base = vec2<i32>(floor(p)); let f = fract(p); "
                        "let a = mix(morphologyLoad(base), morphologyLoad(base + vec2<i32>(1, 0)), f.x); "
                        "let b = mix(morphologyLoad(base + vec2<i32>(0, 1)), morphologyLoad(base + vec2<i32>(1, 1)), f.x); "
                        "return mix(a, b, f.y);"
                    ),
                },
            )
        )
        self._pipeline = device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "agentStep"})
        self._splat_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "splatChemicalState"}
        )
        self._splat_bind_group = device.create_bind_group(
            layout=self._splat_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 1, "resource": {"buffer": core.positions, "offset": 0, "size": core.positions.size}},
                {"binding": 2, "resource": {"buffer": core.active_count_uniform, "offset": 0, "size": core.active_count_uniform.size}},
                {"binding": 5, "resource": {"buffer": environment.deposit_scratch, "offset": 0, "size": environment.deposit_scratch.size}},
                {"binding": 6, "resource": {"buffer": self._physics_uniform, "offset": 0, "size": self._physics_uniform.size}},
                {"binding": 7, "resource": {"buffer": self._agent_state_buffer, "offset": 0, "size": self._agent_state_buffer.size}},
                {"binding": 11, "resource": {"buffer": core.rest, "offset": 0, "size": core.rest.size}},
            ],
        )

        self._step_mode_uniforms = [
            device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
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
                    # state at split time rather than starting undeformed
                    # with zero rest history). core.C is binding 8, completing APIC-state
                    # inheritance across division.
                    {"binding": 10, "resource": {"buffer": core.F, "offset": 0, "size": core.F.size}},
                    {"binding": 11, "resource": {"buffer": core.rest, "offset": 0, "size": core.rest.size}},
                    {"binding": 12, "resource": core.morphology_texture.create_view()},
                    {"binding": 13, "resource": {"buffer": self._step_mode_uniforms[commit_lifecycle]}},
                    *([{"binding": 14, "resource": morphology_sampler}] if morphology_sampler is not None else []),
                ],
            )

        self._communication_bind_groups = [bind_group(0, 0), bind_group(1, 0)]
        self._commit_bind_groups = [bind_group(0, 1), bind_group(1, 1)]

    @property
    def max_active_particles(self) -> int:
        """Current rollout growth cap, bounded by ``particle_capacity``."""
        return self._max_active_particles

    @property
    def particle_capacity(self) -> int:
        """Construction-time slot capacity shared by every density rollout."""
        return self._particle_capacity

    def load_weights(self, flat_weights: np.ndarray) -> None:
        """`flat_weights` is a flat (total_floats,) float array already
        laid out fc1w/fc1b/fc2w/fc2b, row-major within each — exactly
        UpdateRule.flat_parameters()'s own output shape. The Python policy
        has separate semantic output heads, but concatenates their weights
        and then biases into the same fc2 row order agents.wgsl's
        FC1W_OFFSET/FC1B_OFFSET/FC2W_OFFSET/FC2B_OFFSET indexing expects.
        evolve.py's get_weights()/mutate() already produce exactly this
        representation, so this is a straight write_buffer, no restructuring
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
        self.split_displacement = float(split_displacement)
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
            data = np.zeros(8, dtype=np.uint32)
            data[0] = commit
            data.view(np.float32)[1] = max(0.0, float(dt))
            data.view(np.float32)[2] = self._internal_state_speed
            data.view(np.float32)[3] = self._division_directionality
            self.device.queue.write_buffer(buffer, 0, data)

    def set_internal_state_speed(self, speed: float) -> None:
        """Scale only gated private-state residuals; 1 preserves baseline."""
        self._internal_state_speed = max(0.0, float(speed))
        for buffer in self._step_mode_uniforms:
            self.device.queue.write_buffer(
                buffer, 8, np.array([self._internal_state_speed], dtype=np.float32)
            )

    def set_division_directionality(self, strength: float) -> None:
        """Cap policy-polarized daughter placement; 0 is symmetric."""
        self._division_directionality = max(0.0, min(1.0, float(strength)))
        for buffer in self._step_mode_uniforms:
            self.device.queue.write_buffer(
                buffer, 12, np.array([self._division_directionality], dtype=np.float32)
            )

    def set_spawn_center(self, spawn_x: float, spawn_y: float) -> None:
        """Writes the legacy AgentPhysics spawn slots at byte offset 48.

        Spawn coordinates still configure rollout initialization, but the
        policy no longer reads them. The slots remain to preserve the uniform
        ABI while old and new frontend/backend processes overlap.
        """
        self.device.queue.write_buffer(self._physics_uniform, 48, np.array([spawn_x, spawn_y], dtype=np.float32))

    def set_max_active_particles(self, max_active_particles: int) -> None:
        """Writes AgentPhysics.maxActiveParticles at byte offset 56."""
        cap = max(1, int(max_active_particles))
        if cap > self._particle_capacity:
            raise ValueError(f"particle cap {cap} exceeds allocated capacity {self._particle_capacity}")
        self._max_active_particles = cap
        self.device.queue.write_buffer(
            self._physics_uniform,
            56,
            np.array([cap], dtype=np.uint32),
        )

    def set_density_geometry(self, split_displacement: float, deposit_sigma: float) -> None:
        """Update the two particle-scale lengths without disturbing other physics."""
        if split_displacement <= 0.0 or deposit_sigma <= 0.0:
            raise ValueError("density geometry lengths must be positive")
        self.split_displacement = float(split_displacement)
        self.device.queue.write_buffer(
            self._physics_uniform, 28, np.array([split_displacement], dtype=np.float32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 40, np.array([deposit_sigma], dtype=np.float32)
        )

    def set_elastic_strain_scale(self, scale: float) -> None:
        """Writes AgentPhysics.elasticStrainScale at byte offset 60."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            60,
            np.array([max(float(scale), 1e-6)], dtype=np.float32),
        )

    def set_chemical_gradient_input_scale(self, scale: float) -> None:
        """Writes density-resolved AgentPhysics scale at byte offset 64."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            64,
            np.array([max(float(scale), 1e-6)], dtype=np.float32),
        )

    def set_chemical_value_input_multiplier(self, multiplier: float) -> None:
        """Write the live chemical-concentration neural gain at byte offset 112."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            112,
            np.array([max(float(multiplier), 0.0)], dtype=np.float32),
        )

    def set_division_drive_boost(self, boost: float) -> None:
        """Blend signed division drive toward [-1,1] -> [0,1] at byte 116."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            116,
            np.array([max(0.0, min(1.0, float(boost)))], dtype=np.float32),
        )

    def set_chemical_projection_weight(self, weight: float) -> None:
        """Writes represented chemical area at AgentPhysics byte offset 68."""
        if weight <= 0.0:
            raise ValueError("chemical projection weight must be positive")
        self.device.queue.write_buffer(
            self._physics_uniform,
            68,
            np.array([float(weight)], dtype=np.float32),
        )

    def set_rollout_seed(self, seed: int) -> None:
        """Write the common spatial-random-field seed at byte offset 72."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            72,
            np.array([int(seed) & 0xFFFFFFFF], dtype=np.uint32),
        )

    def set_boundary_tangent_min_gradient(self, threshold: float) -> None:
        """Write the flat-interior cutoff at AgentPhysics byte offset 76."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            76,
            np.array([max(0.0, float(threshold))], dtype=np.float32),
        )

    def set_growth_compression_feedback(
        self, start: float, stop: float, strength: float
    ) -> None:
        """Write contact-inhibition controls into trailing AgentPhysics ABI slots."""
        if start < 0.0 or stop < start:
            raise ValueError("growth compression thresholds require 0 <= start <= stop")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("growth compression feedback must be in [0, 1]")
        self.device.queue.write_buffer(
            self._physics_uniform,
            100,
            np.array([start, stop, strength], dtype=np.float32),
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

    def reset_heading(self, seed: int, initial_positions: np.ndarray | None = None) -> None:
        """Randomizes persistent heading state (uniform over [-pi, pi],
        one independent draw per particle slot), initializes the persistent
        world target to that same angle, and zeroes cooldown, EVERY slot up to max_active_particles
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
        existing "resetHeading also resets heading control state" precedent for
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
        The legacy angularVelocity field is not a turn rate anymore; it stores
        the world target and therefore starts equal to heading. cooldown
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
        # This field is a lineage-generation counter in density model v3.
        # Threshold and fallback-angle randomness comes from a common spatial
        # field in WGSL, so numerical particle slot identity never enters it.
        particle_meta["rng"] = 0
        particle_meta["heading"] = (
            _spawn_uniform01_batch(seed, np.arange(count, dtype=np.uint32) + np.uint32(5)) * (2.0 * np.pi) - np.pi
        ).astype(np.float32)
        if initial_positions is not None:
            initial_positions = np.asarray(initial_positions, dtype=np.float32)
            n = min(len(initial_positions), count)
            particle_meta["heading"][:n] = (
                _spatial_uniform01_batch(seed, initial_positions[:n], _SPATIAL_HEADING_DOMAIN)
                * (2.0 * np.pi) - np.pi
            ).astype(np.float32)
        # The legacy angularVelocity lane now stores the persistent world-space
        # heading target. Starting it at heading makes the initial controller
        # error exactly zero. cooldown remains zero ("not on cooldown").
        particle_meta["angularVelocity"] = particle_meta["heading"]
        self.device.queue.write_buffer(self._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, particle_meta.tobytes())
        self.set_rollout_seed(seed)

    def set_headings(self, headings: np.ndarray) -> None:
        """Overwrites the FIRST len(headings) heading fields directly, a
        small follow-up write on top of whatever reset_heading() above
        just wrote there (every slot, independently randomized) — for
        callers that need a handful of slots' own heading coordinated
        with each other instead of independent (currently: training_sim.py's
        own TrainingRollout, hardcoded 2-particle "back to back" start).
        Not folded into reset_heading() itself, which stays a general,
        per-slot-independent utility. Written as individual per-index
        strided byte writes (heading and its target are f32 fields inside
        ParticleMeta's own aligned stride, not a standalone tightly-
        packed array anymore) rather than one bulk write, to touch ONLY
        those two fields while leaving rng/cooldown unchanged.
        Only ever called with a couple of headings in practice, so the
        extra per-index write_buffer() calls cost nothing that matters."""
        heading_offset = self._particle_meta_dtype.fields["heading"][1]
        target_offset = self._particle_meta_dtype.fields["angularVelocity"][1]
        stride = self._particle_meta_dtype.itemsize
        headings32 = headings.astype(np.float32)
        for i, h in enumerate(headings32):
            self.device.queue.write_buffer(
                self._agent_state_buffer,
                PARTICLE_META_BUFFER_OFFSET + i * stride + heading_offset,
                np.array([h], dtype=np.float32),
            )
            self.device.queue.write_buffer(
                self._agent_state_buffer,
                PARTICLE_META_BUFFER_OFFSET + i * stride + target_offset,
                np.array([h], dtype=np.float32),
            )

    def encode_step(self, encoder: wgpu.GPUCommandEncoder, parity: int, *, commit_lifecycle: bool = True) -> None:
        """Encodes the NN forward pass — reads environment's current
        parity buffer (must match `parity`), writes the policy's growth
        direction into MpmCore's particle-rest buffer, optionally applies
        that signal to velocity through maxStrafe, and integrates chemical
        deltas into cell-owned state. Does not submit."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._pipeline)
        groups = self._commit_bind_groups if commit_lifecycle else self._communication_bind_groups
        p.set_bind_group(0, groups[parity])
        p.dispatch_workgroups(self._dispatch)
        p.end()

    def encode_splat_chemical_state(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Publish persistent cell chemistry into the cleared transient field."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._splat_pipeline)
        p.set_bind_group(0, self._splat_bind_group)
        p.dispatch_workgroups(self._dispatch)
        p.end()

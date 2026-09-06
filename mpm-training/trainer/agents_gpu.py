"""GPU policy and adaptive refinement pipelines shared with the browser through core WGSL. Particle metadata contains color, alignment, growth magnitude, private state, and chemistry."""
from __future__ import annotations

from config import CONFIG
import numpy as np
import wgpu

from simulation_settings import (
    CHEMICAL_GRADIENT_INPUT_SCALE,
    CHEMICAL_VALUE_INPUT_MULTIPLIER,
    ELASTIC_STRAIN_INPUTS_ENABLED,
    ELASTIC_STRAIN_SCALE,
    INTERNAL_STATE_SPEED,
    MORPHOLOGY_GRADIENT_INPUT_SCALE,
)

from environment_gpu import EnvironmentGPU, ceil_div
from mpm_core import GROWTH_FIELD_CHANNELS, GRID_N, INV_DX, NODE_COUNT, MpmCore, REPULSION_FIELD_N
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

_SPAWN_HASH_DOMAIN = np.uint32(0xC0FFEE00)

def _spawn_uniform01(seed: int, index: int) -> float:
    combined = np.uint32(seed) ^ _hash_u32(np.array([_SPAWN_HASH_DOMAIN ^ np.uint32(index)]))[0]
    hashed = _hash_u32(np.array([combined]))[0]
    return float(hashed >> np.uint32(8)) / 16777216.0

def weight_layout(
    channels: int, hidden_dim: int, architecture: str = CONFIG["run"]["policyArchitecture"]
) -> dict[str, int]:
    # core/agents.wgsl's own IN_DIM: value + density-frame forward/lateral
    # gradients per channel, with no positional inputs.
    architecture = normalize_architecture(architecture)
    in_dim = policy_input_dim(channels, architecture)
    # Chemical deltas + local 2-D growth vector, then architecture-specific tail.
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
        max_env_write: float,
        max_active_particles: int,
        sample_spacing: float,
        friction: float,
        growth_enabled: float,
        spawn_x: float,
        spawn_y: float,
        elastic_strain_scale: float = ELASTIC_STRAIN_SCALE,
        elastic_strain_inputs_enabled: bool = ELASTIC_STRAIN_INPUTS_ENABLED,
        policy_architecture: str = CONFIG["run"]["policyArchitecture"],
        internal_state_speed: float = INTERNAL_STATE_SPEED,
        chemical_communication_architecture: str = CONFIG["chemistry"]["chemicalCommunicationArchitecture"],
    ) -> None:
        self.unresolved_samples = 0
        self.capacity_blocked = False
        self.device = device
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.policy_architecture = normalize_architecture(policy_architecture)
        self.chemical_communication_architecture = normalize_chemical_communication_architecture(
            chemical_communication_architecture
        )
        self._internal_state_speed = max(0.0, float(internal_state_speed))
        self._particle_capacity = max(1, int(max_active_particles))
        self._max_active_particles = self._particle_capacity
        # Public rollout geometry setting: TrainingRollout uses the same
        # displacement configured on this agent instance for its coordinated
        # two-particle seed, keeping diagnostic/replay overrides consistent.
        self.sample_spacing = float(sample_spacing)

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
        self._physics_uniform = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_physics(
            max_env_write,
            sample_spacing,
            friction,
            growth_enabled,
        )
        self.set_spawn_center(spawn_x, spawn_y)
        self.set_max_active_particles(max_active_particles)
        self.set_elastic_strain_scale(elastic_strain_scale)
        self.set_chemical_gradient_input_scale(CHEMICAL_GRADIENT_INPUT_SCALE)
        self.set_chemical_value_input_multiplier(CHEMICAL_VALUE_INPUT_MULTIPLIER)


        # Lab-only override is disabled during training. This trailing ABI
        # slot is shared with the browser's scheduled-scenario support.
        self.device.queue.write_buffer(
            self._physics_uniform, 36, np.array([0xFFFFFFFF], dtype=np.uint32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 40, np.array([0], dtype=np.uint32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 48, np.array([1.0, 0.0], dtype=np.float32)
        )
        self.device.queue.write_buffer(
            self._physics_uniform, 56, np.array([0xFFFFFFFF], dtype=np.uint32)
        )
        self._forced_growth_field_override = False
        self.set_forced_growth_field_override(False)

        chemical_padding_floats = (-(60 + channels * 4)) % 16 // 4
        self._particle_meta_dtype = np.dtype([
            ("color", "<f4", (4,)),
            ("alignment", "<f4", (2,)),
            ("growthMagnitude", "<f4"),
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
                    "MORPHOLOGY_GRADIENT_INPUT_SCALE": repr(MORPHOLOGY_GRADIENT_INPUT_SCALE),
                    # WGSL wants lowercase `true`/`false` — Python's own
                    # str(bool) gives "True"/"False", invalid WGSL syntax,
                    # so this can't just be passed through as-is.
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
                        "out.color = vec3<f32>(safeSigmoid(outVec[ENV_WRITE_DIM + 18u]), safeSigmoid(outVec[ENV_WRITE_DIM + 19u]), safeSigmoid(outVec[ENV_WRITE_DIM + 20u]));\n"
                        "  for (var s: u32 = 0u; s < PRIVATE_STATE_DIM; s = s + 1u) {\n"
                            "    out.stateDelta[s] = safeTanh(outVec[ENV_WRITE_DIM + 2u + s]);\n"
                            "    out.stateGate[s] = safeSigmoid(outVec[ENV_WRITE_DIM + 2u + PRIVATE_STATE_DIM + s]);\n"
                        "  }"
                        if policy_has_recurrence(self.policy_architecture) else
                        "out.color = vec3<f32>(\n"
                            "    safeSigmoid(outVec[ENV_WRITE_DIM + 2u]),\n"
                            "    safeSigmoid(outVec[ENV_WRITE_DIM + 3u]),\n"
                            "    safeSigmoid(outVec[ENV_WRITE_DIM + 4u])\n"
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
            device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
        ]
        self.set_communication_timestep(1.0)

        def bind_group(p: int, commit_growth: int):
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
                    {"binding": 9, "resource": {"buffer": core.velocities, "offset": 0, "size": core.velocities.size}},
                    # core.F/core.rest — same buffers core/p2g.wgsl and
                    # core/g2p.wgsl already read/write every physics substep
                    # — see core/agents.wgsl's own module docstring for why
                    # agentStep() now needs them too (a freshly-claimed
                    # particle inherits its parent's CURRENT deformation
                    # state at split time rather than starting undeformed
                    # with zero rest history). core.C is binding 8, completing APIC-state
                    # inheritance across refinement.
                    {"binding": 10, "resource": {"buffer": core.F, "offset": 0, "size": core.F.size}},
                    {"binding": 11, "resource": {"buffer": core.rest, "offset": 0, "size": core.rest.size}},
                    {"binding": 12, "resource": core.morphology_texture.create_view()},
                    {"binding": 13, "resource": {"buffer": self._step_mode_uniforms[commit_growth]}},
                    *([{"binding": 14, "resource": morphology_sampler}] if morphology_sampler is not None else []),
                ],
            )

        self._communication_bind_groups = [bind_group(0, 0), bind_group(1, 0)]
        self._commit_bind_groups = [bind_group(0, 1), bind_group(1, 1)]

        # Growth lives on the MPM grid. MpmCore owns this shared buffer because
        # g2p consumes it directly; Agents only populates it from NN vectors.
        self._growth_field = core.growth_field
        refine_capacity = self._particle_capacity
        refine_hash_size = 1 << (6 * refine_capacity - 1).bit_length()
        refine_words = refine_hash_size + 5 * refine_capacity + 1
        self._refinement = device.create_buffer(
            label="conforming refinement scratch", size=4 * refine_words,
            usage=wgpu.BufferUsage.STORAGE,
        )
        growth_module = device.create_shader_module(code=load_core_shader(
            "growthField.wgsl",
            {
                "CHANNELS": channels,
                "GRID_N": GRID_N,
                "INV_DX": INV_DX,
                "MORPHOLOGY_FIELD_N": REPULSION_FIELD_N,
                "REFINE_CAPACITY": refine_capacity,
                "REFINE_HASH_SIZE": refine_hash_size,
            },
        ))
        # Every stage ends a compute pass, providing a device-wide barrier.
        stages = [
            ("clearGrowthField", (3, 8), ceil_div(GROWTH_FIELD_CHANNELS * NODE_COUNT, 256)),
            ("clearRefinement", (7,), ceil_div(refine_words, 256)),
            ("indexRefinementEdges", (1, 2, 7), None),
            ("scatterGrowthIntent", (1, 2, 8), None),
            ("scatterGrowthBoundary", (1, 2, 7, 8), None),
            ("enforceGrowthField", (8, 9), ceil_div(NODE_COUNT, 256)),
            ("linkRefinementEdges", (1, 2, 7), None),
            *[("propagateRefinement", (1, 7), None)] * (refine_capacity - 1).bit_length(),
            ("requestRefinement", (1, 2, 3, 7, 9), None),
            ("reserveRefinement", (1, 2, 3, 7, 9), None),
            ("commitResample", (0, 1, 2, 3, 4, 5, 6, 7), None),
            ("stopGrowthAtCapacity", (3, 7, 8, 9), ceil_div(GROWTH_FIELD_CHANNELS * NODE_COUNT, 256)),
        ]
        pipelines = {entry: device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={"module": growth_module, "entry_point": entry},
        ) for entry in dict.fromkeys(stage[0] for stage in stages)}
        self._growth_entries = [entry for entry, _, _ in stages]
        self._growth_pipelines = [pipelines[entry] for entry, _, _ in stages]
        resources = {
            0: {"buffer": core.positions, "offset": 0, "size": core.positions.size},
            1: {"buffer": core.active_count_uniform, "offset": 0, "size": core.active_count_uniform.size},
            2: {"buffer": core.rest, "offset": 0, "size": core.rest.size},
            3: {"buffer": self._agent_state_buffer, "offset": 0, "size": self._agent_state_buffer.size},
            4: {"buffer": core.C, "offset": 0, "size": core.C.size},
            5: {"buffer": core.velocities, "offset": 0, "size": core.velocities.size},
            6: {"buffer": core.F, "offset": 0, "size": core.F.size},
            7: {"buffer": self._refinement, "offset": 0, "size": self._refinement.size},
            8: {"buffer": self._growth_field, "offset": 0, "size": self._growth_field.size},
            9: {"buffer": self._physics_uniform, "offset": 0, "size": self._physics_uniform.size},
        }
        groups = {entry: device.create_bind_group(
            layout=pipelines[entry].get_bind_group_layout(0),
            entries=[{"binding": binding, "resource": resources[binding]} for binding in bindings],
        ) for entry, bindings, _ in dict.fromkeys(stages)}
        self._growth_bind_groups = [groups[entry] for entry, _, _ in stages]
        self._growth_dispatches = [dispatch for _, _, dispatch in stages]

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
        max_env_write: float,
        sample_spacing: float,
        friction: float,
        growth_enabled: float,
    ) -> None:
        self.sample_spacing = float(sample_spacing)
        self.device.queue.write_buffer(
            self._physics_uniform,
            0,
            np.array(
                [
                    max_env_write,
                    sample_spacing,
                    friction,
                    growth_enabled,
                ],
                dtype=np.float32,
            ),
        )

    def set_growth_enabled(self, enabled: bool) -> None:
        """Enable publication of neural growth vectors (uniform byte 44).

        Disabling immediately publishes zero intent. Already-created rest
        volume remains, so rollout tails settle mechanically without shrinking.
        """
        self.device.queue.write_buffer(
            self._physics_uniform,
            12,
            np.array([1.0 if enabled else 0.0], dtype=np.float32),
        )

    def set_communication_timestep(self, dt: float) -> None:
        """Set the neural substep clock while retaining two lifecycle modes."""
        for commit, buffer in enumerate(self._step_mode_uniforms):
            data = np.zeros(4, dtype=np.uint32)
            data[0] = commit
            data.view(np.float32)[1] = max(0.0, float(dt))
            data.view(np.float32)[2] = self._internal_state_speed
            self.device.queue.write_buffer(buffer, 0, data)

    def set_internal_state_speed(self, speed: float) -> None:
        """Scale only gated private-state residuals; 1 preserves baseline."""
        self._internal_state_speed = max(0.0, float(speed))
        for buffer in self._step_mode_uniforms:
            self.device.queue.write_buffer(
                buffer, 8, np.array([self._internal_state_speed], dtype=np.float32)
            )

    def set_spawn_center(self, spawn_x: float, spawn_y: float) -> None:
        self.device.queue.write_buffer(self._physics_uniform, 16, np.array([spawn_x, spawn_y], dtype=np.float32))

    def set_max_active_particles(self, max_active_particles: int) -> None:
        """Writes AgentPhysics.maxActiveParticles at byte offset 56."""
        cap = max(1, int(max_active_particles))
        if cap > self._particle_capacity:
            raise ValueError(f"particle cap {cap} exceeds allocated capacity {self._particle_capacity}")
        self._max_active_particles = cap
        self.device.queue.write_buffer(
            self._physics_uniform,
            24,
            np.array([cap], dtype=np.uint32),
        )

    def set_density_geometry(self, sample_spacing: float) -> None:
        if sample_spacing <= 0:
            raise ValueError("sample spacing must be positive")
        self.sample_spacing = float(sample_spacing)
        self.device.queue.write_buffer(self._physics_uniform, 4, np.array([sample_spacing], np.float32))

    def set_elastic_strain_scale(self, scale: float) -> None:
        """Writes AgentPhysics.elasticStrainScale at byte offset 60."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            28,
            np.array([max(float(scale), 1e-6)], dtype=np.float32),
        )

    def set_chemical_gradient_input_scale(self, scale: float) -> None:
        """Writes density-resolved AgentPhysics scale at byte offset 64."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            32,
            np.array([max(float(scale), 1e-6)], dtype=np.float32),
        )

    def set_chemical_value_input_multiplier(self, multiplier: float) -> None:
        """Write the live chemical-concentration neural gain at byte offset 112."""
        self.device.queue.write_buffer(
            self._physics_uniform,
            60,
            np.array([max(float(multiplier), 0.0)], dtype=np.float32),
        )

    def set_forced_growth_field_override(self, enabled: bool) -> None:
        """Toggle the Lab-only radial-inward grid field at byte offset 120."""
        self._forced_growth_field_override = bool(enabled)
        self.device.queue.write_buffer(
            self._physics_uniform,
            64,
            np.array([1 if enabled else 0], dtype=np.uint32),
        )

    def set_active_count(self, active_count: int) -> None:
        """Updates this class's own agentStep() dispatch size AND
        growth's own atomic "next free slot" counter (core/agents.wgsl's
        own module docstring), which always needs to start from the
        current active_count — called once per rollout (training_sim.py's
        own TrainingRollout.__init__) and again every macro step growth
        actually changes the count (that module's own macro_step(), after
        reading read_sample_count() back). Deliberately does NOT touch
        core.active_count_uniform itself — MpmCore.set_active_count() (a
        distinct method, on a distinct object) owns that, since it's
        shared with p2g/gridUpdate-adjacent/g2p/repulsion too, not just
        this class's own dispatch."""
        self._refinement_rounds = max(0, active_count - 1).bit_length()
        self._dispatch = ceil_div(active_count, WORKGROUP)
        self.device.queue.write_buffer(self._agent_state_buffer, 0, np.array([active_count], dtype=np.uint32))

    def read_sample_count(self) -> int:
        """Reads back growth's own atomic counter (core/agents.wgsl's own
        module docstring) — a real, deliberate 12-byte status readback,
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
        raw = self.device.queue.read_buffer(self._agent_state_buffer, 0, 12)
        status = np.frombuffer(raw, dtype=np.uint32)
        self.unresolved_samples = int(status[1])
        self.capacity_blocked = bool(status[2])
        return int(status[0])

    def reset_state(self, chemical_state=None, private_state=None) -> None:
        """Clear rollout-scoped neural state.

        Alignment starts at zero and is reconstructed from chemical channel
        3's gradient by every agentStep; it is never randomized or
        persisted as an independently controlled cell property.
        """
        self.unresolved_samples = 0
        self.capacity_blocked = False
        self.device.queue.write_buffer(self._agent_state_buffer, 4, np.zeros(2, dtype=np.uint32))
        count = (self._agent_state_buffer.size - PARTICLE_META_BUFFER_OFFSET) // self._particle_meta_dtype.itemsize
        # One combined structured array, matching core/agents.wgsl's own
        # ParticleMeta struct exactly (see this class's own __init__
        # comment for why the state fields are packed into one buffer).
        particle_meta = np.zeros(count, dtype=self._particle_meta_dtype)
        if chemical_state is not None:
            particle_meta["chemicalState"][:len(chemical_state)] = chemical_state
        if private_state is not None:
            particle_meta["privateState"][:len(private_state)] = private_state
        self.device.queue.write_buffer(self._agent_state_buffer, PARTICLE_META_BUFFER_OFFSET, particle_meta.tobytes())
        self.device.queue.write_buffer(
            self._growth_field, 0, np.zeros(self._growth_field.size // 4, dtype=np.int32)
        )

    def encode_step(self, encoder: wgpu.GPUCommandEncoder, parity: int, *, commit_growth: bool = True) -> None:
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._pipeline)
        groups = self._commit_bind_groups if commit_growth else self._communication_bind_groups
        p.set_bind_group(0, groups[parity])
        p.dispatch_workgroups(self._dispatch)
        p.end()
        if commit_growth:
            self.encode_growth_field(encoder)

    def encode_growth_field(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Splat growth intent and conservatively refine under-resolved footprints."""
        propagation_round = 0
        for entry, pipeline, bind_group, fixed_dispatch in zip(
            self._growth_entries, self._growth_pipelines, self._growth_bind_groups, self._growth_dispatches
        ):
            if entry == "propagateRefinement":
                propagation_round += 1
                if propagation_round > self._refinement_rounds:
                    continue
            p = encoder.begin_compute_pass()
            p.set_pipeline(pipeline)
            p.set_bind_group(0, bind_group)
            p.dispatch_workgroups(self._dispatch if fixed_dispatch is None else fixed_dispatch)
            p.end()

    def encode_splat_chemical_state(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Publish persistent cell chemistry into the cleared transient field."""
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._splat_pipeline)
        p.set_bind_group(0, self._splat_bind_group)
        p.dispatch_workgroups(self._dispatch)
        p.end()

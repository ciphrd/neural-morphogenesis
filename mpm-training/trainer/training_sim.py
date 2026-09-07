"""GPU rollout orchestration: morphology, neural communication, growth/refinement, and MPM. Production seeds are conforming triangle meshes; material growth and numerical sample count are distinct."""
from __future__ import annotations

import numpy as np
from time import perf_counter
from contextlib import nullcontext

from simulation_settings import DEFAULT_RUN_SETTINGS, COMMUNICATION_SPEED, INITIAL_PARTICLE_COUNT, INITIAL_SPACING, NEURAL_UPDATES_PER_MACRO

from agents_gpu import AgentsGPU
from density import INITIAL_PACKING_SPACING_SCALE
from initial_conditions import InitialCondition, validate_initial_condition
from policy_parameters import policy_has_recurrence
from environment_gpu import EnvironmentGPU
from mpm_core import DT, MpmCore

def seed_blob(count: int, center: tuple[float, float], spacing: float, seed: int) -> tuple:
    """Seed a fixed-orientation circular triangle mesh; mirrored in rng.ts.

    Keep the seed argument for callers; geometry is independent of it.
    """
    from triangle_seed import triangulate_seed_disk
    positions, domain, weights = triangulate_seed_disk(
        count, center, spacing * INITIAL_PACKING_SPACING_SCALE, 0.0)
    samples = len(positions)
    return (positions, np.zeros((samples, 2), np.float32),
            np.tile(np.array([1, 0, 0, 1], np.float32), (samples, 1)),
            np.zeros((samples, 4), np.float32), np.ones(samples, np.float32),
            domain, weights, 'triangle-vertices')

class TrainingRollout:
    """One rollout's worth of *state*. `core`/`agents`/`environment` (all
    their GPU buffers/pipelines) are owned and reused by the caller
    across many rollouts — see evolve.py's own module docstring — since
    rebuilding wgpu pipelines per candidate would be real, avoidable
    overhead; this constructor only seeds `core`'s particle buffers (one
    the configured initial particle count; --particles remains a growth cap) and
    resets `agents`/`environment`'s own persistent state back to a fresh
    (empty field and zero alignment cache — see AgentsGPU.reset_state()'s
    own docstring) starting point for this rollout. The base mesh has a fixed
    orientation; `seed` only affects optional initial-condition perturbations.
    `mpm_enabled` (default True) is simulation_settings.py's own
    MPM_ENABLED, threaded through — see macro_step()'s own comment for
    exactly what setting it False skips."""

    def __init__(
        self,
        core: MpmCore,
        agents: AgentsGPU,
        environment: EnvironmentGPU,
        spawn_center: tuple[float, float],
        gravity: float,
        seed: int,
        mpm_enabled: bool = True,
        neural_updates_per_macro: int = NEURAL_UPDATES_PER_MACRO,
        communication_speed: float = COMMUNICATION_SPEED,
        initial_particle_count: int = INITIAL_PARTICLE_COUNT,
        initial_condition: str = DEFAULT_RUN_SETTINGS["initialCondition"],
        initial_condition_strength: float = DEFAULT_RUN_SETTINGS["initialConditionStrength"],
        initial_condition_channel: int = DEFAULT_RUN_SETTINGS["initialConditionChannel"],
        initial_spacing: float = INITIAL_SPACING,
    ) -> None:
        self.core = core
        self.agents = agents
        self.environment = environment
        # See macro_step()'s own comment for exactly what this skips —
        # simulation_settings.py's own MPM_ENABLED (that constant's own
        # comment has the full "why," including the fitness-scoring
        # caveat: with this off, a shape-matching fitness against a
        # spread-out target becomes close to meaningless).
        self.mpm_enabled = mpm_enabled
        self.neural_updates_per_macro = max(1, int(neural_updates_per_macro))
        self.communication_speed = max(0.0, float(communication_speed))
        communication_dt = environment.set_communication_timestep(
            self.neural_updates_per_macro, self.communication_speed
        )
        agents.set_communication_timestep(communication_dt)

        core.set_gravity(gravity)
        agents.set_spawn_center(*spawn_center)
        if agents.max_active_particles < 2:
            raise ValueError('A tiled triangle seed requires capacity for at least two samples')
        initial_cells = min(agents.max_active_particles // 2, max(1, int(initial_particle_count)))
        if not np.isfinite(initial_spacing) or initial_spacing <= 0:
            raise ValueError("Initial spacing must be finite and positive")
        scene = seed_blob(
            initial_cells, spawn_center, initial_spacing, seed
        )
        validate_initial_condition(initial_condition, initial_condition_strength,
                                   initial_condition_channel, agents.channels,
                                   policy_has_recurrence(agents.policy_architecture))
        radius = np.sqrt(initial_cells*(initial_spacing*INITIAL_PACKING_SPACING_SCALE)**2*np.sqrt(3)/(2*np.pi))
        perturbation = InitialCondition(initial_condition, initial_condition_strength,
                                        initial_condition_channel, seed, spawn_center, radius)
        perturbation.deform(scene)
        initial_count = len(scene[0])
        core.reset_growth_buffers(agents.max_active_particles)
        core.load_scene(*scene)
        # Every slot beyond the genuinely seeded particles is destined to
        # become a real particle via growth, at some unknown point in
        # this rollout — see reset_growth_buffers()'s own docstring for
        # why this has to run every rollout (not just once, ever) despite
        # seed_blob() already giving genuinely-seeded particles these
        # exact same fresh defaults. agents.max_active_particles (not a
        # parameter of this constructor — see AgentsGPU's own property
        # docstring for why) is --particles, the growth cap.

        environment.reset()
        agents.set_active_count(initial_count)
        chemistry, private = perturbation.states(scene[0], agents.channels)
        agents.reset_state(chemistry, private)
        if environment.chemical_communication_architecture == "persistent-environment":
            perturbation.seed_environment(environment)

    def macro_step(self, substeps_per_macro: int, *, growth_enabled: bool = True) -> None:
        core = self.core
        timings = getattr(self, "timings", None)
        stage_started = perf_counter()
        gpu = getattr(self, "gpu_timings", None)
        if gpu is not None:
            gpu.begin_sample()
        def measure(encoder, name):
            return gpu.measure(encoder, name) if gpu is not None else nullcontext()

        # Only gates entry into a new cell cycle. Cycles already underway
        # finish normally, leaving the remaining macro steps for elastic
        # relaxation with no fresh growth events being initiated.
        self.agents.set_growth_enabled(growth_enabled)

        # Sense -> NN forward pass -> deposit/decay, one encoder/submit —
        # matches simulation.ts's own step() ordering exactly. A SEPARATE
        # submit from core.step()'s own physics substeps below (not
        # folded into the same encoder) because MpmCore.step() has its
        # own real, load-bearing constraint here: wgpu-native's Metal
        # backend hits a hard cap on outstanding command buffers past a
        # few thousand compute passes (see that method's own docstring),
        # so it already has to chunk large substep counts into multiple
        # submits with a forced sync between them — folding a 5-pass
        # sense/act/deposit block into that same chunking logic would
        # only complicate it for no benefit.
        encoder = core.device.create_command_encoder()
        core.encode_morphology(encoder)
        self.environment.set_advection_timestep(
            substeps_per_macro * DT if self.mpm_enabled else 0.0
        )
        # Transport the old persistent substrate through the preceding MPM
        # motion before the first neural read of this tick.
        for communication_round in range(self.neural_updates_per_macro):
            final_round = communication_round == self.neural_updates_per_macro - 1
            self.environment.encode_prepare_persistent(encoder, transport=communication_round == 0, gpu_timings=gpu)
            self.environment.encode_clear(encoder, gpu_timings=gpu)
            if self.environment.chemical_communication_architecture == "cell-owned-projection":
                with measure(encoder, "gpuChemistrySplat"):
                    self.agents.encode_splat_chemical_state(encoder)
            self.environment.encode_sense(encoder, gpu_timings=gpu)
            self.agents.encode_step(
                encoder,
                self.environment.parity,
                commit_growth=final_round,
                gpu_timings=gpu,
            )
            self.environment.encode_merge_persistent(encoder, gpu_timings=gpu)
        core.device.queue.submit([encoder.finish()])

        # Growth's own readback — see this module's own module docstring
        # for why this (not "zero host round-trips") is correct: the
        # agentStep() pass just submitted may have grown activeCount on
        # the GPU, and nothing else finds out unless this class reads it
        # back and propagates it before core.step() below sizes ITS OWN
        # dispatches. A plain != check, not unconditional writes, so a
        # macro step where nothing actually split (the overwhelmingly
        # common case early in a rollout, or for a policy that never
        # requests growth) costs one 4-byte read
        # and nothing else.
        # min(...) — the atomic itself can overshoot max_active_particles
        # slightly (several agents claiming a slot the same step, right
        # at the cap — see core/agents.wgsl's own agentStep() comment for
        # why that's left unguarded rather than compare-exchanged away);
        # clamping the *reported* count here is what actually enforces
        # the cap, since core/agents.wgsl itself already refuses to WRITE
        # a claimed slot past max_active_particles either way.
        if timings is not None:
            timings.add("neuralCommands", perf_counter()-stage_started)
        stage_started = perf_counter()
        grown = min(self.agents.read_sample_count(), self.agents.max_active_particles)
        if timings is not None:
            timings.add("growthSync", perf_counter()-stage_started)
        stage_started = perf_counter()
        if grown != core.active_count:
            core.set_active_count(grown)
            self.agents.set_active_count(grown)

        # Skippable via self.mpm_enabled (see __init__'s own comment) —
        # everything above (sense/act/deposit/growth readback) always
        # runs regardless; only the actual elastic-material/gravity/
        # repulsion substeps below are skipped. Positions then never
        # advance except where growth itself wrote a brand-new child's
        # own spawn position (core/agents.wgsl's own agentStep()) —
        # mirrors gpu/simulation.ts's own step(), which has the identical
        # toggle for the frontend's own live replay (that one is view-
        # only; this one, driven by simulation_settings.py's own
        # MPM_ENABLED, is what the actual worker-pool population
        # evaluation runs under too).
        if timings is not None:
            timings.add("growthStatusUpdate", perf_counter()-stage_started)
        if self.mpm_enabled:
            stage_started = perf_counter()
            # The next macro's required growth-status readback (or a terminal
            # fitness readback) retires this final chunk. Queue ordering keeps
            # the following neural pass behind physics without a host stall.
            core.step(substeps_per_macro, wait_for_completion=False)
            if timings is not None:
                timings.add("physics", perf_counter()-stage_started)

        if gpu is not None and gpu.active:
            stage_started = perf_counter()
            gpu.finish_sample()
            if timings is not None:
                timings.add("gpuProfilingReadback", perf_counter()-stage_started)

    def positions(self) -> np.ndarray:
        return self.core.read_positions()

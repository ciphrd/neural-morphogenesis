"""Ties MpmCore (headless MLS-MPM physics) + EnvironmentGPU (GPU-resident
chemical field) + AgentsGPU (the evolved policy's GPU-resident forward
pass) into one rollout — the mpm-training analogue of envnca/simulation.py,
adapted to a real physics substrate instead of a bare point-agent grid.

Fully GPU-resident, matching ../viewer/src/gpu/simulation.ts's own
"GPU-resident is the whole point" design for every data-related buffer
(positions, velocities, F/C/ParticleRest, the transient chemical field, weights): each macro
step rebuilds the field from cell-owned chemistry before every agents.encode_step,
then submits once, immediately followed
by core.step(substeps_per_macro)'s own physics submission — see
macro_step()'s own comment for why that's two submits, not one. The ONE
exception is activeCount itself, now that growth exists (see this
module's own "Growth" paragraph below) — macro_step() reads back a
single 4-byte atomic counter every macro step to learn whether the count
changed, a real, deliberate, budgeted host round-trip, not an oversight.

This is a real architectural change from an earlier revision of this
module, which ran the chemical field + NN forward pass entirely in torch
(on MPS/CUDA) while MpmCore's own physics ran on wgpu-native/Metal — two
separate GPU compute frameworks sharing no buffers, so every handoff
between "what the network senses" and "what MpmCore's physics did"
needed a real, blocking host round-trip: read_positions()/
read_velocities() (wgpu GPU -> CPU), torch.from_numpy(...).to(device)
(CPU -> torch GPU) for positions/heading, several more torch GPU -> CPU
downloads for the network's own outputs, then write_buffer() (CPU ->
wgpu GPU) to push the result back into MpmCore. At realistic training
settings (tens of macro steps x population x generations) this dominated
the actual compute cost. EnvironmentGPU/AgentsGPU are Python (wgpu-py)
ports of the exact same WGSL shaders (../core/environment.wgsl,
../core/agents.wgsl) the browser viewer already runs fully GPU-resident
— reusing them here, on MpmCore's own wgpu device/queue, removes that
crossing entirely rather than trying to paper over it with zero-copy
interop between two unrelated GPU frameworks.

The other host round-trip in this module is positions() — evolve.py's
own rollout() still needs particle positions back on the host for
raster.py's own numpy/scipy fitness scoring (rotation search + a
Euclidean distance transform, neither of which has an obvious WGSL
equivalent worth chasing), but that's called at only the ~5
CAPTURE_OFFSETS snapshots near the end of a rollout, not every macro
step — a fixed, small cost per rollout rather than one paid
`macro_steps` times.

Local-frame (heading-relative) sensing/action rotation, and the
persistent per-particle heading/angularVelocity state that drives it
(NOT derived from velocity — a real, confirmed source of chaotic spin
before this class's own heading state was introduced) now live entirely
inside AgentsGPU/core/agents.wgsl — see that file's own module docstring
for the full reasoning (this module used to own that state as
self.heading/self.angular_velocity numpy arrays, computing the rotation
in torch every macro step; there's nothing left for this module to do
there now).

The two former strafe channels propose a local tensor-growth direction. The
agent shader smoothly turns a persistent heading-relative growth angle toward
that target and relaxes persistent anisotropy toward its sigmoid target; a
separate sigmoid controls signed division placement. MAX_STRAFE independently
controls whether the reconstructed world direction also acts as physical
acceleration and is zero by default.

Growth: every rollout starts with the configured initial particle count;
core/agents.wgsl's own agentStep() may spawn
new ones from there (splitting, based on the last chemical channel's own
sensed value — see that file's own module docstring for the full
design), up to evolve.py's own --particles (now a CAP, not a fixed
starting count — see that module's own module docstring for why; a
policy that never learns to use the growth channel stays at that initial
count). This is the
ONE exception to "zero host round-trips for anything data-related" this
module's own docstring boasts about above: macro_step() reads back a
single 4-byte atomic counter every macro step (agents.read_grown_count())
to learn whether growth changed the count, and if so propagates it to
core/agents' own dispatch sizing before this step's own physics
substeps run — a real, deliberate host round-trip (not an oversight),
needed because dispatch sizing for every pass (P2G/gridUpdate/G2P/
repulsion, and Agents' own next agentStep()) is decided on the CPU, and
nothing else would ever learn growth happened purely on the GPU
otherwise. See agents_gpu.py's own read_grown_count()/set_active_count()
for why this is cheap (mpm_core.py's own step() already pays an
equivalent 4-byte sync once per macro step, for a different reason —
see that method's own docstring) rather than a new, unbudgeted cost
class. A newly split particle only starts getting its own agentStep()/
physics one macro step after it split (this readback+propagate happens
AFTER agentStep() already ran for this step) — see core/agents.wgsl's
own module docstring for why that one-step activation lag was a
deliberate choice, not a limitation worth working around.
"""
from __future__ import annotations

import numpy as np

from simulation_settings import COMMUNICATION_SPEED, INITIAL_PARTICLE_COUNT, NEURAL_UPDATES_PER_MACRO

from agents_gpu import AgentsGPU, _spawn_uniform01
from density import INITIAL_PACKING_SPACING_SCALE
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore


def seed_blob(count: int, center: tuple[float, float], spacing: float, seed: int) -> tuple[np.ndarray, ...]:
    """Clip a perfect hexagonal lattice to the most circular ``count`` sites.

    Sites fill by exact Euclidean-radius shells. In axial coordinates their
    squared radius is the integer ``q² + q*r + r²``; this produces a circular
    disk rather than the visibly hexagonal contour produced by axial rings.
    Sites on a partial final shell are selected evenly around its circumference.
    ``spacing`` remains the later daughter split distance; initial nearest
    neighbors are ``spacing * INITIAL_PACKING_SPACING_SCALE`` apart so the seed
    disk starts compact without changing subsequent growth geometry.
    viewer/src/gpu/rng.ts mirrors this construction.
    """
    packed_spacing = spacing * INITIAL_PACKING_SPACING_SCALE
    limit = int(np.ceil(np.sqrt(count))) + 2
    shells: dict[int, list[tuple[float, float]]] = {}
    for q in range(-limit, limit + 1):
        for r in range(-limit, limit + 1):
            radius_squared = q * q + q * r + r * r
            shells.setdefault(radius_squared, []).append(
                (packed_spacing * (q + 0.5 * r), packed_spacing * (np.sqrt(3.0) * 0.5 * r))
            )

    offsets: list[tuple[float, float]] = []
    for radius_squared in sorted(shells):
        if len(offsets) >= count:
            break
        shell = shells[radius_squared]
        shell.sort(key=lambda p: np.arctan2(p[1], p[0]))
        take = min(count - len(offsets), len(shell))
        if take == len(shell):
            offsets.extend(shell)
        else:
            indices = [int(np.floor((j + 0.5) * len(shell) / take)) for j in range(take)]
            offsets.extend(shell[i] for i in indices)

    mean_x = sum(p[0] for p in offsets) / count
    mean_y = sum(p[1] for p in offsets) / count
    offsets = [(x - mean_x, y - mean_y) for x, y in offsets]
    # Rotation is a property of the rollout seed, not of its numerical
    # sampling density.  Using ``2 * count`` made the same seed start from a
    # different world orientation whenever density changed its initial count.
    theta = (_spawn_uniform01(seed, 2) * 2.0 - 1.0) * np.pi
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    positions = np.empty((count, 2), dtype=np.float32)
    for i, (x, y) in enumerate(offsets):
        positions[i, 0] = (center[0] + x * cos_t - y * sin_t) % 1.0
        positions[i, 1] = (center[1] + x * sin_t + y * cos_t) % 1.0
    velocities = np.zeros((count, 2), dtype=np.float32)
    F = np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (count, 1))
    C = np.zeros((count, 4), dtype=np.float32)
    Jp = np.ones((count,), dtype=np.float32)
    return positions, velocities, F, C, Jp


class TrainingRollout:
    """One rollout's worth of *state*. `core`/`agents`/`environment` (all
    their GPU buffers/pipelines) are owned and reused by the caller
    across many rollouts — see evolve.py's own module docstring — since
    rebuilding wgpu pipelines per candidate would be real, avoidable
    overhead; this constructor only seeds `core`'s particle buffers (one
    the configured initial particle count; --particles remains a growth cap) and
    resets `agents`/`environment`'s own persistent state back to a fresh
    (empty field, randomized heading — see AgentsGPU.reset_heading()'s
    own docstring) starting point for this rollout — every bit of that
    starting condition (packed-disk rotation, heading,
    growth's own seed) is now a pure, bit-exact function of `seed` alone
    (see seed_blob()'s/agents_gpu._spawn_uniform01()'s own docstrings),
    no numpy Generator needed anywhere in this constructor anymore.
    `mpm_enabled` (default True) is simulation_settings.py's own
    MPM_ENABLED, threaded through — see macro_step()'s own comment for
    exactly what setting it False skips."""

    def __init__(
        self,
        core: MpmCore,
        agents: AgentsGPU,
        environment: EnvironmentGPU,
        spawn_center: tuple[float, float],
        spawn_half_width: float,
        gravity: float,
        seed: int,
        mpm_enabled: bool = True,
        neural_updates_per_macro: int = NEURAL_UPDATES_PER_MACRO,
        communication_speed: float = COMMUNICATION_SPEED,
        initial_particle_count: int = INITIAL_PARTICLE_COUNT,
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
        # Every rollout — same "run-constant in practice today, but a
        # rollout-scoped setter regardless" reasoning set_gravity() above
        # already follows. The Agents uniform retains legacy spawn slots for
        # wire compatibility, although position is no longer a policy input.
        agents.set_spawn_center(*spawn_center)
        # Retained in the constructor/checkpoint schema for compatibility;
        # compact multi-cell seeding is now governed by split_displacement.
        _ = spawn_half_width
        initial_count = min(agents.max_active_particles, max(1, int(initial_particle_count)))
        positions, velocities, F, C, Jp = seed_blob(
            initial_count, spawn_center, agents.split_displacement, seed
        )
        core.load_scene(positions, velocities, F, C, Jp)
        # Every slot beyond the genuinely seeded particles is destined to
        # become a real particle via growth, at some unknown point in
        # this rollout — see reset_growth_buffers()'s own docstring for
        # why this has to run every rollout (not just once, ever) despite
        # seed_blob() already giving genuinely-seeded particles these
        # exact same fresh defaults. agents.max_active_particles (not a
        # parameter of this constructor — see AgentsGPU's own property
        # docstring for why) is --particles, the growth cap.
        core.reset_growth_buffers(agents.max_active_particles)

        environment.reset()
        agents.set_active_count(initial_count)
        agents.reset_heading(seed, positions)

    def macro_step(self, substeps_per_macro: int, *, growth_enabled: bool = True) -> None:
        core = self.core

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
        for communication_round in range(self.neural_updates_per_macro):
            final_round = communication_round == self.neural_updates_per_macro - 1
            if (
                self.environment.chemical_communication_architecture == "cell-owned-projection"
                or final_round
            ):
                self.environment.encode_clear(encoder)
            if self.environment.chemical_communication_architecture == "cell-owned-projection":
                self.agents.encode_splat_chemical_state(encoder)
            self.environment.encode_sense(encoder)
            self.agents.encode_step(
                encoder,
                self.environment.parity,
                commit_lifecycle=final_round,
            )
        # Persistent mode evolves the frozen field once and merges only the
        # final neural round's direct deposits.
        self.environment.encode_advance_persistent(encoder)
        core.device.queue.submit([encoder.finish()])

        # Growth's own readback — see this module's own module docstring
        # for why this (not "zero host round-trips") is correct: the
        # agentStep() pass just submitted may have grown activeCount on
        # the GPU, and nothing else finds out unless this class reads it
        # back and propagates it before core.step() below sizes ITS OWN
        # dispatches. A plain != check, not unconditional writes, so a
        # macro step where nothing actually split (the overwhelmingly
        # common case early in a rollout, or for a policy that never
        # learns to use the growth channel at all) costs one 4-byte read
        # and nothing else.
        # min(...) — the atomic itself can overshoot max_active_particles
        # slightly (several agents claiming a slot the same step, right
        # at the cap — see core/agents.wgsl's own agentStep() comment for
        # why that's left unguarded rather than compare-exchanged away);
        # clamping the *reported* count here is what actually enforces
        # the cap, since core/agents.wgsl itself already refuses to WRITE
        # a claimed slot past max_active_particles either way.
        grown = min(self.agents.read_grown_count(), self.agents.max_active_particles)
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
        if self.mpm_enabled:
            core.step(substeps_per_macro)

    def positions(self) -> np.ndarray:
        return self.core.read_positions()

"""Ties MpmCore (headless MLS-MPM physics) + EnvironmentGPU (GPU-resident
chemical field) + AgentsGPU (the evolved policy's GPU-resident forward
pass) into one rollout — the mpm-training analogue of envnca/simulation.py,
adapted to a real physics substrate instead of a bare point-agent grid.

Fully GPU-resident, matching ../viewer/src/gpu/simulation.ts's own
"GPU-resident is the whole point" design for every data-related buffer
(positions, velocities, F/C/ParticleRest, the chemical field, weights): each macro
step is one command encoder (environment.encode_sense -> agents.encode_step
-> environment.encode_merge_and_decay), one submit, immediately followed
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

The two former strafe channels now drive a normalized tensor-growth direction.
The agent shader rotates that local axis into world space and stores it with
two independent sigmoid controls: the former acceleration outputs now select
anisotropy and signed division-placement bias. A zero direction gives exactly
isotropic growth and symmetric random-axis division regardless of those
controls. MAX_STRAFE independently controls whether the same direction also
acts as physical acceleration and is zero by default.

Growth: every rollout currently starts with a coordinated two-particle
seed — core/agents.wgsl's own agentStep() may spawn
new ones from there (splitting, based on the last chemical channel's own
sensed value — see that file's own module docstring for the full
design), up to evolve.py's own --particles (now a CAP, not a fixed
starting count — see that module's own module docstring for why; a
policy that never learns to use the growth channel stays at 2
particles). This is the
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

from agents_gpu import AgentsGPU, _spawn_uniform01
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore


def seed_blob(count: int, center: tuple[float, float], half_width: float, seed: int) -> tuple[np.ndarray, ...]:
    """Same scene-seeding shape as feasibility_check.py's own seed_blob —
    duplicated rather than imported (that module is a standalone spike
    script, not a library this one should depend on). Jittered via
    agents_gpu._spawn_uniform01(seed, 2*i)/2*i+1 for particle i's own
    x/y — bit-exact with ../viewer/src/gpu/rng.ts's own seedBlob(), not
    just a plausible replay (that used to be a numpy Generator/PCG64 vs
    TS mulberry32 gap, same as reset_heading()'s own history — see that
    method's own docstring for the fuller "why bit-exact now" reasoning).
    Indices 0..2*count-1 are reserved for this function's own draws —
    TrainingRollout.__init__'s own theta draw (this file's own back-to-
    back placement) starts at index 2*count to never collide, regardless
    of `count`."""
    positions = np.empty((count, 2), dtype=np.float32)
    for i in range(count):
        jx = _spawn_uniform01(seed, 2 * i) * 2.0 - 1.0
        jy = _spawn_uniform01(seed, 2 * i + 1) * 2.0 - 1.0
        positions[i, 0] = center[0] + jx * half_width
        positions[i, 1] = center[1] + jy * half_width
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
    overhead; this constructor only seeds `core`'s particle buffers (2
    particles, back to back — see __init__'s own comment; a HARDCODED
    experiment, not yet CLI-configurable — --particles remains a growth
    CAP, not a starting count, see evolve.py's own module docstring) and
    resets `agents`/`environment`'s own persistent state back to a fresh
    (empty field, randomized heading — see AgentsGPU.reset_heading()'s
    own docstring) starting point for this rollout — every bit of that
    starting condition (spawn jitter, back-to-back theta, heading,
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

        core.set_gravity(gravity)
        # Every rollout — same "run-constant in practice today, but a
        # rollout-scoped setter regardless" reasoning set_gravity() above
        # already follows. The Agents uniform retains legacy spawn slots for
        # wire compatibility, although position is no longer a policy input.
        agents.set_spawn_center(*spawn_center)
        # HARDCODED experiment: start with 2 particles, back to back,
        # instead of the usual single starting particle — particle 1 is
        # placed agents.split_displacement behind particle 0 along a shared
        # random axis (same displacement/direction convention growth's
        # own split uses, agents.wgsl's own agentStep() `behindDir`), but
        # with its heading FLIPPED (theta + pi) rather than copied, so
        # the two face away from each other. Not a general N-agent or
        # CLI-configurable start yet — see this class's own docstring for
        # why a single particle was the rule until now.
        positions, velocities, F, C, Jp = seed_blob(2, spawn_center, spawn_half_width, seed)
        # Index 4 — seed_blob(2, ...) above claims indices 0-3 for its own
        # 2 particles' x/y jitter (see that function's own docstring),
        # this is the next one over. Bit-exact with
        # ../viewer/src/gpu/simulation.ts's own theta draw.
        theta = _spawn_uniform01(seed, 4) * (2.0 * np.pi) - np.pi
        behind_dir = np.array([-np.cos(theta), -np.sin(theta)], dtype=np.float32)
        positions[1] = (positions[0] + behind_dir * agents.split_displacement) % 1.0
        core.load_scene(positions, velocities, F, C, Jp)
        # Every slot beyond these 2 starting particles is destined to
        # become a real particle via growth, at some unknown point in
        # this rollout — see reset_growth_buffers()'s own docstring for
        # why this has to run every rollout (not just once, ever) despite
        # seed_blob() already giving genuinely-seeded particles these
        # exact same fresh defaults. agents.max_active_particles (not a
        # parameter of this constructor — see AgentsGPU's own property
        # docstring for why) is --particles, the growth cap.
        core.reset_growth_buffers(agents.max_active_particles)

        environment.reset()
        agents.set_active_count(2)
        agents.reset_heading(seed)
        # reset_heading() above already randomized every slot's own
        # heading independently (including these 2) — overwrite just
        # slots 0/1 with the coordinated back-to-back pair computed
        # above (theta / theta+pi), same "follow-up small write" agents_gpu.py's
        # own set_headings() docstring describes.
        agents.set_headings(np.array([theta, theta + np.pi], dtype=np.float32))

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
        self.environment.encode_sense(encoder)
        self.agents.encode_step(encoder, self.environment.parity)
        self.environment.encode_merge_and_decay(encoder)
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

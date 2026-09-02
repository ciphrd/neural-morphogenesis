"""A persistent multiprocessing.ProcessPoolExecutor of worker processes,
each with its own wgpu device/MpmCore/AgentsGPU/EnvironmentGPU, for
evolve.py's own run_generation() to fan candidate rollouts out across —
see that module's own module docstring for why this replaced the earlier
in-process batching experiment (profiling showed the actual bottleneck
was a single Python process being CPU-bound on wgpu-py's own FFI call
overhead and GPU-sync poll loop, neither of which reducing sync *count*
addresses — but neither uses more than one CPU core, or more than a
sliver of the GPU's own real throughput for a 400-particle rollout, so
several independent OS processes each doing the exact same
one-candidate-at-a-time work concurrently, on separate cores, sharing
the same underlying GPU, was the fix that actually matched the profile).

Each worker is initialized ONCE (build_pool()'s own initializer,
_worker_init below) with its own wgpu device and a single MpmCore/
AgentsGPU/EnvironmentGPU triple, plus the run's own fixed target/
target_raster/target_distance_field/args — all baked into worker-local
globals rather than re-sent with every task, since none of it changes
generation to generation. Only `weights` (~KB) and `seed` (an int) cross
the process boundary per task; `_worker_rollout` below is the only thing
actually pickled/sent per candidate.

Uses the 'spawn' start method (multiprocessing's own macOS/Windows
default, and deliberately not overridden to 'fork' here even on Linux):
'fork' after a wgpu device/Metal context already exists in the parent
process is a well-known way to corrupt native library state in the
child — spawning fresh processes that each pick their own device from
scratch is the only safe option for a native GPU handle like this.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

from agents_gpu import AgentsGPU
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore
from simulation_settings import (
    ANGULAR_DAMPING,
    CHEM_CHANNELS,
    CHIRALITY,
    CHEMICAL_CHANNEL_PROFILES,
    DECAY,
    DEPOSIT_DISTANCE,
    DEPOSIT_RATE,
    NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
    DEPOSIT_DENSITY_REFERENCE,
    DEPOSIT_SIGMA,
    DIVISION_COOLDOWN,
    FIELD_N,
    FRICTION,
    MAX_ACCEL,
    MAX_ANGULAR_ACCEL,
    MAX_ANGULAR_VELOCITY,
    MAX_ENV_WRITE,
    MAX_STRAFE,
    SPLIT_DISPLACEMENT,
)
from policy_parameters import policy_hidden_dim
from targets import TargetShape

# Worker-local globals — set once per process by _worker_init(), read by
# every _worker_rollout() call that process handles for the rest of the
# run. Deliberately module-level rather than passed around: this is
# exactly the state a persistent worker process is FOR, matching
# multiprocessing's own documented pattern for a per-worker initializer.
_core: Optional[MpmCore] = None
_agents: Optional[AgentsGPU] = None
_environment: Optional[EnvironmentGPU] = None
_target: Optional[TargetShape] = None
_target_raster: Optional[np.ndarray] = None
_target_distance_field: Optional[np.ndarray] = None
_args: Optional[argparse.Namespace] = None


def _worker_init(
    particles: int,
    target: TargetShape,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Runs once per worker process, at pool creation — picks this
    worker's own wgpu device (never shared across processes; see this
    module's own module docstring for why) and builds its own single
    MpmCore/AgentsGPU/EnvironmentGPU triple, reused across every
    candidate this worker ever evaluates for the rest of the run (same
    "wgpu pipeline compilation is real, avoidable overhead" reasoning
    evolve.py's own module docstring already applies to a single
    process)."""
    global _core, _agents, _environment, _target, _target_raster, _target_distance_field, _args
    # verbose=False — build_pool() already logged this once, in the main
    # process, before spawning any worker (see pick_device()'s own
    # docstring for why one line is enough on a single machine).
    wgpu_device = pick_device(verbose=False)
    _core = MpmCore(wgpu_device)
    _environment = EnvironmentGPU(
        wgpu_device, CHEM_CHANNELS, FIELD_N, FIELD_N, DECAY, DEPOSIT_RATE,
        args.chemical_communication_architecture,
        NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY,
        DEPOSIT_DENSITY_REFERENCE,
        grid_velocity=_core.grid_vel,
        channel_profiles=CHEMICAL_CHANNEL_PROFILES,
    )
    _agents = AgentsGPU(
        wgpu_device,
        _core,
        _environment,
        CHEM_CHANNELS,
        policy_hidden_dim(args.policy_architecture),
        MAX_ACCEL,
        MAX_STRAFE,
        MAX_ENV_WRITE,
        MAX_ANGULAR_ACCEL,
        ANGULAR_DAMPING,
        MAX_ANGULAR_VELOCITY,
        CHIRALITY,
        DEPOSIT_DISTANCE,
        particles,
        SPLIT_DISPLACEMENT,
        DIVISION_COOLDOWN,
        FRICTION,
        DEPOSIT_SIGMA,
        1.0,
        args.spawn_x,
        args.spawn_y,
        policy_architecture=args.policy_architecture,
        chemical_communication_architecture=args.chemical_communication_architecture,
    )
    _target = target
    _target_raster = target_raster
    _target_distance_field = target_distance_field
    _args = args


def worker_rollout(weights: np.ndarray, seed: int, density_multiplier: float = 1.0) -> float:
    """The only thing actually sent to a worker per candidate — `weights`
    and `seed`. Public (not `_`-prefixed, unlike this module's other
    worker-local state) because evolve.py's own run_generation() needs a
    module-level, picklable reference to hand to ProcessPoolExecutor.map()
    — imports evolve lazily (not at module top level) purely to avoid a
    circular import (evolve.py itself imports build_pool() from this
    module) — has no effect on worker startup cost since this runs once
    per task, not once per process, and rollout() itself is cheap to look
    up."""
    from evolve import rollout

    return rollout(
        weights, _target, _target_raster, _target_distance_field, _args, seed,
        _core, _agents, _environment, density_multiplier=density_multiplier,
    )


def build_pool(
    num_workers: int,
    particles: int,
    target: TargetShape,
    target_raster: np.ndarray,
    target_distance_field: np.ndarray,
    args: argparse.Namespace,
    log_device: bool = True,
) -> ProcessPoolExecutor:
    """Builds a ProcessPoolExecutor of `num_workers` persistent worker
    processes, each running _worker_init() exactly once before handling
    any tasks. Callers own the returned pool for the lifetime of a whole
    training run (see evolve.py's own main()) — building it fresh per
    generation would pay every worker's own device-pick + pipeline-
    compilation cost every generation, exactly the "real, avoidable
    overhead" this project's own MpmCore/AgentsGPU/EnvironmentGPU already
    take care not to repeat for a single process.

    `log_device`, on by default, logs the "[device] adapter: ..."
    confirmation exactly once, here in the main process, before any
    worker exists — see pick_device()'s own docstring for why every
    worker's OWN pick stays silent instead. This device itself isn't
    used for anything (every worker below picks its own, separately, in
    its own process); it's only requested to learn+log which backend/
    adapter this machine will hand out, since on one machine every
    worker gets the same answer. Callers that already logged their own
    device elsewhere before calling this (train_server.py's own
    _setup(), which needs a real device of its own for debug-image
    rendering — see that module's own comment) should pass False, so
    startup shows that one line once, not twice."""
    if log_device:
        pick_device()
    return ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(particles, target, target_raster, target_distance_field, args),
    )

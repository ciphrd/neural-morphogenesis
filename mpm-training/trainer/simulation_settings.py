"""Single source of truth for every constant that shapes how a training
rollout actually behaves once it's running — the mpm-training analogue
of envnca/constants.py. train_server.py imports every one of these to
forward into its own per-generation broadcast message, so the frontend's
WebGPU replay reproduces the exact same simulation a checkpoint was
trained under, rather than the frontend guessing at its own hardcoded
copy. That guessing was a real, silent-drift risk before this file
existed: these values used to live split across mpm_core.py's own
DEFAULT_* module constants, core/constants.json's DEFAULT_SPLAT_RADIUS/
DEFAULT_REPULSION_STRENGTH, and this file's own former identity as
training_constants.py — nothing forced any of them to agree with
whatever the viewer's own gpu/mpmCore.ts independently hardcoded, and
they didn't ride along in the broadcast message at all until now.

Deliberately NOT here: anything CLI-configurable per training run
(--particles, --macro-steps, --gravity, --population, --elites,
--mutation-sigma, ... — see evolve.py's own argparse) — those are
already overridable without editing code, and already forwarded to the
frontend the same way (train_server.py's own broadcast message pulls
both this file's constants and evolve.py's own args into one payload).
"""
from __future__ import annotations

import json
from pathlib import Path

_CORE_CONSTANTS = json.loads((Path(__file__).parent.parent / "core" / "constants.json").read_text())
_GRID_N: int = _CORE_CONSTANTS["GRID_N"]

# --- Policy architecture (UpdateRule) ---
HIDDEN_DIM = 128
CHEM_CHANNELS = 8  # last channel is always growth's own split-probability field
ANGULAR_DIM = 1
ACCEL_DIM = 2
STRAFE_DIM = 2

# --- Chemical field (environment_gpu.py / core/environment.wgsl) ---
FIELD_N = 256
DECAY = 0.91

# Motion
MAX_ACCEL = 0.0 # 0.1 # not used rn
# Optional physical acceleration scale for the two policy channels that
# now direct tensor growth. Growth reads their raw bounded direction and
# remains active when this is zero.
MAX_STRAFE = 0.0 # 5.3
FRICTION = 0.9
MAX_ANGULAR_ACCEL = 1.4
ANGULAR_DAMPING = 0.8
MAX_ANGULAR_VELOCITY = 0.1

# Deposit
DEPOSIT_RATE = 1.0
# Retained in settings/checkpoint metadata and the AgentPhysics wire layout
# for compatibility. Chemical writes are now always centered under the
# particle, so this value is intentionally unused.
DEPOSIT_DISTANCE = 0.0
DEPOSIT_SIGMA = 0.4
MAX_ENV_WRITE = 1.0

# Behavior
CHIRALITY = True
MPM_ENABLED = True
MATERIAL_E = 1e4
MATERIAL_NU = 0.2
MATERIAL_HARDENING = 3.0
MATERIAL_ELASTICITY = 0.2

# Growth
SPLIT_DISPLACEMENT = 0.01
DIVISION_COOLDOWN = 1.0
# Retained in broadcasts for compatibility with older viewers. The
# conservative grow-then-divide model no longer fades mass in after a
# split; mass is accumulated before division through g instead.
MASS_RAMP_MACRO_STEPS = 1.0

# --- Kinematic growth (multiplicative decomposition F = Fe*Fg) ---
# The mechanism that lets a shape actually GROW as particles are added,
# instead of elasticity fighting to restore its original volume: growth
# accumulates in a full per-particle stress-free tensor Fg, and
# core/p2g.wgsl evaluates the constitutive law on Fe = F*inverse(Fg) rather
# than raw F. A zero policy direction preserves exact isotropic scalar-model
# equivalence; a nonzero direction produces anisotropic rest growth. See
# core/g2p.wgsl and core/agents.wgsl's ParticleRest.growthF comment.
#
# Exponential stress-free area growth rate while a substrate-triggered
# cell cycle is active. Growth is imposed first; elastic relaxation then
# expands the compatible body. 0 disables growth.
GROWTH_RATE = 50.0
# Division area ratio. 2 makes one g=2 parent exactly equivalent in mass
# and rest area to two g=1 daughters.
GROWTH_MAX = 2.0
# Compression reference for continuous mechanical feedback. Below this
# elastic volume Je, growth is slowed by Je/reference; it is never
# switched off at a threshold. 0 disables inhibition.
GROWTH_THRESHOLD = 0.85

# Simulation. This is stronger than the previous 0.999 per-substep
# retention so kinetic energy from completed divisions decays instead of
# surviving for many later fitness snapshots.
SUBSTEPS_PER_DAMPING_FRAME = round(10 * (_GRID_N / 80))
DAMPING_LOSS_FRACTION = 1 - 0.995**SUBSTEPS_PER_DAMPING_FRAME

# Repulsion
SPLAT_RADIUS = 0.004
REPULSION_STRENGTH = 0.2
# Hard cap on the MAGNITUDE of one physics substep's own repulsion
# velocity delta — see core/repulsion.wgsl's own RepulsionParams.maxDelta
# field comment for the full reasoning (an unclamped delta at the
# strength needed to beat MATERIAL_E's own continuous elastic resistance
# is exactly what produces a single-substep MLS-MPM stability violation
# — confirmed empirically to blow up to NaN with REPULSION_STRENGTH>=100
# and zero growth/elasticity involved). ~1/3 of DX/DT (core/
# constants.json's own DX=0.015625, DT=0.0000625 => DX/DT=250, the
# theoretical "moves exactly one grid cell in one substep" bound for
# velocity) — comfortable margin below that bound, not tuned to the edge.
REPULSION_MAX_DELTA = 40.0

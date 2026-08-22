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
MAX_STRAFE = 0.0 # 5.3
FRICTION = 0.9
MAX_ANGULAR_ACCEL = 1.4
ANGULAR_DAMPING = 0.8
MAX_ANGULAR_VELOCITY = 0.1

# Deposit
DEPOSIT_RATE = 1.0
DEPOSIT_SPOTS = 4
DEPOSIT_DISTANCE = 2.0
DEPOSIT_SIGMA = 0.4
MAX_ENV_WRITE = 1.0

# Behavior
CHIRALITY = True
MPM_ENABLED = True
MATERIAL_E = 1e4
MATERIAL_NU = 0.2
MATERIAL_HARDENING = 3.0
MATERIAL_ELASTICITY = 0.0

# Growth
SPLIT_DISPLACEMENT = 0.01
DIVISION_COOLDOWN = 100.0

# Simuation
SUBSTEPS_PER_DAMPING_FRAME = round(10 * (_GRID_N / 80))
DAMPING_LOSS_FRACTION = 1 - 0.999**SUBSTEPS_PER_DAMPING_FRAME

# Repulsion
SPLAT_RADIUS = 0.004
REPULSION_STRENGTH = 0.05

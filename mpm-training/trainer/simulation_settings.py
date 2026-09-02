"""Typed Python access to the shared defaults that shape a training rollout.

Wire-visible run defaults live in ``core/default_run_settings.json`` so the
trainer and the browser's offline random-brain mode consume the same values.
train_server.py imports the constants resolved here to
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

Run-selected values are not fixed here: --particles, --initial-particles,
--macro-steps, --gravity, --population, --elites, --mutation-sigma, ...
(see evolve.py's own argparse) are overridable without editing code and forwarded to the
frontend the same way (train_server.py's own broadcast message pulls
both this file's constants and evolve.py's own args into one payload).
"""
from __future__ import annotations

import json
from pathlib import Path

from chemical_channels import default_channel_profiles

_CORE_CONSTANTS = json.loads((Path(__file__).parent.parent / "core" / "constants.json").read_text())
DEFAULT_RUN_SETTINGS = json.loads(
    (Path(__file__).parent.parent / "core" / "default_run_settings.json").read_text()
)
_GRID_N: int = _CORE_CONSTANTS["GRID_N"]

# --- Policy architecture (UpdateRule) ---
HIDDEN_DIM = int(DEFAULT_RUN_SETTINGS["hiddenDim"])
CHEM_CHANNELS = int(DEFAULT_RUN_SETTINGS["channels"])
CHEMICAL_CHANNEL_PROFILES = default_channel_profiles(CHEM_CHANNELS)
# Per-physics-substep elastic-shear relaxation. Zero preserves the legacy
# solid-like material response exactly.
MATERIAL_FLUIDITY = float(DEFAULT_RUN_SETTINGS["materialFluidity"])
# Default number of genuinely seeded cells before any policy-driven division.
# Shared with the browser through core/constants.json; individual runs and
# historical checkpoint metadata can still override it.
INITIAL_PARTICLE_COUNT = int(DEFAULT_RUN_SETTINGS["initialParticleCount"])
# Robust zero-centered scales from the absolute pooled distribution across all
# chemical channels (and both gradient directions) in the 5,005-sample live
# report. One shared scale preserves channel-permutation symmetry: no channel
# gets a privileged gain based on the accidental behavior of one random brain.
CHEMICAL_VALUE_INPUT_MULTIPLIER = float(DEFAULT_RUN_SETTINGS["chemicalValueInputMultiplier"])
CHEMICAL_GRADIENT_INPUT_SCALE = float(DEFAULT_RUN_SETTINGS["chemicalGradientInputScale"])
MORPHOLOGY_GRADIENT_INPUT_SCALE: float = _CORE_CONSTANTS["MORPHOLOGY_GRADIENT_INPUT_SCALE"]
BOUNDARY_TANGENT_MIN_GRADIENT = float(DEFAULT_RUN_SETTINGS["boundaryTangentMinGradient"])
GROWTH_DIRECTION_RESPONSE_RATE: float = _CORE_CONSTANTS["GROWTH_DIRECTION_RESPONSE_RATE"]
GROWTH_ANISOTROPY_RESPONSE_RATE: float = _CORE_CONSTANTS["GROWTH_ANISOTROPY_RESPONSE_RATE"]
DIRECTION_CONFIDENCE_SCALE: float = _CORE_CONSTANTS["DIRECTION_CONFIDENCE_SCALE"]
# Normal training architecture selection. This is broadcast with the run's
# settings; the paired comparison utility overrides it internally only while
# launching its two controlled experiment subprocesses.
POLICY_ARCHITECTURE = str(DEFAULT_RUN_SETTINGS["policyArchitecture"])
CELL_MEMORY = str(DEFAULT_RUN_SETTINGS["cellMemory"])
HIDDEN_LAYERS = tuple(int(width) for width in DEFAULT_RUN_SETTINGS["hiddenLayers"])
CHEMICAL_COMMUNICATION_ARCHITECTURE = str(
    DEFAULT_RUN_SETTINGS["chemicalCommunicationArchitecture"]
)
INTERNAL_STATE_SPEED = float(DEFAULT_RUN_SETTINGS["internalStateSpeed"])
# Playback/training defaults grant the policy full directional authority.
DIVISION_DIRECTIONALITY = float(DEFAULT_RUN_SETTINGS["divisionDirectionality"])
# Blend between signed division drive and a probability remap. At 0, only
# positive drive contributes; at 1, [-1,1] becomes [0,1].
DIVISION_DRIVE_BOOST = float(DEFAULT_RUN_SETTINGS["divisionDriveBoost"])
# NN-controlled velocity diffusion performed entirely on the MPM grid.
# Cells publish the signal through chemical channel 0; zero is an exact no-op.
# Neural evaluations performed before each mechanical macro step. These are
# numerical communication substeps; increasing this improves temporal
# resolution without multiplying chemical/turning speed.
NEURAL_UPDATES_PER_MACRO = int(DEFAULT_RUN_SETTINGS["neuralUpdatesPerMacro"])
# Chemical/orientation time elapsed per mechanical macro step. 1 preserves the
# original single-round clock; this is independent from evaluation resolution.
COMMUNICATION_SPEED = float(DEFAULT_RUN_SETTINGS["communicationSpeed"])
# Desired local heading vector. The shader derives angular acceleration from
# its signed angular error instead of accepting acceleration directly.
ANGULAR_DIM = 2
ACCEL_DIM = 2
STRAFE_DIM = 2

# --- Chemical field (environment_gpu.py / core/environment.wgsl) ---
FIELD_N = int(DEFAULT_RUN_SETTINGS["fieldN"])
# Used by persistent-environment communication. Cell-owned projection ignores
# decay because the spatial field is rebuilt from cell state every round.
DECAY = float(DEFAULT_RUN_SETTINGS["decay"])

# Motion
MAX_ACCEL = float(DEFAULT_RUN_SETTINGS["maxAccel"])
# Optional physical acceleration scale for the two policy channels that
# now direct tensor growth. Growth reads their raw bounded direction and
# remains active when this is zero.
MAX_STRAFE = float(DEFAULT_RUN_SETTINGS["maxStrafe"])
FRICTION = float(DEFAULT_RUN_SETTINGS["friction"])
MAX_ANGULAR_ACCEL = float(DEFAULT_RUN_SETTINGS["maxAngularAccel"])
ANGULAR_DAMPING = float(DEFAULT_RUN_SETTINGS["angularDamping"])
MAX_ANGULAR_VELOCITY = float(DEFAULT_RUN_SETTINGS["maxAngularVelocity"])
# Heading-relative Hencky strain is divided by this before tanh enters the
# policy. Approximately 15% logarithmic strain therefore produces a strong,
# still-unsaturated mechanosensory signal.
ELASTIC_STRAIN_SCALE = float(DEFAULT_RUN_SETTINGS["elasticStrainScale"])
# Retain this explicit switch so the three volume/axial/shear mechanosensory
# lanes can be ablated without another weight-shape migration.
ELASTIC_STRAIN_INPUTS_ENABLED = bool(DEFAULT_RUN_SETTINGS["elasticStrainInputsEnabled"])

# Deposit
DEPOSIT_RATE = float(DEFAULT_RUN_SETTINGS["depositRate"])
# Optional capacity-normalized deposit mode. Below the configured represented-
# material capacity, raw Gaussian deposition is preserved; above it, matching
# local density divides away any amplification from overcrowding. The legacy
# wire name "reference" is retained for checkpoint compatibility.
NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY = bool(DEFAULT_RUN_SETTINGS["normalizeDepositsByLocalDensity"])
DEPOSIT_DENSITY_REFERENCE = float(DEFAULT_RUN_SETTINGS["depositDensityReference"])
# Retained in the AgentPhysics/settings ABI for checkpoint compatibility.
# Single deposits are always centered underneath the particle.
DEPOSIT_DISTANCE = float(DEFAULT_RUN_SETTINGS["depositDistance"])
# Normalized [0,1] world-domain sigma. Per-scale multipliers preserve this
# physical unit when the shader projects onto differently sized native grids.
DEPOSIT_SIGMA = float(DEFAULT_RUN_SETTINGS["depositSigma"])
# Amplitude of the NN's signed chemical delta rate.
MAX_ENV_WRITE = float(DEFAULT_RUN_SETTINGS["maxEnvWrite"])

# Behavior
CHIRALITY = bool(DEFAULT_RUN_SETTINGS["chirality"])
MPM_ENABLED = bool(DEFAULT_RUN_SETTINGS["mpmEnabled"])
MATERIAL_E = float(DEFAULT_RUN_SETTINGS["materialE"])
MATERIAL_NU = float(DEFAULT_RUN_SETTINGS["materialNu"])
MATERIAL_HARDENING = float(DEFAULT_RUN_SETTINGS["materialHardening"])
MATERIAL_ELASTICITY = float(DEFAULT_RUN_SETTINGS["materialElasticity"])

# Growth
SPLIT_DISPLACEMENT = float(DEFAULT_RUN_SETTINGS["splitDisplacement"])
DIVISION_COOLDOWN = float(DEFAULT_RUN_SETTINGS["divisionCooldown"])
# Retained in broadcasts for compatibility with older viewers. The
# conservative grow-then-divide model no longer fades mass in after a
# split; mass is accumulated before division through g instead.
MASS_RAMP_MACRO_STEPS = float(DEFAULT_RUN_SETTINGS["massRampMacroSteps"])

# --- Kinematic growth (multiplicative decomposition F = Fe*Fg) ---
# The mechanism that lets a shape actually GROW as particles are added,
# instead of elasticity fighting to restore its original volume: growth
# accumulates in a full per-particle stress-free tensor Fg, and
# core/p2g.wgsl evaluates the constitutive law on Fe = F*inverse(Fg) rather
# than raw F. A zero policy direction preserves exact isotropic scalar-model
# equivalence; a nonzero direction produces anisotropic rest growth. See
# core/g2p.wgsl and core/agents.wgsl's ParticleRest.growthF comment.
#
# Approximate number of mechanical macro steps an uncompressed particle takes
# between entering a cell cycle and doubling its stress-free area. The host
# derives the shader's per-physics-substep
# exponential rate from this and the run's actual substeps-per-macro, so
# increasing substeps for numerical stability no longer accelerates growth
# relative to communication and control. 0 disables growth.
GROWTH_DURATION_MACRO_STEPS = float(DEFAULT_RUN_SETTINGS["growthDuration"])
# Division area ratio. 2 makes one g=2 parent exactly equivalent in mass
# and rest area to two g=1 daughters.
GROWTH_MAX = float(DEFAULT_RUN_SETTINGS["growthMax"])
GROWTH_ANISOTROPY_AUTHORITY = float(DEFAULT_RUN_SETTINGS["growthAnisotropy"])
# Mechanical contact inhibition. Elastic areal compression is measured as
# c=max(0,-log(det(Fe))). Growth is unaffected below the start threshold,
# smoothly suppressed between distinct thresholds, and fully paused at/above
# the stop threshold. Equal thresholds select a hard cutoff. Strength 0 is an
# exact legacy-mode escape hatch.
GROWTH_COMPRESSION_START = float(DEFAULT_RUN_SETTINGS["growthCompressionStart"])
GROWTH_COMPRESSION_STOP = float(DEFAULT_RUN_SETTINGS["growthCompressionStop"])
GROWTH_COMPRESSION_FEEDBACK = float(DEFAULT_RUN_SETTINGS["growthCompressionFeedback"])

# CLI default shared with MpmCore's construction-time material initialization.
# Individual runs still override it through --substeps-per-macro.
DEFAULT_SUBSTEPS_PER_MACRO = int(DEFAULT_RUN_SETTINGS["substepsPerMacro"])

# Simulation. This is stronger than the previous 0.999 per-substep
# retention so kinetic energy from completed divisions decays instead of
# surviving for many later fitness snapshots.
SUBSTEPS_PER_DAMPING_FRAME = round(10 * (_GRID_N / 80))
DAMPING_LOSS_FRACTION = float(DEFAULT_RUN_SETTINGS["damping"])

# Repulsion
SPLAT_RADIUS = float(DEFAULT_RUN_SETTINGS["splatRadius"])
REPULSION_STRENGTH = float(DEFAULT_RUN_SETTINGS["repulsionStrength"])
# Policy morphology sensing uses a simulation-owned smoothed density field.
# Sigma is in normalized domain units, independent of texture resolution.
MORPHOLOGY_BLUR_SIGMA = float(DEFAULT_RUN_SETTINGS["morphologyBlurSigma"])
# Blurred density rho becomes bounded occupancy rho/(rho+reference).
MORPHOLOGY_DENSITY_REFERENCE = float(DEFAULT_RUN_SETTINGS["morphologyDensityReference"])
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
REPULSION_MAX_DELTA = float(DEFAULT_RUN_SETTINGS["repulsionMaxDelta"])

# Boundary-localized cohesion sampled from the blurred morphology occupancy.
# Zero strength preserves historical mechanics; the cap uses the same safe
# per-substep velocity scale as repulsion for live experimentation.

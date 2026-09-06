"""Typed projections of core/config.json; no independent defaults."""
from __future__ import annotations

import json
from pathlib import Path

from chemical_channels import default_channel_profiles

from config import CONFIG
_CORE_CONSTANTS = CONFIG["simulation"]
DEFAULT_RUN_SETTINGS = CONFIG["run"]
_GRID_N: int = _CORE_CONSTANTS["GRID_N"]
HIDDEN_DIM = int(DEFAULT_RUN_SETTINGS["hiddenDim"])
CHEM_CHANNELS = int(DEFAULT_RUN_SETTINGS["channels"])
CHEMICAL_CHANNEL_PROFILES = default_channel_profiles(CHEM_CHANNELS)
INITIAL_PARTICLE_COUNT = int(DEFAULT_RUN_SETTINGS["initialParticleCount"])
CHEMICAL_VALUE_INPUT_MULTIPLIER = float(DEFAULT_RUN_SETTINGS["chemicalValueInputMultiplier"])
CHEMICAL_GRADIENT_INPUT_SCALE = float(DEFAULT_RUN_SETTINGS["chemicalGradientInputScale"])
MORPHOLOGY_GRADIENT_INPUT_SCALE: float = _CORE_CONSTANTS["MORPHOLOGY_GRADIENT_INPUT_SCALE"]
BOUNDARY_TANGENT_MIN_GRADIENT = float(DEFAULT_RUN_SETTINGS["boundaryTangentMinGradient"])
POLICY_ARCHITECTURE = str(DEFAULT_RUN_SETTINGS["policyArchitecture"])
CELL_MEMORY = str(DEFAULT_RUN_SETTINGS["cellMemory"])
HIDDEN_LAYERS = tuple(int(width) for width in DEFAULT_RUN_SETTINGS["hiddenLayers"])
CHEMICAL_COMMUNICATION_ARCHITECTURE = str(
    DEFAULT_RUN_SETTINGS["chemicalCommunicationArchitecture"]
)
INTERNAL_STATE_SPEED = float(DEFAULT_RUN_SETTINGS["internalStateSpeed"])
NEURAL_UPDATES_PER_MACRO = int(DEFAULT_RUN_SETTINGS["neuralUpdatesPerMacro"])
COMMUNICATION_SPEED = float(DEFAULT_RUN_SETTINGS["communicationSpeed"])
FIELD_N = int(DEFAULT_RUN_SETTINGS["fieldN"])
DECAY = float(DEFAULT_RUN_SETTINGS["decay"])
FRICTION = float(DEFAULT_RUN_SETTINGS["friction"])
ELASTIC_STRAIN_SCALE = float(DEFAULT_RUN_SETTINGS["elasticStrainScale"])
ELASTIC_STRAIN_INPUTS_ENABLED = bool(DEFAULT_RUN_SETTINGS["elasticStrainInputsEnabled"])
DEPOSIT_RATE = float(DEFAULT_RUN_SETTINGS["depositRate"])
NORMALIZE_DEPOSITS_BY_LOCAL_DENSITY = bool(DEFAULT_RUN_SETTINGS["normalizeDepositsByLocalDensity"])
MAX_ENV_WRITE = float(DEFAULT_RUN_SETTINGS["maxEnvWrite"])
MPM_ENABLED = bool(DEFAULT_RUN_SETTINGS["mpmEnabled"])
MATERIAL_E = float(DEFAULT_RUN_SETTINGS["materialE"])
MATERIAL_NU = float(DEFAULT_RUN_SETTINGS["materialNu"])
MATERIAL_HARDENING = float(DEFAULT_RUN_SETTINGS["materialHardening"])
MATERIAL_ELASTICITY = float(DEFAULT_RUN_SETTINGS["materialElasticity"])
SAMPLE_SPACING = float(DEFAULT_RUN_SETTINGS["sampleSpacing"])
GROWTH_DURATION_MACRO_STEPS = float(DEFAULT_RUN_SETTINGS["growthDuration"])
GROWTH_MODEL_VERSION = int(DEFAULT_RUN_SETTINGS["growthModelVersion"])
MATERIAL_AREA_BUDGET = float(DEFAULT_RUN_SETTINGS["materialAreaBudget"])
GROWTH_ANISOTROPY_AUTHORITY = float(DEFAULT_RUN_SETTINGS["growthAnisotropy"])
GROWTH_COMPRESSION_START = float(DEFAULT_RUN_SETTINGS["growthCompressionStart"])
GROWTH_COMPRESSION_STOP = float(DEFAULT_RUN_SETTINGS["growthCompressionStop"])
GROWTH_COMPRESSION_FEEDBACK = float(DEFAULT_RUN_SETTINGS["growthCompressionFeedback"])
DEFAULT_SUBSTEPS_PER_MACRO = int(DEFAULT_RUN_SETTINGS["substepsPerMacro"])
SUBSTEPS_PER_DAMPING_FRAME = round(10 * (_GRID_N / 80))
DAMPING_LOSS_FRACTION = float(DEFAULT_RUN_SETTINGS["damping"])
SPLAT_RADIUS = float(DEFAULT_RUN_SETTINGS["splatRadius"])
REPULSION_STRENGTH = float(DEFAULT_RUN_SETTINGS["repulsionStrength"])
MORPHOLOGY_BLUR_SIGMA = float(DEFAULT_RUN_SETTINGS["morphologyBlurSigma"])
MORPHOLOGY_DENSITY_REFERENCE = float(DEFAULT_RUN_SETTINGS["morphologyDensityReference"])
REPULSION_MAX_DELTA = float(DEFAULT_RUN_SETTINGS["repulsionMaxDelta"])

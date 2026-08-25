"""Logical policy-head layout and shared parameter bucketing.

The GPU intentionally consumes one concatenated output matrix.  This module
defines how the Python model's separate heads map into that stable wire format,
and applies the initialization/mutation policy from core/policy_parameters.json.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


_CONFIG = json.loads((Path(__file__).parent.parent / "core" / "policy_parameters.json").read_text())
STATELESS_ARCHITECTURE = "stateless-128"
STATEFUL_ARCHITECTURE = "stateful-64"
STATEFUL_128_ARCHITECTURE = "stateful-128"
POLICY_ARCHITECTURES = (STATELESS_ARCHITECTURE, STATEFUL_ARCHITECTURE, STATEFUL_128_ARCHITECTURE)
NO_CELL_MEMORY = "none"
RECURRENT_CELL_MEMORY = "recurrent"
CELL_MEMORY_OPTIONS = (NO_CELL_MEMORY, RECURRENT_CELL_MEMORY)
PRIVATE_STATE_DIM = 8
PERSISTENT_ENVIRONMENT_ARCHITECTURE = "persistent-environment"
CELL_OWNED_PROJECTION_ARCHITECTURE = "cell-owned-projection"
CHEMICAL_COMMUNICATION_ARCHITECTURES = (
    PERSISTENT_ENVIRONMENT_ARCHITECTURE,
    CELL_OWNED_PROJECTION_ARCHITECTURE,
)


@dataclass(frozen=True)
class PolicyHead:
    name: str
    size: int
    weight_gain: float
    bias_center: tuple[float, ...]
    bias_jitter: float
    mutation_scale: float


def normalize_architecture(architecture: str | None) -> str:
    architecture = architecture or STATELESS_ARCHITECTURE
    if architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"unknown policy architecture {architecture!r}; expected one of {POLICY_ARCHITECTURES}")
    return architecture


def policy_has_recurrence(architecture: str) -> bool:
    return normalize_architecture(architecture) in (STATEFUL_ARCHITECTURE, STATEFUL_128_ARCHITECTURE)


def cell_memory_for_architecture(architecture: str) -> str:
    return RECURRENT_CELL_MEMORY if policy_has_recurrence(architecture) else NO_CELL_MEMORY


def architecture_for_cell_memory(cell_memory: str | None) -> str:
    cell_memory = cell_memory or RECURRENT_CELL_MEMORY
    if cell_memory not in CELL_MEMORY_OPTIONS:
        raise ValueError(f"unknown cell memory {cell_memory!r}; expected one of {CELL_MEMORY_OPTIONS}")
    return STATEFUL_128_ARCHITECTURE if cell_memory == RECURRENT_CELL_MEMORY else STATELESS_ARCHITECTURE


def normalize_chemical_communication_architecture(architecture: str | None) -> str:
    architecture = architecture or CELL_OWNED_PROJECTION_ARCHITECTURE
    if architecture not in CHEMICAL_COMMUNICATION_ARCHITECTURES:
        raise ValueError(
            f"unknown chemical communication architecture {architecture!r}; "
            f"expected one of {CHEMICAL_COMMUNICATION_ARCHITECTURES}"
        )
    return architecture


def resolve_chemical_communication_architecture(
    architecture: str | None, decay: float = 0.0
) -> str:
    """Infer the lifecycle used by checkpoints created before it was tagged."""
    if architecture is None:
        architecture = (
            PERSISTENT_ENVIRONMENT_ARCHITECTURE
            if decay > 0.0
            else CELL_OWNED_PROJECTION_ARCHITECTURE
        )
    return normalize_chemical_communication_architecture(architecture)


def policy_hidden_dim(architecture: str) -> int:
    return 64 if normalize_architecture(architecture) == STATEFUL_ARCHITECTURE else 128


def policy_input_dim(num_channels: int, architecture: str) -> int:
    return 3 * num_channels + 6 + (PRIVATE_STATE_DIM if policy_has_recurrence(architecture) else 0)


def policy_heads(num_channels: int, architecture: str = STATELESS_ARCHITECTURE) -> tuple[PolicyHead, ...]:
    architecture = normalize_architecture(architecture)
    sizes: dict[str, int] = {
        "chemical": num_channels,
        "heading": 2,
        "anisotropy": 1,
        "division": 1,
        "growthDirection": 2,
    }
    if policy_has_recurrence(architecture):
        sizes.update({"stateDelta": PRIVATE_STATE_DIM, "stateGate": PRIVATE_STATE_DIM})
    else:
        sizes["color"] = 3
    heads: list[PolicyHead] = []
    for name, size in sizes.items():
        raw = _CONFIG["heads"][name]
        center = tuple(float(v) for v in raw["biasCenter"])
        if len(center) == 1:
            center = center * size
        if len(center) != size:
            raise ValueError(f"policy head {name} has {len(center)} bias centers for {size} outputs")
        heads.append(
            PolicyHead(
                name=name,
                size=size,
                weight_gain=float(raw["weightGain"]),
                bias_center=center,
                bias_jitter=float(raw["biasJitter"]),
                mutation_scale=float(raw["mutationScale"]),
            )
        )
    return tuple(heads)


def trunk_initialization() -> tuple[float, float]:
    trunk = _CONFIG["trunk"]
    return float(trunk["weightGain"]), float(trunk["biasJitter"])


def _xavier_bound(fan_in: int, fan_out: int, gain: float) -> float:
    return gain * np.sqrt(6.0 / float(fan_in + fan_out))


def random_flat_policy_weights(
    num_channels: int,
    hidden_dim: int,
    rng: np.random.Generator,
    architecture: str = STATELESS_ARCHITECTURE,
) -> np.ndarray:
    """Create the canonical fc1w/fc1b/fc2w/fc2b flat GPU layout."""
    architecture = normalize_architecture(architecture)
    expected_hidden = policy_hidden_dim(architecture)
    if hidden_dim != expected_hidden:
        raise ValueError(f"{architecture} requires hidden_dim={expected_hidden}, got {hidden_dim}")
    input_dim = policy_input_dim(num_channels, architecture)
    trunk_gain, trunk_bias_jitter = trunk_initialization()
    trunk_bound = _xavier_bound(input_dim, hidden_dim, trunk_gain)
    fc1w = rng.uniform(-trunk_bound, trunk_bound, (hidden_dim, input_dim))
    fc1b = rng.uniform(-trunk_bias_jitter, trunk_bias_jitter, hidden_dim)

    head_weights: list[np.ndarray] = []
    head_biases: list[np.ndarray] = []
    for head in policy_heads(num_channels, architecture):
        bound = _xavier_bound(hidden_dim, head.size, head.weight_gain)
        head_weights.append(rng.uniform(-bound, bound, (head.size, hidden_dim)))
        center = np.asarray(head.bias_center, dtype=np.float64)
        jitter = rng.uniform(-head.bias_jitter, head.bias_jitter, head.size)
        head_biases.append(center + jitter)

    return np.concatenate(
        [fc1w.ravel(), fc1b, *(w.ravel() for w in head_weights), *head_biases]
    ).astype(np.float32)


def mutation_scale_vector(
    num_channels: int, hidden_dim: int, architecture: str = STATELESS_ARCHITECTURE
) -> np.ndarray:
    """Per-parameter multiplier for the CLI's global mutation sigma."""
    architecture = normalize_architecture(architecture)
    input_dim = policy_input_dim(num_channels, architecture)
    trunk_scale = float(_CONFIG["trunk"]["mutationScale"])
    chunks: list[np.ndarray] = [
        np.full(hidden_dim * input_dim, trunk_scale),
        np.full(hidden_dim, trunk_scale),
    ]
    heads = policy_heads(num_channels, architecture)
    chunks.extend(np.full(head.size * hidden_dim, head.mutation_scale) for head in heads)
    chunks.extend(np.full(head.size, head.mutation_scale) for head in heads)
    return np.concatenate(chunks).astype(np.float32)


def mutation_scales(architecture: str = STATELESS_ARCHITECTURE) -> dict[str, float]:
    """Human/metadata-friendly summary of the fixed scale buckets."""
    return {
        "trunk": float(_CONFIG["trunk"]["mutationScale"]),
        **{head.name: head.mutation_scale for head in policy_heads(1, architecture)},
    }

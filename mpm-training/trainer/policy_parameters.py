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


@dataclass(frozen=True)
class PolicyHead:
    name: str
    size: int
    weight_gain: float
    bias_center: tuple[float, ...]
    bias_jitter: float
    mutation_scale: float


def policy_heads(num_channels: int) -> tuple[PolicyHead, ...]:
    sizes = {
        "chemical": num_channels,
        "heading": 2,
        "anisotropy": 1,
        "division": 1,
        "growthDirection": 2,
        "color": 3,
    }
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
) -> np.ndarray:
    """Create the canonical fc1w/fc1b/fc2w/fc2b flat GPU layout."""
    input_dim = 3 * num_channels + 6
    trunk_gain, trunk_bias_jitter = trunk_initialization()
    trunk_bound = _xavier_bound(input_dim, hidden_dim, trunk_gain)
    fc1w = rng.uniform(-trunk_bound, trunk_bound, (hidden_dim, input_dim))
    fc1b = rng.uniform(-trunk_bias_jitter, trunk_bias_jitter, hidden_dim)

    head_weights: list[np.ndarray] = []
    head_biases: list[np.ndarray] = []
    for head in policy_heads(num_channels):
        bound = _xavier_bound(hidden_dim, head.size, head.weight_gain)
        head_weights.append(rng.uniform(-bound, bound, (head.size, hidden_dim)))
        center = np.asarray(head.bias_center, dtype=np.float64)
        jitter = rng.uniform(-head.bias_jitter, head.bias_jitter, head.size)
        head_biases.append(center + jitter)

    return np.concatenate(
        [fc1w.ravel(), fc1b, *(w.ravel() for w in head_weights), *head_biases]
    ).astype(np.float32)


def mutation_scale_vector(num_channels: int, hidden_dim: int) -> np.ndarray:
    """Per-parameter multiplier for the CLI's global mutation sigma."""
    input_dim = 3 * num_channels + 6
    trunk_scale = float(_CONFIG["trunk"]["mutationScale"])
    chunks: list[np.ndarray] = [
        np.full(hidden_dim * input_dim, trunk_scale),
        np.full(hidden_dim, trunk_scale),
    ]
    heads = policy_heads(num_channels)
    chunks.extend(np.full(head.size * hidden_dim, head.mutation_scale) for head in heads)
    chunks.extend(np.full(head.size, head.mutation_scale) for head in heads)
    return np.concatenate(chunks).astype(np.float32)


def mutation_scales() -> dict[str, float]:
    """Human/metadata-friendly summary of the fixed scale buckets."""
    return {
        "trunk": float(_CONFIG["trunk"]["mutationScale"]),
        **{head.name: head.mutation_scale for head in policy_heads(1)},
    }

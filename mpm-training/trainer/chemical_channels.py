"""Chemical channel profiles and packed grid layout, resolved from core/config.json."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from config import CONFIG
_CONFIG = CONFIG["chemistry"]

@dataclass(frozen=True)
class ChemicalChannelProfile:
    scale: str
    resolution_scale: float = _CONFIG["homogeneous"]["resolutionScale"]
    relaxation_time: float = _CONFIG["homogeneous"]["relaxationTime"]
    field_response_time: float = _CONFIG["homogeneous"]["fieldResponseTime"]
    decay_exponent: float = _CONFIG["homogeneous"]["decayExponent"]
    diffusion_multiplier: float = _CONFIG["homogeneous"]["diffusionMultiplier"]
    role: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "resolutionScale": self.resolution_scale,
            "relaxationTime": self.relaxation_time,
            "fieldResponseTime": self.field_response_time,
            "decayExponent": self.decay_exponent,
            "diffusionMultiplier": self.diffusion_multiplier,
            **({"role": self.role} if self.role is not None else {}),
        }

def _positive(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"chemical channel {name} must be positive, got {result}")
    return result

def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"chemical channel {name} must be nonnegative, got {result}")
    return result

def _parse_profile(raw: dict[str, Any], defaults: dict[str, Any] | None = None) -> ChemicalChannelProfile:
    merged = {**_CONFIG["homogeneous"], **(defaults or {}), **raw}
    return ChemicalChannelProfile(
        scale=str(merged.get("scale", "local")),
        resolution_scale=_positive(merged["resolutionScale"], "resolutionScale"),
        relaxation_time=_positive(merged["relaxationTime"], "relaxationTime"),
        field_response_time=_positive(merged["fieldResponseTime"], "fieldResponseTime"),
        decay_exponent=_positive(merged["decayExponent"], "decayExponent"),
        diffusion_multiplier=_nonnegative(merged["diffusionMultiplier"], "diffusionMultiplier"),
        role=str(merged["role"]) if merged.get("role") is not None else None,
    )

def default_channel_profiles(channels: int) -> tuple[ChemicalChannelProfile, ...]:
    if channels <= 0:
        raise ValueError("chemical channel count must be positive")
    configured = _CONFIG["channels"]
    if channels != len(configured):
        return tuple(ChemicalChannelProfile(scale="local") for _ in range(channels))
    scale_defaults = _CONFIG["profiles"]
    return tuple(_parse_profile(raw, scale_defaults.get(raw["scale"], {})) for raw in configured)

def homogeneous_channel_profiles(channels: int) -> tuple[ChemicalChannelProfile, ...]:
    if channels <= 0:
        raise ValueError("chemical channel count must be positive")
    return tuple(ChemicalChannelProfile(scale="local") for _ in range(channels))

def resolve_channel_profiles(
    channels: int,
    raw_profiles: Iterable[dict[str, Any] | ChemicalChannelProfile] | None = None,
) -> tuple[ChemicalChannelProfile, ...]:
    profiles = default_channel_profiles(channels) if raw_profiles is None else tuple(
        raw if isinstance(raw, ChemicalChannelProfile) else _parse_profile(dict(raw))
        for raw in raw_profiles
    )
    if len(profiles) != channels:
        raise ValueError(f"expected {channels} chemical channel profiles, got {len(profiles)}")
    return profiles

def profiles_to_wire(profiles: Iterable[ChemicalChannelProfile]) -> list[dict[str, Any]]:
    return [profile.to_wire() for profile in profiles]

def resolved_dimensions(
    width: int,
    height: int,
    profiles: Iterable[ChemicalChannelProfile],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if width <= 0 or height <= 0:
        raise ValueError("chemical field dimensions must be positive")
    profiles = tuple(profiles)
    widths = tuple(max(1, round(width * profile.resolution_scale)) for profile in profiles)
    heights = tuple(max(1, round(height * profile.resolution_scale)) for profile in profiles)
    return widths, heights

def packed_offsets(widths: Iterable[int], heights: Iterable[int]) -> tuple[tuple[int, ...], int]:
    offsets: list[int] = []
    total = 0
    for width, height in zip(widths, heights, strict=True):
        offsets.append(total)
        total += int(width) * int(height)
    return tuple(offsets), total

def _wgsl_array(kind: str, values: Iterable[int | float]) -> str:
    values = tuple(values)
    suffix = "u" if kind == "u32" else ""
    body = ", ".join(f"{value}{suffix}" for value in values)
    return f"array<{kind}, {len(values)}>({body})"

def channel_shader_constants(
    width: int,
    height: int,
    profiles: Iterable[ChemicalChannelProfile],
) -> dict[str, object]:
    profiles = tuple(profiles)
    widths, heights = resolved_dimensions(width, height, profiles)
    offsets, total = packed_offsets(widths, heights)
    return {
        "FIELD_WIDTHS": _wgsl_array("u32", widths),
        "FIELD_HEIGHTS": _wgsl_array("u32", heights),
        "FIELD_OFFSETS": _wgsl_array("u32", offsets),
        "FIELD_RELAXATION_TIMES": _wgsl_array("f32", (p.relaxation_time for p in profiles)),
        "FIELD_RESPONSE_TIMES": _wgsl_array("f32", (p.field_response_time for p in profiles)),
        "FIELD_DECAY_EXPONENTS": _wgsl_array("f32", (p.decay_exponent for p in profiles)),
        "FIELD_DIFFUSION_MULTIPLIERS": _wgsl_array("f32", (p.diffusion_multiplier for p in profiles)),
        "FIELD_TOTAL": total,
        "FIELD_MAX_WIDTH": max(widths),
        "FIELD_MAX_HEIGHT": max(heights),
    }

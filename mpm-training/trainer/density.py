"""Pure particle-sampling-density resolution shared conceptually with the viewer.

The public multiplier changes numerical sampling density, not the material's
physical mass density.  All actual shader/rollout values are resolved here so a
training worker can switch density through uniform writes without reconstructing
its GPU pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

_MODEL = json.loads((Path(__file__).parent.parent / "core" / "density.json").read_text())

DENSITY_MODEL_VERSION: int = int(_MODEL["MODEL_VERSION"])
REFERENCE_SPACING: float = float(_MODEL["REFERENCE_SPACING"])
INITIAL_PACKING_SPACING_SCALE: float = float(_MODEL["INITIAL_PACKING_SPACING_SCALE"])
SPATIAL_RANDOM_CELLS: int = int(_MODEL["SPATIAL_RANDOM_CELLS"])
REPULSION_RADIUS_IN_CELLS: float = float(_MODEL["REPULSION_RADIUS_IN_CELLS"])
MIN_SUPPORTED_MULTIPLIER: float = float(_MODEL["MIN_SUPPORTED_MULTIPLIER"])
MAX_SUPPORTED_MULTIPLIER: float = float(_MODEL["MAX_SUPPORTED_MULTIPLIER"])


@dataclass(frozen=True)
class DensityReference:
    particle_cap: int
    initial_particles: int
    chemical_field_n: int
    particle_mass: float
    particle_volume: float
    deposit_sigma: float
    chemical_gradient_input_scale: float
    repulsion_strength: float
    repulsion_max_delta: float


@dataclass(frozen=True)
class ResolvedDensity:
    model_version: int
    multiplier: float
    spacing_scale: float
    spacing: float
    initial_particles: int
    particle_cap: int
    particle_mass: float
    particle_volume: float
    deposit_sigma: float
    chemical_projection_weight: float
    splat_radius: float
    chemical_gradient_input_scale: float
    repulsion_strength: float
    repulsion_max_delta: float

    def metadata(self) -> dict[str, int | float]:
        return {
            "density_model_version": self.model_version,
            "particle_density_multiplier": self.multiplier,
            "split_displacement": self.spacing,
            "initial_particle_count": self.initial_particles,
            "particles": self.particle_cap,
            "particle_mass": self.particle_mass,
            "particle_volume": self.particle_volume,
            "deposit_sigma": self.deposit_sigma,
            "chemical_projection_weight": self.chemical_projection_weight,
            "splat_radius": self.splat_radius,
            "chemical_gradient_input_scale": self.chemical_gradient_input_scale,
            "repulsion_strength": self.repulsion_strength,
            "repulsion_max_delta": self.repulsion_max_delta,
        }


def _round_positive(value: float) -> int:
    """Round non-negative values half-up, identically to TypeScript."""
    return math.floor(value + 0.5)


def validate_multiplier(multiplier: float, *, allow_unsafe: bool = False) -> float:
    q = float(multiplier)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(f"particle density multiplier must be finite and positive, got {multiplier!r}")
    if not allow_unsafe and not MIN_SUPPORTED_MULTIPLIER <= q <= MAX_SUPPORTED_MULTIPLIER:
        raise ValueError(
            f"particle density multiplier {q:g} is outside the supported range "
            f"[{MIN_SUPPORTED_MULTIPLIER:g}, {MAX_SUPPORTED_MULTIPLIER:g}]"
        )
    return q


def resolve_density(
    reference: DensityReference,
    multiplier: float,
    *,
    allow_unsafe: bool = False,
) -> ResolvedDensity:
    q = validate_multiplier(multiplier, allow_unsafe=allow_unsafe)
    if reference.particle_cap < 1 or reference.initial_particles < 1:
        raise ValueError("reference particle counts must be positive")
    if reference.initial_particles > reference.particle_cap:
        raise ValueError("reference initial particle count cannot exceed its cap")
    if reference.chemical_field_n < 1:
        raise ValueError("chemical field resolution must be positive")

    spacing_scale = 1.0 / math.sqrt(q)
    spacing = REFERENCE_SPACING * spacing_scale
    return ResolvedDensity(
        model_version=DENSITY_MODEL_VERSION,
        multiplier=q,
        spacing_scale=spacing_scale,
        spacing=spacing,
        initial_particles=max(1, _round_positive(reference.initial_particles * q)),
        particle_cap=max(1, _round_positive(reference.particle_cap * q)),
        particle_mass=reference.particle_mass / q,
        particle_volume=reference.particle_volume / q,
        # The chemical field is a fixed numerical observation grid.  Shrinking
        # the already-sub-texel q=1 kernel made higher-density particles alias
        # onto the same texels.  Keep its grid-space support fixed and weight
        # each particle by the represented material area instead.
        deposit_sigma=reference.deposit_sigma,
        chemical_projection_weight=1.0 / q,
        splat_radius=REPULSION_RADIUS_IN_CELLS * spacing,
        chemical_gradient_input_scale=reference.chemical_gradient_input_scale,
        repulsion_strength=reference.repulsion_strength * spacing_scale * spacing_scale,
        repulsion_max_delta=reference.repulsion_max_delta * spacing_scale,
    )


def parse_multipliers(values: list[float] | tuple[float, ...], *, allow_unsafe: bool = False) -> tuple[float, ...]:
    if not values:
        raise ValueError("at least one particle density multiplier is required")
    resolved = tuple(validate_multiplier(value, allow_unsafe=allow_unsafe) for value in values)
    if len(set(resolved)) != len(resolved):
        raise ValueError("particle density multipliers must be unique")
    return resolved


def resolve_checkpoint_density(
    metadata: Mapping[str, Any],
    reference: DensityReference,
    *,
    legacy_split_displacement: float,
    legacy_deposit_sigma: float,
    legacy_splat_radius: float,
) -> ResolvedDensity:
    """Resolve modern reference metadata while preserving legacy actual values."""
    density = resolve_density(
        reference,
        float(metadata.get("winner_density_multiplier", 1.0)),
        allow_unsafe=True,
    )
    # v3 changes stochastic forcing only; its resolved physical/chemical
    # scaling is identical to v2, so v2 checkpoints remain modern here.
    if int(metadata.get("density_model_version", 0)) in (2, DENSITY_MODEL_VERSION):
        return density
    return replace(
        density,
        multiplier=float(metadata.get("winner_density_multiplier", metadata.get("particle_density_multiplier", 1.0))),
        spacing_scale=1.0 / math.sqrt(float(metadata.get(
            "winner_density_multiplier", metadata.get("particle_density_multiplier", 1.0)
        ))),
        spacing=float(metadata.get("split_displacement", legacy_split_displacement)),
        initial_particles=int(metadata.get("initial_particle_count", reference.initial_particles)),
        particle_cap=int(metadata.get("particles", reference.particle_cap)),
        particle_mass=float(metadata.get("particle_mass", reference.particle_mass)),
        particle_volume=float(metadata.get("particle_volume", reference.particle_volume)),
        deposit_sigma=float(metadata.get("deposit_sigma", legacy_deposit_sigma)),
        chemical_projection_weight=float(metadata.get("chemical_projection_weight", 1.0)),
        splat_radius=float(metadata.get("splat_radius", legacy_splat_radius)),
        chemical_gradient_input_scale=float(metadata.get(
            "chemical_gradient_input_scale", reference.chemical_gradient_input_scale
        )),
        repulsion_strength=float(metadata.get("repulsion_strength", reference.repulsion_strength)),
        repulsion_max_delta=float(metadata.get("repulsion_max_delta", reference.repulsion_max_delta)),
    )

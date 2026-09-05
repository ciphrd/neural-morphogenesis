"""Particle-level morphoelastic diagnostics for reproducible comparisons.

Growth state is a full row-major 2x2 ``Fg`` and ``Fe = F inv(Fg)``. The
increment may be isotropic or network-directed; diagnostics make no isotropy
assumption. Raw F would incorrectly count stress-free growth as elastic strain.

Metric names describe physical quantities rather than a particular storage
layout, allowing preserved scalar and tensor snapshots to be compared.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mpm_core import PARTICLE_MASS, VOL, MpmCore, lame_params

PERCENTILES = (50, 90, 95, 99)


def policy_elastic_strain_input(
    deformation: np.ndarray,
    growth_f: np.ndarray,
    heading: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Reference implementation of agents.wgsl's three mechanosensory inputs.

    Returns tanh-normalized ``[volume, forward-minus-lateral, shear]``
    components of spatial elastic Hencky strain ``H=0.5 log(Fe Fe.T)`` in
    each particle's heading frame. This intentionally excludes rigid rotation
    and stress-free growth.
    """
    f = np.asarray(deformation, dtype=np.float64).reshape(-1, 2, 2)
    fg = np.asarray(growth_f, dtype=np.float64).reshape(-1, 2, 2)
    angles = np.asarray(heading, dtype=np.float64).reshape(-1)
    if f.shape != fg.shape or len(f) != len(angles):
        raise ValueError("deformation, growth_f, and heading counts must match")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    fe = f @ np.linalg.inv(fg)
    b = fe @ np.swapaxes(fe, 1, 2)
    out = np.empty((len(f), 3), dtype=np.float64)
    for i, angle in enumerate(angles):
        eigenvalues, eigenvectors = np.linalg.eigh(b[i])
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        h = (eigenvectors * (0.5 * np.log(eigenvalues))) @ eigenvectors.T
        forward = np.array([np.cos(angle), np.sin(angle)])
        lateral = np.array([-np.sin(angle), np.cos(angle)])
        q = np.column_stack([forward, lateral])
        local = q.T @ h @ q
        out[i] = np.tanh(
            np.array([local.trace(), local[0, 0] - local[1, 1], 2.0 * local[0, 1]]) / scale
        )
    return out


@dataclass(frozen=True)
class ElasticParticleState:
    elastic_f: np.ndarray
    elastic_volume_ratio: np.ndarray
    principal_stretch_max: np.ndarray
    principal_stretch_min: np.ndarray
    log_areal_strain: np.ndarray
    deviatoric_log_strain: np.ndarray
    corotated_strain: np.ndarray
    pressure: np.ndarray
    elastic_energy_density: np.ndarray
    elastic_energy: np.ndarray
    growth_area_ratio: np.ndarray
    quadrature_weight: np.ndarray
    growth_f: np.ndarray
    growth_principal_stretch_max: np.ndarray
    growth_principal_stretch_min: np.ndarray
    growth_deviatoric_log_strain: np.ndarray
    growth_direction: np.ndarray
    growth_anisotropy: np.ndarray
    division_bias: np.ndarray
    plastic_jacobian: np.ndarray
    cycle_active: np.ndarray


def particle_elastic_state(
    deformation: np.ndarray,
    rest_state: np.ndarray,
    *,
    material_e: float,
    material_nu: float,
    material_hardening: float,
    particle_volume: float = VOL,
) -> ElasticParticleState:
    """Compute per-particle quantities used by the tensor-growth material.

    ``deviatoric_log_strain`` is ``||dev(log Ue)||_F``. In 2D this is
    ``abs(log(s_max/s_min))/sqrt(2)``. It cleanly separates anisotropic
    residual stretch from area change. ``corotated_strain`` is
    ``||Fe-Re||_F``, identical to the viewer's shear diagnostic.

    Elastic energy density is the fixed-corotated potential whose derivative
    produces core/p2g.wgsl's stress:
    ``mu ||Fe-Re||² + lambda/2 (Je-1)²``. ``elastic_energy`` multiplies it by
    the sample's represented grown rest area ``VOL*q*g``.
    """
    f = np.asarray(deformation, dtype=np.float64)
    rest = np.asarray(rest_state, dtype=np.float64)
    if f.ndim == 2 and f.shape[1] == 4:
        f = f.reshape(-1, 2, 2)
    if f.ndim != 3 or f.shape[1:] != (2, 2):
        raise ValueError(f"deformation must have shape (n,4) or (n,2,2), got {f.shape}")
    if rest.ndim != 2 or (rest.shape[0] != f.shape[0] or rest.shape[1] not in (12, 16)):
        raise ValueError(f"rest_state must have shape ({f.shape[0]},12), got {rest.shape}")
    if not np.isfinite(f).all() or not np.isfinite(rest).all():
        raise ValueError("deformation and rest_state must be finite")

    fg = rest[:, :4].reshape(-1, 2, 2)
    jp = rest[:, 4]
    growth = np.linalg.det(fg)
    quadrature_weight = rest[:, 11]
    if np.any(growth <= 0.0):
        raise ValueError("growth area ratios must be strictly positive")
    if np.any(quadrature_weight <= 0.0):
        raise ValueError("quadrature weights must be strictly positive")

    fe = f @ np.linalg.inv(fg)
    growth_singular = np.linalg.svd(fg, compute_uv=False)
    growth_s_max = growth_singular[:, 0]
    growth_s_min = growth_singular[:, 1]
    growth_dev_log = np.abs(np.log(growth_s_max / growth_s_min)) / np.sqrt(2.0)
    je = np.linalg.det(fe)
    singular = np.linalg.svd(fe, compute_uv=False)
    s_max = singular[:, 0]
    s_min = singular[:, 1]
    safe_max = np.maximum(s_max, np.finfo(np.float64).tiny)
    safe_min = np.maximum(s_min, np.finfo(np.float64).tiny)
    log_area = np.log(np.maximum(np.abs(je), np.finfo(np.float64).tiny))
    dev_log = np.abs(np.log(safe_max / safe_min)) / np.sqrt(2.0)
    corotated = np.sqrt((s_max - 1.0) ** 2 + (s_min - 1.0) ** 2)

    mu0, lambda0 = lame_params(material_e, material_nu)
    hardening_scale = np.exp(material_hardening * (1.0 - jp))
    mu = mu0 * hardening_scale
    lam = lambda0 * hardening_scale
    pressure = -lam * (je - 1.0)
    energy_density = mu * corotated**2 + 0.5 * lam * (je - 1.0) ** 2
    energy = particle_volume * quadrature_weight * growth * energy_density

    return ElasticParticleState(
        elastic_f=fe,
        elastic_volume_ratio=je,
        principal_stretch_max=s_max,
        principal_stretch_min=s_min,
        log_areal_strain=log_area,
        deviatoric_log_strain=dev_log,
        corotated_strain=corotated,
        pressure=pressure,
        elastic_energy_density=energy_density,
        elastic_energy=energy,
        growth_area_ratio=growth,
        quadrature_weight=quadrature_weight,
        growth_f=fg,
        growth_principal_stretch_max=growth_s_max,
        growth_principal_stretch_min=growth_s_min,
        growth_deviatoric_log_strain=growth_dev_log,
        growth_direction=np.column_stack((
            np.cos(rest[:, 6] + rest[:, 9]),
            np.sin(rest[:, 6] + rest[:, 9]),
        )),
        growth_anisotropy=rest[:, 7],
        division_bias=rest[:, 8],
        plastic_jacobian=jp,
        cycle_active=rest[:, 5],
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = percentile / 100.0 * cumulative[-1]
    return float(sorted_values[min(np.searchsorted(cumulative, target, side="left"), len(values) - 1)])


def distribution_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Stable grown-mass-weighted summary used in JSON baseline files."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.shape != weights.shape:
        raise ValueError("values and weights must be non-empty vectors with the same shape")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("summary values must be finite and weights strictly positive")
    total_weight = float(weights.sum())
    out = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(np.sum(values * weights) / total_weight),
        "rms": float(np.sqrt(np.sum(values**2 * weights) / total_weight)),
    }
    out.update({f"p{p}": _weighted_percentile(values, weights, p) for p in PERCENTILES})
    return out


def summarize_elastic_state(
    state: ElasticParticleState,
    *,
    velocities: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    particle_mass: float = PARTICLE_MASS,
    particle_volume: float = VOL,
) -> dict[str, Any]:
    represented_area = state.quadrature_weight * state.growth_area_ratio
    weights = particle_mass * represented_area
    metrics = {
        "elastic_volume_ratio": state.elastic_volume_ratio,
        "principal_stretch_max": state.principal_stretch_max,
        "principal_stretch_min": state.principal_stretch_min,
        "log_areal_strain": state.log_areal_strain,
        "deviatoric_log_strain": state.deviatoric_log_strain,
        "corotated_strain": state.corotated_strain,
        "pressure": state.pressure,
        "absolute_pressure": np.abs(state.pressure),
        "elastic_energy_density": state.elastic_energy_density,
        "growth_area_ratio": state.growth_area_ratio,
        "quadrature_weight": state.quadrature_weight,
        "represented_area_ratio": represented_area,
        "growth_principal_stretch_max": state.growth_principal_stretch_max,
        "growth_principal_stretch_min": state.growth_principal_stretch_min,
        "growth_deviatoric_log_strain": state.growth_deviatoric_log_strain,
        "growth_direction_magnitude": np.linalg.norm(state.growth_direction, axis=1),
        "growth_anisotropy": state.growth_anisotropy,
        "division_bias": state.division_bias,
        "plastic_jacobian": state.plastic_jacobian,
    }
    kinetic_energy: float | None = None
    if velocities is not None:
        velocity = np.asarray(velocities, dtype=np.float64)
        if velocity.shape != (weights.size, 2) or not np.isfinite(velocity).all():
            raise ValueError(f"velocities must have finite shape ({weights.size},2)")
        speed = np.linalg.norm(velocity, axis=1)
        metrics["speed"] = speed
        kinetic_energy = float(np.sum(0.5 * weights * speed**2))

    geometry: dict[str, Any] | None = None
    if positions is not None:
        position = np.asarray(positions, dtype=np.float64)
        if position.shape != (weights.size, 2) or not np.isfinite(position).all():
            raise ValueError(f"positions must have finite shape ({weights.size},2)")
        # Circular mean and minimum-image offsets respect the simulation's
        # toroidal domain, unlike an ordinary centroid near a wrapped edge.
        angles = position * (2.0 * np.pi)
        center = np.mod(
            np.arctan2(
                np.sum(np.sin(angles) * weights[:, None], axis=0),
                np.sum(np.cos(angles) * weights[:, None], axis=0),
            )
            / (2.0 * np.pi),
            1.0,
        )
        offsets = (position - center + 0.5) % 1.0 - 0.5
        radius = np.linalg.norm(offsets, axis=1)
        geometry = {
            "toroidal_center": [float(center[0]), float(center[1])],
            "rms_radius": float(np.sqrt(np.sum(radius**2 * weights) / weights.sum())),
            "radius": distribution_summary(radius, weights),
        }

    result = {
        "particle_count": int(weights.size),
        "active_cycle_count": int(np.count_nonzero(state.cycle_active > 0.5)),
        "total_rest_area": float(particle_volume * represented_area.sum()),
        "total_mass": float(weights.sum()),
        "total_elastic_energy": float(state.elastic_energy.sum()),
        "metrics": {name: distribution_summary(values, weights) for name, values in metrics.items()},
    }
    if kinetic_energy is not None:
        result["total_kinetic_energy"] = kinetic_energy
    if geometry is not None:
        result["geometry"] = geometry
    return result


def measure_core(
    core: MpmCore,
    *,
    material_e: float,
    material_nu: float,
    material_hardening: float,
) -> dict[str, Any]:
    state = particle_elastic_state(
        core.read_deformation(),
        core.read_rest_state(),
        material_e=material_e,
        material_nu=material_nu,
        material_hardening=material_hardening,
        particle_volume=core.particle_volume,
    )
    return summarize_elastic_state(
        state,
        velocities=core.read_velocities(),
        positions=core.read_positions(),
        particle_mass=core.particle_mass,
        particle_volume=core.particle_volume,
    )

"""Focused validation for the packed multi-scale chemical field."""
from __future__ import annotations

import numpy as np

from agents_gpu import AgentsGPU, weight_layout
from chemical_channels import (
    channel_shader_constants,
    default_channel_profiles,
    packed_offsets,
    profiles_to_wire,
    resolve_channel_profiles,
    resolved_dimensions,
)
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore
from policy_parameters import STATEFUL_128_ARCHITECTURE
from simulation_settings import (
    ANGULAR_DAMPING,
    CHEM_CHANNELS,
    CHEMICAL_CHANNEL_PROFILES,
    CHIRALITY,
    DECAY,
    DEPOSIT_DISTANCE,
    DEPOSIT_RATE,
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
from training_sim import TrainingRollout


def check_layout() -> None:
    profiles = default_channel_profiles(CHEM_CHANNELS)
    assert [profile.scale for profile in profiles] == [
        "global", "global", "global", "regional", "regional", "regional", "local", "local", "local"
    ]
    assert all(profile.role is None for profile in profiles)
    widths, heights = resolved_dimensions(512, 512, profiles)
    assert widths == heights
    assert widths[0] == widths[1] == widths[2]
    assert widths[3] == widths[4] == widths[5]
    assert widths[6] == widths[7] == widths[8]
    assert 1 <= widths[0] < widths[3] < widths[6] == 512
    offsets, total = packed_offsets(widths, heights)
    expected_offsets = []
    expected_total = 0
    for width, height in zip(widths, heights, strict=True):
        expected_offsets.append(expected_total)
        expected_total += width * height
    assert offsets == tuple(expected_offsets)
    assert total == expected_total
    wire = profiles_to_wire(profiles)
    assert resolve_channel_profiles(CHEM_CHANNELS, wire) == profiles
    constants = channel_shader_constants(512, 512, profiles)
    assert constants["FIELD_TOTAL"] == total
    assert constants["FIELD_MAX_WIDTH"] == 512
    print("[PASS] 3/3/3 profiles resolve and round-trip through run metadata")


def check_gpu_pipelines() -> None:
    device = pick_device()
    layout = weight_layout(CHEM_CHANNELS, 128, STATEFUL_128_ARCHITECTURE)
    weights = np.zeros(layout["total_floats"], dtype=np.float32)
    weights[layout["fc2b_offset"]:layout["fc2b_offset"] + CHEM_CHANNELS] = 0.25
    for architecture in ("persistent-environment", "cell-owned-projection"):
        core = MpmCore(device)
        environment = EnvironmentGPU(
            device, CHEM_CHANNELS, FIELD_N, FIELD_N, DECAY, DEPOSIT_RATE,
            architecture, grid_velocity=core.grid_vel,
            channel_profiles=CHEMICAL_CHANNEL_PROFILES,
        )
        agents = AgentsGPU(
            device, core, environment, CHEM_CHANNELS, 128,
            MAX_ACCEL, MAX_STRAFE, MAX_ENV_WRITE, MAX_ANGULAR_ACCEL,
            ANGULAR_DAMPING, MAX_ANGULAR_VELOCITY, CHIRALITY,
            DEPOSIT_DISTANCE, 16, SPLIT_DISPLACEMENT, DIVISION_COOLDOWN,
            FRICTION, DEPOSIT_SIGMA, 1.0, 0.5, 0.5,
            policy_architecture=STATEFUL_128_ARCHITECTURE,
            chemical_communication_architecture=architecture,
        )
        assert environment.buffers[0].size == environment.total_values * 4
        assert environment.deposit_scratch.size == (environment.total_values * 2 + 1) * 4
        agents.load_weights(weights)
        sim = TrainingRollout(
            core, agents, environment, spawn_center=(0.5, 0.5), spawn_half_width=0.0,
            gravity=0.0, seed=7, mpm_enabled=False, initial_particle_count=1,
        )
        # Cell-owned mode publishes the state produced by the first step on
        # the second step; persistent mode is already nonzero after one.
        for _ in range(2 if architecture == "cell-owned-projection" else 1):
            sim.macro_step(1, growth_enabled=False)
        raw = device.queue.read_buffer(environment.buffers[environment.parity])
        field = np.frombuffer(raw, dtype=np.float32)
        for offset, width, height in zip(
            environment.channel_offsets, environment.channel_widths, environment.channel_heights,
            strict=True,
        ):
            plane = field[offset:offset + width * height]
            assert np.isfinite(plane).all() and np.max(plane) > 0.0
    print("[PASS] both chemical lifecycles update every packed native grid")


def main() -> None:
    check_layout()
    check_gpu_pipelines()


if __name__ == "__main__":
    main()

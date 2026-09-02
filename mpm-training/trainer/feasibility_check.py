"""Runnable feasibility spike, not a test framework — proves the one
thing this whole trainer/ package rests on: that ../core/'s WGSL physics
runs correctly, headlessly, via wgpu-py on this machine. See
../README.md and the top-level plan for context.

Usage:
    python feasibility_check.py

Checks, in order (each prints an explicit PASS/FAIL/SKIP verdict):
  1. Atomics smoke test — standalone, before touching the real pipeline.
  2. No-op-gravity sanity.
  3. Settling under gravity (blocks.ts-style reference config, repulsion
     on at its default strength — repulsion is a standard, always-on part
     of the simulation here, not something toggled for testing).
  4. Cross-check against the mls-mpm browser sandbox — MANUAL. This
     machine's session has no browser-automation tool available, so this
     check cannot run automatically here: it compares against a small
     reference JSON (see CROSS_CHECK_REFERENCE_PATH below) that a human
     has to capture by hand from the actual browser sandbox first. Prints
     SKIP with instructions if that file doesn't exist yet, rather than
     silently passing or crashing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import wgpu

from device import pick_device
from mpm_core import DT, MAX_PARTICLES, MpmCore
from simulation_settings import MATERIAL_E, MATERIAL_HARDENING, MATERIAL_NU, SUBSTEPS_PER_DAMPING_FRAME

CROSS_CHECK_REFERENCE_PATH = Path(__file__).parent / "cross_check_reference.json"

RESULTS: list[tuple[str, str]] = []  # (name, verdict) — printed as a summary at the end


def record(name: str, ok: bool | None, detail: str) -> None:
    verdict = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((name, verdict))
    print(f"[{verdict}] {name}: {detail}")


# --- Check 1: atomics smoke test ---------------------------------------


def check_atomics(device: wgpu.GPUDevice) -> bool:
    n = 10_000
    shader = device.create_shader_module(
        code=f"""
        const N: u32 = {n}u;
        @group(0) @binding(0) var<storage, read_write> counter: array<atomic<i32>>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
          if (gid.x >= N) {{ return; }}
          atomicAdd(&counter[0], 1);
        }}
        """
    )
    buf = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    device.queue.write_buffer(buf, 0, np.array([0], dtype=np.int32))

    pipeline = device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": shader, "entry_point": "main"})
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0), entries=[{"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": 4}}]
    )

    encoder = device.create_command_encoder()
    p = encoder.begin_compute_pass()
    p.set_pipeline(pipeline)
    p.set_bind_group(0, bind_group)
    p.dispatch_workgroups((n + 63) // 64)
    p.end()
    device.queue.submit([encoder.finish()])

    result = int(np.frombuffer(device.queue.read_buffer(buf), dtype=np.int32)[0])
    ok = result == n
    record("atomics", ok, f"expected {n}, got {result}")
    return ok


# --- Scene seeding (mirrors mls-mpm/src/worlds/util.ts's allocateScene/setRestState) ---


def seed_blob(count: int, center: tuple[float, float], half_width: float, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    positions = center + rng.uniform(-half_width, half_width, size=(count, 2))
    velocities = np.zeros((count, 2), dtype=np.float32)
    F = np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (count, 1))  # identity
    C = np.zeros((count, 4), dtype=np.float32)
    Jp = np.ones((count,), dtype=np.float32)
    return positions.astype(np.float32), velocities, F, C, Jp


# --- Check 2: no-op-gravity sanity --------------------------------------


def check_no_gravity_drift(device: wgpu.GPUDevice) -> bool:
    core = MpmCore(device)
    core.set_gravity(0.0)
    core.set_material(MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, elasticity=0.0)

    rng = np.random.default_rng(seed=0)
    positions, velocities, F, C, Jp = seed_blob(500, (0.5, 0.5), 0.08, rng)
    start = positions.copy()
    core.load_scene(positions, velocities, F, C, Jp)

    core.step(50)
    pos = core.read_positions()

    finite = bool(np.isfinite(pos).all())
    in_bounds = bool(np.all((pos >= 0.0) & (pos <= 1.0)))
    mean_drift = float(np.mean(np.linalg.norm(pos - start, axis=1)))
    ok = finite and in_bounds and mean_drift < 0.05
    record(
        "no_gravity_drift",
        ok,
        f"finite={finite} in_bounds={in_bounds} mean_drift={mean_drift:.5f} (threshold 0.05)",
    )
    return ok


# --- Check 3: settling under center gravity -----------------------------


def check_settle(device: wgpu.GPUDevice) -> bool:
    # Start above center and verify the radial gravity field draws the blob
    # inward, then remains bounded rather than drifting around the torus.
    core = MpmCore(device)
    core.set_gravity(200.0)
    core.set_material(MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, elasticity=0.0)
    # Repulsion left at its constructor defaults (simulation_settings.py's
    # SPLAT_RADIUS/REPULSION_STRENGTH) — on by default, not toggled for this
    # check, same as the real simulation always runs it.

    rng = np.random.default_rng(seed=1)
    count = 2000
    positions, velocities, F, C, Jp = seed_blob(count, (0.5, 0.8), 0.08, rng)
    core.load_scene(positions, velocities, F, C, Jp)

    total_substeps = 4000
    batch = SUBSTEPS_PER_DAMPING_FRAME * 25  # 200 substeps/batch at SUBSTEPS_PER_DAMPING_FRAME=8
    batches = total_substeps // batch

    mean_radii: list[float] = []
    positions_by_batch: list[np.ndarray] = []
    all_finite = True
    all_in_bounds = True
    min_pos = 0.015625  # DX, see core/g2p.wgsl's MIN_POS
    max_pos = 1.0 - min_pos

    for i in range(batches):
        core.step(batch)
        pos = core.read_positions()
        vel = core.read_velocities()
        all_finite = all_finite and bool(np.isfinite(pos).all()) and bool(np.isfinite(vel).all())
        all_in_bounds = all_in_bounds and bool(np.all((pos >= min_pos - 1e-6) & (pos <= max_pos + 1e-6)))
        mean_radius = float(np.mean(np.linalg.norm(pos - 0.5, axis=1)))
        mean_radii.append(mean_radius)
        positions_by_batch.append(pos)
        if i % 4 == 0 or i == batches - 1:
            print(f"    step {(i + 1) * batch:5d}  mean_radius={mean_radius:.4f}  mean_speed={float(np.mean(np.linalg.norm(vel, axis=1))):.5f}")

    # NOTE on the criterion below: raw velocity magnitude does NOT settle
    # to near-zero here even once the blob is visibly center-bounded —
    # confirmed by direct investigation (see mpm_core.py's own repulsion
    # comment): repulsion.wgsl's direct-position-write push has no decay
    # mechanism, so it keeps nudging already-settled particles back and
    # forth every substep, which reads as persistent nonzero "speed" (an
    # oscillating ~0.2-0.35 in this config) despite near-zero *net*
    # displacement. What actually indicates "settled" in this simulation
    # (repulsion on, as it always is) is bounded, small displacement over
    # a trailing window — checked directly below — not an absolute
    # low-velocity threshold, which would be the wrong signal here and
    # was confirmed to be by testing (an earlier version of this check
    # used speed < 0.01 and failed despite position being flat to
    # within 0.002 over the same window).
    tail_disp = np.linalg.norm(positions_by_batch[-1] - positions_by_batch[-5], axis=1)
    mean_tail_disp = float(np.mean(tail_disp))
    max_speed = float(np.max(np.linalg.norm(core.read_velocities(), axis=1)))

    decreased = mean_radii[-1] < mean_radii[0]
    settled = mean_tail_disp < 0.02  # over the last 4 batches (800 substeps)
    not_exploding = max_speed < 20.0  # loose sanity bound, not a "calm" threshold — see note above

    ok = all_finite and all_in_bounds and decreased and settled and not_exploding
    record(
        "settle_under_center_gravity",
        ok,
        f"finite={all_finite} in_bounds={all_in_bounds} radius_decreased={decreased} "
        f"mean_tail_displacement={mean_tail_disp:.5f} (threshold 0.02) max_speed={max_speed:.3f} (sanity bound 20.0)",
    )
    return ok


# --- Check 4: cross-check against the browser sandbox (manual) ---------


def check_cross_reference(device: wgpu.GPUDevice) -> bool | None:
    if not CROSS_CHECK_REFERENCE_PATH.exists():
        print(
            "    No browser-captured reference found at "
            f"{CROSS_CHECK_REFERENCE_PATH.name} — this check needs a human to manually\n"
            "    capture particle positions from the actual mls-mpm browser sandbox (see\n"
            "    mls-mpm/src/main.ts's own TEMP DEBUG HOOK, window.__mpmDebug) before and\n"
            "    after a fixed substep count, and save them as JSON:\n"
            '    {"gravity": 200.0, "substeps": N, "initial_positions": [[x,y],...],\n'
            '     "final_positions": [[x,y],...]}.\n'
            "    This automated session has no browser-automation tool available, so this\n"
            "    step could not be completed here — checks 1-3 do not substitute for it."
        )
        record("cross_check_vs_browser", None, "SKIPPED — no reference file, needs manual capture (see above)")
        return None

    reference = json.loads(CROSS_CHECK_REFERENCE_PATH.read_text())
    substeps = int(reference["substeps"])
    gravity = float(reference["gravity"])
    initial = np.array(reference["initial_positions"], dtype=np.float32)
    browser_final = np.array(reference["final_positions"], dtype=np.float32)
    count = initial.shape[0]
    assert browser_final.shape == initial.shape, "initial_positions/final_positions count mismatch"

    core = MpmCore(device)
    core.set_gravity(gravity)
    core.set_material(MATERIAL_E, MATERIAL_NU, MATERIAL_HARDENING, elasticity=0.0)
    # Repulsion left at MpmCore's own constructor defaults — same as the
    # browser sandbox's own defaults, untouched during manual capture.
    velocities = np.zeros((count, 2), dtype=np.float32)
    F = np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (count, 1))
    C = np.zeros((count, 4), dtype=np.float32)
    Jp = np.ones((count,), dtype=np.float32)
    core.load_scene(initial, velocities, F, C, Jp)

    core.step(substeps)
    python_final = core.read_positions()

    delta = np.linalg.norm(python_final - browser_final, axis=1)
    mean_delta = float(np.mean(delta))
    max_delta = float(np.max(delta))
    # Cross-backend (naga vs tint), not cross-run-same-backend — envnca's
    # own README documents ~1e-6 relative drift for the SAME backend
    # across runs; two independently-compiled shader backends accumulating
    # float32 rounding differently over `substeps` nonlinear iterations
    # can plausibly diverge more than that, so this tolerance is a "same
    # ballpark, not a fluke" check, not a bit-exactness claim — mean/max
    # are both printed regardless of verdict so a human can judge the
    # actual magnitude themselves rather than trusting this cutoff blindly.
    ok = mean_delta < 0.01 and max_delta < 0.05
    record(
        "cross_check_vs_browser",
        ok,
        f"substeps={substeps} gravity={gravity} mean_delta={mean_delta:.5f} (threshold 0.01) "
        f"max_delta={max_delta:.5f} (threshold 0.05)",
    )
    return ok


def main() -> int:
    device = pick_device()

    print("\n--- 1. atomics ---")
    atomics_ok = check_atomics(device)
    if not atomics_ok:
        print("\nAtomics failed — this is a hard gate, stopping here (see plan's own Risks section).")
        print_summary()
        return 1

    print("\n--- 2. no-gravity drift ---")
    check_no_gravity_drift(device)

    print("\n--- 3. settle under gravity ---")
    check_settle(device)

    print("\n--- 4. cross-check vs browser (manual) ---")
    check_cross_reference(device)

    print_summary()
    return 0 if all(v != "FAIL" for _, v in RESULTS) else 1


def print_summary() -> None:
    print("\n=== Summary ===")
    for name, verdict in RESULTS:
        print(f"  {verdict:5s}  {name}")


if __name__ == "__main__":
    sys.exit(main())

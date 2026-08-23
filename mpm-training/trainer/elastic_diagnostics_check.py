"""Focused checks for elastic_diagnostics.py.

Run from trainer/ with ``.venv/bin/python elastic_diagnostics_check.py``.
The GPU cross-check independently evaluates the same constitutive quantities
in WGSL, catching buffer-layout or CPU/GPU formula drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import wgpu

from device import pick_device
from elastic_diagnostics import distribution_summary, measure_core, particle_elastic_state, summarize_elastic_state
from mpm_core import MpmCore, lame_params
from shader_template import template_shader

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "tensor_growth_isotropic_equivalence.json"
SCALAR_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "scalar_growth_elastic_baseline.json"
DIRECTIONAL_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "tensor_growth_directional_strafe.json"
E = 1.0e4
NU = 0.2
HARDENING = 3.0


def check_analytic_invariants() -> None:
    theta = 0.73
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    g = 1.7
    deformation = list(
        [
            np.eye(2),
            rotation,
            np.sqrt(g) * rotation,
            np.diag([2.0, 0.5]),
            0.8 * np.eye(2),
        ]
    )
    anisotropic_fg = np.array([[1.3, 0.2], [0.1, 0.9]])
    known_fe = np.array([[1.08, 0.04], [-0.03, 0.94]])
    deformation.append(known_fe @ anisotropic_fg)
    deformation = np.stack(deformation)
    rest = np.zeros((6, 12))
    rest[:, 0] = 1.0
    rest[:, 3] = 1.0
    rest[:, 4] = 1.0
    rest[2, :4] = np.array([np.sqrt(g), 0.0, 0.0, np.sqrt(g)])
    rest[4, 5] = 1.0
    rest[5, :4] = anisotropic_fg.reshape(4)
    state = particle_elastic_state(
        deformation,
        rest,
        material_e=E,
        material_nu=NU,
        material_hardening=HARDENING,
    )
    # Identity, rotation, and stress-free isotropic growth all have Fe=R.
    assert np.allclose(state.elastic_volume_ratio[:3], 1.0, atol=1e-12)
    assert np.allclose(state.corotated_strain[:3], 0.0, atol=1e-12)
    assert np.allclose(state.elastic_energy_density[:3], 0.0, atol=1e-20)
    # Area-preserving anisotropy has no volumetric strain but nonzero shear.
    assert np.isclose(state.elastic_volume_ratio[3], 1.0)
    assert np.isclose(state.log_areal_strain[3], 0.0)
    assert np.isclose(state.deviatoric_log_strain[3], np.log(4.0) / np.sqrt(2.0))
    assert state.elastic_energy_density[3] > 0.0
    # Isotropic compression has zero deviatoric strain and positive pressure.
    assert np.isclose(state.deviatoric_log_strain[4], 0.0)
    assert state.pressure[4] > 0.0
    assert np.allclose(state.elastic_f[5], known_fe, atol=1e-12)
    assert state.growth_deviatoric_log_strain[5] > 0.0
    assert np.allclose(state.growth_deviatoric_log_strain[:5], 0.0, atol=1e-12)
    print("[PASS] analytic invariants: rotation/growth stress-free, tensor decomposition, volumetric/deviatoric separation")


def check_weighted_summary() -> None:
    summary = distribution_summary(np.array([1.0, 3.0, 10.0]), np.array([1.0, 8.0, 1.0]))
    assert np.isclose(summary["mean"], 3.5)
    assert summary["p50"] == 3.0
    assert summary["p90"] == 3.0
    assert summary["p95"] == 10.0
    assert summary["min"] == 1.0 and summary["max"] == 10.0
    print("[PASS] grown-mass-weighted summary and percentile boundaries")


def _gpu_constitutive_probe(
    device: wgpu.GPUDevice,
    deformation: np.ndarray,
    rest: np.ndarray,
) -> np.ndarray:
    mu0, lambda0 = lame_params(E, NU)
    shader = device.create_shader_module(
        code=f"""
        const N: u32 = {len(deformation)}u;
        const MU0: f32 = {mu0};
        const LAMBDA0: f32 = {lambda0};
        const HARDENING: f32 = {HARDENING};
        struct Rest {{ growthF: vec4<f32>, jp: f32, cycleActive: f32, growthDirection: vec2<f32>, growthControls: vec2<f32>, }}
        @group(0) @binding(0) var<storage, read> particleF: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read> particleRest: array<Rest>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
        fn matMul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {{
          return vec4<f32>(
            a.x*b.x+a.y*b.z, a.x*b.y+a.y*b.w,
            a.z*b.x+a.w*b.z, a.z*b.y+a.w*b.w
          );
        }}
        fn matTranspose(m: vec4<f32>) -> vec4<f32> {{ return vec4<f32>(m.x,m.z,m.y,m.w); }}
        fn polarRotation(m: vec4<f32>) -> vec4<f32> {{
          let x = m.x + m.w;
          let y = m.z - m.y;
          let d = sqrt(x*x + y*y);
          if (d < 1e-6) {{ return vec4<f32>(1.0,0.0,0.0,1.0); }}
          let c = x/d;
          let s = y/d;
          return vec4<f32>(c,-s,s,c);
        }}
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
          if (gid.x >= N) {{ return; }}
          let rest = particleRest[gid.x];
          let fg = rest.growthF;
          let detFg = fg.x*fg.w-fg.y*fg.z;
          let invFg = vec4<f32>(fg.w,-fg.y,-fg.z,fg.x)/detFg;
          let f = particleF[gid.x];
          let fe = vec4<f32>(
            f.x*invFg.x+f.y*invFg.z, f.x*invFg.y+f.y*invFg.w,
            f.z*invFg.x+f.w*invFg.z, f.z*invFg.y+f.w*invFg.w
          );
          let je = fe.x*fe.w-fe.y*fe.z;
          let r = polarRotation(fe);
          let delta = fe-r;
          let shear = sqrt(dot(delta,delta));
          let scale = exp(HARDENING*(1.0-rest.jp));
          let pressure = -(LAMBDA0*scale)*(je-1.0);
          let energy = MU0*scale*shear*shear + 0.5*LAMBDA0*scale*(je-1.0)*(je-1.0);
          output[gid.x] = vec4<f32>(je,shear,pressure,energy);
        }}
        """
    )
    f_buffer = device.create_buffer(size=deformation.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    rest_buffer = device.create_buffer(size=rest.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    out_buffer = device.create_buffer(size=len(deformation) * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    device.queue.write_buffer(f_buffer, 0, deformation)
    device.queue.write_buffer(rest_buffer, 0, rest)
    pipeline = device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": shader, "entry_point": "main"})
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": f_buffer}},
            {"binding": 1, "resource": {"buffer": rest_buffer}},
            {"binding": 2, "resource": {"buffer": out_buffer}},
        ],
    )
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, bind_group)
    compute.dispatch_workgroups((len(deformation) + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])
    return np.frombuffer(device.queue.read_buffer(out_buffer), np.float32).reshape(-1, 4).copy()


def check_gpu_consistency(device: wgpu.GPUDevice) -> None:
    rng = np.random.default_rng(4281)
    count = 97
    angles_left = rng.uniform(-np.pi, np.pi, count)
    angles_right = rng.uniform(-np.pi, np.pi, count)
    stretches = rng.uniform(0.72, 1.35, (count, 2))
    growth = rng.uniform(0.8, 1.95, count)
    deformation = np.empty((count, 2, 2))
    for i in range(count):
        cl, sl = np.cos(angles_left[i]), np.sin(angles_left[i])
        cr, sr = np.cos(angles_right[i]), np.sin(angles_right[i])
        left = np.array([[cl, -sl], [sl, cl]])
        right = np.array([[cr, -sr], [sr, cr]])
        fe = left @ np.diag(stretches[i]) @ right
        deformation[i] = fe * np.sqrt(growth[i])
    deformation32 = deformation.astype(np.float32).reshape(-1, 4)
    rest32 = np.zeros((count, 12), dtype=np.float32)
    root_growth = np.sqrt(growth).astype(np.float32)
    rest32[:, 0] = root_growth
    rest32[:, 3] = root_growth
    rest32[:, 4] = rng.uniform(0.9, 1.1, count)
    cpu = particle_elastic_state(
        deformation32,
        rest32,
        material_e=E,
        material_nu=NU,
        material_hardening=HARDENING,
    )
    gpu = _gpu_constitutive_probe(device, deformation32, rest32)
    expected = np.column_stack(
        (
            cpu.elastic_volume_ratio,
            cpu.corotated_strain,
            cpu.pressure,
            cpu.elastic_energy_density,
        )
    )
    assert np.allclose(gpu, expected, rtol=4e-5, atol=3e-3), np.max(np.abs(gpu - expected), axis=0)
    print("[PASS] CPU diagnostics match independent WGSL constitutive probe (97 randomized states)")


def check_core_readback(device: wgpu.GPUDevice) -> None:
    core = MpmCore(device)
    growth = np.array([1.0, 1.3, 1.8], dtype=np.float32)
    fe = np.array([np.eye(2), np.diag([1.1, 0.9]), np.diag([0.85, 1.2])], dtype=np.float32)
    f = fe * np.sqrt(growth)[:, None, None]
    core.load_scene(
        np.array([[0.49, 0.5], [0.5, 0.5], [0.51, 0.5]], dtype=np.float32),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -2.0]], dtype=np.float32),
        f.reshape(-1, 4),
        np.zeros((3, 4), dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )
    rest = np.zeros((3, 12), dtype=np.float32)
    root_growth = np.sqrt(growth)
    rest[:, 0] = root_growth
    rest[:, 3] = root_growth
    rest[:, 4] = 1.0
    rest[:, 5] = [0.0, 1.0, 0.0]
    device.queue.write_buffer(core.rest, 0, rest)
    summary = measure_core(core, material_e=E, material_nu=NU, material_hardening=HARDENING)
    direct = summarize_elastic_state(
        particle_elastic_state(f, rest, material_e=E, material_nu=NU, material_hardening=HARDENING),
        velocities=core.read_velocities(),
        positions=core.read_positions(),
    )
    assert summary == direct
    assert summary["particle_count"] == 3 and summary["active_cycle_count"] == 1
    assert np.isclose(summary["total_rest_area"], growth.sum())
    print("[PASS] MpmCore F/rest/velocity/position readback and complete summary")


def check_viewer_diagnostic_shader(device: wgpu.GPUDevice) -> None:
    source = (Path(__file__).parent.parent / "viewer" / "src" / "gpu" / "fieldDiagnostics.wgsl").read_text()
    code = template_shader(source, {"GRID_N": 64, "DX": 0.015625, "INV_DX": 64, "PARTICLE_MASS": 1.0})
    module = device.create_shader_module(code=code)
    device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "clearDiagnostics"})
    device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "scatterDiagnostics"})
    print("[PASS] viewer tensor-Fg field diagnostic shader compiles for both entry points")


def check_viewer_render_shader(device: wgpu.GPUDevice) -> None:
    source = (Path(__file__).parent.parent / "viewer" / "src" / "gpu" / "render.wgsl").read_text()
    module = device.create_shader_module(code=source)
    for vertex, fragment in (
        ("particleVertex", "particleFragment"),
        ("activationParticleVertex", "activationParticleFragment"),
        ("triangleVertex", "triangleFragment"),
        ("growthAxisVertex", "growthAxisFragment"),
    ):
        device.create_render_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            vertex={"module": module, "entry_point": vertex},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={
                "module": module,
                "entry_point": fragment,
                "targets": [{"format": wgpu.TextureFormat.bgra8unorm}],
            },
        )
    print("[PASS] viewer white-dot, activation-dot, heading, and signed growth-axis render pipelines compile")


def check_saved_snapshot() -> None:
    if not SNAPSHOT_PATH.exists():
        raise AssertionError(f"missing baseline snapshot: {SNAPSHOT_PATH}")
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshot["schema_version"] == 1
    assert snapshot["growth_model"] == "tensor_Fg_with_isotropic_increment"
    assert SCALAR_SNAPSHOT_PATH.exists(), "the preserved scalar baseline must not be overwritten"
    assert snapshot["scenario"]["substeps_per_macro"] == 16
    checkpoints = snapshot["checkpoints"]
    assert [row["macro_step"] for row in checkpoints] == sorted(row["macro_step"] for row in checkpoints)
    assert checkpoints[0]["macro_step"] == 0 and checkpoints[-1]["phase"] == "settling"
    for row in checkpoints:
        assert row["particle_count"] >= 2
        assert np.isfinite(row["total_elastic_energy"])
        assert row["metrics"]["principal_stretch_min"]["min"] > 0.0
        assert row["metrics"]["growth_principal_stretch_min"]["min"] > 0.0
        assert row["metrics"]["growth_deviatoric_log_strain"]["max"] < 1e-7
    scalar = json.loads(SCALAR_SNAPSHOT_PATH.read_text())
    scalar_checkpoints = scalar["checkpoints"]
    assert len(scalar_checkpoints) == len(checkpoints)
    # Isotropic tensor increments should reproduce the scalar trajectory;
    # tiny differences are expected from general matrix inverse/multiply
    # roundoff replacing the old scalar divide.
    for old, new in zip(scalar_checkpoints, checkpoints, strict=True):
        assert old["macro_step"] == new["macro_step"]
        assert old["particle_count"] == new["particle_count"]
        assert old["active_cycle_count"] == new["active_cycle_count"]
        assert abs(old["geometry"]["rms_radius"] - new["geometry"]["rms_radius"]) < 5e-7
        assert abs(old["metrics"]["elastic_volume_ratio"]["mean"] - new["metrics"]["elastic_volume_ratio"]["mean"]) < 3e-5
        assert abs(old["metrics"]["deviatoric_log_strain"]["mean"] - new["metrics"]["deviatoric_log_strain"]["mean"]) < 3e-5
        assert abs(old["total_elastic_energy"] - new["total_elastic_energy"]) < 2e-3
        assert abs(old["total_kinetic_energy"] - new["total_kinetic_energy"]) < 1e-3
    print(f"[PASS] saved snapshot schema and finite/invertible checkpoint measurements ({len(checkpoints)} checkpoints)")
    print("[PASS] tensor isotropic trajectory matches preserved scalar baseline within roundoff tolerance")

    directional = json.loads(DIRECTIONAL_SNAPSHOT_PATH.read_text())
    assert directional["growth_model"] == "tensor_Fg_with_network_direction"
    directional_checkpoints = directional["checkpoints"]
    assert [row["particle_count"] for row in directional_checkpoints] == [row["particle_count"] for row in checkpoints]
    assert max(row["metrics"]["growth_deviatoric_log_strain"]["max"] for row in directional_checkpoints) > 0.2
    assert directional_checkpoints[-1]["geometry"]["rms_radius"] > checkpoints[-1]["geometry"]["rms_radius"] * 1.5
    assert directional_checkpoints[-1]["metrics"]["deviatoric_log_strain"]["mean"] < 2e-4
    print("[PASS] directional snapshot develops anisotropic Fg, elongates morphology, then settles")


def main() -> None:
    check_analytic_invariants()
    check_weighted_summary()
    device = pick_device()
    check_gpu_consistency(device)
    check_core_readback(device)
    check_viewer_diagnostic_shader(device)
    check_viewer_render_shader(device)
    check_saved_snapshot()


if __name__ == "__main__":
    main()

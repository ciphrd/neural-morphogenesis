"""Headless MLS-MPM simulation — the MpmSimulation-equivalent for this
project's Python trainer, scoped to exactly core/'s passes (clearDensity,
splatDensity, densityToTexture, applyRepulsion, clearGrid, p2g,
gridUpdate, g2p). Mirrors mls-mpm/src/gpu/mpm.ts's own MpmSimulation
class structure (buffer layout, bind groups, step() ordering) as closely
as possible so the two stay easy to compare by eye, minus everything
sandbox-only (Mouse uniform, attract-to-point, field-visualize
diagnostic channels — see ../core/README.md).

Repulsion (clearDensity/splatDensity/densityToTexture/applyRepulsion,
all from core/repulsion.wgsl) runs FIRST each substep, before
clearGrid/p2g/gridUpdate/g2p — applyRepulsion nudges particleVel from
THIS substep's own freshly-built density field, at each particle's own
exact position, so the push reaches the grid through the very same
substep's own P2G->gridUpdate->G2P transfer immediately rather than
sitting stale for one substep. See core/repulsion.wgsl's own module
docstring for the full 3-revision history of this mechanism, including
why a 4th revision (moving the push into gridUpdate.wgsl as a per-node
acceleration, to fully eliminate P2G's own momentum-cancellation for
overlapping particles) was tried and reverted: it traded that partial
cancellation for a worse problem, capping the push's effective spatial
resolution at the physics grid's own cell size — coarser than
core/agents.wgsl's own growth-spawn displacement — confirmed empirically
to leave freshly-spawned overlapping particles barely separated at all.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import wgpu

from shader_template import load_core_shader
from simulation_settings import (
    DAMPING_LOSS_FRACTION,
    DEFAULT_SUBSTEPS_PER_MACRO,
    GROWTH_DURATION_MACRO_STEPS,
    GROWTH_MAX,
    GROWTH_THRESHOLD,
    MATERIAL_E,
    MATERIAL_ELASTICITY,
    MATERIAL_HARDENING,
    MATERIAL_NU,
    MORPHOLOGY_BLUR_SIGMA,
    MORPHOLOGY_DENSITY_REFERENCE,
    REPULSION_MAX_DELTA,
    REPULSION_STRENGTH,
    SPLAT_RADIUS,
    SUBSTEPS_PER_DAMPING_FRAME,
)

CORE_DIR = Path(__file__).parent.parent / "core"
CONSTANTS = json.loads((CORE_DIR / "constants.json").read_text())

GRID_N: int = CONSTANTS["GRID_N"]
DX: float = CONSTANTS["DX"]
INV_DX: int = CONSTANTS["INV_DX"]
DT: float = CONSTANTS["DT"]
PARTICLE_MASS: float = CONSTANTS["PARTICLE_MASS"]
VOL: float = CONSTANTS["VOL"]
MAX_PARTICLES: int = CONSTANTS["MAX_PARTICLES"]
# core/constants.json's own FIELD_N is the repulsion density texture's
# resolution — renamed on import to avoid any ambiguity with the
# chemical field's own (unrelated) FIELD_N in simulation_settings.py.
REPULSION_FIELD_N: int = CONSTANTS["FIELD_N"]


def growth_rate_for_duration(duration_macro_steps: float, substeps_per_macro: int) -> float:
    """Return the internal continuous rate for a controller-tick duration.

    With no compression, applying this rate for ``substeps_per_macro`` G2P
    updates per macro step doubles det(Fg) after ``duration_macro_steps``.
    The conversion is deliberately centralized here so numerical physics
    cadence cannot silently change developmental cadence.
    """
    if duration_macro_steps <= 0.0:
        return 0.0
    return math.log(2.0) / (duration_macro_steps * max(int(substeps_per_macro), 1) * DT)

NODE_COUNT = (GRID_N + 1) * (GRID_N + 1)
WORKGROUP = 64
FIELD_WORKGROUP = 16
GRID_ACCUM_CHANNELS = 3  # mom_x, mom_y, mass — see core/clearGrid.wgsl
# growthF(4), jp, cycleActive, growthDirection(2), growthControls(2), then
# two implicit alignment floats — core/agents.wgsl's 48-byte array stride.
REST_FIELDS = 12
REST_GROWTH_F = slice(0, 4)
REST_JP = 4
REST_CYCLE_ACTIVE = 5


def _pack_rest(jp: np.ndarray) -> np.ndarray:
    """Expands a flat (count,) Jp array into ParticleRest's own
    (count, 12) tensor-rest layout, defaulting growthF=I (baseline rest
    configuration), cycleActive=0, direction/controls=0, and padding=0.

    Exists so load_scene()/reset_growth_buffers() can keep their original
    scalar-Jp signatures — every scene seeder in this project
    (training_sim.py's seed_blob(), feasibility_check.py, render_check.py,
    and the viewer's own rng.ts) still hands over a plain (count,) ones
    array, unaware that the underlying buffer grew two siblings."""
    count = jp.shape[0]
    packed = np.zeros((count, REST_FIELDS), dtype=np.float32)
    packed[:, 0] = 1.0
    packed[:, 3] = 1.0
    packed[:, REST_JP] = jp
    return packed

SNOW_YIELD_LOW = 1.0 - 2.5e-2
SNOW_YIELD_HIGH = 1.0 + 7.5e-3
WIDE_YIELD_LOW = 0.5
WIDE_YIELD_HIGH = 2.0


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def per_substep_damping(loss_fraction: float, substeps: int) -> float:
    """Port of mpm.ts's perSubstepDamping() — verbatim, not
    reimplemented from description: converts a per-rendered-frame loss
    fraction into the per-substep multiplier gridUpdate.wgsl's `damping`
    uniform actually wants."""
    clamped = min(max(loss_fraction, 0.0), 0.999)
    return (1 - clamped) ** (1 / max(substeps, 1))


def lame_params(e: float, nu: float) -> tuple[float, float]:
    """Port of mpm.ts's lameParams() — verbatim: (mu0, lambda0)."""
    mu0 = e / (2 * (1 + nu))
    lambda0 = (e * nu) / ((1 + nu) * (1 - 2 * nu))
    return mu0, lambda0


def yield_bounds(elasticity: float) -> tuple[float, float]:
    """Port of mpm.ts's yieldBounds() — verbatim: (yieldLow, yieldHigh)."""
    t = min(max(elasticity, 0.0), 1.0)
    yield_low = SNOW_YIELD_LOW + t * (WIDE_YIELD_LOW - SNOW_YIELD_LOW)
    yield_high = SNOW_YIELD_HIGH + t * (WIDE_YIELD_HIGH - SNOW_YIELD_HIGH)
    return yield_low, yield_high


class MpmCore:
    """Owns every GPU resource for the core MLS-MPM simulation and the
    one step(substeps) entry point — runs `substeps` full advance()
    iterations (clearDensity -> splatDensity -> densityToTexture ->
    applyRepulsion -> clearGrid -> p2g -> gridUpdate -> g2p) in a single
    submitted command buffer, each pass its own begin/end compute pass
    (WebGPU gives no cross-dispatch visibility guarantee *within* one
    pass, only across pass boundaries — same reasoning mpm.ts's own class
    docstring documents).

    Particle buffers are sized to MAX_PARTICLES (fixed capacity); load_scene()
    writes into the head of each buffer and updates the small activeCount
    uniform p2g/g2p gate their per-particle work on."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self._active_count = 0

        f32 = 4
        self.positions = device.create_buffer(
            size=MAX_PARTICLES * 2 * f32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        self.velocities = device.create_buffer(
            size=MAX_PARTICLES * 2 * f32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        self.F = device.create_buffer(size=MAX_PARTICLES * 4 * f32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        self.C = device.create_buffer(size=MAX_PARTICLES * 4 * f32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        # Per-particle rest state — growthF(4), jp, cycleActive, direction(2), see
        # core/agents.wgsl's own ParticleRest struct. Was a bare
        # array<f32> of Jp alone; widened rather than adding sibling
        # buffers because core/agents.wgsl is at the hard
        # 10-storage-buffer Dawn ceiling and needs all three at its
        # split site (see that file's own binding comments).
        self.rest = device.create_buffer(
            size=MAX_PARTICLES * REST_FIELDS * f32,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )

        # COPY_SRC supports the focused stability/headroom regressions; it
        # does not add a transfer to the hot path unless a check reads it.
        self.grid_accum = device.create_buffer(
            size=NODE_COUNT * GRID_ACCUM_CHANNELS * f32,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self.grid_vel = device.create_buffer(size=NODE_COUNT * 2 * f32, usage=wgpu.BufferUsage.STORAGE)

        # Purely a GPU-sync barrier for step()'s own chunking — see
        # _MAX_SUBSTEPS_PER_SUBMIT's docstring. STORAGE|COPY_SRC (not
        # UNIFORM) so read_buffer() is actually legal against it.
        self._sync_buffer = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

        self.gravity_uniform = device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_gravity(0.0)  # not a "simulation setting" — gravity is CLI-configurable (evolve.py's --gravity)

        # Material: nine floats plus uniform-struct padding = 48 bytes —
        # must match p2g.wgsl's/g2p.wgsl's identical Material struct
        # declarations exactly. p2g reads the first five, g2p reads
        # yieldLow/yieldHigh plus the growth params; one shared
        # struct regardless, same convention this buffer already followed.
        self.material_uniform = device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_material(
            MATERIAL_E,
            MATERIAL_NU,
            MATERIAL_HARDENING,
            MATERIAL_ELASTICITY,
            growth_max=GROWTH_MAX,
            growth_threshold=GROWTH_THRESHOLD,
            growth_duration_macro_steps=GROWTH_DURATION_MACRO_STEPS,
            substeps_per_macro=DEFAULT_SUBSTEPS_PER_MACRO,
        )

        self.active_count_uniform = device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.damping_uniform = device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_damping(DAMPING_LOSS_FRACTION, SUBSTEPS_PER_DAMPING_FRAME)

        template_vars = {"GRID_N": GRID_N, "DX": DX, "INV_DX": INV_DX, "DT": DT, "PARTICLE_MASS": PARTICLE_MASS, "VOL": VOL}

        clear_grid_module = device.create_shader_module(code=load_core_shader("clearGrid.wgsl", {"GRID_N": GRID_N}))
        self.clear_grid_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": clear_grid_module, "entry_point": "clearGrid"}
        )
        self.clear_grid_bind_group = device.create_bind_group(
            layout=self.clear_grid_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self.grid_accum, "offset": 0, "size": self.grid_accum.size}}],
        )

        p2g_module = device.create_shader_module(code=load_core_shader("p2g.wgsl", template_vars))
        self.p2g_pipeline = device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": p2g_module, "entry_point": "p2g"})
        self.p2g_bind_group = device.create_bind_group(
            layout=self.p2g_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.positions, "offset": 0, "size": self.positions.size}},
                {"binding": 1, "resource": {"buffer": self.velocities, "offset": 0, "size": self.velocities.size}},
                {"binding": 2, "resource": {"buffer": self.F, "offset": 0, "size": self.F.size}},
                {"binding": 3, "resource": {"buffer": self.C, "offset": 0, "size": self.C.size}},
                {"binding": 4, "resource": {"buffer": self.rest, "offset": 0, "size": self.rest.size}},
                {"binding": 5, "resource": {"buffer": self.grid_accum, "offset": 0, "size": self.grid_accum.size}},
                {"binding": 6, "resource": {"buffer": self.material_uniform, "offset": 0, "size": self.material_uniform.size}},
                {"binding": 7, "resource": {"buffer": self.active_count_uniform, "offset": 0, "size": self.active_count_uniform.size}},
            ],
        )

        grid_update_module = device.create_shader_module(code=load_core_shader("gridUpdate.wgsl", {"GRID_N": GRID_N, "DT": DT}))
        self.grid_update_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": grid_update_module, "entry_point": "gridUpdate"}
        )
        self.grid_update_bind_group = device.create_bind_group(
            layout=self.grid_update_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.grid_accum, "offset": 0, "size": self.grid_accum.size}},
                {"binding": 1, "resource": {"buffer": self.grid_vel, "offset": 0, "size": self.grid_vel.size}},
                {"binding": 2, "resource": {"buffer": self.gravity_uniform, "offset": 0, "size": self.gravity_uniform.size}},
                {"binding": 3, "resource": {"buffer": self.damping_uniform, "offset": 0, "size": self.damping_uniform.size}},
            ],
        )

        g2p_module = device.create_shader_module(code=load_core_shader("g2p.wgsl", {"GRID_N": GRID_N, "INV_DX": INV_DX, "DT": DT}))
        self.g2p_pipeline = device.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto, compute={"module": g2p_module, "entry_point": "g2p"})
        self.g2p_bind_group = device.create_bind_group(
            layout=self.g2p_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.positions, "offset": 0, "size": self.positions.size}},
                {"binding": 1, "resource": {"buffer": self.velocities, "offset": 0, "size": self.velocities.size}},
                {"binding": 2, "resource": {"buffer": self.F, "offset": 0, "size": self.F.size}},
                {"binding": 3, "resource": {"buffer": self.C, "offset": 0, "size": self.C.size}},
                {"binding": 4, "resource": {"buffer": self.rest, "offset": 0, "size": self.rest.size}},
                {"binding": 5, "resource": {"buffer": self.grid_vel, "offset": 0, "size": self.grid_vel.size}},
                {"binding": 6, "resource": {"buffer": self.active_count_uniform, "offset": 0, "size": self.active_count_uniform.size}},
                {"binding": 7, "resource": {"buffer": self.material_uniform, "offset": 0, "size": self.material_uniform.size}},
            ],
        )

        self.grid_dispatch = ceil_div(NODE_COUNT, WORKGROUP)

        # --- Repulsion (see core/repulsion.wgsl) ---
        texel_count = REPULSION_FIELD_N * REPULSION_FIELD_N
        self.density_accum = device.create_buffer(size=texel_count * f32, usage=wgpu.BufferUsage.STORAGE)
        self.density_texture = device.create_texture(
            size=(REPULSION_FIELD_N, REPULSION_FIELD_N, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )
        density_texture_view = self.density_texture.create_view()
        self.morphology_texture = device.create_texture(
            size=(REPULSION_FIELD_N, REPULSION_FIELD_N, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )
        self.morphology_blur_texture = device.create_texture(
            size=(REPULSION_FIELD_N, REPULSION_FIELD_N, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.morphology_params_uniform = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.set_morphology(MORPHOLOGY_BLUR_SIGMA, MORPHOLOGY_DENSITY_REFERENCE)

        self.splat_params_uniform = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_splat_radius(SPLAT_RADIUS)
        self.repulsion_params_uniform = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.set_repulsion_strength(REPULSION_STRENGTH, REPULSION_MAX_DELTA)

        repulsion_module = device.create_shader_module(
            code=load_core_shader("repulsion.wgsl", {"FIELD_N": REPULSION_FIELD_N, "DT": DT})
        )

        self.clear_density_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": repulsion_module, "entry_point": "clearDensity"}
        )
        self.clear_density_bind_group = device.create_bind_group(
            layout=self.clear_density_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.density_accum, "offset": 0, "size": self.density_accum.size}},
            ],
        )

        self.splat_density_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": repulsion_module, "entry_point": "splatDensity"}
        )
        self.splat_density_bind_group = device.create_bind_group(
            layout=self.splat_density_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.density_accum, "offset": 0, "size": self.density_accum.size}},
                {"binding": 1, "resource": {"buffer": self.positions, "offset": 0, "size": self.positions.size}},
                {"binding": 2, "resource": {"buffer": self.active_count_uniform, "offset": 0, "size": self.active_count_uniform.size}},
                {"binding": 3, "resource": {"buffer": self.splat_params_uniform, "offset": 0, "size": self.splat_params_uniform.size}},
            ],
        )

        self.density_to_texture_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": repulsion_module, "entry_point": "densityToTexture"}
        )
        self.density_to_texture_bind_group = device.create_bind_group(
            layout=self.density_to_texture_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.density_accum, "offset": 0, "size": self.density_accum.size}},
                {"binding": 1, "resource": density_texture_view},
            ],
        )

        self.apply_repulsion_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto, compute={"module": repulsion_module, "entry_point": "applyRepulsion"}
        )
        self.apply_repulsion_bind_group = device.create_bind_group(
            layout=self.apply_repulsion_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 1, "resource": {"buffer": self.positions, "offset": 0, "size": self.positions.size}},
                {"binding": 2, "resource": {"buffer": self.active_count_uniform, "offset": 0, "size": self.active_count_uniform.size}},
                {"binding": 4, "resource": {"buffer": self.velocities, "offset": 0, "size": self.velocities.size}},
                {"binding": 5, "resource": density_texture_view},
                {"binding": 7, "resource": {"buffer": self.repulsion_params_uniform, "offset": 0, "size": self.repulsion_params_uniform.size}},
            ],
        )

        self.density_clear_dispatch = ceil_div(texel_count, WORKGROUP)
        self.density_texture_dispatch = (ceil_div(REPULSION_FIELD_N, FIELD_WORKGROUP), ceil_div(REPULSION_FIELD_N, FIELD_WORKGROUP))

        morphology_module = device.create_shader_module(
            code=load_core_shader("morphology.wgsl", {"FIELD_N": REPULSION_FIELD_N})
        )
        self.morphology_horizontal_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={"module": morphology_module, "entry_point": "blurHorizontal"},
        )
        self.morphology_vertical_pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={"module": morphology_module, "entry_point": "blurVerticalAndNormalize"},
        )
        self.morphology_horizontal_bind_group = device.create_bind_group(
            layout=self.morphology_horizontal_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": density_texture_view},
                {"binding": 1, "resource": self.morphology_blur_texture.create_view()},
                {"binding": 2, "resource": {"buffer": self.morphology_params_uniform, "offset": 0, "size": 16}},
            ],
        )
        self.morphology_vertical_bind_group = device.create_bind_group(
            layout=self.morphology_vertical_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.morphology_blur_texture.create_view()},
                {"binding": 1, "resource": self.morphology_texture.create_view()},
                {"binding": 2, "resource": {"buffer": self.morphology_params_uniform, "offset": 0, "size": 16}},
            ],
        )

    @property
    def active_count(self) -> int:
        return self._active_count

    def set_morphology(self, sigma_domain: float, density_reference: float) -> None:
        self.device.queue.write_buffer(
            self.morphology_params_uniform,
            0,
            np.asarray([sigma_domain, density_reference, 0.0, 0.0], dtype=np.float32),
        )

    def encode_morphology(self, encoder: wgpu.GPUCommandEncoder) -> None:
        """Rebuild the blurred occupancy field from current particle positions.

        Called once per controller tick, immediately before policy sensing.
        Physics may rebuild the raw density again per substep for repulsion.
        """
        particle_dispatch = ceil_div(self._active_count, WORKGROUP)
        for pipeline, bind_group, dispatch in (
            (self.clear_density_pipeline, self.clear_density_bind_group, (self.density_clear_dispatch,)),
            (self.splat_density_pipeline, self.splat_density_bind_group, (particle_dispatch,)),
            (self.density_to_texture_pipeline, self.density_to_texture_bind_group, self.density_texture_dispatch),
            (self.morphology_horizontal_pipeline, self.morphology_horizontal_bind_group, self.density_texture_dispatch),
            (self.morphology_vertical_pipeline, self.morphology_vertical_bind_group, self.density_texture_dispatch),
        ):
            p = encoder.begin_compute_pass()
            p.set_pipeline(pipeline)
            p.set_bind_group(0, bind_group)
            p.dispatch_workgroups(*dispatch)
            p.end()

    def read_morphology(self) -> np.ndarray:
        """Diagnostic-only synchronous readback of the occupancy texture."""
        byte_count = REPULSION_FIELD_N * REPULSION_FIELD_N * 4
        staging = self.device.create_buffer(
            size=byte_count, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self.morphology_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": staging, "offset": 0, "bytes_per_row": REPULSION_FIELD_N * 4, "rows_per_image": REPULSION_FIELD_N},
            (REPULSION_FIELD_N, REPULSION_FIELD_N, 1),
        )
        self.device.queue.submit([encoder.finish()])
        result = np.frombuffer(self.device.queue.read_buffer(staging), dtype=np.float32).copy()
        staging.destroy()
        return result.reshape(REPULSION_FIELD_N, REPULSION_FIELD_N)

    def load_scene(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        F: np.ndarray,
        C: np.ndarray,
        Jp: np.ndarray,
    ) -> None:
        """Writes a scene into the head of every particle buffer and
        updates activeCount — mirrors mpm.ts's own loadScene()."""
        count = positions.shape[0]
        assert count <= MAX_PARTICLES
        self.device.queue.write_buffer(self.positions, 0, positions.astype(np.float32))
        self.device.queue.write_buffer(self.velocities, 0, velocities.astype(np.float32))
        self.device.queue.write_buffer(self.F, 0, F.astype(np.float32))
        self.device.queue.write_buffer(self.C, 0, C.astype(np.float32))
        # Scene API deliberately unchanged: callers still hand over a
        # flat (count,) Jp array. Expanded here into ParticleRest's own
        # tensor layout — growthF=I and cycleActive=0 for
        # genuinely-seeded particles, which unlike growth-spawned
        # children have no ramp to serve.
        self.device.queue.write_buffer(self.rest, 0, _pack_rest(np.asarray(Jp, dtype=np.float32)))
        self.set_active_count(count)

    def set_active_count(self, count: int) -> None:
        """Updates both the Python-side count (dispatch sizing — see
        step()'s own particle_dispatch) and the shared GPU uniform
        (p2g/gridUpdate-adjacent/g2p/repulsion, and agents_gpu.py's own
        AgentsGPU — all bind this EXACT buffer, not a copy, so a single
        write here reaches every one of them at once). load_scene() is
        the usual caller (a fresh rollout's own starting count), but
        growth needs this too — training_sim.py's own macro_step() calls
        this after reading back core/agents.wgsl's own atomic growth
        counter, since GPU-side splitting changes the true count without
        this class ever finding out on its own (see that module's own
        module docstring for the full readback/propagation design, and
        why it's a real, deliberate host round-trip, not an oversight)."""
        self._active_count = count
        self.device.queue.write_buffer(self.active_count_uniform, 0, np.array([count], dtype=np.uint32))

    def reset_growth_buffers(self, max_active: int) -> None:
        """Zero/identity-fills velocities/F/C/ParticleRest for [0, max_active) —
        call once per rollout, after load_scene(). Every rollout starts
        with its configured number of real particles (see training_sim.py's own module
        docstring for why --particles is a growth CAP now, not a fixed
        starting count) — every slot beyond those particles is destined to
        become a real particle via growth (core/agents.wgsl's own
        agentStep() may claim any slot, so every per-particle physics/rest
        buffer must be initialized before that happens; division then
        overwrites the claimed slot with its inherited live state), and needs
        to start from the
        exact same fresh MPM state seed_blob() already gives that one
        genuinely-seeded particle — WITHOUT this, a slot THIS rollout's
        own growth later claims could inherit a PREVIOUS rollout's stale,
        possibly heavily-deformed state instead (MpmCore/AgentsGPU/
        EnvironmentGPU are reused across rollouts within a worker process
        — see evolve.py's own module docstring — load_scene() only ever
        writes the HEAD of each buffer, never the tail a previous
        rollout's growth may have touched). Safe (idempotent) to run over
        the one index load_scene() ALSO just wrote — seed_blob()'s own
        velocity/F/C/ParticleRest defaults are identical to these — so this can
        unconditionally cover the whole [0, max_active) range rather than
        needing to carefully skip the one already-real particle."""
        zeros2 = np.zeros((max_active, 2), dtype=np.float32)
        identity_f = np.tile(np.array([1, 0, 0, 1], dtype=np.float32), (max_active, 1))
        zeros4 = np.zeros((max_active, 4), dtype=np.float32)
        ones1 = np.ones((max_active,), dtype=np.float32)
        self.device.queue.write_buffer(self.velocities, 0, zeros2)
        self.device.queue.write_buffer(self.F, 0, identity_f)
        self.device.queue.write_buffer(self.C, 0, zeros4)
        self.device.queue.write_buffer(self.rest, 0, _pack_rest(ones1))

    def set_gravity(self, gravity: float) -> None:
        self.device.queue.write_buffer(self.gravity_uniform, 0, np.array([gravity], dtype=np.float32))

    def set_damping(self, loss_fraction: float, substeps: int) -> None:
        self.device.queue.write_buffer(self.damping_uniform, 0, np.array([per_substep_damping(loss_fraction, substeps)], dtype=np.float32))

    def set_material(
        self,
        e: float,
        nu: float,
        hardening: float,
        elasticity: float,
        growth_rate: float | None = None,
        growth_max: float = GROWTH_MAX,
        growth_threshold: float = GROWTH_THRESHOLD,
        growth_anisotropy: float = 1.0,
        growth_duration_macro_steps: float = GROWTH_DURATION_MACRO_STEPS,
        substeps_per_macro: int = DEFAULT_SUBSTEPS_PER_MACRO,
    ) -> None:
        """Write elastic material and the derived internal growth rate.

        Production callers specify a controller-tick duration and their real
        substep count. ``growth_rate`` remains as an explicit low-level escape
        hatch for analytical tests and legacy checkpoints that recorded the
        old rate directly; when supplied it takes precedence.
        """
        mu0, lambda0 = lame_params(e, nu)
        yield_low, yield_high = yield_bounds(elasticity)
        effective_growth_rate = (
            growth_rate
            if growth_rate is not None
            else growth_rate_for_duration(growth_duration_macro_steps, substeps_per_macro)
        )
        self.device.queue.write_buffer(
            self.material_uniform,
            0,
            np.array(
                [
                    mu0,
                    lambda0,
                    hardening,
                    yield_low,
                    yield_high,
                    effective_growth_rate,
                    growth_max,
                    growth_threshold,
                    growth_anisotropy,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

    def set_splat_radius(self, sigma: float) -> None:
        self.device.queue.write_buffer(
            self.splat_params_uniform, 0, np.array([sigma, 0.0, 0.0, 0.0], dtype=np.float32)
        )

    def set_repulsion_strength(
        self, strength: float, max_delta: float,
    ) -> None:
        """`max_delta` is core/repulsion.wgsl's own RepulsionParams.maxDelta
        — see that field's own comment for what it bounds and why."""
        self.device.queue.write_buffer(
            self.repulsion_params_uniform, 0,
            np.array([strength, max_delta, 0.0, 0.0], dtype=np.float32),
        )

    # Max substeps encoded into a single command encoder before an
    # intermediate submit() — a real, load-bearing limit discovered by
    # testing, not a guess: wgpu-native's Metal backend apparently opens
    # a new MTLCommandBuffer per begin_compute_pass()/end() pair rather
    # than deferring everything to encoder.finish(), and those accumulate
    # as "outstanding" (not yet retired by the GPU) rather than being
    # freed at each submit() — a single step(3000) call (3000 substeps *
    # 8 passes/substep = 24,000 passes in one encoder) hit "refusing to
    # create new command buffer; 4097 outstanding command buffers exceeds
    # the limit of 4096" and killed the device; chunking into smaller
    # encoders alone was NOT enough to fix it either — the count is
    # cumulative ACROSS submits too when nothing makes the host wait for
    # the GPU to catch up, so step() also blocks on
    # on_submitted_work_done_sync() after each chunk (see below) to force
    # that catch-up. Both confirmed live, not hypothetical. This is a
    # genuine difference from the browser sandbox's own Dawn/tint
    # backend, which doesn't hit this at the substep counts mls-mpm's own
    # step() calls per rendered frame. 128 substeps/chunk (1024 passes)
    # is a comfortable margin under the observed ~4096 ceiling, chosen to
    # stay correct regardless of exactly where that limit sits on a given
    # machine/wgpu-native version, not tuned to the edge. (A batched-
    # rollout caller once tried pushing this further across several
    # MpmCore instances sharing one encoder — see evolve.py's own module
    # docstring for why that was tried and reverted, including a real
    # confirmed crash from under-counting this exact budget.)
    _MAX_SUBSTEPS_PER_SUBMIT = 128

    def step(self, substeps: int) -> None:
        """Runs `substeps` full advance() iterations — same pass ordering
        as mpm.ts's own step(): clearDensity -> splatDensity ->
        densityToTexture -> applyRepulsion -> clearGrid -> p2g ->
        gridUpdate -> g2p, each its own begin/end compute pass. Repulsion
        runs FIRST (not after g2p) so applyRepulsion's own velocity nudge
        — computed from THIS substep's own freshly-built density field,
        at each particle's own exact position — reaches the grid through
        the very same substep's own transfer immediately, rather than
        sitting stale in particleVel for one substep. See
        core/repulsion.wgsl's own module docstring for the full
        revision history of this mechanism and why it stays a
        per-particle pass rather than a per-grid-node one. Internally
        chunked into multiple command encoders/submits, each followed by
        an explicit GPU sync, when `substeps` exceeds
        _MAX_SUBSTEPS_PER_SUBMIT (see that constant's own docstring) —
        callers can pass any substep count without needing to know about
        that limit themselves. The per-chunk sync is a real host-device
        stall (this is not free, unlike the browser sandbox's own
        fire-and-forget step()) — acceptable for this feasibility spike
        and for a future ES training loop's own per-episode cadence, but
        worth knowing about before assuming step() is as cheap here as it
        is in the browser."""
        particle_dispatch = ceil_div(self._active_count, WORKGROUP)
        remaining = substeps
        while remaining > 0:
            chunk = min(remaining, self._MAX_SUBSTEPS_PER_SUBMIT)
            remaining -= chunk
            encoder = self.device.create_command_encoder()
            for _ in range(chunk):
                p = encoder.begin_compute_pass()
                p.set_pipeline(self.clear_density_pipeline)
                p.set_bind_group(0, self.clear_density_bind_group)
                p.dispatch_workgroups(self.density_clear_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.splat_density_pipeline)
                p.set_bind_group(0, self.splat_density_bind_group)
                p.dispatch_workgroups(particle_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.density_to_texture_pipeline)
                p.set_bind_group(0, self.density_to_texture_bind_group)
                p.dispatch_workgroups(*self.density_texture_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.apply_repulsion_pipeline)
                p.set_bind_group(0, self.apply_repulsion_bind_group)
                p.dispatch_workgroups(particle_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.clear_grid_pipeline)
                p.set_bind_group(0, self.clear_grid_bind_group)
                p.dispatch_workgroups(self.grid_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.p2g_pipeline)
                p.set_bind_group(0, self.p2g_bind_group)
                p.dispatch_workgroups(particle_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.grid_update_pipeline)
                p.set_bind_group(0, self.grid_update_bind_group)
                p.dispatch_workgroups(self.grid_dispatch)
                p.end()

                p = encoder.begin_compute_pass()
                p.set_pipeline(self.g2p_pipeline)
                p.set_bind_group(0, self.g2p_bind_group)
                p.dispatch_workgroups(particle_dispatch)
                p.end()

            self.device.queue.submit([encoder.finish()])
            # Forces the GPU to retire this chunk's command buffers
            # before the next chunk starts accumulating more — see
            # _MAX_SUBSTEPS_PER_SUBMIT's own docstring for why this is
            # required, not just a safety margin. The "correct" API for
            # this, queue.on_submitted_work_done_sync(), is confirmed
            # broken in this wgpu-py/wgpu-native combination (a real
            # callback-signature TypeError inside wgpu-py's own binding
            # code, not a usage mistake — reproduces even in total
            # isolation, one buffer, one empty encoder). A trivial 4-byte
            # read_buffer() achieves the same wait (WebGPU's queue is a
            # single in-order timeline, so reading anything back
            # necessarily blocks until every submission issued before it
            # has been processed) as a working substitute.
            self.device.queue.read_buffer(self._sync_buffer, 0, 4)

    def read_positions(self) -> np.ndarray:
        raw = self.device.queue.read_buffer(self.positions, 0, self._active_count * 2 * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()

    def read_velocities(self) -> np.ndarray:
        raw = self.device.queue.read_buffer(self.velocities, 0, self._active_count * 2 * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()

    def read_deformation(self) -> np.ndarray:
        raw = self.device.queue.read_buffer(self.F, 0, self._active_count * 4 * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).copy()

    def read_affine(self) -> np.ndarray:
        raw = self.device.queue.read_buffer(self.C, 0, self._active_count * 4 * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).copy()

    def read_rest_state(self) -> np.ndarray:
        """Returns active particles' raw tensor-growth rest-state rows.

        Rows are ``[Fg00,Fg01,Fg10,Fg11,jp,cycleActive,dirX,dirY,
        anisotropy,divisionBias,padding,padding]``.
        This is diagnostic-only: COPY_SRC is present on the buffer, but the
        normal simulation path performs no readback. Keeping the raw layout
        visible here also makes scalar-vs-tensor growth snapshots explicit
        when ParticleRest is upgraded later.
        """
        raw = self.device.queue.read_buffer(self.rest, 0, self._active_count * REST_FIELDS * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, REST_FIELDS).copy()

"""Capture exact per-particle neural inputs over a growth rollout.

Produces a self-contained HTML dashboard and a sibling JSON data file.

Usage (from trainer/):
    .venv/bin/python capture_policy_inputs.py
    .venv/bin/python capture_policy_inputs.py --steps 500 --sample-every 2
    .venv/bin/python capture_policy_inputs.py --weights checkpoints/best.npy \
        --meta checkpoints/best_meta.json --output policy_inputs.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import wgpu

from agents_gpu import PARTICLE_META_BUFFER_OFFSET, AgentsGPU, weight_layout
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore, REPULSION_FIELD_N, ceil_div
from shader_template import load_core_shader
from simulation_settings import (
    ANGULAR_DAMPING, CHEM_CHANNELS, CHIRALITY, COMMUNICATION_SPEED,
    CHEMICAL_GRADIENT_INPUT_SCALE, CHEMICAL_VALUE_INPUT_SCALE,
    DAMPING_LOSS_FRACTION, DECAY, DEPOSIT_DISTANCE, DEPOSIT_RATE,
    DEPOSIT_SIGMA, DIVISION_COOLDOWN, DIVISION_DIRECTIONALITY, ELASTIC_STRAIN_INPUTS_ENABLED,
    ELASTIC_STRAIN_SCALE, FIELD_N, FRICTION, GROWTH_DURATION_MACRO_STEPS,
    GROWTH_ANISOTROPY_AUTHORITY, GROWTH_COMPRESSION_INHIBITION, GROWTH_MAX, GROWTH_THRESHOLD, INITIAL_PARTICLE_COUNT,
    INTERIOR_SUPPORT_STRENGTH,
    INTERNAL_STATE_SPEED,
    MATERIAL_E, MATERIAL_ELASTICITY, MATERIAL_HARDENING, MATERIAL_NU,
    MAX_ACCEL, MAX_ANGULAR_ACCEL, MAX_ANGULAR_VELOCITY, MAX_ENV_WRITE,
    MAX_STRAFE, MORPHOLOGY_BLUR_SIGMA, MORPHOLOGY_DENSITY_REFERENCE,
    MORPHOLOGY_GRADIENT_INPUT_SCALE,
    NEURAL_UPDATES_PER_MACRO, REPULSION_MAX_DELTA, REPULSION_STRENGTH,
    SPLAT_RADIUS, SPLIT_DISPLACEMENT,
)
from training_sim import TrainingRollout
from policy_parameters import (
    STATEFUL_ARCHITECTURE, STATELESS_ARCHITECTURE,
    policy_hidden_dim, policy_input_dim, random_flat_policy_weights,
)

META_NAMES = [
    "valid", "position_x", "position_y", "heading", "cooldown",
    "division_hazard", "division_threshold", "cycle_active",
    "growth_area", "growth_direction_angle", "growth_anisotropy",
    "division_bias",
]


def feature_names(channels: int, architecture: str = STATELESS_ARCHITECTURE) -> list[str]:
    names = (
        [f"chemical_value_{c}" for c in range(channels)]
        + [f"chemical_gradient_forward_{c}" for c in range(channels)]
        + [f"chemical_gradient_lateral_{c}" for c in range(channels)]
        + ["morphology_occupancy", "morphology_gradient_forward", "morphology_gradient_lateral"]
        + ["elastic_volume", "elastic_axial", "elastic_shear"]
    )
    if architecture == STATEFUL_ARCHITECTURE:
        names += [f"private_state_{i}" for i in range(8)]
    return names


def random_policy_weights(
    layout: dict[str, int], hidden_dim: int, rng: np.random.Generator,
    architecture: str = STATELESS_ARCHITECTURE,
) -> np.ndarray:
    """Use the same logical-head initialization as training and the viewer."""
    channels = (layout["in_dim"] - 6) // 3
    out = random_flat_policy_weights(channels, hidden_dim, rng, architecture)
    if out.size != layout["total_floats"]:
        raise ValueError(f"policy initializer produced {out.size} values, expected {layout['total_floats']}")
    return out


class PolicyInputProbe:
    def __init__(
        self, device: wgpu.GPUDevice, core: MpmCore, agents: AgentsGPU,
        environment: EnvironmentGPU, tracked: int, elastic_scale: float,
        elastic_enabled: bool,
    ) -> None:
        self.device = device
        self.agents = agents
        self.tracked = tracked
        self.input_dim = policy_input_dim(environment.channels, agents.policy_architecture)
        self.stride = len(META_NAMES) + 2 * self.input_dim
        self.indices = device.create_buffer(
            size=max(tracked, 1) * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        device.queue.write_buffer(self.indices, 0, np.arange(tracked, dtype=np.uint32))
        self.output = device.create_buffer(
            size=max(tracked * self.stride, 1) * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        module = device.create_shader_module(code=load_core_shader(
            "policyInputProbe.wgsl",
            {
                "CHANNELS": environment.channels,
                "FIELD_WIDTH": environment.width,
                "FIELD_HEIGHT": environment.height,
                "MORPHOLOGY_FIELD_N": REPULSION_FIELD_N,
                "TRACKED": tracked,
                "IN_DIM": self.input_dim,
                "PRIVATE_STATE_PROBE": (
                    "for (var s=0u; s<8u; s=s+1u) { output[rawBase+3u*CHANNELS+6u+s]=agentState.privateState[s]; output[inputBase+3u*CHANNELS+6u+s]=tanh(agentState.privateState[s]); }"
                    if agents.policy_architecture == STATEFUL_ARCHITECTURE else ""
                ),
                "ELASTIC_SCALE": repr(float(elastic_scale)),
                "ELASTIC_ENABLED": "true" if elastic_enabled else "false",
                "CHEMICAL_VALUE_INPUT_SCALE": repr(CHEMICAL_VALUE_INPUT_SCALE),
                "CHEMICAL_GRADIENT_INPUT_SCALE": repr(CHEMICAL_GRADIENT_INPUT_SCALE),
                "MORPHOLOGY_GRADIENT_INPUT_SCALE": repr(MORPHOLOGY_GRADIENT_INPUT_SCALE),
            },
        ))
        self.pipeline = device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={"module": module, "entry_point": "probe"},
        )
        meta_size = agents._agent_state_buffer.size - PARTICLE_META_BUFFER_OFFSET
        self.bind_groups = []
        for parity in (0, 1):
            self.bind_groups.append(device.create_bind_group(
                layout=self.pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": core.positions}},
                    {"binding": 1, "resource": {"buffer": core.active_count_uniform}},
                    {"binding": 2, "resource": {"buffer": environment.buffers[parity]}},
                    {"binding": 3, "resource": {"buffer": environment.gradient}},
                    {"binding": 4, "resource": core.morphology_texture.create_view()},
                    {"binding": 5, "resource": {"buffer": core.F}},
                    {"binding": 6, "resource": {"buffer": core.rest}},
                    {"binding": 7, "resource": {
                        "buffer": agents._agent_state_buffer,
                        "offset": PARTICLE_META_BUFFER_OFFSET,
                        "size": meta_size,
                    }},
                    {"binding": 8, "resource": {"buffer": self.indices}},
                    {"binding": 9, "resource": {"buffer": self.output}},
                ],
            ))

    def capture(self, core: MpmCore, environment: EnvironmentGPU) -> np.ndarray:
        encoder = self.device.create_command_encoder()
        # Rebuild exactly the fields/input gradients agents.wgsl would sample
        # on the next controller evaluation at the current particle state.
        core.encode_morphology(encoder)
        environment.encode_clear(encoder)
        self.agents.encode_splat_chemical_state(encoder)
        environment.encode_sense(encoder)
        p = encoder.begin_compute_pass()
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, self.bind_groups[environment.parity])
        p.dispatch_workgroups(ceil_div(self.tracked, 8))
        p.end()
        self.device.queue.submit([encoder.finish()])
        raw = self.device.queue.read_buffer(self.output, 0, self.tracked * self.stride * 4)
        return np.frombuffer(raw, np.float32).reshape(self.tracked, self.stride).copy()


def percentile_summary(values: list[list[float]], names: list[str]) -> list[dict[str, float | str | int]]:
    if not values:
        return []
    a = np.asarray(values, dtype=np.float64)
    rows: list[dict[str, float | str | int]] = []
    for i, name in enumerate(names):
        x = a[:, i]
        rows.append({
            "feature": name, "samples": int(len(x)), "min": float(x.min()),
            "p05": float(np.percentile(x, 5)), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std()),
        })
    return rows


def build_html(report: dict[str, Any]) -> str:
    data = json.dumps(report, separators=(",", ":"))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Policy input report</title>
<style>
:root{{--bg:#0b0e14;--panel:#141923;--text:#dce3ee;--muted:#8d99aa;--line:#283143;--cyan:#7dd3fc}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
header{{padding:20px 24px;border-bottom:1px solid var(--line)}} h1,h2{{margin:0 0 10px}} .muted{{color:var(--muted)}}
main{{padding:18px;display:grid;gap:16px}} .panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
.controls{{display:flex;gap:16px;flex-wrap:wrap;align-items:center}} select{{background:#0e131c;color:var(--text);border:1px solid #39445a;padding:6px}}
canvas{{width:100%;height:320px;background:#0d1119;border-radius:6px}} #population{{height:190px}} #heatmap{{height:360px;image-rendering:pixelated}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:var(--panel)}} .scroll{{max-height:520px;overflow:auto}}
</style></head><body><header><h1>Policy input report</h1><div id="meta" class="muted"></div></header>
<main><section class="panel controls"><label>Particle <select id="particle"></select></label><label>Input space <select id="space"><option value="inputs">Policy-normalized</option><option value="raw_inputs">Raw sensors</option></select></label><label>Group <select id="group"></select></label><span id="spawn" class="muted"></span></section>
<section class="panel"><h2>Agent population</h2><canvas id="population"></canvas><div id="population-summary" class="muted"></div></section>
<section class="panel"><h2>Raw input traces</h2><canvas id="lines"></canvas><div id="legend" class="muted"></div></section>
<section class="panel"><h2>Time × input heatmap</h2><p class="muted">Each feature column is independently scaled to its global p05–p95 range; hover for raw values.</p><canvas id="heatmap"></canvas><div id="hover" class="muted"></div></section>
<section class="panel"><h2>Raw-value distribution</h2><div class="scroll"><table><thead><tr><th>Feature</th><th>N</th><th>Min</th><th>P05</th><th>Median</th><th>P95</th><th>Max</th><th>Mean</th><th>Std</th></tr></thead><tbody id="stats"></tbody></table></div></section></main>
<script>
const R={data}; const names=R.feature_names; const particles=R.particles;
const groups={{"Chemical values":[0,R.channels],"Chemical gradients — forward":[R.channels,2*R.channels],"Chemical gradients — lateral":[2*R.channels,3*R.channels],"Morphology":[3*R.channels,3*R.channels+3],"Elastic strain":[3*R.channels+3,3*R.channels+6],"All inputs":[0,names.length]}};
const psel=document.querySelector('#particle'),gsel=document.querySelector('#group'),space=document.querySelector('#space');
particles.forEach((p,i)=>psel.add(new Option(`slot ${{p.slot}}`,i))); Object.keys(groups).forEach(k=>gsel.add(new Option(k,k)));
const finalCount=R.population.at(-1)?.active_count??0,peakCount=Math.max(...R.population.map(p=>p.active_count)),searchText=R.search.enabled?` · randomized policy found on attempt ${{R.search.attempt}}, split at search step ${{R.search.split_step}}`:' · checkpoint policy';
document.querySelector('#meta').textContent=`${{R.samples}} input samples · steps 0–${{R.settings.steps}} · every ${{R.settings.sample_every}} step(s) · final ${{finalCount}} agents · peak ${{peakCount}} · ${{R.channels}} channels · seed ${{R.settings.seed}}${{searchText}}`;
const fmt=v=>Math.abs(v)>=1000||Math.abs(v)<1e-3&&v!==0?v.toExponential(3):v.toFixed(4);
const sampleValues=s=>s[space.value]||s.inputs,activeSummary=()=>space.value==='raw_inputs'?R.raw_summary:R.summary;
function renderStats(){{document.querySelector('#stats').innerHTML=activeSummary().map(s=>`<tr><td>${{s.feature}}</td><td>${{s.samples}}</td><td>${{fmt(s.min)}}</td><td>${{fmt(s.p05)}}</td><td>${{fmt(s.median)}}</td><td>${{fmt(s.p95)}}</td><td>${{fmt(s.max)}}</td><td>${{fmt(s.mean)}}</td><td>${{fmt(s.std)}}</td></tr>`).join('')}}
const colors=['#7dd3fc','#fb923c','#c084fc','#4ade80','#facc15','#f472b6','#38bdf8','#a3e635'];
function setup(c){{const d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=Math.max(2,r.width*d);c.height=Math.max(2,r.height*d);const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,r.height]}}
function viridis(t){{const s=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],u=Math.max(0,Math.min(.999999,t))*(s.length-1),i=Math.floor(u),f=u-i,a=s[i],b=s[i+1];return `rgb(${{a[0]+(b[0]-a[0])*f}},${{a[1]+(b[1]-a[1])*f}},${{a[2]+(b[2]-a[2])*f}})`}}
function drawPopulation(){{const P=R.population,[x,w,h]=setup(document.querySelector('#population')),max=Math.max(1,...P.map(p=>p.active_count)),lastStep=Math.max(1,P.at(-1)?.step??1);x.strokeStyle='#293247';x.beginPath();x.moveTo(44,10);x.lineTo(44,h-28);x.lineTo(w-8,h-28);x.stroke();x.strokeStyle='#7dd3fc';x.lineWidth=2;x.beginPath();P.forEach((p,i)=>{{const px=44+(w-54)*p.step/lastStep,py=10+(h-38)*(max-p.active_count)/max;i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke();x.fillStyle='#8995a8';x.fillText(String(max),5,16);x.fillText('0',25,h-29);x.fillText('0',44,h-10);x.fillText(String(lastStep),w-40,h-10);const firstGrowth=P.find(p=>p.active_count>P[0].active_count);document.querySelector('#population-summary').textContent=`initial ${{P[0].active_count}} · final ${{finalCount}} · peak ${{peakCount}}${{firstGrowth?` · first increase at step ${{firstGrowth.step}}`:' · no population increase detected'}}`;}}
function draw(){{const p=particles[+psel.value],range=groups[gsel.value]||groups['Chemical values'],idx=[...Array(range[1]-range[0])].map((_,i)=>i+range[0]),samples=p.samples;
 document.querySelector('#spawn').textContent=p.spawn_step==null?'not spawned during capture':`spawned by step ${{p.spawn_step}} · ${{samples.length}} samples`;
 if(!samples.length){{setup(document.querySelector('#lines'));setup(document.querySelector('#heatmap'));document.querySelector('#legend').textContent='This particle slot was not active during the captured interval.';document.querySelector('#hover').textContent='';return}}
 let vals=[];samples.forEach(s=>idx.forEach(i=>vals.push(sampleValues(s)[i])));let lo=Math.min(...vals),hi=Math.max(...vals);if(!(hi>lo)){{lo-=1;hi+=1}}
 let [x,w,h]=setup(document.querySelector('#lines'));x.strokeStyle='#293247';x.beginPath();x.moveTo(44,10);x.lineTo(44,h-28);x.lineTo(w-8,h-28);x.stroke();
 idx.forEach((fi,k)=>{{x.strokeStyle=colors[k%colors.length];x.beginPath();samples.forEach((s,j)=>{{const px=44+(w-54)*(samples.length<2?0:j/(samples.length-1)),py=10+(h-38)*(hi-sampleValues(s)[fi])/(hi-lo);j?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke()}});
 x.fillStyle='#8995a8';x.fillText(fmt(hi),2,16);x.fillText(fmt(lo),2,h-29);x.fillText(samples[0]?.step??0,44,h-10);x.fillText(samples.at(-1)?.step??0,w-40,h-10);
 document.querySelector('#legend').innerHTML=idx.map((fi,k)=>`<span style="color:${{colors[k%colors.length]}}">■</span> ${{names[fi]}}`).join(' · ');
 drawHeat(p);
}}
function drawHeat(p){{const c=document.querySelector('#heatmap'),[x,w,h]=setup(c),S=p.samples;if(!S.length)return;const cw=w/names.length,ch=h/S.length;
 const summary=activeSummary();S.forEach((s,y)=>names.forEach((_,i)=>{{const q=summary[i],den=Math.max(q.p95-q.p05,1e-9),t=(sampleValues(s)[i]-q.p05)/den;x.fillStyle=viridis(t);x.fillRect(i*cw,y*ch,cw+.5,ch+.5)}}));
 c.onmousemove=e=>{{const r=c.getBoundingClientRect(),i=Math.min(names.length-1,Math.floor((e.clientX-r.left)/r.width*names.length)),j=Math.min(S.length-1,Math.floor((e.clientY-r.top)/r.height*S.length)),s=S[j];document.querySelector('#hover').textContent=`step ${{s.step}} · ${{names[i]}} = ${{fmt(sampleValues(s)[i])}}`}};
}}
psel.onchange=gsel.onchange=draw;space.onchange=()=>{{renderStats();draw()}};addEventListener('resize',()=>{{drawPopulation();draw()}});gsel.value='Chemical values';renderStats();drawPopulation();draw();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=root / "checkpoints" / "best.npy")
    p.add_argument("--meta", type=Path, default=root / "checkpoints" / "best_meta.json")
    p.add_argument("--output", type=Path, default=root / "policy_input_report.html")
    p.add_argument("--steps", type=int, default=None, help="default: checkpoint macro_steps")
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--tracked", type=int, default=5)
    p.add_argument("--seed", type=int, default=None, help="default: checkpoint winner_seed/seed")
    p.add_argument(
        "--initial-particles", type=int, default=None,
        help="override best_meta.json initial_particle_count for both search and measured replay",
    )
    p.add_argument(
        "--search-for-split", action=argparse.BooleanOptionalAction, default=True,
        help="randomize policies until one divides, then restart it for capture (default: enabled)",
    )
    p.add_argument("--search-steps", type=int, default=400, help="macro steps allowed per randomized policy")
    p.add_argument(
        "--max-search-attempts", type=int, default=0,
        help="stop unsuccessfully after N policies; 0 keeps searching until interrupted",
    )
    p.add_argument(
        "--randomization-seed", type=int, default=None,
        help="seed for reproducible policy randomization (default: rollout seed)",
    )
    return p.parse_args()


def resolve_checkpoint_path(path: Path, root: Path, label: str) -> Path:
    """Resolve inputs from either the caller's cwd or trainer/ and provide
    an actionable archived-checkpoint hint when the current run has not
    produced its first checkpoint yet."""
    candidates = [path]
    if not path.is_absolute():
        candidates.append(root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    runs_dir = root / "checkpoints" / "runs"
    archived = sorted(
        (
            run for run in runs_dir.iterdir()
            if run.is_dir()
            and (run / "best.npy").is_file()
            and (run / "best_meta.json").is_file()
        ),
        reverse=True,
    ) if runs_dir.is_dir() else []
    hint = ""
    if archived:
        latest = archived[0]
        hint = (
            "\nA new training run archives the previous best files under checkpoints/runs/."
            f"\nLatest complete archived checkpoint: {latest}"
        )
    raise SystemExit(f"{label} file not found: {path}{hint}")


def main() -> int:
    args = parse_args()
    if (
        args.sample_every < 1 or args.tracked < 1 or args.tracked > 32
        or args.search_steps < 1 or args.max_search_attempts < 0
        or (args.initial_particles is not None and args.initial_particles < 1)
    ):
        raise SystemExit("--sample-every/search-steps must be >=1, --tracked in [1,32], --initial-particles >=1, and --max-search-attempts >=0")
    root = Path(__file__).parent
    args.meta = resolve_checkpoint_path(args.meta, root, "metadata")
    args.weights = resolve_checkpoint_path(args.weights, root, "weights")
    meta = json.loads(args.meta.read_text())
    weights = np.load(args.weights).astype(np.float32)
    channels = int(meta.get("channels", CHEM_CHANNELS))
    architecture = meta.get("policy_architecture", STATELESS_ARCHITECTURE)
    hidden = int(meta.get("hidden_dim", policy_hidden_dim(architecture)))
    layout = weight_layout(channels, hidden, architecture)
    expected = layout["total_floats"]
    if len(weights) != expected:
        raise SystemExit(f"incompatible weights: checkpoint has {len(weights)} floats, current {channels}×{hidden} policy expects {expected}")
    steps = int(args.steps if args.steps is not None else meta["macro_steps"])
    seed = int(args.seed if args.seed is not None else meta.get("winner_seed", meta.get("seed", 0)))
    growth_steps = meta.get("growth_steps")
    initial_particle_count = int(
        args.initial_particles
        if args.initial_particles is not None
        else meta.get("initial_particle_count", INITIAL_PARTICLE_COUNT)
    )
    particle_cap = int(meta["particles"])
    if initial_particle_count > particle_cap:
        raise SystemExit(
            f"initial particle count {initial_particle_count} exceeds checkpoint particle cap {particle_cap}"
        )
    device = pick_device()
    core = MpmCore(device)
    core.set_morphology(meta.get("morphology_blur_sigma", MORPHOLOGY_BLUR_SIGMA), meta.get("morphology_density_reference", MORPHOLOGY_DENSITY_REFERENCE))
    substeps = int(meta.get("substeps_per_macro", 1))
    core.set_material(
        meta.get("material_e", MATERIAL_E), meta.get("material_nu", MATERIAL_NU),
        meta.get("material_hardening", MATERIAL_HARDENING),
        elasticity=meta.get("material_elasticity", MATERIAL_ELASTICITY),
        growth_duration_macro_steps=meta.get("growth_duration_macro_steps", GROWTH_DURATION_MACRO_STEPS),
        growth_max=meta.get("growth_max", GROWTH_MAX), growth_threshold=meta.get("growth_threshold", GROWTH_THRESHOLD),
        growth_compression_inhibition=meta.get(
            "growth_compression_inhibition", GROWTH_COMPRESSION_INHIBITION
        ),
        growth_anisotropy=meta.get(
            "growth_anisotropy_authority", GROWTH_ANISOTROPY_AUTHORITY
        ),
        substeps_per_macro=substeps,
    )
    core.set_damping(meta.get("damping_loss_fraction", meta.get("damping", DAMPING_LOSS_FRACTION)), substeps)
    core.set_splat_radius(meta.get("splat_radius", SPLAT_RADIUS))
    core.set_repulsion_strength(
        meta.get("repulsion_strength", REPULSION_STRENGTH), meta.get("repulsion_max_delta", REPULSION_MAX_DELTA),
    )
    field_n = int(meta.get("field_n", FIELD_N))
    environment = EnvironmentGPU(device, channels, field_n, field_n, meta.get("decay", DECAY), meta.get("deposit_rate", DEPOSIT_RATE))
    agents = AgentsGPU(
        device, core, environment, channels, hidden,
        meta.get("max_accel", MAX_ACCEL), meta.get("max_strafe", MAX_STRAFE), meta.get("max_env_write", MAX_ENV_WRITE),
        meta.get("max_angular_accel", MAX_ANGULAR_ACCEL), meta.get("angular_damping", ANGULAR_DAMPING),
        meta.get("max_angular_velocity", MAX_ANGULAR_VELOCITY), meta.get("chirality", CHIRALITY),
        meta.get("deposit_distance", DEPOSIT_DISTANCE), particle_cap,
        meta.get("split_displacement", SPLIT_DISPLACEMENT), meta.get("division_cooldown", DIVISION_COOLDOWN),
        meta.get("friction", FRICTION), meta.get("deposit_sigma", DEPOSIT_SIGMA), 1.0,
        meta.get("spawn_x", 0.5), meta.get("spawn_y", 0.5),
        meta.get("elastic_strain_scale", ELASTIC_STRAIN_SCALE),
        meta.get("elastic_strain_inputs_enabled", ELASTIC_STRAIN_INPUTS_ENABLED),
        policy_architecture=architecture,
        internal_state_speed=meta.get("internal_state_speed", INTERNAL_STATE_SPEED),
        interior_support_strength=meta.get("interior_support_strength", INTERIOR_SUPPORT_STRENGTH),
        division_directionality=meta.get("division_directionality", DIVISION_DIRECTIONALITY),
    )
    def restart_rollout() -> TrainingRollout:
        return TrainingRollout(
            core, agents, environment,
            spawn_center=(meta.get("spawn_x", 0.5), meta.get("spawn_y", 0.5)),
            spawn_half_width=meta.get("spawn_half_width", 0.08), gravity=meta.get("gravity", 0.0), seed=seed,
            neural_updates_per_macro=meta.get("neural_updates_per_macro", NEURAL_UPDATES_PER_MACRO),
            communication_speed=meta.get("communication_speed", COMMUNICATION_SPEED),
            initial_particle_count=initial_particle_count,
        )

    def growth_log_state() -> str:
        raw = device.queue.read_buffer(
            agents._agent_state_buffer,
            PARTICLE_META_BUFFER_OFFSET,
            agents._particle_meta_dtype.itemsize,
        )
        state = np.frombuffer(raw, dtype=agents._particle_meta_dtype, count=1)[0]
        rest = core.read_rest_state()[0]
        growth_area = float(rest[0] * rest[3] - rest[1] * rest[2])
        return (
            f"hazard={float(state['divisionHazard']):.5f}/"
            f"{float(state['divisionThreshold']):.5f}, "
            f"cycle={'on' if rest[5] > 0.5 else 'off'}, growth_area={growth_area:.5f}"
        )

    search_attempt: int | None = None
    search_split_step: int | None = None
    found_weights_path: Path | None = None
    if args.search_for_split:
        random_seed = seed if args.randomization_seed is None else args.randomization_seed
        policy_rng = np.random.default_rng(random_seed)
        attempt = 0
        while True:
            attempt += 1
            candidate = random_policy_weights(layout, hidden, policy_rng, architecture)
            agents.load_weights(candidate)
            search_sim = restart_rollout()
            initial_count = core.active_count
            print(f"[search] attempt {attempt}: randomized brain; scanning up to {args.search_steps} steps (initial={initial_count})")
            for search_step in range(1, args.search_steps + 1):
                search_sim.macro_step(
                    substeps,
                    growth_enabled=growth_steps is None or search_step - 1 < growth_steps,
                )
                if core.active_count > initial_count:
                    weights = candidate
                    search_attempt = attempt
                    search_split_step = search_step
                    print(f"[search] SUCCESS attempt {attempt}: population {initial_count} -> {core.active_count} at step {search_step}")
                    break
                if search_step % 100 == 0:
                    print(
                        f"[search] attempt {attempt}: step {search_step}/{args.search_steps}, "
                        f"active={core.active_count}, {growth_log_state()}"
                    )
            if search_split_step is not None:
                break
            print(f"[search] attempt {attempt}: no split after {args.search_steps} steps; randomizing again")
            if args.max_search_attempts and attempt >= args.max_search_attempts:
                raise SystemExit(f"no randomized policy split in {attempt} attempts")

        found_weights_path = args.output.with_suffix(".weights.npy")
        found_weights_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(found_weights_path, weights)
        print(f"[search] saved successful brain to {found_weights_path}")
        agents.load_weights(weights)
        sim = restart_rollout()
        print(f"[measure] restarted successful brain from step 0 with rollout seed {seed}")
        if steps < search_split_step:
            print(f"[measure] extending capture from {steps} to {search_split_step} steps so the verified split is included")
            steps = search_split_step
    else:
        agents.load_weights(weights)
        sim = restart_rollout()
        print(f"[measure] using checkpoint brain {args.weights}")
    probe = PolicyInputProbe(
        device, core, agents, environment, args.tracked,
        meta.get("elastic_strain_scale", ELASTIC_STRAIN_SCALE),
        meta.get("elastic_strain_inputs_enabled", ELASTIC_STRAIN_INPUTS_ENABLED),
    )
    names = feature_names(channels, architecture)
    particles = [{"slot": i, "spawn_step": None, "samples": []} for i in range(args.tracked)]
    all_values: list[list[float]] = []
    all_raw_values: list[list[float]] = []
    population: list[dict[str, int]] = [{"step": 0, "active_count": core.active_count}]
    sampled_times = 0

    def capture(step: int) -> None:
        nonlocal sampled_times
        rows = probe.capture(core, environment)
        sampled_times += 1
        for slot, row in enumerate(rows):
            if row[0] < 0.5:
                continue
            particle = particles[slot]
            if particle["spawn_step"] is None:
                particle["spawn_step"] = step
            raw_start = len(META_NAMES)
            input_start = raw_start + len(names)
            raw_inputs = row[raw_start:input_start].astype(float).tolist()
            inputs = row[input_start:].astype(float).tolist()
            metadata = {name: float(row[i]) for i, name in enumerate(META_NAMES[1:], start=1)}
            particle["samples"].append({
                "step": step, "active_count": core.active_count,
                "meta": metadata, "raw_inputs": raw_inputs, "inputs": inputs,
            })
            all_raw_values.append(raw_inputs)
            all_values.append(inputs)

    capture(0)
    measurement_first_split: int | None = None
    for step in range(1, steps + 1):
        sim.macro_step(substeps, growth_enabled=growth_steps is None or step - 1 < growth_steps)
        population.append({"step": step, "active_count": core.active_count})
        if measurement_first_split is None and core.active_count > population[0]["active_count"]:
            measurement_first_split = step
            print(f"[measure] split replayed: population {population[0]['active_count']} -> {core.active_count} at step {step}")
        if step % args.sample_every == 0 or step == steps:
            capture(step)
        if step % max(1, steps // 20) == 0:
            print(f"[measure] step {step}/{steps}: active={core.active_count}")

    report = {
        "version": 1, "channels": channels, "policy_architecture": architecture, "feature_names": names,
        "metadata_names": META_NAMES[1:], "particles": particles, "population": population,
        "summary": percentile_summary(all_values, names),
        "raw_summary": percentile_summary(all_raw_values, names), "samples": sampled_times,
        "normalization": {
            "chemical_value_scale": CHEMICAL_VALUE_INPUT_SCALE,
            "chemical_gradient_scale": CHEMICAL_GRADIENT_INPUT_SCALE,
            "morphology_occupancy": "clamp(2*x-1,-1,1)",
            "morphology_gradient_scale": MORPHOLOGY_GRADIENT_INPUT_SCALE,
            "elastic": "already normalized; unchanged",
            "private_state": "tanh(state), state clamped to [-4,4]" if architecture == STATEFUL_ARCHITECTURE else "not present",
        },
        "search": {
            "enabled": args.search_for_split, "attempt": search_attempt,
            "split_step": search_split_step,
            "steps_per_attempt": args.search_steps,
            "randomization_seed": seed if args.randomization_seed is None else args.randomization_seed,
            "found_weights": str(found_weights_path) if found_weights_path else None,
        },
        "settings": {"steps": steps, "sample_every": args.sample_every, "tracked": args.tracked, "seed": seed,
                     "initial_particles": initial_particle_count,
                     "weights": str(args.weights), "meta": str(args.meta)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    args.output.write_text(build_html(report))
    print(f"wrote {args.output}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

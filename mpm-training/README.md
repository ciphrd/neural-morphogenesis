# MPM tissue growth

Research system for evolving local neural policies that grow a continuous material toward a target shape. Python runs headless WebGPU training; the React viewer replays policies with the same WGSL simulation shaders.

## Run

```sh
cd trainer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_server.py --target circle
```

In another terminal:

```sh
cd viewer
npm install
npm run dev
```

The viewer requires WebGPU. It starts with a random policy from the shared defaults and connects to the training server. For training without the server, use `trainer/.venv/bin/python trainer/evolve.py --help` and run `evolve.py` from the trainer directory.

## Configuration

**`core/config.json` is the source of all shared default configuration.**

| Section | Responsibility |
| --- | --- |
| `simulation` | Grid, time step, numerical limits, refinement and material yield bounds |
| `run` | Physics, growth, policy selection, chemistry, training, fitness and stop settings |
| `density` | Numerical sampling-density scaling and supported range |
| `chemistry` | Channel profiles and homogeneous experimental profile |
| `policy` | Head initialization and mutation scales |
| `initialConditions` | Available perturbation presets |
| `viewer` | Playback, rendering, tools and lab defaults |
| `server` | Default server host and port |

Python `config.py` loads the JSON; `simulation_settings.py` exposes typed values and derived quantities. TypeScript imports the JSON directly. CLI arguments and viewer controls override a run's settings. Saved run settings record that run's actual configuration; they are outputs, not another default source. `core/density_cases.json` contains parity test fixtures.

The current growth schema is version 13. Older settings and checkpoints are not migrated or adapted; start new runs after this cleanup.

## Simulation

The unit-square domain is periodic. Material is represented by a conforming triangular mesh with one numerical sample per triangle. A sample carries physical deformation, plastic state, a stress-free growth tensor, transported vertices, original area, and quadrature weight.

Each macro step builds the morphology field, runs the configured neural communication rounds, publishes a two-dimensional growth vector, projects growth onto the mechanics grid, and refines triangles. MPM substeps transfer particle momentum and stress to the grid, update grid velocity, and gather velocity and constitutive updates back to the material. Shared vertices use the same velocity interpolation.

Growth continuously changes stress-free material area. Refinement changes numerical resolution while conserving represented material and inheriting chemistry/private state. Refinement has a geometric demand threshold, minimum child weight, and capacity handling; it is not a biological division clock. A global material-area budget and compression feedback can arrest growth.

The policy senses chemical values and gradients in a local frame, morphology, and optional elastic strain. It outputs chemical delta rates and a local growth vector. All policies output three sigmoid RGB values in [0, 1] for the “NN output” particle color mode; recurrent policies also update eight private state values. The default policy is stateless with 128 hidden units. Recurrent 64/128-unit variants remain available for experiments.

The default chemical architecture is a persistent environment with diffusion, decay, material advection, and triangle-area deposition. Cell-owned projection remains an explicit alternative. Nine default channels use three global, three regional, and three local profiles. Chemistry does not control fluidity or motility.

Optional controls retained include repulsion, initial perturbations, recurrent memory, strain sensing, multi-density evaluation, material budgets and stable-shape stopping. Rendering includes material domains, growth vectors/magnitude, neural and chemical state, fields, and constitutive diagnostics.

## Multi-density evaluation

A density multiplier changes how finely the same physical material is sampled. At multiplier `q`, spacing scales as `1/sqrt(q)`, seed count and numerical capacity scale with `q`, and each sample's mass/volume scales as `1/q`. Chemical observation-grid resolution stays fixed; repulsion parameters are adjusted with spacing.

Training can evaluate each candidate at several multipliers and aggregate fitness using the worst score or their mean. This checks whether a learned behavior survives a change in numerical resolution. It does not mean multiple tissue types or physical densities. The default list is `[1.0]`, so ordinary training evaluates only one density.

## Validation and code map

- `core/`: shared compute shaders and canonical config; see `core/README.md` for buffer layouts.
- `trainer/`: GPU orchestration, evolution, fitness, server and diagnostic checks.
- `viewer/`: browser GPU orchestration, React controls and rendering.
- `CORE_FEATURE_SURVEY.md`: historical survey made immediately before the cleanup.

Run Python checks from `trainer/` with `.venv/bin/python <check>.py`. Focused suites include `continuous_growth_check.py`, `conforming_refinement_check.py`, `vertex_transport_check.py`, `split_weight_check.py`, `growth_check.py`, `chemical_transfer_check.py`, `chemical_channels_check.py`, `density_check.py`, `density_gpu_check.py`, `initial_conditions_check.py`, `seed_blob_check.py`, `elastic_diagnostics_check.py`, `domain_render_check.py`, `policy_parameters_check.py`, and `rollout_control_check.py`. GPU checks require a native GPU adapter. Build the viewer with `npm run build --prefix viewer` from the repository root.

Other design notes and saved research snapshots describe experiments at their capture time; use the source and this README for the current model.

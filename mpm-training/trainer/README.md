# Python trainer

Headless WebGPU orchestration for the shared MPM tissue-growth shaders. Defaults come from `../core/config.json` through `config.py`; `simulation_settings.py` exposes typed and derived values without maintaining a second set of defaults.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python evolve.py --help
.venv/bin/python train_server.py --target circle
```

`evolve.py` evaluates and mutates policy populations. `parallel_workers.py` owns worker-local GPU systems. `training_sim.py` seeds and advances rollouts. `mpm_core.py`, `agents_gpu.py`, and `environment_gpu.py` allocate buffers and schedule the shared shaders. `domain_fitness.py` scores the transported material geometry and provides stable-shape stopping. `train_server.py` publishes run settings, generation records, weights and diagnostic images.

Growth and numerical refinement are separate: continuous stress-free tensor growth creates material; conforming triangle subdivision conserves it. Sample counts are synchronized after refinement to size subsequent GPU dispatches. Native GPU submission is chunked where necessary to avoid excessive outstanding work.

Run focused checks with `.venv/bin/python <name>_check.py`. See the root README for the main regression suites. A short end-to-end smoke run is:

```sh
.venv/bin/python evolve.py --generations 1 --population 2 --elites 1 \
  --macro-steps 4 --particles 64 --workers 1 --checkpoint-every 1 \
  --checkpoint-dir /tmp/mpm-smoke
```

Run metadata and weights must match the current model schema; there is no checkpoint migration layer. Archived research snapshots retain their original meaning and are not acceptance baselines for the current layout.

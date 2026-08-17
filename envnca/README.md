# envnca

A from-scratch sibling to `../trainer`, same broad concept (agents sense a
chemical gradient and run it through a neural update rule) with one
deliberate architectural inversion: **chemicals live in the environment,
not in the agents.** This version has a fixed population (no growth) —
see "Not built yet".

In `trainer/`, each node carries its own chemical vector, and the field
any node senses is an implicit Gaussian sum over every other node's
vector, recomputed from scratch on every query. Here, the environment is
a real, dense `(C, H, W)` tensor — a 512×512 grid by default — resident
on the GPU. Agents have continuous positions inside that grid; each step
they sense the field's value and gradient at their own position (bilinear
interpolation), run that through a neural network, and the network's
output is what they *deposit back into the grid* — not a delta to their
own state, since they no longer have chemical state of their own.

The grid has its own dynamics independent of the agents: after writes are
deposited, it diffuses (mass-preserving blur) and decays slightly each
step — the grid-native replacement for the smooth, long-range influence
the old Gaussian-sum field gave for free. Without it, the field would only
ever be nonzero exactly where an agent has stood, with nothing for a
*nearby* agent to sense.

## Layout

- `environment.py` — the GPU chemical grid: `sample_value_and_gradient()`
  (bilinear gather, gradient via a whole-grid Sobel conv2d), `deposit()`
  (bilinear scatter-add), `step_dynamics()` (diffuse + decay).
- `agent_state.py` — per-agent tensors (position, velocity), all
  batched, fixed population — no chemicals, no id vector, no energy (see
  "Not built yet" below).
- `update_rule.py` — the network itself: `Dense(128) -> tanh -> Dense`,
  input = sensed value + local-frame gradient, output = environment
  write + local-frame acceleration.
- `simulation.py` — `Simulation.step()`: one fully-batched, GPU-resident
  step tying the three together — sense, decide, move, write.
- `target.py`, `distance.py`, `alignment.py` — target-shape loading
  (`envnca/targets/*.json`, same pixel-export format and files as
  `trainer/backend/targets/`) and rotation/translation-invariant Chamfer
  distance — ported essentially verbatim from `trainer/backend`, generic
  over what "points" means (there, node positions; here, agent
  positions).
- `evolve.py` — evolutionary training CLI: same (mu, lambda) + elitism
  approach as `trainer/backend/evolve.py` (random search over
  `UpdateRule`'s weights, scored by Chamfer distance against a target),
  adapted to run rollouts sequentially on the GPU instead of fanning them
  out across CPU worker processes — see its own module docstring for why.
  Headless: console progress + checkpointing only, no window — see
  "Visualization" below.
- `device.py` — `pick_device()`, shared so every entry point picks
  CUDA -> MPS -> CPU the same way.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evolve.py --target circle --population 24 --generations 100
```

Trains `UpdateRule`'s weights against a target shape from `targets/`,
checkpointing the best-so-far to `checkpoints/best.npy` +
`checkpoints/best_meta.json` every `--checkpoint-every` generations.
Device is auto-selected: CUDA if available, otherwise Apple's MPS
backend, otherwise CPU.

## Visualization

This project used to have its own matplotlib window (`run.py`, plus a
live fitness-chart-and-replay window built into `evolve.py`) — both have
been removed in favor of a separate web-based frontend (built elsewhere)
driving this same simulation/training code. `evolve.py` itself no longer
opens anything; it's purely a console-logged, checkpointed training run,
the same role `trainer/backend/evolve.py`'s own CLI plays for that
project.

Note on reproducibility: rollouts are deterministic given the same
weights + seed *up to float32 GPU rounding* (observed ~1e-6 relative
difference between two runs of the same seed on MPS — negligible next to
real generation-to-generation fitness differences, but not bit-exact).
The only actual source of randomness in a rollout is the initial agent
jitter (`agent_state.py`'s `seed()`, driven by an explicit
`torch.Generator`) — the simulation itself has no other stochastic step.

## Carried over from `trainer/backend` unchanged in spirit

Velocity-based motion (a single persistent 2D velocity; heading is always
derived as `atan2(vy, vx)`, never stored) and local-frame sensing/
acceleration rotation — neither has anything to do with where chemicals
live, so there was no reason to redesign them. See `update_rule.py`'s and
`simulation.py`'s docstrings for the details.

## Not built yet

This is a first cut, scoped to what was actually asked — an env-based
chemical field, gradient sensing, NN-driven writes, GPU-resident, 512×512
— not a full port of every `trainer/backend` mechanic:

- **No growth.** The population is fixed at whatever `--population` seeds
  it with; no splitting, no energy budget, no death. This is what keeps
  every step a fixed-shape GPU op with zero host syncs (see
  `simulation.py`'s module docstring) — reintroducing growth means either
  paying a per-step sync to find out how many agents split, or a
  fixed-buffer-plus-alive-mask redesign to avoid it.
- **No physics/collision pass.** Agents can and do overlap; nothing
  pushes them apart. `trainer/backend/physics.py`'s tension + collision
  solver is a substantial separate subsystem — porting it to grid-space
  on GPU is a reasonable next step, not something folded in here.
- **No `id` vector / homophilic adhesion** — that's a physics concept,
  and there's no physics pass yet.
- **Training runs sequentially, not batched-across-candidates on the
  GPU.** `evolve.py` evaluates one candidate's rollout at a time in a
  single process rather than fanning the population out across CPU
  worker processes (`trainer/backend/evolve.py`'s approach, which
  doesn't map cleanly onto one shared GPU device) — fast enough at
  current population/step sizes, but batching the whole *candidate
  population* into an extra GPU tensor dimension is the natural next
  step if this ever becomes the bottleneck. See `evolve.py`'s own module
  docstring.

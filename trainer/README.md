# trainer — spec

A particle-based growth system: a population of nodes in continuous 2D
world space, no persistent graph topology, that grows by nodes splitting
and settles under simple physics. The end goal is for growth to be driven
by a per-node neural update rule, evaluated identically at every node
(a Neural Cellular Automata–style local rule), so that a target shape
(loaded from `sculpture`'s pixel export) emerges from repeated local
decisions rather than being built top-down.

This document specs the intended system. See "Implementation status" at
the end for what's actually built vs. still to do.

## Why no graph

Earlier iterations of this idea (see `../trainer-3d`) represented growth
as an explicit graph: nodes, edges, triangular faces, each face growable
exactly once. That fixes topology at creation time and makes growth a
discrete combinatorial choice (which face to extrude).

This version drops that: nodes are just points with per-node state.
Connectivity is never stored — every step, physics looks at raw pairwise
distances and figures out who's touching whom from scratch. Splitting a
node doesn't wire up an edge, it just places a new point nearby and lets
physics (surface tension + collision) sort out the consequences. This
trades discrete/combinatorial growth for continuous/emergent growth, and
is a better fit for a rule that's meant to be learned rather than
hand-designed.

## Node state

Each node carries a persistent state vector of `N` dimensions, split into:

| Field       | Size  | Meaning                                                                 |
| ----------- | ----- | ----------------------------------------------------------------------- |
| `id`        | 3     | Encodes the node's identity in the embedding space.                     |
| `chemicals` | `N-3` | Free-form signal channels the network uses to evolve complex behaviors. |

Physical position (`x, y` in world space, driven by physics) is tracked
separately from this state vector — `id` is part of the _learned_ state,
not the node's literal coordinates.

A node also carries **energy**, a single scalar tracked separately from
this `id`/`chemicals` vector (`Graph.energy`, not something the network
writes to directly) — see "The per-node update rule" and "Splitting"
below for how it gates growth.

## World-space physics

Two purely distance-based forces, recomputed fresh from raw positions
every relax pass — no cached topology, no notion of "this pair used to be
connected":

- **Surface tension** (soft, breakable, short-range): any pair within
  tension range but not yet touching is pulled toward contact distance,
  only partially per iteration (tension stiffness < 1 → elastic, not
  rigid). Outside tension range there's no force at all, which is what
  makes bonds breakable rather than permanent.
- **Collision** (hard, unconditional): nodes are circles of fixed radius
  and must never overlap. Resolved as a separate cleanup pass after
  tension, so a hard inequality constraint never fights a soft attraction
  in the same solve. Purely geometric — never modulated by chemistry.

Both passes use Jacobi-style updates (each node's correction is the
average of all its simultaneously-active pairwise corrections from one
consistent snapshot) rather than Gauss-Seidel, which is what keeps the
solve stable once more than a few nodes crowd the same region.

Tension is further modulated per-pair by the cosine similarity of the
two nodes' `id` vectors, clipped to `[0, 1]` — homophilic adhesion:
similarly-"identified" nodes pull together at full strength, orthogonal
or opposed identities feel no pull at all even in range, rather than
becoming repulsive. Proximity decides whether a pair is *eligible* for
tension; `id` compatibility decides how strongly they actually feel it —
which gives the update rule's `id` deltas an actual physical consequence
to learn to exploit (e.g. clustering same-identity nodes together).

_(Current implementation: `backend/physics.py`. Tension/collision
strength are currently hardcoded constants, not runtime-tunable
parameters — see "Open questions".)_

## The per-node update rule

At each simulation step, every node stochastically evaluates itself
against its local neighborhood and decides whether to change its own
state and/or split:

1. **Sense.** Compute the gradient of each of the `N-3` chemical channels
   along the two world-space in-plane directions, `∂/∂x` and `∂/∂y`, at
   the node's own position. This is a spatial derivative of the
   _chemical values_ held by nearby nodes (e.g. via a kernel-weighted
   local estimate over neighbors) — distinct from `backend/substrate.py`'s
   existing multi-scale Gaussian field, which encodes _node density_
   around a point, not the gradient of any chemical channel's value. That
   module's exact-gradient-of-a-Gaussian-sum machinery is directly
   reusable for this, but the field it currently builds isn't the one
   this step needs.

2. **Decide.** Feed `[chemicals, ∂chemicals/∂x, ∂chemicals/∂y, energy]`
   — a `3·(N-3) + 1`-dimensional vector, energy normalized to roughly
   `[-1, 1]` first so it's on the same scale as everything else — into a
   small MLP:

   ```
   Dense(128) → tanh → Dense(18)
   ```

3. **Act.** The 18-wide output is split into:
   - `1` — split *desire*: **not** used directly as a Bernoulli
     parameter anymore — see "Splitting" below, energy gates it first
   - `N-3` — a delta added to the chemical channels
   - `3` — a delta added to the `id` channels
   - `2` — a spawn-direction vector: which way this node would place a
     child *if* it split this step, computed every step regardless of
     whether the energy-gated draw actually lands (it can't be computed
     only after the fact — see "Splitting"). Not part of the persistent
     `id`/`chemicals` state above: it's a fresh reading each step, not
     something added to a running value, so there's no accumulation to
     bound the way `id`/`chemicals` need. `Graph.spawn_directions` holds
     the latest one per node, normalized, purely for spawning and
     display (see the "Spawn direction" node color mode).

Since the output width is fixed at 18, `1 + (N-3) + 3 + 2 = 18 → N = 15`:
3 `id` channels and 12 `chemicals` channels, giving a `37`-dimensional
MLP input (`12` chemicals + `12` × `∂/∂x` + `12` × `∂/∂y` + `1` energy).

_(Implemented in `backend/update_rule.py` — a PyTorch `nn.Module` used
purely for inference (`@torch.no_grad()`), not (yet) trained by
backprop; see "Training". Sensing uses
`substrate.weighted_field_and_gradient` at a single fixed bandwidth,
`SENSING_SIGMA`, rather than a learned or multi-scale neighborhood. All
nodes are updated from one pre-step snapshot — Jacobi-style, like
physics — and a node that splits passes its post-delta state to its
child, matching "Splitting" below.)_

## Splitting

Splitting is **energy-gated**, not a free-standing Bernoulli draw on the
network's raw output — motivated by growth otherwise happening far too
fast for any temporally-extended behavior to matter (a single node could
reach 24 in 8 steps with an untrained/random network): every node has an
energy budget (`Graph.energy`, starts at `INITIAL_ENERGY = 100`),
regenerated by a small flat amount each step
(`ENERGY_INJECTION ± ENERGY_INJECTION_NOISE`, clamped to
`[0, MAX_ENERGY]`) — a per-node flat rate, not something that scales
with population, which is what actually bounds how fast the *whole
organism* can grow regardless of how many nodes want to split at once.

1. Below `MIN_SPLIT_ENERGY` (`75`), a node cannot split at all — a hard
   gate, zero chance, independent of what the network's split output says.
2. Above that threshold, the network's own split probability is scaled
   by an `energy_weight` ramping linearly from `0` at `MIN_SPLIT_ENERGY`
   to `1` at `MAX_ENERGY` — "the higher the energy, the more chances,"
   without ever letting the network exceed its own stated probability.
   `effective_split_prob = split_prob * energy_weight`, then a single
   Bernoulli draw against that.
3. On an actual split: a new node is added at contact distance, in the
   direction given by this step's spawn-direction output (see "The
   per-node update rule") — falling back to a random angle only if that
   vector is too close to zero to normalize (the network expressed no
   preference this step). The manual "Add node" tool
   (`Graph.split_node`) has no learned direction to offer and always
   uses a random angle, same as before this existed. The child copies
   the parent's full state (`id` + `chemicals`), **and the parent's
   (post-injection) energy is divided evenly between the two** —
   mirroring cell division spending the resources it took to reach the
   threshold in the first place, and the direct mechanism that paces
   further splits: a freshly-split node starts around half its parent's
   energy, usually well under `MIN_SPLIT_ENERGY`, so it needs another
   full regeneration cycle before it's eligible again.
- Physics (tension + collision) resolves whatever overlap or
  rearrangement the new node causes on the next relax pass — splitting
  itself never checks for or avoids overlap. Skipped entirely on a step
  where nothing split (and, in `evolve.py`, no damage removed a node
  either) — an unchanged node set is already at equilibrium from the
  previous relax, so re-running it would be pure wasted `O(n²)` work; see
  `update_rule.step()`'s and `evolve.py`'s `rollout()`'s return/tracking
  of whether anything actually changed this step.

_(Constants: `backend/update_rule.py`. The manual "Add node" tool
(`Graph.split_node`) also splits energy 50/50 on use, but bypasses the
threshold — an explicit user action always succeeds regardless of
budget.)_

## Target matching

Growth is scored against a target point cloud loaded from `sculpture`'s
pixel export (`{nx, ny, pixels: [{x,y}, ...]}`, recentered and scaled so
one pixel = one graph edge length — `backend/target.py`):

- `chamfer_distance` (`backend/distance.py`) — symmetric nearest-neighbor
  distance in both directions (grown-to-target and target-to-grown), so
  neither "grew somewhere the target doesn't cover" nor "target has
  uncovered regions" scores well by accident.
- `best_fit_distance` (`backend/alignment.py`) — precise rotation +
  translation search (12-restart Nelder-Mead) for the _best possible_
  Chamfer fit, used for one-off reporting/visualization calls where
  search cost doesn't matter.
- `training_alignment_distance` (`backend/alignment.py`) — cheaper
  rotation/translation alignment (analytic centroid-matching translation
  + coarse rotation grid search) used as the actual evolutionary fitness
  signal in `evolve.py`'s `rollout()`. Growth isn't pinned to any
  particular pose (nothing anchors the structure during physics relax,
  and the target's own orientation is an arbitrary artifact of however it
  was drawn), so fitness rewards getting the _shape_ right independent of
  pose, not accidentally landing in the target's exact orientation.

## Training

Two approaches are in play; **random evolution is the current, first
attempt** — no backprop, no differentiable-physics porting required, and
it works with `update_rule.py` exactly as it's written today (inference
only). The differentiable-unroll approach further below remains the
documented plan for later, if/once evolution's results or speed hit a
ceiling worth spending the JAX-porting effort to push past.

### Random evolution (implemented — `backend/evolve.py`)

Population-based `(μ, λ)` evolution strategy with elitism, no gradients
anywhere: each generation, every candidate weight vector is loaded into
`UpdateRule`, run through a full seed-to-`--steps` rollout
(`update_rule.step` + `physics.relax`, same as the live websocket `step`
message), and scored by (non-aligned) Chamfer distance to a target —
lower is better. The best `--elites` candidates survive unmutated into
the next generation; the rest of the population is refilled with
Gaussian-noise-mutated copies of a randomly-chosen elite. `torch.nn.utils
.parameters_to_vector`/`vector_to_parameters` flatten a model's weights
for mutation and load them back in — no autograd involved at any point.

```
python evolve.py --target circle --generations 100 --population 24
```

Key flags: `--target` (name in `backend/targets/`), `--population`,
`--elites`, `--generations`, `--steps` (sim steps per rollout),
`--mutation-sigma`, `--damage-prob`/`--damage-fraction` (see below),
`--seed`. Saves the best weight vector + a metadata JSON to
`backend/checkpoints/` periodically (`--checkpoint-every`) and at the
end. Population size, mutation sigma, and rollout length are starting
guesses, not tuned — expect to adjust once there's a sense of whether
fitness is actually improving generation over generation.

**`backend/train_server.py`** runs the same loop (same flags, plus
`--port`, default `8001`) as its own FastAPI/websocket server instead of
a one-shot CLI run, so it can be watched live in the frontend's
"Training" tab. Population evaluation runs via `asyncio.to_thread` so
the websocket stays responsive while a generation's worth of
physics-heavy rollouts computes. `evolve.py`'s CLI and `train_server.py`
share the exact same flag parser and per-generation logic
(`build_arg_parser`, `run_generation`) so they can never drift apart on
what a flag means.

Growth visualization is entirely **client-side**, not streamed from the
server: each generation, `train_server.py` broadcasts one
`{"type": "generation", weights, steps, maxNodes, sensingSigma, ...}`
message (`UpdateRule.export_weights()`, `{fc1w, fc1b, fc2w, fc2b}` as
nested lists — `nn.Linear`'s `(out_features, in_features)` orientation,
`y = x @ Wᵀ + b`), and the frontend's `frontend/src/sim/` — a hand-rolled
TypeScript reimplementation of sensing (`substrate.ts`), the MLP forward
pass (`updateRule.ts`), splitting (`graph.ts`), and physics
(`physics.ts`, restructured as a generator so `requestAnimationFrame`
can drive the Jacobi settling motion frame-by-frame) — replays the
winning weights entirely in the browser (`sim/runner.ts` orchestrates
one full seed-to-`steps` replay; `sim/useLocalSimulation.ts` is the
React hook driving it). No per-frame network dependency, so growth
animates smoothly regardless of latency. An earlier version streamed a
`{"type": "state", ...}` message per server-side simulation step
(reusing `main.py`'s shape) — that's gone; the server no longer runs any
rollout for visualization, only for scoring.

Numerical parity between the two implementations (Python authoritative,
TypeScript for display only) was spot-checked before trusting this: a
fixed set of positions/id-vectors/weights run through both
`physics.relax()`/`UpdateRule.step_numpy()` and their TS counterparts,
outputs compared within float32/float64 rounding tolerance (~1e-8). No
fixture lives in the repo — it was a throwaway comparison script, not a
permanent test — so re-derive the same kind of check by hand after any
change to either `physics.py`/`update_rule.py` or their TS counterparts,
since nothing currently catches the two silently drifting apart.

- **Damage training** — implemented as `--damage-prob`/`--damage-fraction`
  (both default to effectively off: `--damage-prob 0.0`). Each
  simulation step, with probability `--damage-prob`, a random
  `--damage-fraction` of non-pinned nodes are deleted
  (`Graph.remove_nodes`) before that step's physics relax, and the
  rollout continues from the damaged state.
- **Growth regularization** — still deliberately *not* a node-count or
  split-rate term added to the *fitness*; the intent is still to see
  whether the network learns to stop growing at the target boundary on
  its own, from Chamfer distance alone. What changed: growth is now
  energy-gated (see "Splitting"), which is a mechanistic rate limit on
  the *simulation itself*, not a scoring penalty — it doesn't tell
  evolution "overgrowing is bad," it just makes overgrowing physically
  slower, so a fast-and-sloppy strategy no longer wins by default before
  a more deliberate one gets enough steps to even be expressible. Whether
  the network learns restraint *within* that budget is still an open
  question the fitness function alone is responsible for answering.
- **Max node count** — implemented as `MAX_NODES = 200` in
  `update_rule.py`, independent of fitness/loss: past that count, split
  decisions are still computed but simply not acted on. A safety valve
  against runaway growth blowing up rollout compute (`physics.relax` is
  `O(n²)`), not a training signal — kept deliberately separate from
  "growth regularization" above. Currently a module constant, not yet a
  CLI flag.

### Differentiable unroll (future — not implemented)

Growing-NCA style: from a seed, run update-rule + physics-relax for a
randomly-sampled number of steps, backprop the Chamfer loss through the
whole rollout, repeat with a new random rollout length each time. This
requires porting `physics.py` (and `update_rule.py`) off plain
numpy/`@torch.no_grad()` onto something that tracks gradients through
the *simulation*, not just the network — JAX is the natural fit, since
the physics code is already written in the vectorized, stateless style
that ports to `jax.numpy` with minimal changes.

Splitting can't be backpropagated through as a literal Bernoulli sample
the way `update_rule.step` draws it today — the plan is the same trick
Growing NCA uses for its alive-mask: every node always spawns a child,
but the child's "existence weight" starts at the split probability and
fades in continuously rather than being a hard yes/no. Growth stays
differentiable throughout, and a binary outcome emerges naturally once
weights saturate near 0 or 1 through training pressure.

## API surface (current)

- `WS /ws` — pushes `{type: "state", nodes: [...]}` on connect and after
  each processed message (each node carrying `id`, `position`, and
  `spawnDirection` — small enough to broadcast for every node, unlike
  `idVector`/`chemicals` which stay `GET /nodes/{id}`-only); handles
  `{type: "split_node", nodeId}` (manual split) and `{type: "step"}`
  (run one autonomous `update_rule.step` + physics relax for every node
  at once), both followed by a relax and a fresh state push.
- `GET /nodes/{id}` — a single node's position, `idVector`, `chemicals`,
  `energy`, `spawnDirection`.
- `GET /substrate` — the density field + gradient at every node's own
  position (`substrate.field_and_gradient`) — the multi-scale one, for
  inspection; not the chemical-gradient sensing the update rule actually
  uses internally, which isn't exposed over the API.
- `POST /target/load`, `GET /targets`, `POST /targets/{name}/load`,
  `GET /targets/{name}/distance`, `POST /target/clear`,
  `GET /target/points` — target management, backed by files in
  `backend/targets/`.

## Implementation status

Built:

- Physics (surface tension + `id`-modulated collision-free cohesion),
  `id`/`chemicals` node state (`cell_state.py`, `graph.py`), `Graph
  .split_node`/`add_child`/`remove_nodes`, target loading, Chamfer +
  best-fit distance, websocket state sync.
- `backend/substrate.py`: both the general multi-scale density field
  (`field_and_gradient`, still just exposed for inspection via
  `GET /substrate`) and the chemical-value-gradient sensing the update
  rule actually consumes (`weighted_field_and_gradient`).
- `backend/update_rule.py`: the MLP (`Dense(128)→tanh→Dense(18)`),
  sense → decide → act each step (`step()`), the `MAX_NODES` safety cap,
  the learned spawn-direction output steering where a split lands.
  Random, untrained weights by default — behavior is undirected until
  training shapes it.
- `backend/evolve.py`: random-evolution training loop (see "Training").

Not yet built:

- Any trained weights — `evolve.py` exists and runs, but hasn't been run
  long enough (or tuned enough) yet to produce a checkpoint worth calling
  "trained."
- The differentiable-unroll (backprop) training path — porting physics
  and the update rule to a framework that supports gradients through the
  simulation, the soft-split relaxation, the pool/persistence and
  variable-rollout-length training tricks. See "Training" → "future."
- Runtime-tunable physics constants (tension strength/range) — still
  hardcoded.

## Open questions

Split timing, update ordering, split-probability semantics, and the
sensing neighborhood definition were open here previously — all now
resolved, see `update_rule.py`'s own docstring and "The per-node update
rule" above. Still open:

- **Tunable physics.** "Surface tension strength" is named as a rule
  alongside collisions, implying it should be a runtime-adjustable
  parameter — today `TENSION_STIFFNESS`/`TENSION_RANGE` are hardcoded
  constants in `physics.py`.
- **`MAX_NODES` as a CLI flag.** Currently a module constant in
  `update_rule.py`, shared by both the live websocket server and
  `evolve.py` — worth promoting to a parameter if different runs (live
  demo vs. an evolution sweep) turn out to want different caps.
- **Evolution hyperparameter defaults.** `--population`, `--elites`,
  `--mutation-sigma`, `--steps` in `evolve.py` are first-guess starting
  points, not tuned against any actual training run yet.

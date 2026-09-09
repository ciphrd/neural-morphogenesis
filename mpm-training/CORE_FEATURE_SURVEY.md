# Core feature survey — before cleanup, 2026-09-06

> Historical inventory captured before the requested cleanup. Removed features and old configuration paths below are not current behavior. See README.md and core/config.json for the cleaned system.

This surveys the current working tree, including existing uncommitted changes. Scope: shared simulation shaders, Python training orchestration, browser simulation counterparts, configuration, and closely related research utilities. Rendering and application plumbing are covered only where they affect the simulation or explain retained code.

“Active” means reachable under the checked-in normal training defaults, not proven useful to fitness or used in every historical run. “Optional” means implemented but not selected by those defaults. “Obsolete” means the current production computation does not consume the control. Findings come from tracing callers and shader reads; no training runs, benchmarks, or behavioral ablations were performed. No existing files were modified for this survey.

## Current execution model

The system is now a **continuously growing material represented by adaptive triangle samples**, rather than a population of independently dividing cells.

1. Seed a conforming circular triangle mesh, with optional initial perturbations.
2. Build a smoothed morphology field from represented material density.
3. For each neural communication round: prepare the chemical field, sense it, evaluate the policy, and update chemistry.
4. On the final round: apply optional motility and friction, publish growth vectors, integrate them onto the MPM grid, and refine triangles.
5. Apply sampling-capacity and material-area limits; read back the sample count.
6. Run MPM substeps: optional repulsion, clear grid, P2G, grid update, G2P. G2P advances geometry, constitutive state, and rest growth.
7. At selected macro steps, score transported triangle coverage and update the external shape-stop controller.

Sources: [training_sim.py](trainer/training_sim.py), [agents_gpu.py](trainer/agents_gpu.py), [simulation.ts](viewer/src/gpu/simulation.ts), [agents.ts](viewer/src/gpu/agents.ts), [evolve.py](trainer/evolve.py).

## Mechanics and material representation

| Feature | Status | Actual behavior / controlling settings |
|---|---|---|
| GPU MLS-MPM with APIC transfer | Active | Shared WGSL in Python wgpu and browser WebGPU. Quadratic 3×3 transfers; centroid-based momentum/stress. `mpmEnabled=true`. |
| Periodic domain | Active | Positions, triangle vertices, transfer stencils, chemistry and morphology wrap on the unit torus. There are no sticky walls in the current grid-update shader. |
| Explicit triangle geometry | Active | Three advected vertices are authoritative; particle position is their periodic centroid. Vertices sample MPM grid velocity but carry no separate momentum. This is not domain-integrated/CPDI stress transfer. |
| Corotated elasticity | Active | Stress uses `Fe = F * inverse(Fg)`. Defaults: `materialE=10000`, `materialNu=0.2`. |
| Plasticity and hardening | Active, state-dependent | SVD clamps elastic stretches and updates `Jp`; hardening multiplies stiffness. `materialElasticity=1` selects wide bounds `[0.5, 2]`, not “plasticity disabled.” `materialHardening=3`. |
| Growth-scaled mass and volume | Active | Both scale with quadrature weight and `det(Fg)`. Refinement halves represented weight and original area while copying state. Base mass 10, volume 1. |
| Grid damping | Active | `damping≈0.039307`, converted to per-substep retention. |
| Agent velocity friction | Active | `friction=0.9` multiplies velocity on the final communication round, even when strafe is zero. It is additional to grid damping. |
| Gravity | Optional, zero by default | Implemented in grid update; `gravity=0`. |
| Grid speed guard | Active | Hardcoded maximum displacement of half a grid cell per axis per substep. This is a numerical bound, not adaptive timestep selection. |
| Particle repulsion | Optional, zero by default | `repulsionStrength=0` skips all four per-physics-substep repulsion passes. Nonzero strength enables density-gradient forces capped by `repulsionMaxDelta=40`. |
| Density/morphology prepass | Active | Uses the density-building passes from `repulsion.wgsl` even when repulsion forces are off. `splatRadius` therefore remains meaningful. Gaussian morphology blur and bounded occupancy feed the policy. |
| Chemical-controlled fluidity | Optional, zero by default | `materialFluidity=0`; when enabled, principal stretches relax toward their geometric mean, multiplied by positive cell-owned chemical channel index 6. |
| Physics bypass | Optional | `mpmEnabled=false` skips mechanical substeps, not neural updates, chemistry, or refinement passes. |

Fluidity has an architectural dependency worth deciding explicitly: G2P reads **cell-owned** chemical state. The default persistent-environment path does not integrate neural output into that state. Raising fluidity alone therefore does not create neural fluidity control from the evolving environmental channel in a normal zero-state reset.

Sources: [p2g.wgsl](core/p2g.wgsl), [g2p.wgsl](core/g2p.wgsl), [gridUpdate.wgsl](core/gridUpdate.wgsl), [repulsion.wgsl](core/repulsion.wgsl), [morphology.wgsl](core/morphology.wgsl), [mpm_core.py](trainer/mpm_core.py).

## Growth, refinement, and stopping

| Feature | Status | Actual behavior / controlling settings |
|---|---|---|
| Continuous neural growth vector | Active | Two linear outputs define a local vector; its magnitude requests growth and is capped at one before growth-field scatter. Rotated into world space and published once per macro step. No separate division probability head. |
| Vector-first field integration | Active | Volume-weighted signed vectors are averaged on the MPM grid before construction of the expansion tensor. Opposing proposals cancel. |
| Tensor rest growth | Active | G2P exponentiates the sampled tensor into `Fg`, transformed through the elastic rotation. `growthDuration=12` sets the base rate; duration zero disables material growth. |
| Anisotropy control | Active | `growthAnisotropy=1`; interpolates between isotropic and directional tensors without changing trace. The former temporal anisotropy relaxation is absent. |
| Compression inhibition | Active | `growthCompressionFeedback=1`, start/stop both `0.1`: a hard cutoff based on elastic areal compression. Distinct thresholds select smooth inhibition; feedback zero bypasses it. |
| Conforming longest-edge refinement | Active, conditional | Geometry-based demand, edge hashing/linking, propagation, paired slot reservation, midpoint splits, and state copying. Hardcoded demand threshold `1.75`; minimum daughter grown weight `0.3125`. |
| Sampling limits | Active | Runtime sample cap plus allocated buffer capacity. Default cap 400, density-resolved. **Capacity exhaustion or blocked reservation clears the growth field and stops physical growth too.** |
| External stable-shape stop | Active | Every 10 steps, require 3 successful checks, then 20 settling steps with growth off. Failed settling resumes growth. Missing/spill/overlap tolerances are each 0.02. Capacity-blocked or unresolved sampling cannot count as success. |
| Explicit growth horizon | Optional | `growthSteps=null`; a specified cutoff stops the growth command while mechanics and chemistry continue. |
| Deterministic forced growth | Lab/test only | Forced sample ranges, fixed directions and a radial-inward field override. The radial case deliberately injects an isotropic expansion tensor. |
| Biological cell-cycle admission / stochastic division | Superseded | No current hazard-based admission, cooldown decision, signed division-drive remap, or independent division event. Sample creation is numerical refinement. |
| Birth mass ramp | Superseded | Refinement copies physical state immediately; no post-birth mass fade. `appearanceScale` is a separate rendering-related retained field. |
| Coarsening / death / tearing model | Not present in the production path surveyed | Refinement adds samples; there is no corresponding merge/removal or explicit tear detector. Geometric gaps are not a modeled biological tearing operation. |

Sources: [growthField.wgsl](core/growthField.wgsl), [g2p.wgsl](core/g2p.wgsl), [agents.wgsl](core/agents.wgsl), [domain_fitness.py](trainer/domain_fitness.py), [evolve.py](trainer/evolve.py).

## Policy, sensing, and activations

| Feature | Status | Actual behavior / controlling settings |
|---|---|---|
| Stateless policy | Default | `stateless-128`: one 128-wide ReLU hidden layer. With 9 channels: 33 inputs and 14 outputs (9 chemical + 2 growth + 3 RGB). |
| Recurrent policy | Optional | `stateful-64` and `stateful-128`, with 8 private state values and residual/gate heads. Normal `cellMemory="recurrent"` selection maps to stateful-128; stateful-64 remains an explicit comparison option. |
| Chemical perception | Active | Nine values plus forward/lateral gradients. Value multiplier 1; gradient normalization scale 0.045. |
| Morphology perception | Active | Occupancy and two local gradients. Blur sigma 0.01, density reference 1, gradient input scale 0.018. Separate from renderer blur. |
| Mechanosensing | Active, ablatable | Three Hencky-strain inputs: volume, axial, shear; `elasticStrainInputsEnabled=true`, scale 0.15. Disabling preserves the input layout. |
| Chemical-derived local frame | Active | Chemical channel index 3 defines orientation using an L2-clipped gradient. Weak gradients suppress directional perception. Growth rotation retains a deterministic world-X fallback when orientation is undefined. |
| Reflection averaging | Optional, off | `chirality=false`. Enabling evaluates the policy on original and reflected input and averages corrected outputs; the setting enforces reflection symmetry despite its name. |
| Color head | Active, visual only | Stateless RGB logits become sigmoid colors. These colors are not physical actions or inputs to current domain fitness. Recurrent color comes from private state instead. |
| Chemical and growth output activations | Active | Linear chemical, growth-vector, and memory residual outputs. Growth scatter caps vector magnitude. Recurrent gates and display colors use sigmoid; private state is clamped after gated residual integration. |
| Sine hidden activation | Superseded | Current implementations use ReLU; no runtime activation selector. |
| Learned heading / angular dynamics | Superseded | No learned heading head or persistent angular velocity. Angular controls remain in the settings/layout surface. |
| Absolute/spawn-relative position inputs | Absent | Spawn coordinates control initialization, not the production policy input vector. |
| Physical motility from growth vector | Optional, off | `maxStrafe=0`. Nonzero applies world growth vector to velocity; this scale does not set material growth rate. |
| Logical-head initialization/mutation | Active | Shared gains, bias priors and mutation buckets in `policy_parameters.json`; GPU output matrix remains concatenated. |
| CPU/JS policy evaluators | Reference/inspection | Torch `UpdateRule` supports initialization/export and reference forward evaluation; JS evaluator supports network inspection. Live rollout inference is WGSL. |

Sources: [policy_parameters.py](trainer/policy_parameters.py), [update_rule.py](trainer/update_rule.py), [agents.wgsl](core/agents.wgsl), [policyEval.ts](viewer/src/gpu/policyEval.ts).

## Chemistry, seeding, and density

| Feature | Status | Actual behavior / controlling settings |
|---|---|---|
| Persistent environmental chemistry | Default | Transport previous field with MPM velocity, diffuse/decay, sense, add signed neural deltas each round. |
| Cell-owned chemistry projection | Optional | Integrate per-sample bounded chemical state; synchronously project old state before each neural round. Environmental decay/deposit-rate controls do not have the same role here. |
| Multiscale channels | Active | 3 global, 3 regional, 3 local. Resolution scales 0.25/0.5/1 on a 512 base grid; distinct response/relaxation times, decay exponents, diffusion multipliers. |
| Chemical transfer and sensing | Active | Fixed quadratic 3×3 projection/gather, world-area weighting, Sobel gradients on native grids. Gaussian deposit-width settings no longer control this stencil. |
| Density-normalized expression | Default | `normalizeDepositsByLocalDensity=true`: area-weighted expression with coverage attenuation. False selects density-proportional secretion. |
| Chemical amplitude / evolution | Active | `maxEnvWrite=1`, `depositRate=4`, base retention `decay=0.91`, profile response scales. |
| Separate communication clock | Active | `neuralUpdatesPerMacro=1`, `communicationSpeed=4`; timestep split across rounds. Physics defaults to 32 substeps per macro, `DT=0.000015625`. |
| Private-state speed | Optional dependency | `internalStateSpeed=4` matters with recurrence; default stateless policy has no private-state update. |
| Circular triangle seed | Active | Default 5 material units produce **10 triangle samples**, subject to capacity. Seed rotation is deterministic. `spawnX/Y=0.5`; spacing and packing scale govern size. |
| Initial asymmetry presets | Optional, default none | Chemical pole, gradient, smooth noise, geometric elongation, internal-state patch, mechanical-strain patch, handed chemical patches. Internal-state patch requires recurrence. |
| Sampling-density scaling | Active machinery, default 1× | Supported ordinary range 0.5–4. Changes spacing, counts, represented mass/volume and repulsion scales; does not mean changing physical material density. |
| Multi-density evaluation | Optional | Default `trainingDensityMultipliers=[1]`; multiple densities can use worst/mean aggregation. A single density makes that choice immaterial. |

Sources: [chemical_channels.json](core/chemical_channels.json), [environment.wgsl](core/environment.wgsl), [environment_gpu.py](trainer/environment_gpu.py), [density.py](trainer/density.py), [initial_conditions.py](trainer/initial_conditions.py), [triangle_seed.py](trainer/triangle_seed.py).

## Training and research paths

Normal training is evolutionary search with elitism and Gaussian mutation, not backpropagation through MPM. Defaults: population 16, elites 3, mutation sigma 0.05, 50 generations, checkpoints every 5. Each generation shares a rotating seed batch across candidates; default one seed per candidate. Worker processes reuse GPU objects.

The live objective integrates explicit triangle/pixel intersection areas, centers material, searches rotation, and combines multiscale coverage, spill, boundary and overlap/crowding losses. Defaults: resolution 256; weights 1, 1, 0.25, 0.05; outside-distance weight 1. Five snapshots over the final 10% of the rollout feed a mean/worst blend (`fitnessTemporalWorstWeight=0.3`), unless successful settling supplies the score window. Default horizon: 160 macro steps; default target: circle. Target JSON files supply alternative shapes.

Earlier point-density raster scoring and Chamfer alignment are no longer the production objective. They are not all disposable: `domain_fitness.py` imports helpers from `raster.py`; `alignment.py`/`distance.py` still serve rendering and research scripts. `material_domain.py` is a standalone domain-transfer reference experiment, not the production transfer algorithm. Shape, growth, chemistry, density, stability and transport checks are separate research/verification entry points, not runtime feature toggles.

Browser Lab scenarios, live material overrides, growth speed multiplier, interactive deformation, diagnostic fields, policy probes, random policies, and parameter sweeps are experimentation tools. They can alter a browser rollout independently of training metadata. Bloom, display blur, color modes and most render controls do not alter the trained dynamics; morphology blur is explicitly a separate simulation input.

## Controls that no longer affect the production computation

These should be distinguished from useful ablations set to zero. Some are still validated, serialized, or written into buffers; removal requires tracing that surface, but changing their value does not restore the old algorithm.

| Setting / retained element | Finding |
|---|---|
| `maxAccel` | No corresponding physics read in the current agent shader; still exposed as a UI slider. |
| `maxAngularAccel`, `angularDamping`, `maxAngularVelocity` | Angular dynamics removed. JS evaluator also accepts ignored angular arguments. |
| `divisionCooldown` | No cooldown decision; still exposed as “Growth cooldown.” |
| `divisionDriveBoost` | No signed-drive probability remap in current growth. |
| `divisionDirectionality` | Still written into step uniforms; current shader does not read it. |
| `massRampMacroSteps` | Retained metadata/settings for removed physical mass fade. |
| `growthMax` | Retained material uniform; continuous growth does not read it as a cap. |
| `depositDistance` | Deposits are centered under the sample. |
| `depositSigma`, channel `depositSigmaMultiplier` | Carried through config/templates but fixed quadratic transfer does not read them. |
| `chemicalProjectionWeight` | Actual original triangle area already handles sampling density; no shader read. |
| `depositDensityReference` | Retained environment uniform; normalization uses actual deposited area. |
| `spawnHalfWidth` | Explicitly ignored by `TrainingRollout`; compact seed sizing comes from spacing. |
| `GROWTH_ANISOTROPY_RESPONSE_RATE` | Declared/template-substituted, no current calculation uses it. |
| `fitnessTargetOccupancy`, `rasterSigma` | Still parsed/validated/recorded, but not passed to current domain fitness. |
| Precomputed `target_distance_field` rollout argument | Still constructed and transported to workers, unused by current rollout scoring; domain fitness computes its own distance field. |
| `mechanicalGrowthGate()` in agents shader | Uncalled helper; actual contact inhibition remains active in G2P. |
| `spatialUniform01()` and its random-field machinery | Uncalled growth helper; do not confuse this with still-active seed RNG or mutation RNG. |

Renaming rather than deletion is needed for several **live** state slots: `cycleActive` and `growthAngle` now carry growth-vector X/Y; `divisionBias` stores original world area; `mitosisPropensity` displays vector magnitude. These names no longer describe their jobs.

## Where configuration actually lives

| Location | Responsibility / duplication |
|---|---|
| `core/default_run_settings.json` | Main shared run defaults, mixing mechanics, chemistry, policy, training, scoring, stopping, and obsolete settings. |
| `core/constants.json` | Grid/timestep/capacity, shader scales and duplicate policy/run defaults. |
| `core/density.json` | Density model rules, spacing, supported range, packing and repulsion ratios. |
| `core/density_cases.json` | Reference/check fixtures, not normal run defaults. Its mass 1 and field 256 differ from current run defaults 10 and 512. |
| `core/chemical_channels.json` | Channel assignments and multiscale profiles; includes obsolete deposit-width multipliers. |
| `core/policy_parameters.json` | Head initialization and mutation priors. |
| `core/initial_conditions.json` | Preset names and defaults; Python CLI also hardcodes strength/channel defaults. |
| `trainer/simulation_settings.py` | Typed aliases plus derived constants and a hardcoded growth model version. Many comments describe prior implementations. |
| `trainer/evolve.py` | CLI override/default resolution, density derivation, validation, capture schedule, checkpoints. |
| `trainer/train_server.py` | Builds run/broadcast records, repeating the settings surface. |
| Python and TypeScript GPU wrappers | Uniform packing, construction fallbacks, derived material settings and dispatch rules, maintained in two languages. |
| `viewer/src/gpu/types.ts`, `viewer/src/net/runs.ts` | Run/replay schema and historical fallback resolution. |
| `viewer/src/net/settingsStorage.ts` | Offline defaults from shared JSON, plus channel profiles. Despite the name it is not a second persistent settings store. |
| `viewer/src/viewerConfig.ts`, views and panels | Display defaults, Lab defaults, live overrides and slider semantics. |
| Shader source | Model choices absent from run settings: heading channel, fluidity channel, refinement thresholds, morphology kernel-radius cap, grid speed cap and numerical clamps. |

Concrete inconsistencies:

- Shared JSON says `growthModelVersion=3`; trainer exports hardcoded version **12**. Offline viewer defaults spread the JSON value unchanged.
- `constants.json` says `CELL_MEMORY="recurrent"`; normal run defaults say `cellMemory="none"`. Internal-state speed also differs: 1 versus 4. Some duplicate values serve fallbacks rather than current trainer defaults.
- `core/README.md` says repulsion always runs and grid boundaries are sticky; both contradict current execution.
- Comments in `agents.wgsl` claim growth continues after sampling capacity is reached; `stopGrowthAtCapacity()` clears the field.
- `training_sim.py` still explains stochastic division and cell-cycle completion, although `growthEnabled` now gates continuous vector publication.
- Compatibility is partial: old settings receive fallbacks, but the browser explicitly rejects old policy weight shapes. Retaining every old control does not preserve every old checkpoint's behavior.

## Decisions to inform the cleanup plan

No removal plan is assumed yet. The main decisions are whether to retain both chemistry architectures, recurrent variants, motility/repulsion/fluidity, symmetry averaging, and Lab overrides; what level of historical replay must remain supported; and whether sampling capacity should intentionally stop biological growth.

There is also a separate scientific question: which active policy observations and outputs improve growth/shape fitness? Reachability alone cannot answer that. Mechanosensing, morphology inputs, channel scales, private memory and visual-only RGB heads are candidates for explicit usefulness decisions or ablations, not automatically dead code.

The clear structural cleanup candidates are the unused control surface, misleading live-state names, stale documentation, redundant config resolution, and separation of production code from reference experiments. The density prepass, raster helpers, seed RNG and actual compression gate are examples where indiscriminate removal by old feature name would break active functionality.

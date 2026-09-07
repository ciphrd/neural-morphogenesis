# Lizard detail investigation — 2026-09-07

Follow-up: [the determinism audit](DETERMINISM_INVESTIGATION.md) isolates three
GPU scheduling sources plus a seeded-chemistry reset leak, fixes stable
allocation/reset, and tests an opt-in ordered reference. The rollout noise
findings below describe the system before those follow-up changes.

The experiments do **not** establish that the current learned dynamics can grow
a crystal-clean lizard. They establish that triangle geometry can encode much
more detail than the learned result, and identify substantial barriers in
evaluation and optimization before a larger network or finer mechanics grid
can be judged fairly. No clean trained lizard was obtained in this investigation.

The strongest finding is rollout noise: the same policy and the same seed gave
fitness **0.8472–0.8883** in four repeats. That variation exceeds the apparent
benefit of the finest mutations tested. The next engineering priority is to
locate and reduce this noise, while separating shape formation from continued
growth and numerical capacity termination.

## What was measured

The parent was copied from
`trainer/checkpoints/runs/20260907_112430_lizard-64_gen69/best.npy`, a recurrent
128-unit policy. Experiments used the current code/configuration with the
matching recurrent architecture, native Metal on Apple M2 Max, the 64-pixel
lizard target, 2,000 requested macro steps, 16 physics substeps, and RGB loss.
The current live run uses the stateless architecture, so these conclusions
about mutation sensitivity are specifically about the archived recurrent
parent, not a completed comparison of architectures.

Artifacts are in [artifacts/lizard-detail](artifacts/lizard-detail). The probe
copies its input weights, saves terminal vertices/colors/rasters, records config
and arguments, and keeps an independent validation run. The final probe version
also records source hashes and supports exact repeat counts. Earlier three
probe reports predate these extra fields; their config/arguments and copied
weights are retained. Live checkpoints and archived training histories were
not modified. There were 50 full-horizon evaluation requests across the four
probe experiments; all stopped early at capacity. A separate two-candidate,
four-step GPU run tested the new training controls.

**Seed interpretation matters:** `seed_blob()` is deterministic and the selected
`initialCondition: none` does not use the seed to perturb the state. Different
seed numbers in these experiments are therefore repeat executions, not different
initial geometries. The validation results must not be interpreted as a test of
generalization to unseen initial conditions.

| Experiment | Observation | Interpretation |
| --- | --- | --- |
| Same parent, seed 12345, four repeats | Mean 0.8669; sample SD 0.01845; range 0.04114 | Small selection gains can be numerical noise. Four repeats are a diagnostic, not a precise distribution estimate. |
| Paired Gaussian directions, sigma 0.02 / 0.002 / 0.0002 | All eight children at the two larger scales lost to the parent. Best 0.0002 child: selection mean 0.8581 vs parent 0.8650; validation 0.9227 vs 0.8425 | The larger steps disrupt this policy; even a small selected improvement did not repeat. |
| Finer sigma 0.00002 / 0.000002 | Selected child mean 0.8555 vs 0.8667; validation 0.8377 vs 0.8414 | Difference is below the observed repeatability range; not evidence of a reliable unlock. |
| Parent, cap 4,000 | First two runs stop at steps 613/618, missing about 43% | The intended late-window checkpoints are never reached. |
| Same parent, cap 8,000 | Stops at 856/854; mean loss 1.1065 vs 0.8650 at 4,000; spill about 42%, overlap 14–18% | Extra capacity allows more growth but does not teach a clean shape or stop rule. |

The configured sample-spacing estimate for the target is about 4,441 samples,
already above the 4,000 cap before allowance for anisotropy and conformity.
This is an allocation estimate, not proof of a hard representation limit.
Increasing the cap alone was demonstrably insufficient for this parent.

![Actual rollout comparisons](artifacts/lizard-detail/rollouts.png)

## Geometry and fitness precision

A binary lizard silhouette was tiled with **2,126 triangles** and scored at
256×256. The oracle uses the thresholded 64-pixel source at 80% canvas scale,
leaving enough padding for rotations without periodic seam ambiguity. It
bypasses neural development and physics; it proves a property of the mesh and
scorer, not reachability from the seed. It also does not prove the smooth
256-pixel colored lizard is expressible with this particular mesh count.

| Oracle rotation | Current raster-alignment loss | Geometry-alignment loss |
| --- | ---: | ---: |
| 0° | 0.000000051 | 0.000000051 |
| 1.25° | 0.072897 | 0.000000051 |
| 7° | 0.015408 | 0.000000051 |
| 13.7° | 0.067960 | 0.000000051 |
| 33° | 0.015347 | 0.000000051 |

The original scorer rasterizes once, then rotates the image with bilinear
interpolation. Its two local thirds leave a 2.5° angular grid. Together these
create error on otherwise correct geometry. The new opt-in scorer uses the
existing search to find a pose, locally optimizes the actual weighted objective,
and rotates triangle vertices before exact integration. RGB follows its
triangle. Subdivision invariance, invalid geometry, real shrinkage, and real
color errors are tested.

This accurate scorer is **slower**: a saved 3,997-triangle colored rollout took
about 5.06 seconds versus 0.45 seconds with raster alignment in one measurement.
Its loss was 0.87275 versus 0.86723: accurate scoring is not guaranteed to lower
loss, since resampling can hide defects. Do not compare absolute scores across
the two modes. Search is still local around the coarse pose, not a guarantee of
global pose optimality. Browser shape stopping retains its existing raster
metrics; training stopping is disabled, and training metadata records the mode.

The source target itself also matters. After both targets are resolved and
centered at 256×256, the 64-pixel alpha target differs from the 256-pixel target
by **9.17% of target area in normalized absolute alpha error**. The 128-pixel
source differs by 3.52%. Increasing fitness raster resolution cannot recover
details absent from its source PNG. Use `lizard-256` for a final-detail objective.

![Target sampling and scoring oracle](artifacts/lizard-detail/oracle/precision.png)

## Changes available for experiments

- `--fitness-alignment geometry`: accurate geometry/RGB pose scoring. Default
  remains `raster`, preserving the existing objective.
- `--initial-weights PATH.npy`: start a new run from a saved parent; validates
  parameter layout and finite values, retains the parent, and evaluates fresh
  scores under the new run's configuration. This is a weight warm start, not
  restoration of the old simulation/configuration or a checkpoint migration.
- `--mutation-factors 1 .1 .01 .001`: cycle offspring across multiples of
  `--mutation-sigma`. Elites are retained. The default `[1]` retains the prior
  mutation behavior. Both CLI and server use the shared initializer and ladder.
- `trainer/detail_probe.py`: paired directions, shared evaluation seeds,
  terminal geometry capture, validation run, and `--probe-repeats` for noise
  measurement. `--probe-directions 0` evaluates only the parent.
- `trainer/detail_precision_check.py`: reproducible CPU mesh/pose oracle and
  target-sampling comparison. `trainer/detail_refinement_check.py` checks the
  new training controls and accurate RGB alignment.

For example, from the repository root, this starts an isolated **experimental**
refinement run. It has not been demonstrated to converge to a clean lizard:

```sh
trainer/.venv/bin/python trainer/evolve.py \
  --target lizard-256 --cell-memory recurrent \
  --initial-weights artifacts/lizard-detail/probe-gpu/parent.npy \
  --fitness-alignment geometry \
  --mutation-sigma .0002 --mutation-factors 1 .1 .01 \
  --seeds-per-candidate 4 --population 16 --elites 3 --workers 2 \
  --checkpoint-dir /tmp/lizard-detail-refinement
```

Using four evaluation seeds with `initialCondition: none` averages execution
noise; it does not create genuine initial-condition diversity. A separate
perturbation experiment is needed to assess developmental robustness. Accurate
scoring plus repeated evaluations materially increases training cost.

## Next experiments, in priority order

1. **Find the first divergent simulation stage for identical inputs.**
   `core/p2g.wgsl:addGridFloat` accumulates floating-point values using CAS;
   addition order can change rounding. `core/agents.wgsl` also accumulates float
   deposits, and `core/growthField.wgsl:reserveRefinement` allocates sample slots
   in scheduling order. These are candidates, not experimentally isolated root
   causes. Compare stepwise states before/after each stage, then replace the
   dominant source with stable ordering/reduction and rerun the exact-repeat
   probe. Also rule out incomplete reset state. Reproducible refinement is a
   prerequisite for trusting very small parameter changes.
2. **Make refinement a stable phase of the behavior.** Compare continued growth
   with explicit growth cutoff plus relaxation, and score the settled sequence.
   A policy should earn fitness by maintaining a shape, not by reaching its
   sample cap at a favorable instant. This investigation did not change the
   existing capacity-stop law or establish the best cutoff.
3. **Run repeated, matched-budget optimizer ablations.** Compare fixed sigma
   against the new ladder from the same parent. Reevaluate apparent winners;
   require improvement larger than the measured execution noise. Separate
   geometry error from RGB and reject a color-only improvement as a silhouette
   breakthrough. Do not promote the saved probe child as a verified better model.
4. **Then measure dynamic spatial capacity.** Compare mechanics/morphology and
   chemical resolution, sampling spacing, and cap independently. The current
   mechanics grid is 64²; local channels are 64², regional channels 32², and
   global channels 16² under the current profiles. The oracle does not establish
   whether these fields can produce stable toes and tail curvature. Introduce
   localized target-feature metrics or a detail-weighted objective only after
   checking that exact geometry improvements consistently improve selection.

Validation completed: existing domain fitness and Python/browser parity,
existing color fitness, shared-seed scheduling, snapshot-preview reuse, new
warm-start/mutation/alignment checks, the lizard oracle, viewer production build,
`git diff --check`, and a native-GPU warm-start/geometry-scoring smoke run.

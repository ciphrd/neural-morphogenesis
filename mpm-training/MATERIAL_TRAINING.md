# Training to fill a material shape

New runs use fitness model 2. Numerical samples describe transported triangles;
subdivision changes sampling without changing the amount of material. The old
Gaussian point scorer remains available for historical diagnostic tools. Scores
from the two fitness versions are not directly comparable.

## Sample capacity

`--particles` remains a reference-density **allocation limit**. It cannot define
what counts as filled. At capacity, the simulator still pauses growth for
numerical safety, and a rollout cannot declare stable success at that limit or
while refinement requests are unresolved.

The startup log and viewer show an allocation estimate of approximately
`2 * target_area / reference_spacing²`, scaled by density in playback. This is
headroom for sampling, not an optimal count: anisotropy, deformation and
conforming refinement can require more. It never silently raises capacity.
The default circle has area 0.070625 and an estimate of about 19,376 reference
samples at spacing 0.0027. The previous cap of 400 is much smaller. For example:

```sh
trainer/.venv/bin/python trainer/evolve.py --target circle --particles 20000 --particle-densities 0.5
```

This allocates 10,000 samples at the coarser density. Increase capacity if
refinement still becomes blocked.

## Material coverage fitness

Triangles are lifted from the periodic domain, aligned by their area-weighted
centroid, and integrated exactly against raster cells using convex clipping.
Splitting a triangle into children preserves its coverage raster, up to floating
point roundoff. Rotation alignment uses bilinear image resampling over 16 angles
and two local refinements. The target uses exact texel/cell intersection areas.

The objective combines multiscale missing area, outside coverage plus an outside
distance penalty, silhouette-edge disagreement, and overlapping material area.
Material clipped outside the image is counted as spill. Point count and capacity
are absent from the normalization. Geometry below raster resolution and the
rotation interpolation still impose an accuracy limit; the simulator also assumes
that individual triangles span less than half a periodic domain.

The stopping pose minimizes the sum of missing, outside and overlap fractions,
independently of the configurable fitness weights. All three fractions use target
area as their denominator. This keeps success tolerances physical even if fitness
weights are changed. The browser implements the same stopping calculation, with
cross-runtime parity tests. Target geometry is saved with run settings, so new
runs can replay and stop without a scoring server connection.

## Stable completion

Defaults check the shape every 10 macro steps. Three consecutive checks must have
at most 2% missing area, 2% outside area and 2% overlap. Growth is then disabled for
20 macro steps while mechanics and policy evaluation continue. Intermediate
checks and the final settling step must still pass. A failed settling check
resumes growth on the next step (unless an explicit growth time cutoff applies).

A successful rollout uses the mean/worst blend over its settling evaluations and
stops early. Otherwise it runs to the configured horizon and retains late-window
fitness. Reaching the horizon during settling is **not** success. Viewer playback
holds a stable completed result; Restart begins it again. Diagnostics report
stable completion, settling and sampling limits separately.

Controls: `--shape-check-interval`, `--shape-confirmations`,
`--shape-settle-steps`, `--shape-missing-tolerance`, `--shape-spill-tolerance`,
`--shape-overlap-tolerance`. `--no-stable-stop` retains fixed-horizon evaluation;
`--growth-steps` remains an independent growth cutoff. Historical runs without
these settings retain their earlier playback behavior. The Lab does not use the
training-target stopping controller.

This is an external stopping controller. It does not establish that the policy
has learned to stop growing autonomously. To study autonomous maintenance, disable
the controller and evaluate sustained shape quality over a longer horizon.

## Verification

```sh
trainer/.venv/bin/python trainer/domain_fitness_check.py
trainer/.venv/bin/python trainer/rollout_control_check.py
trainer/.venv/bin/python trainer/seed_schedule_check.py
trainer/.venv/bin/python trainer/raster_fitness_check.py
npm --prefix viewer run build
```

The domain checks cover area conservation, subdivision, periodic translation,
holes, missing and excess material, overlaps, invalid geometry, and Python/browser
metric and stopping parity. The rollout check exercises actual loop control with
deterministic observations, including failed settling and truncated horizons.

# Polar image fitness

New CLI and server runs default to `--fitness-function polar`. Use
`--fitness-function multiscale` to retain the former weighted coverage, spill,
boundary, crowding and color objective. The existing `--fitness-alignment` flag
only affects that legacy objective. Restart the training server for a new run
to use the new default; archived generations retain their original scores.

The implementation follows the pixel loss from [Growing Isotropic Neural
Cellular Automata](https://arxiv.org/html/2205.01681) and the authors'
[reference notebook](https://github.com/google-research/self-organising-systems/blob/master/isotropic_nca/blogpost_isonca_single_seed_pytorch.ipynb):

1. Integrate material triangles into cell-average coverage and premultiplied
   RGB. SVG and PNG references use their existing cell-average rasterizer.
   Apply the same opaque-image conversion to both: any coverage above clipping
   roundoff (`1e-10`) gives alpha **1**, and empty pixels give alpha **0**.
   Divide premultiplied RGB by coverage to recover straight color; empty pixels
   have RGB zero. Overlapping triangles average their colors. Thus black
   material is `(0, 0, 0, 1)` and void is `(0, 0, 0, 0)`. Source transparency
   cannot reduce the target's occupied alpha. Alpha is binary at this stage;
   the following filtering and interpolation still produce fractional values.
2. Apply `image + 2 * (image - gaussian_blur(image))` independently to each
   channel, using a 5×5 Gaussian with sigma 1 and reflected border padding.
   Do **not** clamp the sharpened values.
3. Bilinearly sample into radius × angle × channel arrays.
4. Use real FFT cross-correlation to compute squared pixel differences at every
   angular shift, for both the target and its angular reflection.
5. Select the minimum of the shape-priority score over the same rotations and
   reflections, so alignment uses the same objective as evolutionary selection.

The selected pose is shared across all channels. Shape-only targets supervise
occupancy; colored targets supervise RGBA with shape taking priority throughout
one continuous optimization. There are no stages or generation thresholds.

For each pose, sum squared differences over angles and average over radii.
Divide each channel's error by the target alpha energy (its squared values,
summed over angles and averaged over radii). Let `S` be normalized alpha error
and `C` the mean normalized RGB error. The default score is:

```
gate = 0.1 + 0.9 / (1 + S / 0.25)
color = 0.1 * fitness_color_weight * gate * C / (1 + C)
loss = S + color
```

Shape has unit weight; total color contribution is bounded by 0.1. The color
term stays active from the beginning and grows smoothly in influence as shape
error decreases. Bounding it also prevents a large RGB mismatch from dominating
geometry. The loss is strictly increasing in `S` for fixed `C`: worsening shape
cannot improve fitness by suppressing color. Both gradients remain useful
without an explicit two-stage process (training uses evolutionary selection).

`core/config.json` → `polarFitness` sets the cap, floor, and shape scale.
Validation requires `cap * (1-floor) < shapeScale` to preserve monotonicity.
`--fitness-color-weight` accepts 0–1 as a multiplier; 0 requests shape alone.
Legacy coverage/spill/boundary/crowding weights do not contribute. Existing match
metrics govern settling/stopping at the pose selected by this loss.
There is no radial Jacobian weighting. Alpha-energy normalization removes the
linear score scaling with angular sample count, though discretization still
matters. Fitness model version 9 scores are not comparable with older runs.

Adaptations to the particle simulation:

- Existing material-centroid translation alignment is retained. No scale fit.
- An exactly periodic angular grid covers [0, 2π), avoiding a duplicate seam.
- The fixed radial support covers every original canvas corner around the
  target center, plus three pixels for sharpening. The reference notebook uses
  an inscribed disk. Our padding avoids ignoring corner features. Material
  outside this fixed support is invalid (infinite fitness), rather than being
  silently cropped. Sampling resolution and radius are fixed per target.
- This introduces the image objective, not new auxiliary target channels or
  neural outputs. Existing RGBA/occupancy are supervised; arbitrary auxiliary
  polar arrays are supported by the matching primitive but not wired to policy
  outputs. Evolutionary selection remains the trainer's optimization method.

## Inspecting a generation

The frontend's **Polar fitness** section shows the unrotated target, candidate,
matched target, and per-pixel mean squared difference. Horizontal position is
angle (CCW from +x); radius increases downward. The RGB/occupancy switch exposes
all four supervised channels. Blue/orange curves show losses against original
and reflected targets; the red marker selects the minimum.

The PNGs come directly from the selected worker snapshot, without replaying or
rescoring it. The viewer defaults to natural colors, clipping sharpened values
to 0…1 for display. **Show expanded diagnostic range (−2 to 3)** exposes
negative and above-one values (zero is 40% gray). This toggle applies to both
RGB and occupancy previews; differences always use 0…1 and clip larger errors
for display only. The PNGs retain their fixed −2…3 encoding; the viewer reverses
that mapping for natural colors, so existing snapshots also support the toggle.
Display settings do not affect scoring. The linked
`gen_NNNNN_polar.npz` contains full-precision inputs, matched reference, squared
differences, both curves, shift, angle, reflection, radius, and `objective_json`.
The raw difference preview is unweighted; its plain mean is not the score.
To reproduce a new score, compute channel errors from
`squared_difference.sum(axis=1).mean(axis=0)`, divide by the saved `shapeEnergy`,
and use the saved cap, floor, shape scale, and color multiplier in the formula
above. `shapeLoss` and `colorContribution` are also saved and displayed in the
viewer. Historical snapshots without `objective_json` use their original loss. The chosen transform acts
on the **target**: reflect across the x axis if selected, then rotate CCW.
Cartesian snapshot previews use the same transform but bilinear Cartesian
resampling, so their squared difference need not equal the polar loss.

Run `trainer/.venv/bin/python trainer/polar_fitness_check.py` from the project
root. It compares every FFT shift against exhaustive differences, then checks
known rotated/reflected colored geometry, translation, malformed input,
sharpening, serialization and the server's exact diagnostic exports.

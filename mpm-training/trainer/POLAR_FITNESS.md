# Polar image fitness

New CLI and server runs default to `--fitness-function polar`. Use
`--fitness-function multiscale` to retain the former weighted coverage, spill,
boundary, crowding and color objective. The existing `--fitness-alignment` flag
only affects that legacy objective. Restart the training server for a new run
to use the new default; archived generations retain their original scores.

The implementation follows the pixel loss from [Growing Isotropic Neural
Cellular Automata](https://arxiv.org/html/2205.01681) and the authors'
[reference notebook](https://github.com/google-research/self-organising-systems/blob/master/isotropic_nca/blogpost_isonca_single_seed_pytorch.ipynb):

1. Integrate material triangles into cell-average occupancy and premultiplied
   RGB. SVG and PNG references use their existing cell-average rasterizer.
2. Apply `image + 2 * (image - gaussian_blur(image))` independently to each
   channel, using a 5×5 Gaussian with sigma 1 and reflected border padding.
   Do **not** clamp the sharpened values.
3. Bilinearly sample into radius × angle × channel arrays.
4. Use real FFT cross-correlation to compute squared pixel differences at every
   angular shift, for both the target and its angular reflection.
5. Select the minimum. As in the reference code, sum across angles and average
   across radii and channels. There is no radial area/Jacobian weighting. This
   is not the Cartesian area-weighted MSE. Scores depend on angular resolution.

The selected pose is shared across all channels. Shape-only targets use one
occupancy channel; colored targets use four equally weighted RGBA channels.
`--fitness-color-weight 0` explicitly requests occupancy alone; polar mode
otherwise requires weight 1. Legacy coverage/spill/boundary/crowding weights do
not contribute. The existing physical match metrics still govern settling and
stopping, and are evaluated at the pose selected by this loss.

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
rescoring it. Sharpened images share a fixed −2…3 display range (zero is 40%
gray); differences use 0…1 and clip larger errors for display only. The linked
`gen_NNNNN_polar.npz` contains full-precision inputs, matched reference, squared
differences, both curves, shift, angle, reflection, and radius. Reproduce the
fitness with `squared_difference.sum(axis=1).mean()`. The chosen transform acts
on the **target**: reflect across the x axis if selected, then rotate CCW.
Cartesian snapshot previews use the same transform but bilinear Cartesian
resampling, so their squared difference need not equal the polar loss.

Run `trainer/.venv/bin/python trainer/polar_fitness_check.py` from the project
root. It compares every FFT shift against exhaustive differences, then checks
known rotated/reflected colored geometry, translation, malformed input,
sharpening, serialization and the server's exact diagnostic exports.

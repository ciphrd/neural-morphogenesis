# Morphoelastic diagnostic snapshots

`scalar_growth_elastic_baseline.json` is the preserved reference for the
current scalar isotropic growth law, `Fg = sqrt(g) I`.

`tensor_growth_isotropic_equivalence.json` is the first full-tensor `Fg`
capture. Its growth increment is deliberately still isotropic, and the test
suite compares it checkpoint-by-checkpoint with the scalar file. The scalar
file must remain unchanged as historical evidence.

`tensor_growth_directional_strafe.json` uses the same saturated-growth
scenario but sets the former local-forward strafe bias to 1. The physical
strafe scale remains zero: the output controls tensor anisotropy and signed
division polarity. The child and the positional center of its daughter pair
are biased toward `+n`, in proportion to the independent division-bias output. This is the
directional-growth comparison trajectory.

The capture is intentionally independent of evolved policy weights. A fixed
policy saturates the dedicated division-drive output, growth admission is
disabled after macro step 32, already-active cycles finish, and the material
settles through macro step 80. Nine checkpoints retain both transient and
residual measurements.

Metrics are particle-level and weighted by grown particle mass (`mass*g`):

- elastic volume ratio `Je = det(Fe)`
- minimum and maximum principal elastic stretch
- logarithmic area strain
- deviatoric logarithmic strain `||dev(log Ue)||F`
- growth-tensor principal stretches and `||dev(log Ug)||F`
- fixed-corotated strain `||Fe-Re||F`
- pressure and fixed-corotated elastic energy
- speed, kinetic energy, particle count, active cycles, and toroidal radius

Regenerate the reference deliberately:

```bash
cd trainer
.venv/bin/python capture_elastic_baseline.py
```

Verify that a fresh GPU run still reproduces it without overwriting it:

```bash
.venv/bin/python capture_elastic_baseline.py --verify
.venv/bin/python capture_elastic_baseline.py --directional --verify
.venv/bin/python elastic_diagnostics_check.py
```

When remodeling is introduced, preserve these files and capture a new snapshot
under the same scenario. Compare checkpoint-by-checkpoint, especially steps
48, 64, and 80 after all cell cycles complete.

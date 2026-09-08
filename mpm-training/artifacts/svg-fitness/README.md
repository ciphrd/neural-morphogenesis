# SVG target alignment benchmark

Run from the repository root:

```sh
trainer/.venv/bin/python trainer/svg_fitness_check.py --report artifacts/svg-fitness/benchmark.json
```

The fixture is an asymmetric colored L made from 128 triangles, scored at 128×128.
The dense fixture subdivides the same geometry to 4,096 triangles. Each timing is
the median of three warmed evaluations, including candidate rasterization. This
is a controlled CPU benchmark, not a full training-throughput measurement.

| Scorer | 128 triangles | 4,096 triangles |
| --- | ---: | ---: |
| Existing raster rotation | 0.104 s | 0.451 s |
| Existing exact geometry rotation | 0.635 s | 6.575 s |
| Continuous SVG target rotation | 0.129 s | 0.493 s |

At 13.7° rotation, raster alignment scored 0.019414 and SVG alignment scored
0.000320. Across six poses (including 359.3°), SVG scores ranged from 0.000255 to
0.000350. The remaining error is finite-resolution Cairo coverage/antialiasing;
SVG rendering is not analytic triangle/pixel intersection. The dense case is
about 13.3× faster than geometry alignment and about 9% slower than raster
alignment. Results depend on SVG complexity, resolution, mesh size, and machine.

SVG targets use their own normalization: alpha-centered, fitted within radius
0.48 so rotations do not crop the target. Both comparison scorers receive that
same normalized target mask in this experiment. PNG/JSON training behavior is
unchanged. Continuous angle refinement searches two coarse local minima and
can still miss a better basin for a sufficiently ambiguous target.

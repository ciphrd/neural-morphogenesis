# Accumulation GPU verification

Start the viewer with `npm run dev`, then open the test pages below in a WebGPU browser. They run automatically and report results in page text.

- `/tests/gpu-accumulation.html`: chemical splats against a CPU reference, both chemical lifecycles, hardware fallback, and culling from 5,000 to 1,000 particles across the accumulation-mode threshold.
- `/tests/gpu-grouped-splats.html`: packing seven channels across multiple resolutions, signed contributions, wrapped boundaries, and one/two-pixel grids.
- `/tests/gpu-workgroup-reduction.html`: signed 1e-11 contributions, cancellation, partial workgroups, and hash saturation with global fallback.
- `/tests/gpu-translation.html`: uniform motion through 100 physics substeps at particle weights from 1 down to 1e-12, with reduction on/off and spread/clustered populations of 1 and 65 particles. Tolerance is 7e-6 position units for float32 integration rounding.
- `/tests/gpu-morphology-reduction.html`: density fields from the Gaussian renderer versus the compute reduction, including wrapped clusters.
- `/tests/gpu-profile.html?count=20000`: a GPU timestamp profile and wall timings. Also supports `count=200000`. Requires timestamp-query support.

## Performance mode

Fast accumulation uses bounded workgroup hashes to combine float32 physics contributions before global writes. Occupied buckets flush through the original float CAS writer; a bounded probe limit falls back to that writer directly. No fixed-point conversion is used for physics or chemistry.

Below 4,096 particles, devices with float32 blending use chemical splats. Each rgba32float target packs up to three same-resolution chemical numerators and one shared area sum. Four periodic images suffice for grids at least three pixels wide/high; smaller grids retain nine. At higher populations (or without float32 blending), chemistry uses workgroup reduction to avoid raster overdraw. Morphology similarly switches from Gaussian splats to a floating-point compute reduction at high counts. Its existing per-contribution rounding is preserved, without the old global 32-bit integer overflow risk.

The same quadratic deposition kernel, signed neural outputs, density normalization, and channel response times are preserved. Performance refinement allocates in parallel; child IDs and capacity arbitration can depend on GPU scheduling. Training and Fast accumulation OFF retain the reference allocation order and kernels. Fast accumulation is enabled by default for fresh Performance configurations.

## Measured regression

In the controlled 20,000-particle benchmark (128-wide policy, 16 physics substeps), median measured wall time fell from approximately 38 ms before these changes to 6.4 ms. Particle-to-grid pass durations fell from approximately 20 ms to 1.1 ms summed across the physics substeps. The optimized 200,000-particle run measured approximately 43 ms per macro step.

The benchmark seeds a dense blob, keeps its approximate footprint consistent across counts, and excludes final scene drawing and UI work. Results are not a guarantee of live FPS. Wall measurements include query readback overhead. Individual GPU pass durations can overlap and should not be summed as a substitute for wall time. Run profiling separately from other GPU test pages or active outputs.

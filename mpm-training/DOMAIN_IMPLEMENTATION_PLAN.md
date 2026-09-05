# Material-domain implementation

The implementation follows `GROWTH_REDESIGN.md` in four ordered stages:

1. Establish a CPU reference for domain geometry and subdivision; test
   conservation and compare point versus finite-domain transfers.
2. Extend the shared rest-state ABI with transported half edges. Seed a tiling,
   advect geometry independently of constitutive clamps, and replace heuristic
   insertion with conservative bisection driven by transported domain area.
   Use the longest transported edge only to choose the partition axis. Preserve
   policy/material state.
3. Retain ordinary point-based MLS-MPM and point-based field projection. Use the
   reconstructed APIC velocity gradient to transport the domain, without using
   the domain to widen grid coupling. Verify trainer/viewer parity.
4. Test subdivision, rigid rotation, affine motion, deformation-invariant sampling,
   growth, capacity handling and seed/reset behavior; build the viewer and
   document numerical limits and performance.

Three-point Gauss-Legendre domain quadrature was implemented as an initial
baseline, then removed after performance testing and comparison with the source
adaptation paper. Finite-domain CPDI remains a possible separate solver design;
it is not approximated inside the current MLS-MPM transfer.

All four stages are implemented with point transfers. Chemistry, morphology and
mechanical field diagnostics use weighted particle centers; rendered sample
glyphs remain user-sized.
The physical world-area budget and numerical capacity status have independent
controls. The retired insertion-ownership grid has been removed.

Validation completed on Metal (Apple M2 Max): CPU moment/transfer checks, GPU
growth and subdivision suite, seed/reset and policy-state inheritance,
chemical/morphology projection, diagnostic/render shader compilation,
high-strain stability and the existing density smoke test. The viewer production
build and Python compilation pass. Detailed measurements and commands are in
`GROWTH_MODEL.md`.

This establishes an executable baseline, not full morphology convergence.
Independent affine domains can develop inter-domain gaps under nonuniform
motion; broader grid/particle/time convergence and severe-shear remapping remain
research work. No new fracture or plasticity model is inferred from subdivision.

# Material-domain implementation

The implementation follows `GROWTH_REDESIGN.md` in four ordered stages:

1. Establish a CPU reference for domain quadrature, moment matrices, stress
   gradients and subdivision; test conservation and quantify quadrature error.
2. Extend the shared rest-state ABI with transported half edges. Seed a tiling,
   advect geometry independently of constitutive clamps, and replace heuristic
   insertion with longest-edge bisection. Preserve policy/material state.
3. Use the same domain quadrature in P2G, G2P and growth integration. Replace
   the point APIC moment with `D = dx² I/4 + H Hᵀ/3`; use integrated basis
   gradients for stress and kinematic deformation. Verify trainer/viewer parity.
4. Test subdivision, rigid rotation, affine motion, passive stretching,
   growth, capacity handling and seed/reset behavior; build the viewer and
   document numerical limits and performance.

Three-point Gauss-Legendre quadrature per coordinate is the initial GPU rule.
It integrates the domain moments exactly, but basis functions are piecewise
polynomials: nodal fields across spline knots are approximate. Conservation of
global moments and convergence of nodal fields are separate acceptance criteria.

All four stages are implemented. Chemistry, morphology and mechanical field
diagnostics use domain quadrature; rendered sample glyphs remain user-sized.
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

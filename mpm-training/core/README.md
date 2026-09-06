# Shared simulation core

Python and the browser compile these same WGSL sources. `config.json` supplies default settings and shader template constants. Shader loaders resolve template parameters; run-specific channel layouts and material overrides remain explicit.

| Shader | Responsibility |
| --- | --- |
| `agents.wgsl` | Chemical/morphology/strain sensing, policy evaluation, private state, growth output and friction |
| `environment.wgsl` | Triangle-area chemical transfer, persistent environment or cell-owned projection, diffusion, advection and gradients |
| `morphology.wgsl` | Smoothed material density used for morphology sensing |
| `growthField.wgsl` | Growth-vector projection, conforming longest-edge refinement, capacity and material-budget handling |
| `clearGrid.wgsl` | Clear MPM accumulators |
| `p2g.wgsl` | APIC/MLS-MPM momentum and fixed-corotated stress transfer |
| `gridUpdate.wgsl` | Grid velocities, gravity and damping |
| `g2p.wgsl` | Velocity gather, vertex transport, plasticity and continuous stress-free growth |
| `repulsion.wgsl` | Optional density-based repulsion |
| `policyInputProbe.wgsl` | Diagnostic policy-input capture |

## Shared layouts

`ParticleRest` is **64 bytes / 16 floats**, in this order:

| Float offset | Field |
| --- | --- |
| 0–3 | `growthF`: stress-free 2×2 growth tensor |
| 4 | `jp`: plastic Jacobian |
| 5–6 | `growthVectorX/Y`: world-space growth vector |
| 7 | `budgetGrowthRatio`: growth-area snapshot for material-budget accounting |
| 8–13 | `verticesAB`, `vertexC`: transported triangle vertices |
| 14 | `originalArea`: original geometric area |
| 15 | `quadratureWeight`: represented numerical material weight |

`ParticleMeta` starts with color (bytes 0–15), alignment (16–23), growth magnitude (24–27), eight private-state floats (28–59), then chemical state (byte 60 onward). Its stride rounds up to 16 bytes: 96 bytes with nine channels. The combined agent-state buffer reserves a 256-byte header for storage-binding alignment; its first three unsigned integers hold sample count, unresolved samples and capacity-blocked status.

`AgentPhysics` occupies an 80-byte aligned buffer. It contains chemical write scale, sample spacing, friction, growth enable, spawn center, numerical capacity, strain/gradient scales, diagnostic forced-growth controls, chemical input multiplier and material budget. `StepMode` contains commit-growth, communication dt, and private-state update speed. There are no lifecycle clocks, direction lotteries, motion outputs or unused compatibility slots.

The shared layout must agree in all shader declarations, Python structured arrays/readback code and TypeScript upload code. GPU diagnostics and render checks exercise this boundary.

`density_cases.json` is a set of cross-language test cases, not a default configuration file.

# Browser viewer

React/Vite application that runs the shared tissue-growth shaders on browser WebGPU. It can start offline with random policy weights from `../core/config.json`, or load complete settings and generation weights from the training server.

```sh
npm install
npm run dev
npm run build
```

The server address, playback, rendering and tool defaults come from `core/config.json`. Run-specific controls override the loaded settings. Old run schemas are rejected rather than guessed or migrated.

`src/gpu/simulation.ts` coordinates rollouts. `mpmCore.ts`, `agents.ts` and `environment.ts` mirror the Python GPU wrappers. `render.ts` and viewer-local shaders display material triangles, markers, growth vectors/magnitude, neural and chemical state, fields, and elastic diagnostics. `src/net` combines complete run settings with generation records. The training view provides evolutionary-run playback; the lab provides controlled growth scenarios and deformation tools.

Physics controls include material response, damping, friction, chemistry, numerical spacing, morphology and optional repulsion. Growth controls include doubling duration, anisotropy, compression feedback and material budget. Removed motion, lifecycle and chemical-fluidity controls have no UI or GPU state.

A real WebGPU browser is required for playback. The production build checks TypeScript and bundling; native GPU regression checks in `trainer/` also compile and exercise rendering shaders.

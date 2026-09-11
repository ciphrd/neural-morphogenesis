# Browser viewer

React/Vite application that runs the shared tissue-growth shaders on browser WebGPU. It can start offline with random policy weights from `../core/config.json`, or load complete settings and generation weights from the training server.

```sh
npm install
npm run dev
npm run build
```

The server address, playback, rendering and tool defaults come from `core/config.json`. Run-specific controls override the loaded settings. Old run schemas are rejected rather than guessed or migrated.

`src/gpu/simulation.ts` coordinates rollouts. `mpmCore.ts`, `agents.ts` and `environment.ts` mirror the Python GPU wrappers. `render.ts` and viewer-local shaders display material triangles, markers, growth vectors/magnitude, neural and chemical state, fields, and elastic diagnostics. `src/net` combines complete run settings with generation records. The training view provides evolutionary-run playback; the lab provides controlled growth scenarios and deformation tools.

Physics controls include material response, damping, friction, chemistry, numerical spacing, morphology and optional repulsion. Growth controls include doubling duration, anisotropy and compression feedback. Removed motion, lifecycle and chemical-fluidity controls have no UI or GPU state.

A real WebGPU browser is required for playback. The production build checks TypeScript and bundling; native GPU regression checks in `trainer/` also compile and exercise rendering shaders.

## Substrate displacement diagnostic

In Lab or Training, select **Rendering → Background → Substrate rings**, then
**Restart** to track from the seed. Enable **Overlay particle domains** to compare
the transported material boundaries against the rings. Selecting the view starts
tracking; switching backgrounds afterwards keeps the marker moving. Restart
restores perfectly concentric rings centered on the configured spawn position.

The diagnostic transports a 512×512 radial texture coordinate using the same
velocity sampling, periodic backtrace, and macro cadence as the persistent
substrate. It draws alternating black/white bands, each 1/256 world units wide,
inside an initial radius of 0.45. Pixel-footprint filtering reduces aliasing.
Transporting coordinates keeps fine bands readable instead of repeatedly
blurring black/white colors. Coordinate interpolation still has numerical error.
There are no neural writes, diffusion, or decay, and the marker never affects
policy inputs or physics. This isolates displacement; it does not reproduce
chemical reaction or smoothing, and its resolution is independent of chemistry.

GPU checks for stationarity, periodic translation, zero timestep, and
anisotropic deformation: `trainer/.venv/bin/python trainer/substrate_marker_check.py`
(from the repository root).

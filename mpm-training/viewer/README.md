# viewer

The training-viewer frontend — mirrors `envnca/frontend`'s own role:
connects to `../trainer/train_server.py`'s websocket and replays each
generation's winning rollout **client-side, on the browser's own
WebGPU**, driven by the weights/config that generation's broadcast
message carries. Same overall shape as `envnca/frontend` (a React
dashboard: fitness chart, run picker, physics panel, playback controls)
plus a from-scratch WebGPU compute/render pipeline, not just a UI layer.

No growth (a fixed particle count per rollout) — this project's own
NN/growth design settled on that simplification (see
`../trainer/training_sim.py`'s own module docstring for why), so unlike
`envnca/frontend/src/gpu/`'s agent simulation there's no dynamic-population
spawn/compaction mechanism here either.

## Running

```bash
npm install
npm run dev
```

Needs `../trainer/train_server.py` running (default `http://localhost:8003`,
hardcoded in `src/TrainingView.tsx`) and a WebGPU-capable browser (Chrome
or another Chromium-based browser; Safari Technology Preview also
works). **Headless/software-WebGPU (SwiftShader) environments are known
to fail at canvas presentation** — the compute pipeline (physics,
chemical field, NN forward pass) and the whole dashboard/networking
layer were verified working in that environment, but the actual on-
screen render was only confirmed in that constrained setting to the
point of "WebGPU device lost" during `getCurrentTexture()`/
`beginRenderPass()` — a canvas-presentation-specific failure, not a
compute/pipeline validation error, and consistent with SwiftShader's own
known headless limitations. Verify the live canvas in a real browser
before trusting it end to end.

## What's here

- `src/gpu/mpmCore.ts` — browser port of `../trainer/mpm_core.py`'s own
  `MpmCore`: same buffer layout, bind groups, pass ordering. The WGSL
  itself isn't re-typed — it's loaded straight out of `../core/*.wgsl`
  via Vite `?raw` imports (see `vite.config.ts`'s own
  `server.fs.allow`), one source of truth for the physics on both the
  Python and browser sides.
- `src/gpu/environment.ts` + `environment.wgsl` — GPU-resident chemical
  field (bilinear sense/deposit, Sobel gradient, blur+decay), a WGSL
  port of `../trainer/environment.py`, **bounded** (clamped), not
  toroidal — MpmCore's domain has real walls.
- `src/gpu/agents.ts` + `agents.wgsl` — the evolved policy's forward
  pass (`Dense(128) -> tanh -> Dense(16)`), a WGSL port of
  `../trainer/update_rule.py`. No heading/local-frame rotation, no
  repulsion sampling (MPM's own `core/repulsion.wgsl` already covers
  that) — see that file's own module docstring.
- `src/gpu/simulation.ts` — `GpuSimulation`, the per-macro-step
  orchestration: sense -> NN forward -> deposit/decay -> physics
  substeps, one command encoder/submit per macro step, fully
  GPU-resident (no host readback at all, unlike the Python trainer's own
  `training_sim.py`, which has to sync every macro step).
- `src/gpu/render.ts` + `render.wgsl` — instanced-quad particle
  rendering (mirrors `mls-mpm/src/gpu/render.wgsl`'s own approach) plus
  a target-point-cloud overlay and a static domain outline.
- `src/net/` — `trainingSocket.ts` (the live-state reducer/hook, shared
  by the websocket and REST backfill), `runs.ts`, `images.ts` — talk to
  `train_server.py`'s actual endpoints (`/history`, `/runs`,
  `/runs/{id}/history`, `/runs/{id}/images/{grown,aligned}.png`, `WS /ws`).
- `src/TrainingView.tsx` + `src/charts/FitnessChart.tsx` +
  `src/ui/{RunPicker,PhysicsPanel}.tsx` — the dashboard, ported closely
  from `envnca/frontend`'s own (near-verbatim for `FitnessChart`/
  `RunPicker`; `PhysicsPanel` and `TrainingView` adapted to this
  project's own broadcast fields). No "Background"/chemical-substrate
  colorize section and no spawn-distribution toggle — neither has an
  equivalent in this project's own render/seeding design.

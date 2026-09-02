# trainer (feasibility spike)

This is **not** the training backend yet — it's a feasibility spike that
proves the one thing the whole `mpm-training/` project rests on: that
`../core/`'s WGSL physics runs correctly, headlessly, via Python `wgpu`
(wgpu-py, wrapping wgpu-native — a sibling implementation to Chrome's own
Dawn, not the same thing). No NN, no growth/spawn mechanism, no
evolutionary search, no websocket server — see the top-level plan for
what's deferred and why.

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python feasibility_check.py
```

Prints an explicit PASS/FAIL/SKIP per check and a summary at the end.
Exit code is 0 iff no check FAILed (a SKIP is not a failure).

## What's here

- `device.py` — GPU adapter/device acquisition (mirrors envnca's own
  `device.py`).
- `shader_template.py` — Python port of `mls-mpm/src/gpu/shaderTemplate.ts`'s
  `templateShader()`, plus `load_core_shader()` which reads straight out
  of `../core/`.
- `mpm_core.py` — `MpmCore`, the headless simulation class. Mirrors
  `mls-mpm/src/gpu/mpm.ts`'s own `MpmSimulation` (buffer layout, bind
  groups, pass ordering) as closely as possible.
- `feasibility_check.py` — the runnable spike itself; see its own
  docstring for the 4 checks.

## Confirmed on this machine (2026, `wgpu==0.32.0`, macOS)

- The adapter picks **Metal** (`adapter.info["backend_type"] == "Metal"`)
  — the risk the plan flagged (headless compute-only Metal being a
  narrower-tested path) did not bite.
- `atomic<i32>` storage-buffer scatter works correctly under `naga`
  (wgpu-native's shader compiler) — the single riskiest, genuinely
  unvalidated bet this whole approach rested on. Confirmed via the
  standalone atomics check, before touching the real pipeline.
- The extracted physics produces qualitatively correct behavior: a
  gravity-free blob barely drifts; a blob under center gravity remains
  bounded around the middle of the domain.

## Two real, load-bearing findings from building this (not hypothetical)

**1. wgpu-native's Metal backend has a hard cap on compute passes per
command encoder before `finish()`, and on outstanding (not-yet-retired)
command buffers across submits — both confirmed by triggering them
directly.** A single `step()` call encoding ~2000+ compute passes (250+
substeps × 8 passes/substep) before `finish()` reliably kills the device
("refusing to create new command buffer; 4097 outstanding command
buffers exceeds the limit of 4096"). This is a genuine difference from
the browser sandbox's own Dawn/tint backend, which doesn't hit this at
the substep counts `mls-mpm`'s own `step()` calls per rendered frame.
**Fixed inside `MpmCore.step()`**: it internally chunks large substep
counts into ≤128-substep encoders, forcing a GPU sync between chunks —
callers can pass any substep count safely. See `step()`'s own docstring
(`_MAX_SUBSTEPS_PER_SUBMIT`) for the exact numbers and reasoning. This
has a real cost (the per-chunk sync is a host-device stall, unlike the
browser's fire-and-forget `step()`) — worth knowing about when designing
the eventual ES training loop's own per-episode/per-generation cadence.

**2. `wgpu.GPUQueue.on_submitted_work_done_sync()` is broken in this
exact `wgpu==0.32.0` build** — a `TypeError` inside wgpu-py's own binding
code (a callback-signature mismatch against the native library),
reproducible in total isolation (one buffer, one empty command encoder,
nothing else). This is exactly the kind of API-churn/version-fragility
risk the top-level plan flagged for `wgpu-py` in general, now confirmed
concretely on a specific API rather than just a general concern. Worked
around by using a trivial `read_buffer()` on a dedicated 4-byte
STORAGE|COPY_SRC buffer as the sync point instead (WebGPU's queue is a
single in-order timeline, so reading anything back necessarily waits for
every prior submission) — see `MpmCore._sync_buffer`. If a future
`wgpu` version fixes this, the workaround can likely be dropped in favor
of the "correct" API, but don't assume it's fixed without re-testing —
this dependency has already shown it can't be assumed stable across
versions, which is why `requirements.txt` pins an **exact** version
(`wgpu==0.32.0`), not a range.

## What this does NOT prove yet

**Check 4 (numerical cross-check against the actual `mls-mpm` browser
sandbox) could not be completed in this session** — it needs a human to
manually capture a handful of particle positions from the real browser
sandbox (Blocks world, gravity=200, a fixed/hardcoded initial jitter, at
a specific substep count) and save them to `cross_check_reference.json`
(`{"substeps": N, "positions": [[x, y], ...]}`) next to
`feasibility_check.py`; the script detects that file and will compare
against it, but the comparison itself still needs to be wired up once a
reference exists. Checks 1-3 show the physics is *qualitatively*
plausible (settles, doesn't explode, matches known stability behavior)
but do **not** rule out a subtly wrong constant that still happens to
look stable (e.g. a swapped `DX`/`INV_DX`) — that's specifically what
check 4 is for, and it remains genuinely unverified until someone
performs that manual capture.

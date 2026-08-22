"""Single source of truth for every constant that shapes how a
Simulation actually behaves once it's running — the physics-ish motion
caps, the diffusion decay, and the network's hidden width. These used to
live one apiece in whichever module owned the mechanism they constrain
(MAX_SPEED in update_rule.py, DECAY in environment.py, ...) — pulled
together here instead because train_server.py imports every single one
of them to forward into the per-generation broadcast message
(SimulationConfig), so the frontend's WebGPU replay reproduces the exact
same simulation a checkpoint was trained under. One file to check for
"what governs this run's behavior" instead of hunting through each
mechanism's own module.

Deliberately NOT here: anything CLI-configurable per training run
(population, spawn spread, step count, mutation sigma, ...) — those are
evolve.py's own argparse defaults, already overridable without editing
code, and already forwarded to the frontend the same way. This file is
only for the constants that aren't exposed as a flag anywhere and are
therefore identical across every run unless this file itself is edited.
"""

# Environment.step_dynamics()'s per-step multiplicative decay, applied
# right after the mass-preserving blur (kernel sums to 1) — see
# environment.py's own module docstring for the diffusion+decay design
# this implements.
DECAY = 0.98

# UpdateRule's single hidden layer width.
HIDDEN_DIM = 128

# Grid-space equivalent of trainer/backend's MAX_SPEED/MAX_ACCEL — that
# project's world is a handful of unit-radius circles, this one is a
# grid of arbitrary size, so the constants are re-derived in this
# project's own units rather than reused: MAX_SPEED is still "a subtle
# nudge," here expressed as a small fraction of one grid cell's diagonal
# per step, and MAX_ACCEL keeps the same MAX_SPEED/4 relationship (reach
# full speed from rest in ~4 steps of sustained acceleration).
MAX_SPEED = 0.01
MAX_ACCEL = MAX_SPEED / 4.0

# Strafe doesn't accumulate step to step the way accel -> velocity does
# (see update_rule.py's "Strafe" docstring section), so there's no
# multi-step buildup toward a ceiling the way velocity has — this is
# directly "how far can one step's strafe move an agent," same order of
# magnitude as MAX_SPEED (the velocity cap) rather than MAX_ACCEL.
MAX_STRAFE = 0.5

# Ceiling on env_write (simulation.py's step(), right before
# Environment.deposit()) — unlike local_accel/local_strafe, this one
# wasn't squashed at all until a gradient-descent training run was
# diagnosed plateauing hard: env_write is deposited every step into a
# grid that decays at only 0.98/step (~50-step effective memory), and
# with nothing bounding it, the sensed value agents read back
# (environment.py's sample_value_and_gradient) grew over the course of
# training until it saturated the network's own first Linear -> Tanh
# layer (confirmed instrumentally: hidden-layer saturation climbed from
# 0% to ~80% over 150 epochs, and the best loss stopped improving
# entirely right around where saturation crossed ~50-60%) — a dead local
# gradient that backprop can't do anything about, no matter the learning
# rate. ES never hit this: it only needs the forward behavior to look
# adequate, never a nonzero local gradient. A starting value, not a
# carefully derived one — re-run the same hidden-saturation diagnostic
# (train_gd.py/train_server_gd.py's UpdateRule.record_diagnostics
# logging) after changing this to confirm it actually keeps saturation
# down over a long run, and adjust if not.
MAX_ENV_WRITE = 1.0

# repulsion.RepulsionField's own dedicated density-field resolution —
# deliberately independent of (and much coarser than) the main (C,H,W)
# grid's own H/W, since a repulsion signal only needs "which direction is
# crowded," not real sensing fidelity — see repulsion.py's module
# docstring for the full O(N)-not-O(N^2) reasoning and why this exists
# at all (a gradient-descent-trained policy was observed collapsing
# every agent onto a single point — a stable fixed point no loss-
# function change alone can break).
REPULSION_RESOLUTION = 128

# Gaussian splat width, in *this field's own* cells (REPULSION_RESOLUTION-
# relative), not main-grid pixels — same "sigma is resolution-relative"
# convention raster.py's own raster_sigma already uses. Tuned live via
# the frontend's Physics panel (against REPULSION_RESOLUTION=128) before
# landing here — no longer a placeholder.
REPULSION_SIGMA = 0.4

# Scales the repulsion force added into velocity alongside accel (see
# simulation.py's step()) — participates in the same MAX_SPEED-clamped
# budget as accel, so this is on the same order of magnitude as
# MAX_ACCEL by design. Tuned live via the frontend's Physics panel
# alongside REPULSION_SIGMA above — no longer a placeholder.
# Set to 0.0 temporarily to A/B test performance impact — see
# RepulsionField.compute()'s own docstring: strength==0.0 short-circuits
# before the splat/Sobel-gradient work even runs, not just before the
# resulting force is applied, so this genuinely disables the computation,
# not just its effect. Restore to a nonzero value (0.005 was the last
# live-tuned one) once the comparison is done.
REPULSION_STRENGTH = 0.0

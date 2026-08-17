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
MAX_SPEED = 0.2
MAX_ACCEL = MAX_SPEED / 4.0

# Strafe doesn't accumulate step to step the way accel -> velocity does
# (see update_rule.py's "Strafe" docstring section), so there's no
# multi-step buildup toward a ceiling the way velocity has — this is
# directly "how far can one step's strafe move an agent," same order of
# magnitude as MAX_SPEED (the velocity cap) rather than MAX_ACCEL.
MAX_STRAFE = 0.5

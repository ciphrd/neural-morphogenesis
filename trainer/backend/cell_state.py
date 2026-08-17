"""Per-node internal state: a persistent N-dimensional vector split into
`id` (encodes the node's identity in an embedding space — distinct from
its physical x,y position, which physics drives separately) and
`chemicals` (free-form signal channels the update rule evolves). See
trainer/README.md.

ID_DIM and NUM_CHEMICAL_CHANNELS are fixed by the update rule's MLP
output width (20 = 1 split logit + NUM_CHEMICAL_CHANNELS + ID_DIM +
SPAWN_DIR_DIM + MOTION_DIM), not picked independently — see
update_rule.py.

SPAWN_DIR_DIM is not part of the persistent state vector above — unlike
id/chemicals it isn't carried forward and accumulated step to step, it's
fresh output the network emits *every* step ("if this node were to split
right now, which way") and Graph.spawn_directions holds the latest
reading for display/consumption, the same way Graph.energy holds a value
the network senses but doesn't itself own.

velocity (see Graph) *is* persistent, accumulated state, same treatment
as id/chemicals — but unlike those, the network doesn't emit it
directly each step, it emits a *local-frame acceleration* (MOTION_DIM's
two output slots) that update_rule.py's step() rotates into world
coordinates and integrates onto the running velocity, mirroring how
chemical_delta integrates onto chemicals rather than replacing them
outright. heading is not separate state at all — it's derived on demand
as atan2(vy, vx) — see update_rule.py's own "Velocity & heading"
docstring section for the full mechanism and why it exists (local-frame
sensing).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

ID_DIM = 3
NUM_CHEMICAL_CHANNELS = 12
SPAWN_DIR_DIM = 2
# Not part of the persistent state vector, same footing as
# SPAWN_DIR_DIM — a 2D local-frame acceleration the network emits every
# step (see update_rule.py's MAX_ACCEL), rotated into world coordinates
# and integrated onto Graph.velocity rather than read back as raw
# network output next step.
MOTION_DIM = 2

# Starting energy budget for the seed node — see update_rule.py for the
# rest of the energy mechanism (injection rate, split threshold).
INITIAL_ENERGY = 100.0


def random_id(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    # `rng` lets a caller that needs reproducibility (evolve.py's
    # rollout(), via Graph.seed()) get a deterministic seed-node id
    # instead of drawing from numpy's global random state — see
    # update_rule.py's step() for the fuller rationale (this is the same
    # pattern, and without it the *seed* node's own random id/chemicals
    # was the dominant source of a rollout being non-reproducible even
    # after step()'s own randomness was fixed, since it seeds the entire
    # nonlinear growth trajectory that follows). Omit for callers that
    # don't care (e.g. the interactive tool's fresh seed each restart).
    draw = rng.uniform if rng is not None else np.random.uniform
    return draw(-1.0, 1.0, size=ID_DIM)


def random_chemical(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    draw = rng.uniform if rng is not None else np.random.uniform
    return draw(-1.0, 1.0, size=NUM_CHEMICAL_CHANNELS)

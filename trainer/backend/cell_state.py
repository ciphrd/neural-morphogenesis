"""Per-node internal state: a persistent N-dimensional vector split into
`id` (encodes the node's identity in an embedding space — distinct from
its physical x,y position, which physics drives separately) and
`chemicals` (free-form signal channels the update rule evolves). See
trainer/README.md.

ID_DIM and NUM_CHEMICAL_CHANNELS are fixed by the update rule's MLP
output width (18 = 1 split logit + NUM_CHEMICAL_CHANNELS + ID_DIM +
SPAWN_DIR_DIM), not picked independently — see update_rule.py.

SPAWN_DIR_DIM is not part of the persistent state vector above — unlike
id/chemicals it isn't carried forward and accumulated step to step, it's
a fresh 2D direction the network emits *every* step ("if this node were
to split right now, which way") and Graph.spawn_directions just holds
the latest one for display/consumption, the same way Graph.energy holds
a value the network senses but doesn't itself own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

ID_DIM = 3
NUM_CHEMICAL_CHANNELS = 12
SPAWN_DIR_DIM = 2

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

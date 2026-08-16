from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import REST_LENGTH

# The resolution sculpture's grid defaulted to when cell_size=REST_LENGTH
# was first chosen as "one pixel = one graph edge length" — kept as the
# fixed physical-scale anchor so any resolution now reproduces that same
# real-world size, per the module docstring above.
REFERENCE_RESOLUTION = 32


@dataclass
class TargetShape:
    """A target shape loaded from a pixel export (the format sculpture's
    `pixelGridToJSON` produces: `{nx, ny, pixels: [{x,y}, ...]}`, integer
    cell indices with no embedded world scale). 2D analogue of trainer-3d's
    TargetVolume.

    Represented as a point cloud of occupied-pixel centers in graph space:
    recentered on the centroid of the filled pixels, and scaled so that the
    overall shape spans the same physical extent regardless of the export's
    resolution — one pixel spacing equals one graph edge length (`cell_size`)
    *at REFERENCE_RESOLUTION*, and finer/coarser exports scale that spacing
    down/up so a higher-resolution export of the same drawing just adds
    point density rather than growing the shape. Without this, a 64x64
    export of the same picture as a 32x32 one would come out twice the
    physical size, since raw pixel indices simply run twice as high at
    double the resolution.
    """

    points: np.ndarray  # (N, 2)
    resolution: tuple[int, int]

    @classmethod
    def from_export(cls, data: dict, cell_size: float = REST_LENGTH) -> "TargetShape":
        nx, ny = data["nx"], data["ny"]
        pixels = data["pixels"]
        if not pixels:
            return cls(points=np.zeros((0, 2)), resolution=(nx, ny))

        # sculpture's grid is always square (App.tsx's buildGrid passes the
        # same value for both dims) but take the max rather than assume it,
        # so a non-square export still scales sanely.
        effective_cell_size = cell_size * REFERENCE_RESOLUTION / max(nx, ny)

        coords = np.array([[p["x"], p["y"]] for p in pixels], dtype=np.float64)
        centers = coords + 0.5  # pixel centers, not corners
        centroid = centers.mean(axis=0)
        points = (centers - centroid) * effective_cell_size

        return cls(points=points, resolution=(nx, ny))

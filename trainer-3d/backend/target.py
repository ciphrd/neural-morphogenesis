from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import REST_LENGTH


@dataclass
class TargetVolume:
    """A target shape loaded from a voxel export (the format sculpture's
    `voxelGridToJSON` produces: `{nx, ny, nz, voxels: [{x,y,z}, ...]}`,
    integer cell indices with no embedded world scale).

    Represented as a point cloud of occupied-voxel centers in graph space:
    recentered on the centroid of the filled voxels, and scaled so that one
    voxel spacing equals one graph edge length (`cell_size`) — i.e. one
    split is meant to advance the structure by roughly one voxel-width, so
    the target's own resolution sets the growth granularity rather than an
    arbitrary fixed size.
    """

    points: np.ndarray  # (N, 3)
    resolution: tuple[int, int, int]

    @classmethod
    def from_export(cls, data: dict, cell_size: float = REST_LENGTH) -> "TargetVolume":
        nx, ny, nz = data["nx"], data["ny"], data["nz"]
        voxels = data["voxels"]
        if not voxels:
            return cls(points=np.zeros((0, 3)), resolution=(nx, ny, nz))

        coords = np.array([[v["x"], v["y"], v["z"]] for v in voxels], dtype=np.float64)
        centers = coords + 0.5  # voxel centers, not corners
        centroid = centers.mean(axis=0)
        points = (centers - centroid) * cell_size

        return cls(points=points, resolution=(nx, ny, nz))

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from physics import REST_LENGTH

# Height from a triangular face to the apex of a *regular* tetrahedron
# built on it, for edge length REST_LENGTH — keeps a freshly-extruded
# tetrahedron close to regular before physics relaxes it the rest of the
# way, same spirit as the old centroid-split's initial guess.
EXTRUDE_HEIGHT = REST_LENGTH * math.sqrt(2.0 / 3.0)


def _regular_tetrahedron_vertices(edge_length: float) -> np.ndarray:
    # alternating corners of a cube — the standard construction for a
    # regular tetrahedron, scaled to the requested edge length
    base = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )
    base_edge = np.linalg.norm(base[0] - base[1])
    return base * (edge_length / base_edge)


@dataclass
class Graph:
    """Owns node positions, edges, and the triangular surface. Triangles
    are the unit of growth: each one can extrude a new apex outward
    exactly once, turning itself into an internal face shared with the
    freshly-created tetrahedron and exposing that tetrahedron's 3 other
    faces as new growable triangles.

    The backend is the sole source of truth for geometry and topology —
    the frontend only ever renders what it's sent.
    """

    positions: list[np.ndarray] = field(default_factory=list)
    edges: set[tuple[int, int]] = field(default_factory=set)
    pinned: set[int] = field(default_factory=set)
    triangles: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # for each triangle, the vertex on the "inside" of the structure it
    # was created relative to — used only to orient its outward normal
    inward_ref: dict[int, int] = field(default_factory=dict)
    grown: set[int] = field(default_factory=set)
    _next_triangle_id: int = field(default=0, repr=False)

    @classmethod
    def seed(cls) -> "Graph":
        """The starting organism: a single regular tetrahedron, its 4
        faces registered as the initial growable triangles."""
        graph = cls()
        v0, v1, v2, v3 = (
            graph._add_node(v) for v in _regular_tetrahedron_vertices(REST_LENGTH)
        )
        graph._add_triangle((v1, v2, v3), inward_ref=v0)
        graph._add_triangle((v0, v2, v3), inward_ref=v1)
        graph._add_triangle((v0, v1, v3), inward_ref=v2)
        graph._add_triangle((v0, v1, v2), inward_ref=v3)
        # Pin 3 (non-collinear) vertices, not 1. A single pinned point
        # removes translation but leaves rotation about it completely
        # unconstrained, so any re-relax can visibly spin the whole
        # structure. 3 fixed points removes all 6 rigid-body degrees of
        # freedom, so nothing downstream can ever rotate the frame again.
        graph.pinned.update((v0, v1, v2))
        return graph

    def _add_node(self, position: np.ndarray) -> int:
        node_id = len(self.positions)
        self.positions.append(np.asarray(position, dtype=np.float64))
        return node_id

    def _add_edge(self, a: int, b: int) -> None:
        self.edges.add((a, b) if a < b else (b, a))

    def _add_triangle(self, vertices: tuple[int, int, int], inward_ref: int) -> int:
        tri_id = self._next_triangle_id
        self._next_triangle_id += 1
        self.triangles[tri_id] = vertices
        self.inward_ref[tri_id] = inward_ref
        a, b, c = vertices
        self._add_edge(a, b)
        self._add_edge(a, c)
        self._add_edge(b, c)
        return tri_id

    def _outward_normal(self, tri_id: int) -> np.ndarray:
        v0, v1, v2 = self.triangles[tri_id]
        p0, p1, p2 = self.positions[v0], self.positions[v1], self.positions[v2]
        centroid = (p0 + p1 + p2) / 3.0

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        ref_pos = self.positions[self.inward_ref[tri_id]]
        if np.dot(normal, ref_pos - centroid) > 0:
            normal = -normal
        return normal

    def apex_preview(self, tri_id: int) -> np.ndarray:
        """Where growing this triangle would place its new node — exposed
        separately from grow_triangle so hover previews can show it
        without mutating anything."""
        v0, v1, v2 = self.triangles[tri_id]
        centroid = (self.positions[v0] + self.positions[v1] + self.positions[v2]) / 3.0
        return centroid + self._outward_normal(tri_id) * EXTRUDE_HEIGHT

    def grow_triangle(self, tri_id: int) -> Optional[int]:
        if tri_id not in self.triangles or tri_id in self.grown:
            return None

        v0, v1, v2 = self.triangles[tri_id]
        new_id = self._add_node(self.apex_preview(tri_id))
        self._add_edge(v0, new_id)
        self._add_edge(v1, new_id)
        self._add_edge(v2, new_id)

        self.grown.add(tri_id)
        self._add_triangle((v0, v1, new_id), inward_ref=v2)
        self._add_triangle((v0, v2, new_id), inward_ref=v1)
        self._add_triangle((v1, v2, new_id), inward_ref=v0)

        return new_id

    def positions_array(self) -> np.ndarray:
        if not self.positions:
            return np.zeros((0, 3))
        return np.stack(self.positions)

    def set_positions(self, positions: np.ndarray) -> None:
        self.positions = [positions[i] for i in range(positions.shape[0])]

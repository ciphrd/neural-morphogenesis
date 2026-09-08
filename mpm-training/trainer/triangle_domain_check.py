"""Independent CPU reference for the live simulator's triangle geometry.

Run: trainer/.venv/bin/python trainer/triangle_domain_check.py
The four domain floats hold row-major edges [B-A, C-A]; x is the centroid.
"""
from dataclasses import dataclass
import numpy as np

from growth_resampling_math_check import totals

@dataclass
class Triangle:
    x: np.ndarray
    edges: np.ndarray

    @classmethod
    def from_vertices(cls, vertices):
        a, b, c = np.asarray(vertices, dtype=float)
        return cls((a+b+c)/3, np.column_stack((b-a, c-a)))

    def vertices(self):
        a = self.x - self.edges.sum(axis=1)/3
        return np.array([a, a+self.edges[:, 0], a+self.edges[:, 1]])

    def signed_area(self):
        return np.linalg.det(self.edges)/2

    def split(self, edge=None):
        vertices = self.vertices()
        if edge is None:
            lengths = np.sum((np.roll(vertices, -1, axis=0)-vertices)**2, axis=1)
            edge = int(np.argmax(lengths))
        a, b, c = np.roll(vertices, -edge, axis=0)
        midpoint = (a+b)/2
        return (Triangle.from_vertices([a, midpoint, c]),
                Triangle.from_vertices([midpoint, b, c]))

    def covariance(self):
        offsets = self.vertices()-self.x
        return offsets.T @ offsets / 12

def triangulate_parallelogram(x, h):
    a, b = h[:, 0], h[:, 1]
    return (Triangle.from_vertices([x-a-b, x+a-b, x+a+b]),
            Triangle.from_vertices([x-a-b, x+a+b, x-a+b]))

def check():
    rng = np.random.default_rng(731)
    cases = 0
    for _ in range(200):
        vertices = rng.normal(size=(3, 2))
        parent = Triangle.from_vertices(vertices)
        if abs(parent.signed_area()) < 1e-5:
            continue
        np.testing.assert_allclose(parent.vertices(), vertices, atol=1e-12)
        velocity = rng.normal(size=2)
        affine = rng.normal(size=(2, 2))
        mass = rng.uniform(.1, 10)
        point_moment = np.eye(2)*(1/256)**2/4
        before = totals(mass, parent.x, velocity, affine, point_moment)
        for edge in range(3):
            children = parent.split(edge)
            a, b, c = np.roll(vertices, -edge, axis=0)
            # These vertex identities establish an exact partition along a median.
            np.testing.assert_allclose(children[0].vertices(), [a, (a+b)/2, c], atol=1e-12)
            np.testing.assert_allclose(children[1].vertices(), [(a+b)/2, b, c], atol=1e-12)
            for child in children:
                np.testing.assert_allclose(child.signed_area(), parent.signed_area()/2, atol=1e-12)
            np.testing.assert_allclose(sum(t.x for t in children)/2, parent.x, atol=1e-12)
            covariance = sum(t.covariance()+np.outer(t.x-parent.x, t.x-parent.x)
                             for t in children)/2
            np.testing.assert_allclose(covariance, parent.covariance(), atol=1e-12)
            # Live point APIC uses copied velocity/C, not affine-offset velocity.
            after = sum(totals(mass/2, t.x, velocity, affine, point_moment) for t in children)
            np.testing.assert_allclose(after, before, atol=1e-12)
            cases += 1
        # Existing G2P left-multiplication transports all three vertices correctly.
        transform = np.eye(2)+.01*affine
        translation = .01*velocity
        transported = Triangle(transform @ parent.x+translation, transform @ parent.edges)
        np.testing.assert_allclose(transported.vertices(), vertices @ transform.T+translation, atol=1e-12)
        x, h = rng.normal(size=2), rng.normal(size=(2, 2))
        pair = triangulate_parallelogram(x, h)
        np.testing.assert_allclose(sum(t.signed_area() for t in pair), 4*np.linalg.det(h), atol=1e-12)
        np.testing.assert_allclose(sum(t.x for t in pair)/2, x, atol=1e-12)

    # Longest-edge selection must include the third edge (C-A)-(B-A).
    p = Triangle.from_vertices([[0, 0], [1, 0], [-1, .1]])
    for automatic, explicit in zip(p.split(), p.split(1)):
        np.testing.assert_allclose(automatic.vertices(), explicit.vertices(), atol=1e-12)

    # Area-only refinement intentionally cannot detect isochoric elongation.
    p = Triangle.from_vertices([[0, 0], [1, 0], [.5, np.sqrt(3)/2]])
    stretched = Triangle(p.x, np.diag([8., 1/8]) @ p.edges)
    np.testing.assert_allclose(stretched.signed_area(), p.signed_area(), atol=1e-12)
    assert np.max(np.linalg.norm(np.roll(stretched.vertices(), -1, axis=0)-stretched.vertices(), axis=1)) > 7.9
    # Bisection need not strictly reduce the maximum length of BOTH children:
    # an equilateral parent's children each retain an original full-length side.
    for child in p.split():
        lengths = np.linalg.norm(np.roll(child.vertices(), -1, axis=0)-child.vertices(), axis=1)
        np.testing.assert_allclose(max(lengths), 1., atol=1e-12)
    print(f'[PASS] {cases} bisections: exact partition, signed area, centroid, covariance, point APIC totals')
    print('[PASS] affine transport, seed conversion, third-edge selection, area/length trigger distinctions')

if __name__ == '__main__':
    check()

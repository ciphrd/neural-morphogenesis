"""Float64 CPU reference for domain-integrated MPM transfers.

Independent of WGSL generation. Grid coordinates are unwrapped so periodic
boundary tests can compare transfers without confusing torus coordinates with
Euclidean angular momentum. Tensor entries use ordinary row-major matrices.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class Domain:
    x: np.ndarray
    h: np.ndarray
    mass: float = 1.0
    rest_area: float = 1.0
    velocity: np.ndarray = None
    affine: np.ndarray = None
    stress: np.ndarray = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.h = np.asarray(self.h, dtype=float).reshape(2, 2)
        self.velocity = np.zeros(2) if self.velocity is None else np.asarray(self.velocity)
        self.affine = np.zeros((2, 2)) if self.affine is None else np.asarray(self.affine)
        self.stress = np.zeros((2, 2)) if self.stress is None else np.asarray(self.stress)

    def split(self, axis=None):
        if axis is None:
            axis = int(np.argmax(np.sum(self.h**2, axis=0)))
        delta = self.h[:, axis] / 2
        child_h = self.h.copy()
        child_h[:, axis] /= 2
        return tuple(replace(self, x=self.x + sign * delta, h=child_h,
                             mass=self.mass/2, rest_area=self.rest_area/2,
                             velocity=self.velocity + sign * self.affine @ delta)
                     for sign in (-1, 1))

    def moment(self, dx):
        return np.eye(2)*dx**2/4 + self.h @ self.h.T/3


def stencil(domain, dx, order=3):
    """Return node -> (integrated normalized basis, integrated gradient)."""
    xi, wi = np.polynomial.legendre.leggauss(order)
    nodes = {}
    for a in range(order):
        for b in range(order):
            sample = domain.x + domain.h @ np.array([xi[a], xi[b]])
            base = np.floor(sample/dx - 0.5).astype(int)
            fx = sample/dx-base
            w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2]
            dw = [fx-1.5, -2*(fx-1), fx-0.5]
            q = wi[a]*wi[b]/4
            for i in range(3):
                for j in range(3):
                    node = tuple(base + [i, j])
                    value = q*np.array([w[i][0]*w[j][1],
                                         dw[i][0]*w[j][1]/dx,
                                         w[i][0]*dw[j][1]/dx])
                    nodes[node] = nodes.get(node, np.zeros(3)) + value
    return nodes


def scatter(domains, dx, order=3, dt=0.0):
    """Node -> mass, momentum x/y, force x/y."""
    grid = {}
    for p in domains:
        for node, (w, gx, gy) in stencil(p, dx, order).items():
            offset = np.array(node)*dx-p.x
            force = -p.rest_area * p.stress @ [gx, gy]
            momentum = p.mass*w*(p.velocity+p.affine@offset) + dt*force
            entry = np.r_[p.mass*w, momentum, force]
            grid[node] = grid.get(node, np.zeros(5)) + entry
    return grid


def gather(domain, grid_velocity, dx, order=3):
    velocity = np.zeros(2)
    moment = np.zeros((2, 2))
    gradient = np.zeros((2, 2))
    for node, (w, gx, gy) in stencil(domain, dx, order).items():
        v = grid_velocity(np.array(node)*dx)
        velocity += w*v
        moment += w*np.outer(v, np.array(node)*dx-domain.x)
        gradient += np.outer(v, [gx, gy])
    return velocity, moment@np.linalg.inv(domain.moment(dx)), gradient


def check_reference():
    dx = 1/256
    p = Domain([0.5031, 0.5027], [[0.002, 0.0003], [0.0002, 0.0015]],
               velocity=[0.2, -0.1], affine=[[2, -7], [7, -1]], stress=[[3, 1], [1, 2]])
    before = scatter([p], dx)
    after = scatter(p.split(), dx)
    for grid in (before, after):
        values = np.array(list(grid.values()))
        np.testing.assert_allclose(values[:, 0].sum(), p.mass, atol=1e-12)
        np.testing.assert_allclose(values[:, 1:3].sum(axis=0), p.mass*p.velocity, atol=1e-12)
        np.testing.assert_allclose(values[:, 3:].sum(axis=0), 0, atol=1e-10)
    def angular(grid, lane):
        return sum((node[0]*dx*row[lane+1] - node[1]*dx*row[lane]) for node, row in grid.items())
    np.testing.assert_allclose(angular(before, 1), angular(after, 1), atol=1e-12)
    np.testing.assert_allclose(angular(before, 3), 0, atol=1e-10)
    for child in (p, *p.split()):
        v, c, l = gather(child, lambda x: p.velocity+p.affine@(x-p.x), dx)
        np.testing.assert_allclose(v, child.velocity, atol=1e-12)
        np.testing.assert_allclose(c, p.affine, atol=1e-10)
        np.testing.assert_allclose(l, p.affine, atol=1e-10)
    def error(a, b):
        return sum(abs(a.get(k, np.zeros(5))[0]-b.get(k, np.zeros(5))[0]) for k in a.keys()|b.keys())
    fine = scatter([p], dx, order=24)
    coarse_error = error(before, fine)
    refined_error = error(after, fine)
    assert refined_error < coarse_error, (coarse_error, refined_error)
    assert coarse_error < 0.01, coarse_error
    print(f'[PASS] CPU domain transfers: mass/linear/angular momentum, stress balance, affine reproduction; '
          f'L1 nodal mass error {coarse_error:.6g} -> {refined_error:.6g} after bisection')


if __name__ == '__main__':
    check_reference()

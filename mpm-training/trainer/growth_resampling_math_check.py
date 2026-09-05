"""CPU algebra checks supporting GROWTH_REDESIGN.md; no simulator changes.

Run: trainer/.venv/bin/python trainer/growth_resampling_math_check.py
"""
import numpy as np


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def affine_spin(c, covariance):
    return (c @ covariance)[1, 0] - (c @ covariance)[0, 1]


def totals(m, x, v, c, covariance):
    return np.array([
        m,
        *(m * x),
        *(m * v),
        m * (cross(x, v) + affine_spin(c, covariance)),
        0.5 * m * (v @ v + np.trace(c @ covariance @ c.T)),
    ])


def check_domain_bisection():
    rng = np.random.default_rng(42)
    for _ in range(100):
        h = rng.normal(size=(2, 2))
        if abs(np.linalg.det(h)) < 1e-5:
            continue
        x, v = rng.normal(size=(2, 2))
        c = rng.normal(size=(2, 2))
        m = rng.uniform(0.1, 10)
        parent_cov = h @ h.T / 3
        for axis in (0, 1):
            delta = h[:, axis] / 2
            child_h = h.copy()
            child_h[:, axis] /= 2
            child_cov = child_h @ child_h.T / 3
            np.testing.assert_allclose(child_cov + np.outer(delta, delta), parent_cov,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(2 * abs(np.linalg.det(child_h)),
                                       abs(np.linalg.det(h)), rtol=1e-12, atol=1e-12)
            before = totals(m, x, v, c, parent_cov)
            after = sum(totals(m / 2, x + sign * delta, v + sign * c @ delta,
                               c, child_cov) for sign in (-1, 1))
            np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
    print('[PASS] domain bisection preserves area, covariance, mass, centroid, '
          'linear/angular momentum and affine kinetic energy')


def check_point_apic_counterexample():
    dx, mass, omega = 1 / 256, 1.0, 3.0
    x, v = np.array([0.5, 0.5]), np.zeros(2)
    delta = np.array([0.0027 / 2, 0.0])
    c = np.array([[0.0, -omega], [omega, 0.0]])
    d = np.eye(2) * dx**2 / 4
    before = totals(mass, x, v, c, d)
    after = sum(totals(mass / 2, x + sign * delta, v + sign * c @ delta, c, d)
                for sign in (-1, 1))
    expected_spin = mass * omega * (delta @ delta)
    expected_energy = 0.5 * mass * ((c @ delta) @ (c @ delta))
    np.testing.assert_allclose(after[:5], before[:5], atol=1e-14)
    np.testing.assert_allclose(after[5:] - before[5:], [expected_spin, expected_energy],
                               rtol=1e-12, atol=1e-14)
    # Independently scatter onto the actual quadratic point stencil, without
    # forces or fixed-point rounding, and measure grid angular momentum.
    def grid_angular_momentum(position, velocity, particle_mass):
        base = np.floor(position / dx - 0.5).astype(int)
        fx = position / dx - base
        weights = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                   0.5 * (fx - 0.5)**2]
        angular = 0.0
        for i in range(3):
            for j in range(3):
                node = (base + np.array([i, j])) * dx
                momentum = (particle_mass * weights[i][0] * weights[j][1]
                            * (velocity + c @ (node - position)))
                angular += cross(node, momentum)
        return angular

    grid_before = grid_angular_momentum(x, v, mass)
    grid_after = sum(grid_angular_momentum(x + sign * delta,
                                          v + sign * c @ delta, mass / 2)
                     for sign in (-1, 1))
    np.testing.assert_allclose(grid_after - grid_before, expected_spin,
                               rtol=1e-10, atol=1e-14)
    assert expected_spin > 0
    print(f'[PASS] point APIC split counterexample: angular momentum jump '
          f'{expected_spin:.9g}, affine energy jump {expected_energy:.9g}')


if __name__ == '__main__':
    check_domain_bisection()
    check_point_apic_counterexample()

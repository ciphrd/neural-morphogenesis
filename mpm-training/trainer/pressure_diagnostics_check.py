"""Analytic checks for the pressure energy inventory and timestep override."""
import numpy as np
from pressure_diagnostics import measure
from elastic_diagnostics import particle_elastic_state
from mpm_core import GROWTH_FIELD_CHANNELS, MpmCore, DT, DX
from device import pick_device

def check_inventory():
    rng = np.random.default_rng(12)
    n = 64
    rest = np.zeros((n, 16), np.float32)
    rest[:, :4] = np.tile([1.3, 0, 0, .9], (n, 1))
    rest[:, 4] = rng.uniform(.6, 1.2, n)
    rest[:, 15] = rng.uniform(.01, .5, n)
    rest[:, 8:14] = [.5, .5, .501, .5, .5, .501]
    f = np.tile([1.2, .1, .02, .95], (n, 1))
    v, c = rng.normal(size=(n, 2)), rng.normal(size=(n, 4))
    snap = dict(rest=rest, deformation=f, velocities=v, affine=c)
    metrics = measure(snap, density=4)
    reference = particle_elastic_state(f, rest, material_e=10000, material_nu=.2,
                                        material_hardening=3, particle_volume=.25)
    np.testing.assert_allclose(metrics['elastic_energy'], reference.elastic_energy.sum(), rtol=1e-6)
    mass = 2.5*rest[:, 15].astype(float)*np.linalg.det(rest[:, :4].astype(float).reshape(-1, 2, 2))
    expected = sum(.5*m*(np.dot(vel, vel)+DX*DX/4*np.dot(aff, aff))
                   for m, vel, aff in zip(mass, v, c))
    np.testing.assert_allclose(metrics['kinetic_energy']+metrics['affine_kinetic_energy'], expected)
    assert metrics['inverted_triangles'] == 0
    rest[0, 10:14] = [.5, .501, .501, .5]
    assert measure(snap)['inverted_triangles'] == 1
    print('[PASS] elastic inventory matches reference; APIC energy and signed inversion counting verified')

def check_timestep(device):
    for bad in (0, -1, float('nan'), float('inf')):
        try:
            MpmCore(device, physics_dt=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid timestep accepted')
    displacements = []
    for divisor in (1, 2, 4):
        core = MpmCore(device, physics_dt=DT/divisor)
        core.set_gravity(0)
        core.set_repulsion_strength(0, 40)
        core.set_damping(.1, 32)
        # Growth duration must still mean the same elapsed time when the
        # caller increases substeps proportionally to the smaller timestep.
        core.set_material(0, 0.2, 0, 1, growth_duration_macro_steps=12, substeps_per_macro=32 * divisor, growth_compression_feedback=0)
        core.load_scene(np.array([[.5, .5]], np.float32), np.array([[1., 0]], np.float32),
                        np.array([[1, 0, 0, 1]], np.float32), np.zeros((1,4), np.float32),
                        np.ones(1, np.float32))
        field = np.zeros((core.growth_field.size//(4*GROWTH_FIELD_CHANNELS), GROWTH_FIELD_CHANNELS), np.float32)
        field[:, 2] = 1.0
        field[:, 5] = 1.0
        device.queue.write_buffer(core.growth_field, 0, field)
        core.step(32*divisor)
        g = core.read_rest_state()[0, :4].reshape(2, 2)
        # Repeated f32 matrix exponentials on Metal accumulate ~6e-5
        # relative error at the smallest timestep; a cadence error is percent-scale.
        np.testing.assert_allclose(np.linalg.det(g), 2**(1/12), rtol=1e-4)
        np.testing.assert_allclose(core.read_velocities(), [[.9, 0]], atol=3e-5)
        displacements.append(float(core.read_positions()[0,0]-.5))
    assert max(displacements)-min(displacements) < 3e-6
    print('[PASS] timestep override preserves physical damping, growth duration, and translation time')

def main():
    check_inventory()
    check_timestep(pick_device())

if __name__ == '__main__':
    main()

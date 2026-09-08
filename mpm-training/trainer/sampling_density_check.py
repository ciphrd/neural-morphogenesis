"""Coarser refinement preserves the initial seed and represented material."""
import numpy as np
from config import DEFAULT_RUN_SETTINGS
from density import INITIAL_SPACING_IN_SAMPLE_SPACINGS
from continuous_growth_check import make_system, run_growth_field, synchronize_count, read_rest
from device import pick_device
from training_sim import TrainingRollout, seed_blob
from triangle_vertices import domain_edges


def main():
    device = pick_device()
    coarse_spacing = DEFAULT_RUN_SETTINGS['sampleSpacing']
    seed_spacing = INITIAL_SPACING_IN_SAMPLE_SPACINGS * coarse_spacing
    core, agents, environment = make_system(device, capacity=8192, include_environment=True)
    initial = seed_blob(5, (.5,.5), seed_spacing, 17)
    counts = []
    for spacing in (seed_spacing, coarse_spacing):
        agents.set_density_geometry(spacing)
        TrainingRollout(core, agents, environment, (.5,.5), gravity=0, seed=17,
                        initial_particle_count=5, initial_spacing=seed_spacing)
        np.testing.assert_array_equal(core.read_positions(), initial[0])
        np.testing.assert_array_equal(core.read_rest_state()[:,8:14], initial[5])
        rest = core.read_rest_state()
        # Same grown tissue for both sampling targets. Its area is 256 times
        # the seed area, with stress-free deformation and zero growth proposal.
        vertices = rest[:,8:14].reshape(-1,3,2)
        vertices[:] = .5 + 16*(vertices-.5)
        rest[:,0] = rest[:,3] = 16
        positions = vertices.mean(axis=1)
        device.queue.write_buffer(core.positions, 0, positions.astype(np.float32))
        device.queue.write_buffer(core.rest, 0, rest)
        device.queue.write_buffer(core.F, 0, rest[:,:4].copy())
        area = .5*np.linalg.det(domain_edges(rest)).sum()
        for _ in range(48):
            run_growth_field(device, agents)
            count = synchronize_count(core, agents)
        after = read_rest(core, count)
        assert not agents.capacity_blocked
        np.testing.assert_allclose(after[:,15].sum(), 5, rtol=1e-5)
        np.testing.assert_allclose(after[:,14].sum(), rest[:,14].sum(), rtol=1e-5)
        np.testing.assert_allclose(.5*np.linalg.det(domain_edges(after)).sum(), area, rtol=1e-4)
        counts.append(count)
    assert counts[1] < counts[0] / 8, counts
    print(f'[PASS] seed geometry unchanged; same grown area uses {counts[0]} → {counts[1]} samples ({counts[0]/counts[1]:.1f}x fewer)')


if __name__ == '__main__':
    main()

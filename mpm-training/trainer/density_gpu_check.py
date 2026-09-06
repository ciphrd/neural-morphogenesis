"""GPU regression: change numerical density while preserving physical material."""
import numpy as np
from agents_gpu import AgentsGPU, weight_layout
from density import DensityReference, resolve_density
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore, PARTICLE_MASS, VOL
from policy_parameters import policy_hidden_dim
from simulation_settings import (CHEM_CHANNELS, CHEMICAL_COMMUNICATION_ARCHITECTURE,
    CHEMICAL_GRADIENT_INPUT_SCALE, DECAY, DEPOSIT_RATE, FIELD_N, FRICTION,
    MAX_ENV_WRITE, POLICY_ARCHITECTURE, REPULSION_MAX_DELTA, REPULSION_STRENGTH, SAMPLE_SPACING)
from training_sim import TrainingRollout


def main():
    reference = DensityReference(particle_cap=20, initial_particles=4,
        chemical_field_n=FIELD_N, particle_mass=PARTICLE_MASS, particle_volume=VOL,
        chemical_gradient_input_scale=CHEMICAL_GRADIENT_INPUT_SCALE,
        repulsion_strength=REPULSION_STRENGTH, repulsion_max_delta=REPULSION_MAX_DELTA)
    cases = [resolve_density(reference, q) for q in (.25, .5, 1, 2, 4)]
    device = pick_device()
    core = MpmCore(device)
    environment = EnvironmentGPU(device, CHEM_CHANNELS, FIELD_N, FIELD_N, DECAY, DEPOSIT_RATE)
    hidden = policy_hidden_dim(POLICY_ARCHITECTURE)
    agents = AgentsGPU(device, core, environment, CHEM_CHANNELS, hidden, MAX_ENV_WRITE,
        max(c.particle_cap for c in cases), SAMPLE_SPACING, FRICTION, 1, .5, .5)
    weights = np.zeros(agents._total_floats, np.float32)
    weights[weight_layout(CHEM_CHANNELS, hidden, POLICY_ARCHITECTURE)['fc2b_offset'] + CHEM_CHANNELS] = 20
    agents.load_weights(weights)
    final_areas = []
    for case in cases:
        agents.set_forced_growth_field_override(False)
        # Isotropic growth without stress isolates physical material accounting
        # from density-dependent differences in geometric discretization.
        core.set_material(0, .2, 0, 1, growth_duration_macro_steps=8,
            substeps_per_macro=1, growth_anisotropy=0, growth_compression_feedback=0,
            particle_mass=case.particle_mass, particle_volume=case.particle_volume)
        agents.set_density_geometry(case.spacing)
        agents.set_max_active_particles(case.particle_cap)
        sim = TrainingRollout(core, agents, environment, gravity=0, seed=11,
            spawn_center=(.5,.5), initial_particle_count=case.initial_particles, initial_spacing=case.initial_spacing)
        rest = core.read_rest_state()
        np.testing.assert_allclose(rest[:,15].sum()*case.particle_volume,
            reference.initial_particles*reference.particle_volume, rtol=1e-6)
        sim.macro_step(1, growth_enabled=False)
        baseline = core.read_rest_state()
        np.testing.assert_allclose(baseline[:,:4], rest[:,:4], atol=1e-7)
        # A constant *vector* now contracts inward-facing boundaries; it no
        # longer specifies isotropic positive growth. Use a prescribed tensor
        # to isolate accounting across different coarse seed geometries. Signed
        # projection/refinement invariance is tested in signed_growth_check.py.
        agents.set_forced_growth_field_override(True)
        for _ in range(12):
            sim.macro_step(1)
        rest = core.read_rest_state()
        det = rest[:,0]*rest[:,3]-rest[:,1]*rest[:,2]
        area = float(np.sum(rest[:,15]*det)*case.particle_volume)
        assert np.isfinite(sim.positions()).all() and np.isfinite(rest).all()
        assert area > reference.initial_particles*reference.particle_volume*1.1
        assert core.active_count <= case.particle_cap and not agents.capacity_blocked
        final_areas.append(area)
    # Prescribed unit trace integrates to the same physical rest area at every density.
    areas = np.array(final_areas)
    expected = reference.initial_particles*reference.particle_volume*2**(12/8)
    np.testing.assert_allclose(areas, expected, rtol=3e-5)
    print(f'[PASS] persistent pipelines switch 0.25/0.5/1/2/4x density: seed material conserved, growth active, final areas={areas}')

if __name__ == '__main__':
    main()

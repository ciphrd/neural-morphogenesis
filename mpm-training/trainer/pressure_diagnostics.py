"""Energy and geometry measurements for the current point-transfer material."""
import numpy as np
from mpm_core import DX, REST_FIELDS, lame_params
from triangle_vertices import domain_edges


def read_snapshot(core):
    n = core.active_count
    def read(buffer, width):
        return np.frombuffer(core.device.queue.read_buffer(buffer, 0, n*width*4),
                             np.float32).reshape(n, width).copy()
    return dict(positions=read(core.positions, 2), velocities=read(core.velocities, 2),
                deformation=read(core.F, 4), affine=read(core.C, 4), rest=read(core.rest, REST_FIELDS))


def measure(snapshot, *, density=1., material_e=10000., hardening=3., nu=.2):
    """Includes APIC affine kinetic energy, using D=dx²/4 I.

    This is an energy inventory, not a conserved-energy claim during growth,
    plasticity/hardening, damping, or particle relocation at refinement.
    """
    rest = snapshot['rest'].astype(float)
    f = snapshot['deformation'].astype(float).reshape(-1, 2, 2)
    fg = rest[:, :4].reshape(-1, 2, 2)
    g = np.linalg.det(fg)
    mass = (10/density)*rest[:, 11]*g
    fe = f @ np.linalg.inv(fg)
    je = np.linalg.det(fe)
    x, y = fe[:, 0, 0]+fe[:, 1, 1], fe[:, 1, 0]-fe[:, 0, 1]
    norm = np.hypot(x, y)
    c = np.divide(x, norm, out=np.ones_like(x), where=norm >= 1e-6)
    s = np.divide(y, norm, out=np.zeros_like(y), where=norm >= 1e-6)
    rotation = np.stack((c, -s, s, c), axis=1).reshape(-1, 2, 2)
    mu0, lam0 = lame_params(material_e, nu)
    scale = np.exp(hardening*(1-rest[:, 4]))
    energy_density = mu0*scale*np.sum((fe-rotation)**2, axis=(1, 2)) + .5*lam0*scale*(je-1)**2
    elastic = np.sum(rest[:, 11]*g*energy_density/density)
    velocity = snapshot['velocities'].astype(float)
    affine = snapshot['affine'].astype(float)
    kinetic = .5*np.sum(mass*np.sum(velocity**2, axis=1))
    affine_kinetic = (DX**2/8)*np.sum(mass*np.sum(affine**2, axis=1))
    edges = domain_edges(rest)
    signed_area = .5*np.linalg.det(edges)
    v = rest[:, 12:18].reshape(-1, 3, 2)
    e = np.roll(v, -1, axis=1)-v
    e -= np.floor(e+.5)
    edge_sum = np.sum(e*e, axis=(1, 2))
    quality = 4*np.sqrt(3)*signed_area/np.maximum(edge_sum, 1e-30)
    return dict(samples=len(rest), elastic_energy=float(elastic), kinetic_energy=float(kinetic),
                affine_kinetic_energy=float(affine_kinetic),
                total_energy=float(elastic+kinetic+affine_kinetic), mass=float(mass.sum()),
                grown_rest_area=float(np.sum(rest[:, 8]*g)),
                max_speed=float(np.linalg.norm(velocity, axis=1).max()),
                max_volumetric_pressure=float(np.max(-lam0*scale*(je-1))),
                min_elastic_det=float(je.min()), min_triangle_area=float(signed_area.min()),
                inverted_triangles=int(np.sum(signed_area <= 0)),
                min_triangle_quality=float(quality.min()), max_edge=float(np.linalg.norm(e, axis=2).max()),
                min_weight=float(rest[:, 11].min()), max_hardening=float(scale.max()))

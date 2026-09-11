"""Passive substrate marker: stationarity, periodic translation, and expansion.

Run with trainer/.venv/bin/python trainer/substrate_marker_check.py.
Uses the same advectOnly WGSL entry point as the viewer diagnostic.
"""
import numpy as np
import wgpu

from chemical_channels import ChemicalChannelProfile, channel_shader_constants
from device import pick_device
from shader_template import load_core_shader


def main():
    device = pick_device()
    n, grid_n = 512, 64
    constants = channel_shader_constants(n, n, (ChemicalChannelProfile(
        scale="local", resolution_scale=1, relaxation_time=1,
        field_response_time=1, decay_exponent=1, diffusion_multiplier=0,
    ),))
    module = device.create_shader_module(code=load_core_shader(
        "environment.wgsl", {"CHANNELS": 1, "GRID_N": grid_n, **constants}))
    pipeline = device.create_compute_pipeline(layout="auto", compute={
        "module": module, "entry_point": "advectOnly"})
    usage = wgpu.BufferUsage
    source = device.create_buffer(size=n*n*4, usage=usage.STORAGE | usage.COPY_DST)
    output = device.create_buffer(size=n*n*4, usage=usage.STORAGE | usage.COPY_SRC)
    velocity = device.create_buffer(size=(grid_n+1)**2*8, usage=usage.STORAGE | usage.COPY_DST)
    physics = device.create_buffer(size=32, usage=usage.UNIFORM | usage.COPY_DST)
    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
        {"binding": binding, "resource": {"buffer": buffer}}
        for binding, buffer in ((0, source), (3, output), (4, physics), (5, velocity))
    ])
    y, x = np.mgrid[:n, :n].astype(np.float32)
    x = (x+.5)/n; y = (y+.5)/n
    radius = np.hypot(x-.5, y-.5).astype(np.float32)

    def advect(field, flow, dt):
        device.queue.write_buffer(source, 0, np.asarray(field, np.float32))
        device.queue.write_buffer(velocity, 0, np.asarray(flow, np.float32))
        # Deliberately nontrivial reaction coefficients: passive entry ignores them.
        device.queue.write_buffer(physics, 0, np.array([.5, 7, 1, 0, dt, 0, 0, 0], np.float32))
        encoder = device.create_command_encoder()
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline); p.set_bind_group(0, group)
        p.dispatch_workgroups(n//16, n//16); p.end()
        device.queue.submit([encoder.finish()])
        return np.frombuffer(device.queue.read_buffer(output), np.float32).reshape(n, n).copy()

    flow = np.zeros((grid_n+1, grid_n+1, 2), np.float32)
    np.testing.assert_array_equal(advect(radius, flow, 1), radius)
    print("[PASS] zero velocity preserves radial coordinates exactly; no decay/diffusion")

    flow[:] = [3/n, -2/n]
    np.testing.assert_allclose(advect(radius, flow, 1), np.roll(radius, (-2, 3), axis=(0, 1)), atol=2e-7)
    np.testing.assert_array_equal(advect(radius, flow, 0), radius)
    print("[PASS] translated rings move with the flow, wrap periodically, and freeze at dt=0")

    # Grid velocity is x-major, unlike the chemical field. Anisotropic expansion
    # must turn circular marker level sets into ellipses, not reseed circles.
    gx, gy = np.meshgrid(np.arange(grid_n+1)/grid_n, np.arange(grid_n+1)/grid_n, indexing="ij")
    flow[..., 0] = .2*(gx-.5); flow[..., 1] = .05*(gy-.5)
    result = advect(radius, flow, 1)
    expected = np.hypot(.8*(x-.5), .95*(y-.5))
    interior = (radius > .08) & (radius < .35)
    np.testing.assert_allclose(result[interior], expected[interior], atol=1.3e-5)
    assert np.max(np.abs(result[interior] - radius[interior])) > .04
    print("[PASS] anisotropic flow deforms the radial texture into ellipses")


if __name__ == "__main__":
    main()

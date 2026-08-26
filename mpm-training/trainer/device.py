"""GPU device acquisition — mirrors envnca/device.py's own one-function
shape (pick_device()), but for a wgpu adapter/device pair instead of a
torch device, since this project's physics runs as WGSL compute via
`wgpu` rather than torch tensor ops.
"""
from __future__ import annotations

import wgpu


def pick_device(verbose: bool = True) -> wgpu.GPUDevice:
    """Requests a high-performance adapter and its default device,
    logging which backend actually got picked — on this project's own
    machine that's expected to be Metal; confirming that (rather than an
    unexpectedly slow software/CPU fallback) is the single most important
    piece of information this whole feasibility spike depends on.

    `verbose=False` skips that log line — every worker process
    parallel_workers.py's own build_pool() spawns calls this once, each
    picking its own separate device (never shared across processes — see
    that module's own docstring for why), and on one machine they all
    report the identical backend/adapter, so build_pool() logs it exactly
    once itself (a throwaway verbose=True call in the main process,
    before any worker exists) rather than every worker repeating the same
    line --workers times over."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    if verbose:
        print(f"[device] adapter: {info['device']!r} backend={info['backend_type']} type={info['adapter_type']}")
    float32_filterable = wgpu.FeatureName.float32_filterable
    required_features = (
        [float32_filterable] if float32_filterable in adapter.features else []
    )
    return adapter.request_device_sync(required_features=required_features)

"""torch device selection — CUDA -> MPS -> CPU precedence, mirrors
envnca/device.py's own pick_device(), renamed to avoid colliding with
this project's own device.py, which picks a *wgpu* adapter/device for
MpmCore's physics — an entirely separate concern from where the evolved
policy's torch tensors live.
"""
from __future__ import annotations

import torch


def pick_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

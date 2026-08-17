"""GPU device selection — one place that decides CUDA -> MPS -> CPU
precedence, so every entry point (evolve.py, and whatever the web
frontend's backend ends up importing this from) picks the same device
the same way."""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

"""Single source of truth for the compute device."""

from __future__ import annotations

import os

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the compute device.

    Order: explicit ``prefer`` → ``CAPTIONER_DEVICE`` env var → CUDA → MPS → CPU.

    Args:
        prefer: Optional device string such as ``"cpu"`` or ``"cuda:0"``.

    Returns:
        The selected ``torch.device``.
    """
    choice = prefer or os.environ.get("CAPTIONER_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

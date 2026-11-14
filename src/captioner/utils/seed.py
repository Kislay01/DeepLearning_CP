"""Seeding for reproducible runs (Python, NumPy, torch, DataLoader workers)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG the project touches.

    Args:
        seed: Global seed.
        deterministic: Ask cuDNN / torch for deterministic kernels where available.

    Returns:
        A ``torch.Generator`` seeded with ``seed`` for DataLoader shuffling.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by some third-party code
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and ``random`` inside a DataLoader worker from torch's per-worker seed."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)

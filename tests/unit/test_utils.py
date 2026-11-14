from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from captioner.utils.device import get_device
from captioner.utils.io import bytes_digest, file_digest, git_sha, read_json, write_json
from captioner.utils.logging import setup_logging
from captioner.utils.seed import seed_everything, worker_init_fn


def test_seed_everything_is_reproducible() -> None:
    seed_everything(42)
    a = (random.random(), np.random.rand(), torch.rand(3))  # noqa: NPY002
    gen = seed_everything(42)
    b = (random.random(), np.random.rand(), torch.rand(3))  # noqa: NPY002
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])
    assert (
        torch.randint(0, 10**6, (1,), generator=gen).item()
        == torch.randint(0, 10**6, (1,), generator=seed_everything(42)).item()
    )
    worker_init_fn(0)


def test_get_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_device("cpu") == torch.device("cpu")
    monkeypatch.setenv("CAPTIONER_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")
    monkeypatch.delenv("CAPTIONER_DEVICE")
    assert isinstance(get_device(), torch.device)


def test_digests_and_json(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert file_digest(p, "md5") == "900150983cd24fb0d6963f7d28e17f72"
    assert file_digest(p) == bytes_digest(b"abc")
    write_json({"k": [1, "é"]}, tmp_path / "sub" / "a.json")
    assert read_json(tmp_path / "sub" / "a.json") == {"k": [1, "é"]}


def test_git_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert git_sha(tmp_path) == "unknown"

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert git_sha() == "unknown"


def test_setup_logging_idempotent() -> None:
    setup_logging("INFO")
    setup_logging("DEBUG")

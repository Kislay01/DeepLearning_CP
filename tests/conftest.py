"""Shared fixtures. Everything runs on CPU against tests/fixtures/mini_flickr."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from captioner.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "mini_flickr"
MINI_COUNTS = {"train": 12, "val": 4, "test": 4}

os.environ.setdefault("CAPTIONER_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

# Small, fast settings for unit/integration tests (no pretrained weights, tiny decoder).
TEST_OVERRIDES: dict[str, Any] = {
    "vocab": {"min_freq": 1},
    "encoder": {"weights": None, "batch_size": 4, "num_workers": 0},
    "model": {"embed_dim": 16, "decoder_dim": 32, "attention_dim": 16, "dropout": 0.0},
    "train": {
        "batch_size": 4,
        "epochs": 2,
        "num_workers": 0,
        "amp": False,
        "log_samples": 2,
        "patience": 5,
    },
    "decode": {"beam": 2},
    "eval": {"beams": [1, 2], "n_random": 2, "n_best_worst": 2, "n_attention": 2, "n_taxonomy": 4},
}


def make_config(tmp_path: Path, extra: dict[str, Any] | None = None) -> Config:
    """Base config pointed at ``tmp_path`` with the test overrides applied."""
    paths = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "reports_dir": str(tmp_path / "reports"),
            "mlruns_dir": str(tmp_path / "mlruns"),
        }
    }
    cfg = load_config(ROOT / "configs" / "base.yaml", overrides=paths)
    cfg = cfg.with_overrides(TEST_OVERRIDES)
    return cfg.with_overrides(extra) if extra else cfg


def install_mini_raw(cfg: Config) -> None:
    """Copy the mini fixture into ``cfg.paths.raw_dir`` as if it had been extracted."""
    shutil.copytree(MINI, cfg.paths.raw_dir, dirs_exist_ok=True)


@pytest.fixture
def mini_cfg(tmp_path: Path) -> Config:
    cfg = make_config(tmp_path)
    install_mini_raw(cfg)
    return cfg


@pytest.fixture(scope="session")
def prepared_mini(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Session-wide mini dataset with manifest, vocab and random-weight features. Read-only."""
    from captioner.data.prepare import prepare_data
    from captioner.features.extract import extract_features

    cfg = make_config(tmp_path_factory.mktemp("prepared"))
    install_mini_raw(cfg)
    prepare_data(cfg)
    torch.manual_seed(1234)  # random-init encoder → make the cached features reproducible
    extract_features(cfg)
    return cfg


@pytest.fixture(scope="session")
def trained_runs(prepared_mini: Config) -> dict[str, Path]:
    """One short training run per model kind on the mini data (session-wide, read-only)."""
    from captioner.training.train import run_training

    runs = {}
    for kind in ("merge", "attention"):
        cfg = prepared_mini.with_overrides(
            {"name": f"t-{kind}", "model": {"kind": kind}, "train": {"epochs": 1}}
        )
        runs[kind] = run_training(cfg).run_dir
    return runs

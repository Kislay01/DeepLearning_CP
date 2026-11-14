"""ADR-024 — full pipeline with the CLIP ViT-B/32 encoder (random weights in CI)."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch
from PIL import Image

from captioner.data.prepare import prepare_data
from captioner.features.extract import extract_features
from captioner.inference import service
from captioner.inference.predictor import Predictor
from captioner.training.train import run_training
from tests.conftest import MINI, install_mini_raw, make_config

pytestmark = pytest.mark.integration


def test_clip_end_to_end(tmp_path: Path) -> None:
    cfg = make_config(
        tmp_path,
        {
            "name": "clip",
            "encoder": {"name": "clip_vit_b32", "resize": 224},
            "model": {"encoder_dim": 768},
            "train": {"epochs": 1},
        },
    )
    install_mini_raw(cfg)
    prepare_data(cfg)
    torch.manual_seed(0)
    paths = extract_features(cfg)
    with h5py.File(paths["spatial"], "r") as fh:
        assert fh["features"].shape == (20, 49, 768)
    run_dir = run_training(cfg).run_dir
    torch.manual_seed(0)
    pred = Predictor.from_run(run_dir)
    image_path = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0]
    cap = pred.caption(image_path, beam=2)
    assert cap.attention is None or cap.attention.shape[1:] == (7, 7)
    with Image.open(image_path) as img:
        maps = service.build_heatmaps(img.convert("RGB"), cap, 12, "clip_vit_b32")
    assert maps is None or all(m.size == (224, 224) for m in maps.images)

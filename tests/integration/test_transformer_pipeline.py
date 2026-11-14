"""ADR-027 — train, calibrate, evaluate and serve the Transformer decoder end to end."""

from __future__ import annotations

import json

import pytest
import torch

from captioner.config import Config, load_config
from captioner.evaluation.evaluate import evaluate_run
from captioner.inference.predictor import Predictor
from captioner.training.train import run_training
from tests.conftest import MINI, ROOT

pytestmark = pytest.mark.integration


def test_transformer_end_to_end(prepared_mini: Config) -> None:
    cfg = prepared_mini.with_overrides(
        {
            "name": "m3",
            "model": {
                "kind": "transformer",
                "decoder_dim": 32,
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ff": 64,
                "dropout": 0.1,
                "alpha_c": 0.0,
            },
            "train": {
                "epochs": 1,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "warmup_steps": 2,
                "select_metric": "cider",
            },
        }
    )
    run_dir = run_training(cfg).run_dir
    assert (run_dir / "reliability.safetensors").exists()
    result = evaluate_run(cfg, run_dir, beams=[1, 2])
    metrics = json.loads((result.report_dir / "metrics.json").read_text())
    assert metrics["model_kind"] == "transformer"
    torch.manual_seed(7)
    pred = Predictor.from_run(run_dir)
    assert pred.has_attention
    cap = pred.caption(sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0], beam=2)
    assert cap.attention is None or cap.attention.shape == (len(cap.tokens), 7, 7)


def test_transformer_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "coco_clip_transformer.yaml")
    assert cfg.model.kind == "transformer" and cfg.encoder.name == "clip_vit_b32"
    assert cfg.model.encoder_dim == 768 and cfg.train.select_metric == "cider"

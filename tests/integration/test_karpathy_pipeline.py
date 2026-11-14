"""ADR-023 — the full pipeline on a COCO-format (Karpathy) dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.data.download import download_dataset
from captioner.data.prepare import prepare_data
from captioner.evaluation.evaluate import evaluate_run
from captioner.features.extract import extract_features
from captioner.inference.predictor import Predictor
from captioner.training.train import run_training
from tests.karpathy_fixture import build

pytestmark = pytest.mark.integration


def test_coco_style_end_to_end(tmp_path: Path) -> None:
    cfg, _ = build(tmp_path)
    cfg = cfg.with_overrides({"vocab": {"gender_neutral": True}, "train": {"epochs": 1}})
    download_dataset(cfg)
    prepare_data(cfg)
    paths = extract_features(cfg)
    assert all(p.parent.name == "coco" for p in paths.values())
    run_dir = run_training(cfg).run_dir
    result = evaluate_run(cfg.with_overrides({"eval": {"source": "coco"}}), run_dir, beams=[1])
    metrics = json.loads((result.report_dir / "metrics.json").read_text())
    assert metrics["source"] == "coco" and metrics["n_images"] == 4
    assert (result.report_dir / "samples.html").exists()
    image = next((cfg.paths.raw_dir / "coco" / "val2014").glob("*.jpg"))
    assert Predictor.from_run(run_dir).caption(image).text is not None

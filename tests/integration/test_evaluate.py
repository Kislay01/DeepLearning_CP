"""FR-7 / D1 — evaluation outputs on the mini test split, and the full CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from captioner.cli import app
from captioner.config import Config
from captioner.evaluation.evaluate import evaluate_run
from captioner.evaluation.metrics import METRIC_KEYS
from tests.conftest import MINI, install_mini_raw, make_config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_regression_D1_evaluation_reports(
    prepared_mini: Config, trained_runs: dict[str, Path], kind: str
) -> None:
    result = evaluate_run(prepared_mini, trained_runs[kind], beams=[1, 2])
    metrics = json.loads((result.report_dir / "metrics.json").read_text())
    assert metrics["split"] == "test" and metrics["n_images"] == 4
    assert set(metrics["results"]) == {"beam_1", "beam_2"}
    for res in metrics["results"].values():
        assert set(METRIC_KEYS) <= set(res)
        assert {"diversity", "failure_taxonomy"} <= set(res)
    preds = pd.read_csv(result.report_dir / "predictions.csv")
    expected = [
        "image_id",
        "split",
        "beam",
        "prediction",
        *[f"ref_{i}" for i in range(5)],
        "bleu4_sentence",
        "cider_sentence",
    ]
    assert list(preds.columns) == expected and len(preds) == 8
    html = (result.report_dir / "samples.html").read_text()
    assert "Failure taxonomy" in html and "best by CIDEr" in html and "<script" not in html
    if kind == "attention":
        assert "Attention maps:" in html


def test_full_cli_pipeline(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, {"name": "cli"})
    install_mini_raw(cfg)
    config_file = tmp_path / "cli.yaml"
    config_file.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    runner = CliRunner()
    steps = [
        ["prepare"],
        ["extract"],
        ["train", "--set", "train.epochs=1"],
        ["evaluate", "--beam", "1", "--beam", "2"],
        ["export", "--out", str(tmp_path / "rel")],
    ]
    for args in steps:
        result = runner.invoke(app, [*args, "--config", str(config_file)])
        assert result.exit_code == 0, (args, result.output)
    image = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0]
    for extra in (["--beam", "3"], ["--release", str(tmp_path / "rel")]):
        result = runner.invoke(
            app, ["predict", "--image", str(image), *extra, "--config", str(config_file)]
        )
        assert result.exit_code == 0, result.output
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    result = runner.invoke(app, ["predict", "--image", str(bad), "--config", str(config_file)])
    assert result.exit_code == 2 and "isn't an image we can read" in result.output
    assert "Model card" in (tmp_path / "rel" / "model_card.md").read_text()
    assert "BLEU-4" in (tmp_path / "rel" / "model_card.md").read_text()
    assert runner.invoke(app, ["version"]).exit_code == 0

"""testing.md §3.1, §3.3, §3.4 — pipeline smoke via the CLI, determinism, checkpoint round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from captioner.cli import app
from captioner.config import Config
from captioner.data.dataset import build_caption_table, open_feature_store
from captioner.data.prepare import load_prepared
from captioner.features.extract import feature_paths
from captioner.inference.decode import greedy_decode
from captioner.training.checkpoint import load_run
from captioner.training.loop import Amp, train_step
from captioner.training.train import build_data, build_optimizer, build_training_model, run_training
from captioner.utils.seed import seed_everything
from tests.conftest import ROOT, install_mini_raw, make_config

pytestmark = pytest.mark.integration


def _write_config(cfg: Config, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


def test_pipeline_smoke_via_cli(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, {"name": "smoke", "model": {"kind": "attention"}})
    install_mini_raw(cfg)
    config_file = _write_config(cfg, tmp_path / "smoke.yaml")
    runner = CliRunner()
    for args in (["prepare"], ["extract"], ["train", "--set", "train.epochs=2"]):
        result = runner.invoke(app, [*args, "--config", str(config_file)])
        assert result.exit_code == 0, result.output
    assert cfg.paths.manifest_path.exists() and cfg.paths.vocab_path.exists()
    assert all(p.exists() for p in feature_paths(cfg).values())
    runs = list(cfg.paths.runs_dir.glob("smoke-*/best.pt"))
    assert len(runs) == 1
    history = json.loads((runs[0].parent / "history.json").read_text())
    assert len(history) == 2


def _first_losses(cfg: Config, steps: int = 20) -> list[float]:
    gen = seed_everything(cfg.seed)
    prepared = load_prepared(cfg)
    data = build_data(cfg, prepared, gen)
    model = build_training_model(cfg, len(prepared.vocab))
    opt = build_optimizer(model, cfg)
    amp = Amp(torch.device("cpu"), enabled=False)
    losses: list[float] = []
    model.train()
    while len(losses) < steps:
        for batch in data.train:
            losses.append(float(train_step(model, batch, opt, amp, 5.0, cfg.model.alpha_c).total))
            if len(losses) == steps:
                break
    return losses


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_determinism_first_20_steps(prepared_mini: Config, kind: str) -> None:
    cfg = prepared_mini.with_overrides({"model": {"kind": kind, "dropout": 0.5}})
    assert _first_losses(cfg) == _first_losses(cfg)


def test_checkpoint_roundtrip_identical_predictions(prepared_mini: Config) -> None:
    cfg = prepared_mini.with_overrides({"model": {"kind": "attention"}, "train": {"epochs": 1}})
    result = run_training(cfg)
    prep = load_prepared(cfg)
    table = build_caption_table(prep.manifest, "test", prep.vocab)
    store = open_feature_store(feature_paths(cfg)["spatial"], table)
    feats = torch.stack([store.get(int(i)) for i in table.feature_idx])
    a = load_run(result.run_dir)
    b = load_run(result.run_dir)
    ha = greedy_decode(a.model, feats, 25)
    hb = greedy_decode(b.model, feats, 25)
    assert [h.ids for h in ha] == [h.ids for h in hb]
    assert [h.logprob for h in ha] == [h.logprob for h in hb]


def test_finetune_stage_runs_from_images(prepared_mini: Config, tmp_path: Path) -> None:
    cfg = prepared_mini.with_overrides(
        {
            "name": "ft",
            "model": {"kind": "attention", "encoder_dim": 2048},
            "train": {"finetune_encoder": True, "epochs": 1, "max_steps_per_epoch": 1},
        }
    )
    result = run_training(cfg)
    loaded = load_run(result.run_dir)
    assert loaded.encoder_state is not None and any(
        k.startswith("body.7.") for k in loaded.encoder_state
    )


def test_configs_load() -> None:
    from captioner.config import load_config

    names = {}
    for name in ("merge_baseline", "attention", "attention_finetune"):
        names[name] = load_config(ROOT / "configs" / f"{name}.yaml")
    assert names["merge_baseline"].model.kind == "merge"
    assert names["merge_baseline"].train.weight_decay == pytest.approx(1e-5)
    assert names["attention"].model.decoder_dim == 512
    assert names["attention_finetune"].train.finetune_encoder
    assert names["attention_finetune"].model.kind == "attention"

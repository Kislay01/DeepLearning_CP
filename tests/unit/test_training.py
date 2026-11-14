from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import typer
import yaml

from captioner.cli import parse_overrides
from captioner.config import Config, ModelConfig
from captioner.evaluation.metrics import corpus_bleu4
from captioner.models.captioner import build_model
from captioner.training.checkpoint import (
    CheckpointError,
    latest_run,
    load_checkpoint,
    load_run,
    save_checkpoint,
)
from captioner.training.early_stopping import EarlyStopping
from captioner.training.loop import Amp
from captioner.training.tracking import RunTracker
from captioner.training.train import build_optimizer, build_training_model, run_training


def test_early_stopping() -> None:
    es = EarlyStopping(patience=2)
    assert es.update(0.1) and not es.should_stop
    assert not es.update(0.1)
    assert es.update(0.2)
    assert not es.update(0.15) and not es.should_stop
    assert not es.update(0.19) and es.should_stop
    assert es.best == 0.2


def test_amp_disabled_on_cpu() -> None:
    amp = Amp(torch.device("cpu"), enabled=True)
    assert not amp.enabled and not amp.scaler.is_enabled()
    with amp.autocast():
        assert torch.ones(1).dtype == torch.float32


def test_corpus_bleu4_identical_and_disjoint() -> None:
    hyp = [["a", "dog", "runs", "on", "the", "grass"]]
    assert corpus_bleu4(hyp, [[hyp[0]]]) == pytest.approx(1.0)
    assert corpus_bleu4(hyp, [[["two", "cats", "sleep", "in", "a", "box"]]]) == 0.0
    assert corpus_bleu4([], []) == 0.0
    with pytest.raises(ValueError, match="same length"):
        corpus_bleu4(hyp, [])


def test_parse_overrides() -> None:
    assert parse_overrides(["train.epochs=2", "name=x", "eval.beams=[1, 3]"]) == {
        "train": {"epochs": 2},
        "name": "x",
        "eval": {"beams": [1, 3]},
    }
    assert parse_overrides(None) == {}
    with pytest.raises(typer.BadParameter):
        parse_overrides(["oops"])


def test_checkpoint_errors(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="Cannot load"):
        load_checkpoint(tmp_path / "junk.pt")
    torch.save({"x": torch.ones(1)}, tmp_path / "other.pt")
    with pytest.raises(CheckpointError, match="not a captioner"):
        load_checkpoint(tmp_path / "other.pt")
    with pytest.raises(CheckpointError, match="No trained runs"):
        latest_run(tmp_path)


def test_tracker_writes_mlflow_and_tensorboard(tmp_path: Path) -> None:
    with RunTracker(tmp_path / "mlruns", "exp", "run-1", tmp_path / "tb") as tr:
        tr.log_params({"a": "1"})
        tr.set_tags({"model": "merge"})
        tr.log_metrics({"val/bleu4": 0.5}, step=1)
        tr.log_text("hello", "s.txt", 1)
        (tmp_path / "f.txt").write_text("x")
        tr.log_artifact(tmp_path / "f.txt")
        tr.log_artifact(tmp_path / "missing.txt")
        assert tr.run_id
    assert any((tmp_path / "mlruns").rglob("bleu4"))
    assert any((tmp_path / "tb").iterdir())


def test_run_training_writes_run_artifacts(prepared_mini: Config) -> None:
    cfg = prepared_mini.with_overrides({"model": {"kind": "merge"}, "train": {"epochs": 2}})
    result = run_training(cfg)
    names = {p.name for p in result.run_dir.iterdir()}
    assert {"best.pt", "last.pt", "config.yaml", "vocab.json", "history.json", "tb"} <= names
    history = json.loads((result.run_dir / "history.json").read_text())
    assert len(history) == result.epochs_run == 2
    assert {"train/loss", "val/loss", "val/bleu4", "lr", "train/perplexity"} <= set(history[0])
    snap = yaml.safe_load((result.run_dir / "config.yaml").read_text())
    assert Config.model_validate(snap) == cfg
    meta = load_checkpoint(result.run_dir / "best.pt")["meta"]
    assert meta["model_kind"] == "merge" and meta["git_sha"] and meta["vocab_sha256"]
    assert "optimizer" in load_checkpoint(result.run_dir / "last.pt")
    loaded = load_run(result.run_dir)
    assert not loaded.model.training and loaded.encoder_state is None
    assert latest_run(cfg.paths.runs_dir) == result.run_dir
    assert any(cfg.paths.mlruns_dir.rglob("val/bleu4")) or any(cfg.paths.mlruns_dir.rglob("bleu4"))


def test_early_stopping_ends_training(prepared_mini: Config) -> None:
    cfg = prepared_mini.with_overrides(
        {"model": {"kind": "merge"}, "train": {"epochs": 6, "patience": 1, "lr": 0.0}}
    )
    result = run_training(cfg)
    assert result.epochs_run == 2  # epoch 1 sets the best; epoch 2 does not improve with lr=0


def test_vocab_mismatch_detected(prepared_mini: Config) -> None:
    cfg = prepared_mini.with_overrides({"model": {"kind": "merge"}, "train": {"epochs": 1}})
    run_dir = run_training(cfg).run_dir
    vocab = json.loads((run_dir / "vocab.json").read_text())
    vocab["freqs"]["a"] = 1
    (run_dir / "vocab.json").write_text(json.dumps(vocab))
    with pytest.raises(CheckpointError, match="does not match"):
        load_run(run_dir)


def test_init_from_and_optimizer_groups(prepared_mini: Config, tmp_path: Path) -> None:
    mcfg = prepared_mini.model.model_copy(update={"kind": "attention"})
    decoder = build_model(mcfg, 50)
    save_checkpoint(tmp_path / "w.pt", decoder, {"x": 1})
    cfg = prepared_mini.with_overrides({"train": {"init_from": str(tmp_path / "w.pt")}})
    model = build_training_model(cfg, 50)
    for a, b in zip(model.parameters(), decoder.parameters(), strict=True):
        assert torch.equal(a, b)
    ft = cfg.with_overrides({"train": {"finetune_encoder": True}})
    wrapped = build_training_model(ft, 50)
    opt = build_optimizer(wrapped, ft)
    assert [g["lr"] for g in opt.param_groups] == [ft.train.lr, ft.train.encoder_lr]
    save_checkpoint(tmp_path / "ft.pt", wrapped, {"x": 1})
    again = build_training_model(
        cfg.with_overrides({"train": {"init_from": str(tmp_path / "ft.pt")}}), 50
    )
    assert isinstance(again, type(decoder))


def test_model_config_kinds() -> None:
    assert ModelConfig(kind="merge").kind == "merge"

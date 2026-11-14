from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from captioner.config import Config, deep_merge, flatten, load_config, save_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_base_config_matches_architecture_hyperparameters() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.seed == 42
    assert cfg.vocab.min_freq == 5
    assert cfg.vocab.max_len == 25
    assert cfg.model.embed_dim == 256
    assert cfg.model.decoder_dim == 512
    assert cfg.model.attention_dim == 512
    assert cfg.model.encoder_dim == 2048
    assert cfg.train.lr == pytest.approx(4e-4)
    assert cfg.train.grad_clip == 5.0
    assert cfg.train.patience == 5
    assert cfg.decode.beam == 3
    assert cfg.decode.length_alpha == pytest.approx(0.7)
    assert cfg.dataset.images_md5 == "f18a1e2920de5bd84dae7cf08ec78978"
    assert cfg.dataset.text_md5 == "bf6c1abcb8e4a833b7f922104de18627"


def test_defaults_inheritance_and_override(tmp_path: Path) -> None:
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "exp.yaml").write_text(
        yaml.safe_dump({"defaults": ["base"], "name": "exp", "model": {"kind": "merge"}})
    )
    cfg = load_config(tmp_path / "exp.yaml", overrides={"train": {"epochs": 2}})
    assert cfg.name == "exp"
    assert cfg.model.kind == "merge"
    assert cfg.model.decoder_dim == 512  # inherited
    assert cfg.train.epochs == 2


def test_circular_defaults_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("defaults: [b]\n")
    (tmp_path / "b.yaml").write_text("defaults: [a]\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_env_overrides_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CAPTIONER_DATA_DIR", str(tmp_path / "d"))
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.paths.data_dir == tmp_path / "d"
    assert cfg.paths.manifest_path == tmp_path / "d" / "processed" / "manifest.parquet"


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "base.yaml", overrides={"model": {"hidden": 3}})


def test_save_roundtrip_and_with_overrides(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    save_config(cfg, tmp_path / "snap" / "config.yaml")
    again = Config.model_validate(yaml.safe_load((tmp_path / "snap" / "config.yaml").read_text()))
    assert again == cfg
    assert cfg.with_overrides({"decode": {"beam": 5}}).decode.beam == 5


def test_deep_merge_and_flatten() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert flatten({"a": {"b": 1}, "c": [1, 2]}) == {"a.b": "1", "c": "[1, 2]"}

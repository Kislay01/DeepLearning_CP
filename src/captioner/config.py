"""Typed experiment configuration loaded from YAML.

Experiment files inherit other files in the same directory through ``defaults: [base]``.
Paths can be overridden with ``CAPTIONER_*`` environment variables so nothing is hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    mlruns_dir: Path = Path("mlruns")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "manifest.parquet"

    @property
    def vocab_path(self) -> Path:
        return self.artifacts_dir / "vocab.json"

    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"


class DatasetConfig(_Strict):
    name: str = "flickr8k"
    split_source: Literal["official", "karpathy"] = "official"
    images_url: str
    images_md5: str
    images_bytes: int
    text_url: str
    text_md5: str
    text_bytes: int
    image_dir_name: str = "Flicker8k_Dataset"
    text_dir_name: str = "Flickr8k_text"
    token_file: str = "Flickr8k.token.txt"
    split_files: dict[str, str]
    captions_per_image: int = 5


class VocabConfig(_Strict):
    min_freq: int = Field(5, ge=1)
    max_len: int = Field(25, ge=3)


class EncoderConfig(_Strict):
    name: Literal["resnet50"] = "resnet50"
    weights: str | None = "IMAGENET1K_V2"
    resize: int = 256
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


class ModelConfig(_Strict):
    kind: Literal["merge", "attention"] = "attention"
    embed_dim: int = 256
    decoder_dim: int = 512
    attention_dim: int = 512
    encoder_dim: int = 2048
    num_regions: int = 49
    dropout: float = Field(0.5, ge=0.0, lt=1.0)
    alpha_c: float = Field(1.0, ge=0.0)


class TrainConfig(_Strict):
    batch_size: int = 64
    captions_per_image: Literal[1, 5] = 1
    lr: float = 4e-4
    weight_decay: float = 0.0
    epochs: int = 30
    patience: int = 5
    lr_factor: float = 0.5
    lr_patience: int = 2
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 2
    val_beam: int = 1
    log_samples: int = 8
    max_steps_per_epoch: int | None = None
    finetune_encoder: bool = False
    encoder_lr: float = 1e-4
    init_from: str | None = None


class DecodeConfig(_Strict):
    beam: int = Field(3, ge=1)
    length_alpha: float = Field(0.7, ge=0.0)
    max_len: int = Field(25, ge=2)


class EvalConfig(_Strict):
    split: Literal["train", "val", "test"] = "test"
    beams: list[int] = Field(default_factory=lambda: [1, 3, 5])
    n_random: int = 25
    n_best_worst: int = 10
    n_attention: int = 10
    n_taxonomy: int = 100


class AppConfig(_Strict):
    max_upload_mb: float = 10
    max_image_pixels: int = 40_000_000
    max_side_px: int = 8000
    inference_timeout_s: float = 10.0
    tts_timeout_s: float = 5.0
    cooldown_s: float = 2.0
    low_confidence_logprob: float = -1.2
    max_heatmap_words: int = 12


class TTSConfig(_Strict):
    """Speech settings. Default is a local neural voice; nothing leaves the server (ADR-020)."""

    engine: Literal["piper", "offline", "gtts", "off"] = "piper"
    allow_online: bool = False
    voice_url: str = (
        "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz"
    )
    voice_sha256: str = "d45ca50084c0558eb9581cd7d26938043bc8853513da47c63b94d95a2367a5c9"
    model_file: str = "en-us-lessac-medium.onnx"
    model_sha256: str = "fa5998eb236e4291f7920d88b897225d0f2ce90ff6193480fbaa279d7814f347"
    config_file: str = "en-us-lessac-medium.onnx.json"
    config_sha256: str = "f3e0b906861cc2fb8a50e12ceca263afe226ff9688f60e9d4ef943d4f047a513"


class Config(_Strict):
    name: str = "base"
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig
    vocab: VocabConfig = Field(default_factory=VocabConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with a nested ``overrides`` dict deep-merged in."""
        merged = deep_merge(self.model_dump(mode="json"), overrides)
        return Config.model_validate(merged)


_ENV_PATHS = {
    "data_dir": "CAPTIONER_DATA_DIR",
    "artifacts_dir": "CAPTIONER_ARTIFACTS_DIR",
    "reports_dir": "CAPTIONER_REPORTS_DIR",
    "mlruns_dir": "CAPTIONER_MLRUNS_DIR",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_with_defaults(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular `defaults` in {path}")
    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    parents = raw.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = path.with_name(f"{parent}.yaml")
        merged = deep_merge(merged, _read_with_defaults(parent_path, seen | {path}))
    return deep_merge(merged, raw)


def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> Config:
    """Load and validate a YAML config, resolving ``defaults`` and env path overrides.

    Args:
        path: YAML file to load.
        overrides: Optional nested dict merged last (e.g. from tests or the CLI).

    Returns:
        The validated, frozen ``Config``.
    """
    data = _read_with_defaults(Path(path))
    if overrides:
        data = deep_merge(data, overrides)
    paths = dict(data.get("paths", {}))
    for key, env in _ENV_PATHS.items():
        if env_value := os.environ.get(env):
            paths[key] = env_value
    data["paths"] = paths
    return Config.model_validate(data)


def save_config(cfg: Config, path: Path) -> None:
    """Write the fully resolved config as YAML (the run snapshot)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict to ``a.b.c -> str(value)`` (for MLflow params)."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{full}."))
        else:
            flat[full] = str(value)
    return flat

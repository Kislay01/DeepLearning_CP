"""Typed experiment configuration loaded from YAML.

Experiment files inherit other files in the same directory through ``defaults: [base]``.
Paths can be overridden with ``CAPTIONER_*`` environment variables so nothing is hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"


class ArchiveConfig(_Strict):
    """One downloadable archive. ``md5: null`` = not pinned upstream: the MD5 seen on first
    download is recorded and enforced afterwards (trust on first use, ADR-023)."""

    url: str
    md5: str | None = None
    dest: str = ""  # sub-folder of data/raw to extract into


class KarpathySource(_Strict):
    """A dataset described by a Karpathy split JSON (``dataset_coco.json`` …)."""

    name: Literal["coco", "flickr30k", "flickr8k"]
    json_file: str  # file inside the caption_datasets archive
    image_root: str  # folder (relative to data/raw, or absolute) holding the ``filepath`` dirs
    archives: list[ArchiveConfig] = Field(default_factory=list)


class DatasetConfig(_Strict):
    name: str = "flickr8k"
    kind: Literal["flickr8k", "karpathy"] = "flickr8k"
    split_source: Literal["official", "karpathy"] = "official"
    karpathy_url: str = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
    karpathy_md5: str = "4fa8c08369d22fe16e41dc124bd1adc2"
    sources: list[KarpathySource] = Field(default_factory=list)
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
    gender_neutral: bool = False  # ADR-021: train on gender-neutral captions

    @property
    def variant(self) -> str:
        """Suffix that keeps vocabularies of different text pipelines apart."""
        parts = [] if self.min_freq == 5 else [f"mf{self.min_freq}"]
        if self.gender_neutral:
            parts.append("neutral")
        return "_".join(parts)


class EncoderConfig(_Strict):
    name: Literal["resnet50", "clip_vit_b32"] = "resnet50"
    weights: str | None = "IMAGENET1K_V2"
    resize: int = 256
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


class ModelConfig(_Strict):
    kind: Literal["merge", "attention", "transformer"] = "attention"
    embed_dim: int = 256
    decoder_dim: int = 512
    attention_dim: int = 512
    encoder_dim: int = 2048
    num_regions: int = 49
    dropout: float = Field(0.5, ge=0.0, lt=1.0)
    alpha_c: float = Field(1.0, ge=0.0)
    # M3 Transformer decoder (ADR-027); decoder_dim is the model width
    transformer_layers: int = Field(3, ge=1)
    transformer_heads: int = Field(8, ge=1)
    transformer_ff: int = Field(2048, ge=1)
    transformer_enc_layers: int = Field(1, ge=0)
    max_positions: int = Field(64, ge=2)


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
    # ADR-025: modern training recipe (defaults reproduce the blueprint)
    objective: Literal["xe", "scst"] = "xe"
    select_metric: Literal["bleu4", "cider"] = "bleu4"
    label_smoothing: float = Field(0.0, ge=0.0, lt=1.0)
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: Literal["plateau", "cosine"] = "plateau"
    warmup_steps: int = Field(0, ge=0)
    ema_decay: float = Field(0.0, ge=0.0, lt=1.0)
    scst_samples: int = Field(5, ge=2)


class DecodeConfig(_Strict):
    beam: int = Field(3, ge=1)
    length_alpha: float = Field(0.7, ge=0.0)
    max_len: int = Field(25, ge=2)
    gender_neutral_output: bool = True  # ADR-021: rewrite gendered words in served captions
    no_repeat_ngram: int = Field(3, ge=0)  # ADR-022: ban repeated n-grams (0 = off)
    block_repeated_words: bool = True  # ADR-022: each content word at most once


class EvalConfig(_Strict):
    split: Literal["train", "val", "test"] = "test"
    beams: list[int] = Field(default_factory=lambda: [1, 3, 5])
    n_random: int = 25
    n_best_worst: int = 10
    n_attention: int = 10
    n_taxonomy: int = 100
    source: str | None = None  # evaluate only one source of a combined dataset (ADR-023)
    tag: str | None = None  # suffix for the report folder (e.g. decoding ablations)
    reliability_centroids: int = Field(64, ge=1)  # ADR-022
    ood_quantile: float = Field(0.99, gt=0.0, lt=1.0)
    confidence_quantile: float = Field(0.10, gt=0.0, lt=1.0)


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

    @property
    def _dataset_tag(self) -> str:
        return "" if self.dataset.name == "flickr8k" else self.dataset.name

    @property
    def manifest_file(self) -> Path:
        """``data/processed/manifest.parquet`` (Flickr8k) or ``data/processed/<dataset>/…``."""
        tag = self._dataset_tag
        base = self.paths.processed_dir / tag if tag else self.paths.processed_dir
        return base / "manifest.parquet"

    @property
    def features_dir(self) -> Path:
        tag = self._dataset_tag
        return self.paths.features_dir / tag if tag else self.paths.features_dir

    @property
    def vocab_file(self) -> Path:
        """``artifacts/vocab.json`` or ``vocab_<dataset>_<variant>.json`` for other pipelines."""
        parts = [p for p in (self._dataset_tag, self.vocab.variant) if p]
        return self.paths.artifacts_dir / (
            f"vocab_{'_'.join(parts)}.json" if parts else "vocab.json"
        )

    @model_validator(mode="after")
    def _encoder_matches_decoder(self) -> Config:
        dims = {"resnet50": 2048, "clip_vit_b32": 768}
        if self.model.encoder_dim != dims[self.encoder.name]:
            raise ValueError(
                f"model.encoder_dim={self.model.encoder_dim} but {self.encoder.name} "
                f"outputs {dims[self.encoder.name]}"
            )
        return self

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

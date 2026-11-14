"""Checkpoints: plain state dicts + metadata, always loaded with ``weights_only=True``."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from captioner.config import Config
from captioner.data.vocab import Vocabulary
from captioner.models.captioner import build_model


class CheckpointError(RuntimeError):
    """A checkpoint is missing, unreadable or inconsistent with its run."""


@dataclass(frozen=True)
class LoadedRun:
    """A trained decoder with the config and vocabulary it was trained with."""

    model: nn.Module
    cfg: Config
    vocab: Vocabulary
    meta: dict[str, Any]
    encoder_state: dict[str, torch.Tensor] | None = None


def save_checkpoint(path: Path, model: nn.Module, meta: dict[str, Any], **states: Any) -> None:
    """Write ``{"model": state_dict, "meta": meta, **states}`` with ``torch.save``.

    Args:
        path: Destination (``best.pt`` / ``last.pt``).
        model: Decoder (or ``ImageCaptioner`` when fine-tuning).
        meta: JSON-like metadata: config, epoch, metric, git SHA, vocab hash.
        **states: Extra state dicts (optimizer, scheduler, scaler) for resuming.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "meta": meta, **states}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint safely (``weights_only=True``, CPU)."""
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    if not zipfile.is_zipfile(path):  # torch.save writes a zip container
        raise CheckpointError(f"Cannot load {path}: not a torch checkpoint archive")
    try:
        data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, OSError, EOFError) as exc:
        raise CheckpointError(f"Cannot load {path}: {exc}") from exc
    if "model" not in data or "meta" not in data:
        raise CheckpointError(f"{path} is not a captioner checkpoint")
    return data


def load_run(run_dir: Path, which: str = "best") -> LoadedRun:
    """Rebuild the decoder of a training run from ``<run_dir>/<which>.pt``.

    The vocabulary hash stored in the checkpoint must match ``<run_dir>/vocab.json``.
    """
    data = load_checkpoint(run_dir / f"{which}.pt")
    meta = data["meta"]
    cfg = Config.model_validate(meta["config"])
    vocab = Vocabulary.load(run_dir / "vocab.json")
    if vocab.sha256() != meta.get("vocab_sha256"):
        raise CheckpointError(f"{run_dir}/vocab.json does not match the checkpoint's vocabulary")
    model = build_model(cfg.model, len(vocab))
    state = data["model"]
    encoder_state = None
    if any(k.startswith("decoder.") for k in state):  # fine-tuned ImageCaptioner checkpoint
        encoder_state = {
            k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")
        }
        state = {
            k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")
        }
    model.load_state_dict(state)
    return LoadedRun(
        model=model.eval(), cfg=cfg, vocab=vocab, meta=meta, encoder_state=encoder_state
    )


def latest_run(runs_dir: Path) -> Path:
    """Most recently modified run folder that has a ``best.pt``."""
    candidates = [p.parent for p in runs_dir.glob("*/best.pt")]
    if not candidates:
        raise CheckpointError(f"No trained runs under {runs_dir} (run `captioner train`)")
    return max(candidates, key=lambda p: (p / "best.pt").stat().st_mtime)

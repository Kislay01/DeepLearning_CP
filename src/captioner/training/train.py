"""Stage `train`: fit a decoder on cached features with early stopping on val BLEU-4 (FR-6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from captioner.config import Config, flatten, save_config
from captioner.data.dataset import (
    CaptionDataset,
    CaptionTable,
    ImageCaptionDataset,
    ImageFeatureDataset,
    ImageTableDataset,
    build_caption_table,
    collate_captions,
    collate_images,
    open_feature_store,
)
from captioner.data.prepare import PreparedData, load_prepared
from captioner.data.sources import image_root
from captioner.data.transforms import build_train_augment
from captioner.features.extract import feature_paths
from captioner.inference.decode import Hypothesis, build_repeat_policy
from captioner.models.captioner import (
    ImageCaptioner,
    build_model,
    count_trainable,
    feature_kind_for,
)
from captioner.models.encoder import ResNetEncoder
from captioner.training.checkpoint import load_checkpoint, save_checkpoint
from captioner.training.early_stopping import EarlyStopping
from captioner.training.loop import (
    Amp,
    beam_decoder,
    bleu4_for,
    decode_split,
    train_epoch,
    validation_loss,
)
from captioner.training.tracking import RunTracker
from captioner.utils.device import get_device
from captioner.utils.io import git_sha, write_json
from captioner.utils.seed import seed_everything, worker_init_fn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """Summary of a finished training run."""

    run_dir: Path
    best_bleu4: float
    best_epoch: int
    epochs_run: int
    history: list[dict[str, float]]


@dataclass(frozen=True)
class _Data:
    train: DataLoader[Any]
    val: DataLoader[Any]
    val_images: DataLoader[Any]
    val_table: CaptionTable


def _loader(
    ds: Dataset[Any], cfg: Config, shuffle: bool, collate: Any, gen: torch.Generator
) -> DataLoader[Any]:
    workers = cfg.train.num_workers
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,
        generator=gen,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def build_data(cfg: Config, prepared: PreparedData, gen: torch.Generator) -> _Data:
    """Train / val loaders over cached features (or over images when fine-tuning)."""
    train_t = build_caption_table(prepared.manifest, "train", prepared.vocab)
    val_t = build_caption_table(prepared.manifest, "val", prepared.vocab)
    k = cfg.train.captions_per_image
    if cfg.train.finetune_encoder:
        image_dir = image_root(cfg)
        train_ds: Dataset[Any] = ImageCaptionDataset(train_t, image_dir, build_train_augment(), k)
        val_ds: Dataset[Any] = ImageCaptionDataset(val_t, image_dir, None, 5)
        val_img: Dataset[Any] = ImageTableDataset(val_t, image_dir)
    else:
        path = feature_paths(cfg)[feature_kind_for(cfg.model)]
        train_ds = CaptionDataset(train_t, open_feature_store(path, train_t), k)
        val_store = open_feature_store(path, val_t)
        val_ds, val_img = CaptionDataset(val_t, val_store, 5), ImageFeatureDataset(val_t, val_store)
    return _Data(
        train=_loader(train_ds, cfg, True, collate_captions, gen),
        val=_loader(val_ds, cfg, False, collate_captions, gen),
        val_images=_loader(val_img, cfg, False, collate_images, gen),
        val_table=val_t,
    )


def build_training_model(cfg: Config, vocab_size: int) -> nn.Module:
    """Decoder (optionally warm-started), wrapped with the encoder when fine-tuning."""
    decoder = build_model(cfg.model, vocab_size)
    if cfg.train.init_from:
        state = load_checkpoint(Path(cfg.train.init_from))["model"]
        if any(k.startswith("decoder.") for k in state):
            state = {
                k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")
            }
        decoder.load_state_dict(state)
        logger.info("Initialised decoder from %s", cfg.train.init_from)
    if not cfg.train.finetune_encoder:
        return decoder
    return ImageCaptioner(ResNetEncoder(cfg.encoder.weights, train_layer4=True), decoder)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Adam with a separate (smaller) learning rate for fine-tuned encoder weights."""
    if isinstance(model, ImageCaptioner):
        groups = [
            {
                "params": [p for p in model.decoder.parameters() if p.requires_grad],
                "lr": cfg.train.lr,
            },
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": cfg.train.encoder_lr,
            },
        ]
        return torch.optim.Adam(groups, weight_decay=cfg.train.weight_decay)
    return torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )


def _new_run_dir(cfg: Config) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = cfg.paths.runs_dir / f"{cfg.name}-{stamp}-s{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _meta(
    cfg: Config, prepared: PreparedData, epoch: int, bleu4: float, run_dir: Path
) -> dict[str, Any]:
    return {
        "config": cfg.model_dump(mode="json"),
        "epoch": epoch,
        "val_bleu4": bleu4,
        "git_sha": git_sha(),
        "vocab_sha256": prepared.vocab.sha256(),
        "manifest_sha256": prepared.manifest_sha256,
        "dataset_md5": f"{cfg.dataset.images_md5}+{cfg.dataset.text_md5}",
        "model_kind": cfg.model.kind,
        "vocab_size": len(prepared.vocab),
        "run_name": run_dir.name,
    }


def _samples(hyps: list[Hypothesis], table: CaptionTable, prepared: PreparedData, n: int) -> str:
    lines = []
    for i, hyp in enumerate(hyps[:n]):
        pred = " ".join(prepared.vocab.decode(hyp.ids))
        lines.append(f"{table.image_ids[i]}\n  pred: {pred}\n  ref : {table.references[i][0]}")
    return "\n".join(lines)


def _tracker(cfg: Config, prepared: PreparedData, run_dir: Path) -> RunTracker:
    tracker = RunTracker(cfg.paths.mlruns_dir, "captioner", run_dir.name, run_dir / "tb")
    tracker.log_params(flatten(cfg.model_dump(mode="json")))
    tracker.set_tags(
        {
            "git_sha": git_sha(),
            "dataset_md5": f"{cfg.dataset.images_md5}+{cfg.dataset.text_md5}",
            "vocab_sha256": prepared.vocab.sha256(),
            "model": cfg.model.kind,
        }
    )
    return tracker


def run_training(cfg: Config, device: torch.device | None = None) -> TrainResult:
    """Train with validation every epoch, ReduceLROnPlateau and early stopping on val BLEU-4.

    Writes ``best.pt``, ``last.pt``, ``config.yaml``, ``vocab.json`` and ``history.json`` to
    ``artifacts/runs/<run>/`` and logs everything to MLflow + TensorBoard.
    """
    device = device or get_device()
    gen = seed_everything(cfg.seed)
    prepared = load_prepared(cfg)
    data = build_data(cfg, prepared, gen)
    model = build_training_model(cfg, len(prepared.vocab)).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.train.lr_factor, patience=cfg.train.lr_patience
    )
    amp = Amp(device, cfg.train.amp)
    run_dir = _new_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")
    prepared.vocab.save(run_dir / "vocab.json")
    logger.info(
        "Run %s: %s, %d trainable params", run_dir.name, cfg.model.kind, count_trainable(model)
    )
    stopper = EarlyStopping(cfg.train.patience)
    with _tracker(cfg, prepared, run_dir) as tracker:
        history = _fit(
            cfg, model, data, optimizer, scheduler, amp, stopper, tracker, prepared, run_dir
        )
        for name in ("config.yaml", "vocab.json", "history.json", "best.pt"):
            tracker.log_artifact(run_dir / name)
    if not cfg.train.finetune_encoder:
        from captioner.evaluation.calibrate import calibrate_run

        calibrate_run(cfg, run_dir, device)
    best = max(history, key=lambda h: h["val/bleu4"])
    return TrainResult(run_dir, best["val/bleu4"], int(best["epoch"]), len(history), history)


def _fit(
    cfg: Config,
    model: nn.Module,
    data: _Data,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    amp: Amp,
    stopper: EarlyStopping,
    tracker: RunTracker,
    prepared: PreparedData,
    run_dir: Path,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    alpha_c = cfg.model.alpha_c if cfg.model.kind == "attention" else 0.0
    repeat = build_repeat_policy(
        prepared.vocab.itos, cfg.decode.no_repeat_ngram, cfg.decode.block_repeated_words
    )
    decoder = beam_decoder(cfg.train.val_beam, cfg.decode.max_len, cfg.decode.length_alpha, repeat)
    for epoch in range(1, cfg.train.epochs + 1):
        tr = train_epoch(
            model,
            data.train,
            optimizer,
            amp,
            cfg.train.grad_clip,
            alpha_c,
            cfg.train.max_steps_per_epoch,
        )
        va = validation_loss(model, data.val, amp, alpha_c)
        hyps = decode_split(model, data.val_images, amp.device, decoder)
        bleu4 = bleu4_for(hyps, data.val_table, prepared.vocab)
        scheduler.step(bleu4)
        improved = stopper.update(bleu4)
        row = {
            "epoch": float(epoch),
            "train/loss": tr.loss,
            "train/ce": tr.ce,
            "train/perplexity": tr.perplexity,
            "val/loss": va.loss,
            "val/perplexity": va.perplexity,
            "val/bleu4": bleu4,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, epoch)
        tracker.log_text(
            _samples(hyps, data.val_table, prepared, cfg.train.log_samples),
            f"epoch_{epoch:03d}.txt",
            epoch,
        )
        meta = _meta(cfg, prepared, epoch, bleu4, run_dir)
        states = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        save_checkpoint(run_dir / "last.pt", model, meta, **states)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, meta)
        write_json(history, run_dir / "history.json")
        logger.info(
            "epoch %d  train %.3f  val %.3f  BLEU-4 %.4f%s",
            epoch,
            tr.loss,
            va.loss,
            bleu4,
            "  *best*" if improved else "",
        )
        if stopper.should_stop:
            logger.info("Early stopping: no val BLEU-4 improvement for %d epochs", stopper.patience)
            break
    return history

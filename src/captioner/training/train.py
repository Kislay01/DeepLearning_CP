"""Stage `train`: fit a decoder on cached features with early stopping on val BLEU-4 (FR-6)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from captioner.config import Config, flatten, save_config
from captioner.data.dataset import (
    Batch,
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
from captioner.data.vocab import Vocabulary
from captioner.evaluation.cider import CiderD
from captioner.evaluation.metrics import reference_tokens
from captioner.features.extract import feature_paths
from captioner.inference.decode import Hypothesis, build_repeat_policy
from captioner.models.captioner import (
    ImageCaptioner,
    build_model,
    count_trainable,
    feature_kind_for,
)
from captioner.models.encoder import build_encoder
from captioner.training.checkpoint import load_checkpoint, save_checkpoint
from captioner.training.early_stopping import EarlyStopping
from captioner.training.loop import (
    Amp,
    beam_decoder,
    bleu4_for,
    decode_split,
    run_epoch,
    train_step,
    validation_loss,
)
from captioner.training.optim import Ema, Schedule, build_optimizer
from captioner.training.scst import scst_step
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
    train_table: CaptionTable


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
        train_table=train_t,
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
    encoder = build_encoder(cfg.encoder.name, cfg.encoder.weights, train_layer4=True)
    return ImageCaptioner(encoder, decoder)


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
    """Train with validation every epoch and early stopping on the selection metric.

    Objective ``xe`` (teacher forcing) or ``scst`` (self-critical, CIDEr-D reward, ADR-026);
    selection metric val BLEU-4 (blueprint) or val CIDEr-D (ADR-025). Writes ``best.pt``,
    ``last.pt``, ``config.yaml``, ``vocab.json`` and ``history.json`` to
    ``artifacts/runs/<run>/`` and logs everything to MLflow + TensorBoard.
    """
    device = device or get_device()
    gen = seed_everything(cfg.seed)
    prepared = load_prepared(cfg)
    data = build_data(cfg, prepared, gen)
    if cfg.train.objective == "scst" and not cfg.train.init_from:
        logger.warning(
            "SCST without train.init_from: starting from random weights is slow/unstable"
        )
    model = build_training_model(cfg, len(prepared.vocab)).to(device)
    optimizer = build_optimizer(model, cfg)
    steps = cfg.train.max_steps_per_epoch or len(data.train)
    schedule = Schedule(optimizer, cfg, total_steps=steps * cfg.train.epochs)
    ema = Ema(model, cfg.train.ema_decay)
    run_dir = _new_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")
    prepared.vocab.save(run_dir / "vocab.json")
    logger.info(
        "Run %s: %s/%s, %d trainable params",
        run_dir.name,
        cfg.model.kind,
        cfg.train.objective,
        count_trainable(model),
    )
    with _tracker(cfg, prepared, run_dir) as tracker:
        loop = _Loop(cfg, model, data, optimizer, schedule, ema, tracker, prepared, run_dir, device)
        history = loop.fit()
        for name in ("config.yaml", "vocab.json", "history.json", "best.pt"):
            tracker.log_artifact(run_dir / name)
    if not cfg.train.finetune_encoder:
        from captioner.evaluation.calibrate import calibrate_run

        calibrate_run(cfg, run_dir, device)
    key = f"val/{cfg.train.select_metric}"
    best = max(history, key=lambda h: h[key])
    return TrainResult(run_dir, best["val/bleu4"], int(best["epoch"]), len(history), history)


class _Loop:
    """One training run: step function per objective, validation, selection, checkpoints."""

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        data: _Data,
        optimizer: torch.optim.Optimizer,
        schedule: Schedule,
        ema: Ema,
        tracker: RunTracker,
        prepared: PreparedData,
        run_dir: Path,
        device: torch.device,
    ) -> None:
        self.cfg, self.model, self.data, self.optimizer = cfg, model, data, optimizer
        self.schedule, self.ema, self.tracker = schedule, ema, tracker
        self.prepared, self.run_dir, self.device = prepared, run_dir, device
        self.amp = Amp(device, cfg.train.amp)
        self.alpha_c = cfg.model.alpha_c if cfg.model.kind == "attention" else 0.0
        vocab = prepared.vocab
        d = cfg.decode
        repeat = build_repeat_policy(vocab.itos, d.no_repeat_ngram, d.block_repeated_words)
        self.decoder = beam_decoder(cfg.train.val_beam, d.max_len, d.length_alpha, repeat)
        self.val_cider = CiderD(_ref_corpus(data.val_table, vocab))
        self.train_cider = (
            CiderD(_ref_corpus(data.train_table, vocab)) if cfg.train.objective == "scst" else None
        )

    def _step(self, batch: Batch) -> dict[str, float]:
        t = self.cfg.train
        if self.train_cider is not None:
            out = scst_step(
                self.model,
                batch,
                self.optimizer,
                self.train_cider,
                self.prepared.vocab,
                t.scst_samples,
                self.cfg.decode.max_len,
                t.grad_clip,
            )
        else:
            loss = train_step(
                self.model,
                batch,
                self.optimizer,
                self.amp,
                t.grad_clip,
                self.alpha_c,
                t.label_smoothing,
            )
            out = {"loss": float(loss.total), "ce": float(loss.ce), "reg": float(loss.reg)}
        self.schedule.on_step()
        self.ema.update()
        return out

    def _validate(self) -> tuple[dict[str, float], list[Hypothesis]]:
        model = self.ema.eval_model
        va = validation_loss(model, self.data.val, self.amp, self.alpha_c)
        hyps = decode_split(model, self.data.val_images, self.device, self.decoder)
        vocab = self.prepared.vocab
        words = [vocab.decode(h.ids) for h in hyps]
        metrics = {
            "val/loss": va.loss,
            "val/perplexity": va.perplexity,
            "val/bleu4": bleu4_for(hyps, self.data.val_table, vocab),
            "val/cider": self.val_cider.corpus_score(words),
        }
        return metrics, hyps

    def fit(self) -> list[dict[str, float]]:
        cfg = self.cfg
        key = f"val/{cfg.train.select_metric}"
        stopper = EarlyStopping(cfg.train.patience)
        history: list[dict[str, float]] = []
        for epoch in range(1, cfg.train.epochs + 1):
            self.model.train()
            tr = run_epoch(self.data.train, self._step, self.device, cfg.train.max_steps_per_epoch)
            val, hyps = self._validate()
            self.schedule.on_epoch(val[key])
            improved = stopper.update(val[key])
            row = {"epoch": float(epoch), **{f"train/{k}": v for k, v in tr.items()}, **val}
            if "train/ce" in row:
                row["train/perplexity"] = math.exp(min(row["train/ce"], 50.0))
            row["lr"] = float(self.optimizer.param_groups[0]["lr"])
            history.append(row)
            self.tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, epoch)
            self.tracker.log_text(
                _samples(hyps, self.data.val_table, self.prepared, cfg.train.log_samples),
                f"epoch_{epoch:03d}.txt",
                epoch,
            )
            self._checkpoint(epoch, val, improved)
            write_json(history, self.run_dir / "history.json")
            logger.info(
                "epoch %d  %s  BLEU-4 %.4f  CIDEr %.4f%s",
                epoch,
                "  ".join(f"{k} {v:.3f}" for k, v in tr.items()),
                val["val/bleu4"],
                val["val/cider"],
                "  *best*" if improved else "",
            )
            if stopper.should_stop:
                logger.info(
                    "Early stopping: no %s improvement for %d epochs", key, stopper.patience
                )
                break
        return history

    def _checkpoint(self, epoch: int, val: dict[str, float], improved: bool) -> None:
        meta = _meta(self.cfg, self.prepared, epoch, val["val/bleu4"], self.run_dir)
        meta["val_cider"] = val["val/cider"]
        model = self.ema.eval_model
        states = {"optimizer": self.optimizer.state_dict(), "scheduler": self.schedule.state_dict()}
        save_checkpoint(self.run_dir / "last.pt", model, meta, **states)
        if improved:
            save_checkpoint(self.run_dir / "best.pt", model, meta)


def _ref_corpus(table: CaptionTable, vocab: Vocabulary) -> list[list[list[str]]]:
    """Tokenised references per image, with the vocabulary's text transform applied."""
    return [[vocab.prepare_tokens(r) for r in reference_tokens(refs)] for refs in table.references]

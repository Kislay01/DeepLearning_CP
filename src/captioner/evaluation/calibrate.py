"""Stage `calibrate`: fit the reliability model (OOD + confidence thresholds) for a run."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from captioner.config import Config
from captioner.data.dataset import (
    ImageFeatureDataset,
    build_caption_table,
    collate_images,
    open_feature_store,
)
from captioner.data.prepare import load_prepared
from captioner.features.extract import feature_paths
from captioner.inference.decode import build_repeat_policy
from captioner.inference.reliability import Reliability, fit_reliability
from captioner.models.captioner import feature_kind_for
from captioner.training.checkpoint import load_run
from captioner.training.loop import beam_decoder, decode_split
from captioner.utils.device import get_device

logger = logging.getLogger(__name__)

RELIABILITY_FILE = "reliability.safetensors"
_MAX_TRAIN_POINTS = 20_000


def _pooled_rows(path: Path, rows: np.ndarray) -> torch.Tensor:
    order = np.argsort(rows)
    with h5py.File(path, "r") as fh:
        data = fh["features"][rows[order]].astype(np.float32)  # h5py needs increasing indices
    out = np.empty_like(data)
    out[order] = data
    return torch.from_numpy(out)


def calibrate_run(
    cfg: Config, run_dir: Path, device: torch.device | None = None
) -> Reliability | None:
    """Fit and save ``<run_dir>/reliability.safetensors``; ``None`` for fine-tuned encoders.

    Cached features come from the frozen encoder, so they don't describe a fine-tuned one.
    """
    device = device or get_device()
    run = load_run(run_dir)
    if run.encoder_state:
        logger.warning("Skipping calibration: fine-tuned encoder does not match cached features")
        return None
    prepared = load_prepared(cfg)
    train_t = build_caption_table(prepared.manifest, "train", run.vocab)
    val_t = build_caption_table(prepared.manifest, "val", run.vocab)
    paths = feature_paths(cfg)
    rng = np.random.default_rng(cfg.seed)
    train_rows = train_t.feature_idx
    if len(train_rows) > _MAX_TRAIN_POINTS:
        train_rows = rng.choice(train_rows, _MAX_TRAIN_POINTS, replace=False)
    train_pooled = _pooled_rows(paths["pooled"], np.asarray(train_rows))
    val_pooled = _pooled_rows(paths["pooled"], val_t.feature_idx)
    store = open_feature_store(paths[feature_kind_for(run.cfg.model)], val_t)
    loader = DataLoader(
        ImageFeatureDataset(val_t, store),
        batch_size=cfg.train.batch_size,
        collate_fn=collate_images,
    )
    d = run.cfg.decode
    repeat = build_repeat_policy(run.vocab.itos, d.no_repeat_ngram, d.block_repeated_words)
    hyps = decode_split(
        run.model.to(device),
        loader,
        device,
        beam_decoder(d.beam, d.max_len, d.length_alpha, repeat),
    )
    e = cfg.eval
    reliability = fit_reliability(
        train_pooled,
        val_pooled,
        torch.tensor([h.mean_logprob for h in hyps]),
        k=e.reliability_centroids,
        ood_quantile=e.ood_quantile,
        confidence_quantile=e.confidence_quantile,
        seed=cfg.seed,
    )
    reliability.save(run_dir / RELIABILITY_FILE)
    logger.info(
        "Reliability: OOD threshold %.4f, confidence threshold %.3f",
        reliability.ood_threshold,
        reliability.confidence_threshold,
    )
    return reliability

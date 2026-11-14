"""Stage `evaluate`: decode a split with several beam widths and write reports (FR-7, D1).

Outputs in ``reports/<run>/``: ``metrics.json`` (every number in the README must come from one
of these), ``predictions.csv`` (database.md §3.5) and ``samples.html``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from captioner.config import Config
from captioner.data.dataset import (
    CaptionTable,
    ImageFeatureDataset,
    ImageTableDataset,
    build_caption_table,
    collate_images,
    open_feature_store,
)
from captioner.data.download import raw_layout
from captioner.data.prepare import load_prepared
from captioner.data.text import detokenize
from captioner.evaluation.metrics import METRIC_KEYS, CocoResult, coco_scores
from captioner.evaluation.report import write_samples_html
from captioner.evaluation.taxonomy import diversity, taxonomy_counts
from captioner.features.extract import feature_paths
from captioner.inference.decode import Hypothesis
from captioner.models.captioner import ImageCaptioner, feature_kind_for
from captioner.models.encoder import ResNetEncoder
from captioner.training.checkpoint import CheckpointError, LoadedRun, load_run
from captioner.training.loop import beam_decoder, decode_split
from captioner.utils.device import get_device
from captioner.utils.io import git_sha, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    """Where the reports went and the per-beam corpus scores."""

    report_dir: Path
    metrics: dict[str, Any]


def _model_and_data(
    cfg: Config, run: LoadedRun, table: CaptionTable
) -> tuple[nn.Module, Dataset[tuple[torch.Tensor, int]]]:
    if run.encoder_state:  # fine-tuned encoder → evaluate from images
        encoder = ResNetEncoder(run.cfg.encoder.weights)
        encoder.load_state_dict(run.encoder_state)
        return ImageCaptioner(encoder, run.model), ImageTableDataset(
            table, raw_layout(cfg).image_dir
        )
    path = feature_paths(cfg)[feature_kind_for(run.cfg.model)]
    return run.model, ImageFeatureDataset(table, open_feature_store(path, table))


def _texts(hyps: list[Hypothesis], run: LoadedRun) -> list[str]:
    return [" ".join(run.vocab.decode(h.ids)) for h in hyps]


def _rows(
    table: CaptionTable, split: str, beam: int, texts: list[str], res: CocoResult
) -> list[dict[str, Any]]:
    rows = []
    for i, image_id in enumerate(table.image_ids):
        row: dict[str, Any] = {
            "image_id": image_id,
            "split": split,
            "beam": beam,
            "prediction": texts[i],
        }
        row.update({f"ref_{j}": r for j, r in enumerate(table.references[i])})
        row.update(
            {"bleu4_sentence": res.bleu4_per_image[i], "cider_sentence": res.cider_per_image[i]}
        )
        rows.append(row)
    return rows


def evaluate_run(
    cfg: Config,
    run_dir: Path,
    split: str | None = None,
    beams: list[int] | None = None,
    device: torch.device | None = None,
) -> EvalResult:
    """Evaluate a trained run.

    Args:
        cfg: Current config (data/feature/report paths, eval settings).
        run_dir: Training run folder with ``best.pt`` and ``vocab.json``.
        split: Split to decode (default ``cfg.eval.split``; the test split is meant to be
            touched once per final model).
        beams: Beam widths (default ``cfg.eval.beams``).
        device: Compute device.

    Returns:
        ``EvalResult`` with the report folder and the metrics dict written to ``metrics.json``.
    """
    device = device or get_device()
    split = split or cfg.eval.split
    beams = beams or list(cfg.eval.beams)
    run = load_run(run_dir)
    prepared = load_prepared(cfg)
    if prepared.vocab.sha256() != run.vocab.sha256():
        raise CheckpointError(
            "The run's vocabulary differs from artifacts/vocab.json; re-run prepare"
        )
    table = build_caption_table(prepared.manifest, split, run.vocab)
    model, ds = _model_and_data(cfg, run, table)
    model = model.to(device).eval()
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=collate_images)
    refs = [list(r) for r in table.references]
    report_dir = cfg.paths.reports_dir / run_dir.name
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    main_beam = cfg.decode.beam if cfg.decode.beam in beams else beams[-1]
    main: tuple[list[Hypothesis], list[str], CocoResult] | None = None
    for beam in beams:
        dec = beam_decoder(beam, run.cfg.decode.max_len, run.cfg.decode.length_alpha)
        hyps = decode_split(model, loader, device, dec)
        texts = _texts(hyps, run)
        res = coco_scores(texts, refs)
        n_tax = min(cfg.eval.n_taxonomy, len(texts))
        results[f"beam_{beam}"] = {
            **res.corpus,
            "diversity": diversity(texts),
            "failure_taxonomy": taxonomy_counts(texts[:n_tax], refs[:n_tax]),
            "failure_taxonomy_n_images": n_tax,
        }
        rows += _rows(table, split, beam, texts, res)
        logger.info("beam %d: %s", beam, {k: res.corpus[k] for k in METRIC_KEYS})
        if beam == main_beam:
            main = (hyps, texts, res)
    metrics = {
        "run": run_dir.name,
        "split": split,
        "n_images": len(table),
        "model_kind": run.cfg.model.kind,
        "checkpoint_epoch": run.meta.get("epoch"),
        "val_bleu4_nltk": run.meta.get("val_bleu4"),
        "java_available": bool(main and main[2].java),
        "tokenizer": main[2].tokenizer if main else None,
        "metric_keys": list(METRIC_KEYS),
        "results": results,
        "failure_taxonomy_method": "keyword heuristics (captioner.evaluation.taxonomy)",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "training_git_sha": run.meta.get("git_sha"),
    }
    write_json(metrics, report_dir / "metrics.json")
    pd.DataFrame(rows).to_csv(report_dir / "predictions.csv", index=False)
    if main is not None:
        hyps, texts, res = main
        write_samples_html(
            report_dir / "samples.html",
            cfg=cfg,
            table=table,
            texts=[detokenize(t.split()) for t in texts],
            hyps=hyps,
            vocab=run.vocab,
            result=res,
            metrics=metrics,
            beam=main_beam,
        )
    return EvalResult(report_dir, metrics)

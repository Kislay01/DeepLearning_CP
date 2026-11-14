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
from captioner.data.neutral import neutralize_aligned, neutralize_text
from captioner.data.prepare import load_prepared
from captioner.data.sources import image_root
from captioner.data.text import clean_caption, detokenize, tokenize
from captioner.evaluation.fairness import gender_metrics
from captioner.evaluation.metrics import METRIC_KEYS, CocoResult, coco_scores
from captioner.evaluation.report import write_samples_html
from captioner.evaluation.taxonomy import diversity, taxonomy_counts
from captioner.features.extract import feature_paths
from captioner.inference.decode import Hypothesis, build_repeat_policy
from captioner.models.captioner import ImageCaptioner, feature_kind_for
from captioner.models.encoder import build_encoder
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
        encoder = build_encoder(run.cfg.encoder.name, run.cfg.encoder.weights)
        encoder.load_state_dict(run.encoder_state)
        return ImageCaptioner(encoder, run.model), ImageTableDataset(table, image_root(cfg))
    path = feature_paths(cfg)[feature_kind_for(run.cfg.model)]
    return run.model, ImageFeatureDataset(table, open_feature_store(path, table))


def _words(
    hyps: list[Hypothesis], run: LoadedRun, neutral_output: bool
) -> tuple[list[list[str]], list[str]]:
    """Raw decoded words (what the model learned) and the served text (after ADR-021 rewrite)."""
    raw = [run.vocab.decode(h.ids) for h in hyps]
    served = [neutralize_aligned(w) if neutral_output else w for w in raw]
    return raw, [" ".join(w) for w in served]


def _neutral_refs(refs: list[list[str]]) -> list[list[str]]:
    return [[neutralize_text(" ".join(tokenize(clean_caption(r)))) for r in rs] for rs in refs]


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
    manifest = prepared.manifest
    if cfg.eval.source:
        manifest = manifest[manifest["image_id"].str.startswith(f"{cfg.eval.source}_")]
    table = build_caption_table(manifest, split, run.vocab)
    model, ds = _model_and_data(cfg, run, table)
    model = model.to(device).eval()
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=collate_images)
    refs = [list(r) for r in table.references]
    neutral_eval = run.vocab.text_transform == "gender_neutral" or cfg.decode.gender_neutral_output
    score_refs = _neutral_refs(refs) if neutral_eval else refs
    suffix = "".join(f"-{s}" for s in (cfg.eval.source, cfg.eval.tag) if s)
    report_dir = cfg.paths.reports_dir / f"{run_dir.name}{suffix}"
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    main_beam = cfg.decode.beam if cfg.decode.beam in beams else beams[-1]
    main: tuple[list[Hypothesis], list[str], CocoResult] | None = None
    d = cfg.decode  # decoding is an inference-time choice: taken from the current config
    repeat = build_repeat_policy(run.vocab.itos, d.no_repeat_ngram, d.block_repeated_words)
    for beam in beams:
        dec = beam_decoder(beam, d.max_len, d.length_alpha, repeat)
        hyps = decode_split(model, loader, device, dec)
        raw_words, texts = _words(hyps, run, d.gender_neutral_output)
        res = coco_scores(texts, score_refs)
        raw_ref_scores = coco_scores(texts, refs).corpus if neutral_eval else res.corpus
        n_tax = min(cfg.eval.n_taxonomy, len(texts))
        results[f"beam_{beam}"] = {
            **res.corpus,
            "raw_references": raw_ref_scores,
            "fairness_model": gender_metrics(raw_words, refs),
            "fairness_served": gender_metrics([t.split() for t in texts], refs),
            "diversity": diversity(texts),
            "failure_taxonomy": taxonomy_counts(texts[:n_tax], score_refs[:n_tax]),
            "failure_taxonomy_n_images": n_tax,
        }
        rows += _rows(table, split, beam, texts, res)
        logger.info("beam %d: %s", beam, {k: res.corpus[k] for k in METRIC_KEYS})
        if beam == main_beam:
            main = (hyps, texts, res)
    metrics = {
        "run": run_dir.name,
        "split": split,
        "source": cfg.eval.source or cfg.dataset.name,
        "n_images": len(table),
        "model_kind": run.cfg.model.kind,
        "checkpoint_epoch": run.meta.get("epoch"),
        "val_bleu4_nltk": run.meta.get("val_bleu4"),
        "java_available": bool(main and main[2].java),
        "tokenizer": main[2].tokenizer if main else None,
        "metric_keys": list(METRIC_KEYS),
        "references": "gender_neutral" if neutral_eval else "raw",
        "text_transform": run.vocab.text_transform,
        "results": results,
        "failure_taxonomy_method": "keyword heuristics (captioner.evaluation.taxonomy)",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "training_git_sha": run.meta.get("git_sha"),
    }
    if main is not None and (run_dir / "reliability.safetensors").exists():
        metrics["reliability"] = _reliability_check(cfg, run_dir, table, main[0], main[2])
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


def _reliability_check(
    cfg: Config, run_dir: Path, table: CaptionTable, hyps: list[Hypothesis], res: CocoResult
) -> dict[str, float]:
    """Do flagged captions really score worse? (validates the ADR-022 calibration on test)."""
    from captioner.evaluation.calibrate import _pooled_rows
    from captioner.inference.reliability import Reliability

    rel = Reliability.load(run_dir / "reliability.safetensors")
    pooled = _pooled_rows(feature_paths(cfg)["pooled"], table.feature_idx)
    ood = rel.is_out_of_domain(pooled).tolist()
    low = [h.mean_logprob < rel.confidence_threshold for h in hyps]
    flagged = [a or b for a, b in zip(ood, low, strict=True)]
    cider = res.cider_per_image

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "flagged_rate": sum(flagged) / len(flagged),
        "out_of_domain_rate": sum(ood) / len(ood),
        "low_confidence_rate": sum(low) / len(low),
        "cider_flagged": mean([c for c, f in zip(cider, flagged, strict=True) if f]),
        "cider_unflagged": mean([c for c, f in zip(cider, flagged, strict=True) if not f]),
    }

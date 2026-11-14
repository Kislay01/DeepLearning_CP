"""``samples.html``: qualitative report (best/worst/random, attention maps, taxonomy, diversity).

Every piece of text is HTML-escaped; images are embedded as base64 so the file is self-contained.
"""

from __future__ import annotations

import base64
import html
import io
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from captioner.config import Config
from captioner.data.dataset import CaptionTable
from captioner.data.sources import image_root
from captioner.data.vocab import Vocabulary
from captioner.evaluation.metrics import METRIC_KEYS, CocoResult
from captioner.inference.decode import Hypothesis
from captioner.inference.heatmap import attention_grid, attention_summary, model_view, overlay

_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;color:#0F172A}"
    "table{border-collapse:collapse}td,th{border:1px solid #CBD5E1;padding:4px 8px;text-align:left}"
    ".card{display:flex;gap:16px;margin:12px 0;padding:12px;background:#F1F5F9;border-radius:12px}"
    ".pred{font-size:1.15em;font-weight:600}.meta{color:#475569;font-size:.9em}"
)


def _img_tag(img: Image.Image, alt: str, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=80)
    data = base64.b64encode(buf.getvalue()).decode()
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f'<img src="data:image/{mime};base64,{data}" alt="{html.escape(alt)}" width="224">'


def _thumb(image_dir: Path, file_name: str, alt: str) -> str:
    path = image_dir / file_name
    if not path.exists():
        return "<em>image not available</em>"
    with Image.open(path) as img:
        return _img_tag(model_view(img), alt)


def _card(image_dir: Path, table: CaptionTable, i: int, text: str, res: CocoResult) -> str:
    refs = "".join(f"<li>{html.escape(r)}</li>" for r in table.references[i])
    return (
        f'<div class="card">{_thumb(image_dir, table.file_names[i], text)}<div>'
        f'<div class="pred">{html.escape(text)}</div>'
        f'<div class="meta">{html.escape(table.image_ids[i])} · CIDEr {res.cider_per_image[i]:.3f}'
        f" · BLEU-4 {res.bleu4_per_image[i]:.3f}</div><ol>{refs}</ol></div></div>"
    )


def _attention_figure(
    image_dir: Path, image_id: str, file_name: str, words: list[str], hyp: Hypothesis, side: int
) -> str:
    if hyp.attention is None or not words:
        return ""
    grids = [attention_grid(a, side) for a in hyp.attention][:12]
    with Image.open(image_dir / file_name) as img:
        view = img.convert("RGB")
        cols = min(6, len(grids))
        rows = (len(grids) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, word, grid in zip(axes.flat, words, grids, strict=False):
            ax.imshow(overlay(view, grid))
            ax.set_title(word, fontsize=10)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=72)
    plt.close(fig)
    summary = attention_summary(words, grids)
    data = base64.b64encode(buf.getvalue()).decode()
    return (
        f"<h3>{html.escape(image_id)}</h3><img src='data:image/png;base64,{data}' "
        f"alt='Attention maps: {html.escape(summary)}'><p class='meta'>{html.escape(summary)}</p>"
    )


def _table(header: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def write_samples_html(
    path: Path,
    *,
    cfg: Config,
    table: CaptionTable,
    texts: list[str],
    hyps: list[Hypothesis],
    vocab: Vocabulary,
    result: CocoResult,
    metrics: dict[str, Any],
    beam: int,
) -> None:
    """Write the qualitative report for one beam width."""
    image_dir = image_root(cfg)
    order = sorted(range(len(texts)), key=lambda i: result.cider_per_image[i])
    rng = random.Random(cfg.seed)
    picks = rng.sample(range(len(texts)), min(cfg.eval.n_random, len(texts)))
    k = cfg.eval.n_best_worst
    beam_res = metrics["results"][f"beam_{beam}"]
    scores = _table(
        ["beam", *METRIC_KEYS],
        [
            [b.removeprefix("beam_"), *[_fmt(r[m]) for m in METRIC_KEYS]]
            for b, r in metrics["results"].items()
        ],
    )
    tax = _table(["category", "count"], list(beam_res["failure_taxonomy"].items()))
    div = _table(["statistic", "value"], [[k2, _fmt(v)] for k2, v in beam_res["diversity"].items()])
    side = round(cfg.model.num_regions**0.5)
    att = "".join(
        _attention_figure(
            image_dir,
            table.image_ids[i],
            table.file_names[i],
            vocab.decode(hyps[i].ids),
            hyps[i],
            side,
        )
        for i in picks[: cfg.eval.n_attention]
    )
    sections = [
        f"<h1>Samples — {html.escape(metrics['run'])} "
        f"({html.escape(metrics['split'])}, beam {beam})</h1>",
        "<h2>Scores</h2>",
        scores,
        f"<h2>Failure taxonomy (first {beam_res['failure_taxonomy_n_images']} images, "
        "keyword heuristics)</h2>",
        tax,
        "<h2>Diversity</h2>",
        div,
        f"<h2>{len(picks)} random images</h2>",
        *[_card(image_dir, table, i, texts[i], result) for i in picks],
        f"<h2>{k} best by CIDEr</h2>",
        *[_card(image_dir, table, i, texts[i], result) for i in order[::-1][:k]],
        f"<h2>{k} worst by CIDEr</h2>",
        *[_card(image_dir, table, i, texts[i], result) for i in order[:k]],
        "<h2>Attention maps</h2>",
        att or "<p>Model has no attention (M1 merge baseline).</p>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Samples</title>"
        f"<style>{_CSS}</style></head><body>{''.join(sections)}</body></html>",
        encoding="utf-8",
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)

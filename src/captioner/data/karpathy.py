"""Larger, more diverse training data via Karpathy split files: MS-COCO, Flickr30k (ADR-023).

Flickr8k is 8k photos dominated by people and dogs. COCO 2014 (123k images of 80 everyday
object categories, indoor and outdoor) is the standard way to broaden what a captioner knows.
Every source is turned into the same manifest schema as Flickr8k (database.md §3.1), so features,
training, evaluation and serving are unchanged. Several sources can be combined.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from captioner.data.manifest import MANIFEST_COLUMNS, ManifestError, _probe_all, _typed_frame
from captioner.data.text import clean_caption, tokenize

logger = logging.getLogger(__name__)

SPLIT_MAP = {"train": "train", "restval": "train", "val": "val", "test": "test"}


@dataclass(frozen=True)
class SourceFiles:
    """Resolved inputs of one Karpathy source."""

    name: str
    json_path: Path
    image_root: Path


def _records(
    src: SourceFiles, raw_dir: Path, n_captions: int
) -> tuple[list[dict[str, object]], int]:
    data = json.loads(src.json_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    dropped = 0
    for img in data["images"]:
        caps = [s["raw"].strip() for s in img["sentences"] if clean_caption(s["raw"])]
        split = SPLIT_MAP.get(img["split"])
        if split is None or len(caps) < n_captions:
            dropped += 1
            continue
        path = src.image_root / img.get("filepath", "") / img["filename"]
        try:
            file_name = str(path.relative_to(raw_dir))
        except ValueError:
            file_name = str(path)
        image_id = f"{src.name}_{Path(img['filename']).stem}"
        for idx, raw in enumerate(caps[:n_captions]):
            clean = clean_caption(raw)
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "split": split,
                    "caption_idx": idx,
                    "caption_raw": raw,
                    "caption_clean": clean,
                    "n_tokens": len(tokenize(clean)),
                }
            )
    return rows, dropped


def build_karpathy_manifest(
    sources: Sequence[SourceFiles], raw_dir: Path, n_captions: int = 5, validate_images: bool = True
) -> pd.DataFrame:
    """Combine Karpathy-format sources into one validated manifest.

    ``restval`` counts as train (the standard 113k COCO training set). Images with fewer than
    ``n_captions`` non-empty captions are dropped; extra captions beyond ``n_captions`` are
    ignored so every image has exactly ``n_captions`` (COCO has 5–7). Image ids are prefixed with
    the source name, so sources can be mixed.

    Raises:
        ManifestError: If an image file is missing/unreadable or ids collide.
    """
    rows: list[dict[str, object]] = []
    for src in sources:
        recs, dropped = _records(src, raw_dir, n_captions)
        logger.info("%s: %d captions kept, %d images dropped", src.name, len(recs), dropped)
        rows += recs
    if not rows:
        raise ManifestError("No usable images in the Karpathy sources")
    df = pd.DataFrame(rows)
    images = df.drop_duplicates("image_id")[["image_id", "file_name"]].sort_values("image_id")
    if images["file_name"].duplicated().any():
        raise ManifestError("Two image ids point to the same file")
    paths = [Path(f) if Path(f).is_absolute() else raw_dir / f for f in images["file_name"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise ManifestError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    if validate_images:
        sizes = _probe_all(Path("/"), [str(p.resolve()) for p in paths])
        size_of = dict(
            zip(images["image_id"], (sizes[str(p.resolve())] for p in paths), strict=True)
        )
    else:
        size_of = {i: (0, 0) for i in images["image_id"]}
    index = {image_id: i for i, image_id in enumerate(images["image_id"])}
    df["width"] = df["image_id"].map(lambda i: size_of[i][0])
    df["height"] = df["image_id"].map(lambda i: size_of[i][1])
    df["img_feature_idx"] = df["image_id"].map(index)
    return _typed_frame(df[list(MANIFEST_COLUMNS)])

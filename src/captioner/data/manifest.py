"""Validated caption manifest: one row per (split image, caption) — database.md §3.1.

Handles the Flickr8k quirks: the orphan caption id ``2258277193_586949ec62.jpg.1`` that has no
image (reference D8) and the 91 images that belong to no split.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from captioner.data.text import clean_caption, tokenize
from captioner.utils.io import bytes_digest

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test")
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "file_name",
    "split",
    "caption_idx",
    "caption_raw",
    "caption_clean",
    "n_tokens",
    "width",
    "height",
    "img_feature_idx",
)


class ManifestError(ValueError):
    """The raw data or a manifest violates the documented invariants."""


def parse_token_file(path: Path) -> dict[str, dict[int, str]]:
    """Parse ``Flickr8k.token.txt`` (``<file>#<n>\\t<caption>`` per line).

    Returns:
        ``{file_name: {caption_idx: raw_caption}}``. Keys are exactly as written in the file,
        so the orphan ``…jpg.1`` key is kept here and dropped later.
    """
    captions: dict[str, dict[int, str]] = defaultdict(dict)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            key, caption = line.split("\t", 1)
            name, idx = key.rsplit("#", 1)
            captions[name][int(idx)] = caption.strip()
        except ValueError as exc:
            raise ManifestError(f"{path.name}:{lineno}: malformed line {line!r}") from exc
    return dict(captions)


def read_split_file(path: Path) -> list[str]:
    """Read a split list (one image file name per line)."""
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def list_images(image_dir: Path) -> list[str]:
    """Sorted JPEG file names in the image folder (macOS ``._*`` files excluded)."""
    if not image_dir.is_dir():
        raise ManifestError(f"Image folder not found: {image_dir}")
    return sorted(
        p.name
        for p in image_dir.iterdir()
        if p.suffix.lower() == ".jpg" and not p.name.startswith(".")
    )


def probe_image(path: Path) -> tuple[int, int]:
    """Check an image decodes and converts to RGB; return ``(width, height)``."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return rgb.width, rgb.height
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise ManifestError(f"Unreadable image {path.name}: {exc}") from exc


def _probe_all(image_dir: Path, names: list[str]) -> dict[str, tuple[int, int]]:
    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = list(pool.map(lambda n: probe_image(image_dir / n), names))
    return dict(zip(names, sizes, strict=True))


def _split_rows(
    split: str,
    names: list[str],
    captions: Mapping[str, Mapping[int, str]],
    sizes: Mapping[str, tuple[int, int]],
    feature_idx: Mapping[str, int],
    n_captions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in names:
        caps = captions.get(name, {})
        if sorted(caps) != list(range(n_captions)):
            raise ManifestError(
                f"{name}: expected captions 0..{n_captions - 1}, got {sorted(caps)}"
            )
        width, height = sizes[name]
        for idx in range(n_captions):
            clean = clean_caption(caps[idx])
            if not clean:
                raise ManifestError(f"{name}#{idx}: caption is empty after cleaning")
            rows.append(
                {
                    "image_id": name.removesuffix(".jpg"),
                    "file_name": name,
                    "split": split,
                    "caption_idx": idx,
                    "caption_raw": caps[idx],
                    "caption_clean": clean,
                    "n_tokens": len(tokenize(clean)),
                    "width": width,
                    "height": height,
                    "img_feature_idx": feature_idx[name],
                }
            )
    return rows


def build_manifest(
    image_dir: Path,
    token_file: Path,
    split_files: Mapping[str, Path],
    n_captions: int = 5,
) -> pd.DataFrame:
    """Build the validated manifest.

    Args:
        image_dir: Folder with the extracted JPEGs (``Flicker8k_Dataset``).
        token_file: ``Flickr8k.token.txt``.
        split_files: ``{"train"|"val"|"test": path}`` official split lists.
        n_captions: Captions required per image.

    Returns:
        DataFrame with ``MANIFEST_COLUMNS``; ``img_feature_idx`` is the position of the image in
        the sorted list of all images in ``image_dir`` (ADR-006).

    Raises:
        ManifestError: On missing/unreadable split images, wrong caption counts, overlapping
            splits or empty cleaned captions.
    """
    captions = parse_token_file(token_file)
    images = list_images(image_dir)
    feature_idx = {name: i for i, name in enumerate(images)}
    orphans = sorted(set(captions) - set(feature_idx))
    if orphans:
        logger.info("Dropping %d caption ids without an image: %s", len(orphans), orphans)
    splits = {s: read_split_file(split_files[s]) for s in SPLITS}
    split_names = [n for s in SPLITS for n in splits[s]]
    if len(split_names) != len(set(split_names)):
        raise ManifestError("Split lists overlap or contain duplicates")
    missing = sorted(set(split_names) - set(feature_idx))
    if missing:
        raise ManifestError(f"{len(missing)} split images missing from {image_dir}: {missing[:5]}")
    unsplit = len(images) - len(split_names)
    logger.info("%d images are in no split and are ignored", unsplit)
    sizes = _probe_all(image_dir, split_names)
    rows = [
        row
        for s in SPLITS
        for row in _split_rows(s, splits[s], captions, sizes, feature_idx, n_captions)
    ]
    return _typed_frame(pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)))


def _typed_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[list(MANIFEST_COLUMNS)].astype(
        {
            "image_id": "string",
            "file_name": "string",
            "split": pd.CategoricalDtype(list(SPLITS)),
            "caption_idx": "int8",
            "caption_raw": "string",
            "caption_clean": "string",
            "n_tokens": "int16",
            "width": "int32",
            "height": "int32",
            "img_feature_idx": "int32",
        }
    )


def manifest_sha256(df: pd.DataFrame) -> str:
    """Content hash of a manifest (independent of the Parquet writer version)."""
    return bytes_digest(df.to_csv(index=False).encode("utf-8"))


def save_manifest(df: pd.DataFrame, path: Path) -> str:
    """Write ``manifest.parquet`` and return its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return manifest_sha256(df)


def load_manifest(path: Path) -> pd.DataFrame:
    """Read ``manifest.parquet`` with the documented dtypes."""
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path} (run `captioner prepare`)")
    return _typed_frame(pd.read_parquet(path))


def check_invariants(
    df: pd.DataFrame,
    expected_images: Mapping[str, int] | None = None,
    n_captions: int = 5,
) -> None:
    """Assert the manifest invariants of database.md §3.1.

    Args:
        df: Manifest.
        expected_images: Optional ``{split: n_images}``; full Flickr8k is 6000/1000/1000.
        n_captions: Captions per image.

    Raises:
        ManifestError: Listing every violated invariant.
    """
    problems: list[str] = []
    if tuple(df.columns) != MANIFEST_COLUMNS:
        problems.append(f"columns {list(df.columns)} != {list(MANIFEST_COLUMNS)}")
    per_split = df.groupby("split", observed=True)["image_id"].unique()
    seen: set[str] = set()
    for split, ids in per_split.items():
        if seen & set(ids):
            problems.append(f"split {split} overlaps another split")
        seen |= set(ids)
        if expected_images and len(ids) != expected_images.get(str(split)):
            problems.append(f"{split}: {len(ids)} images, expected {expected_images[str(split)]}")
    counts = df.groupby("image_id")["caption_idx"].nunique()
    if (counts != n_captions).any():
        problems.append(f"{int((counts != n_captions).sum())} images without {n_captions} captions")
    if len(df) != len(seen) * n_captions:
        problems.append(f"{len(df)} rows != {len(seen)} images x {n_captions}")
    idx = df.groupby("image_id")["img_feature_idx"].nunique()
    if (idx != 1).any() or df.drop_duplicates("image_id")["img_feature_idx"].duplicated().any():
        problems.append("image_id <-> img_feature_idx is not one-to-one")
    if (df["caption_clean"].str.len() == 0).any():
        problems.append("empty cleaned captions")
    if df["file_name"].str.endswith(".jpg.1").any():
        problems.append("orphan caption id present")
    if problems:
        raise ManifestError("; ".join(problems))

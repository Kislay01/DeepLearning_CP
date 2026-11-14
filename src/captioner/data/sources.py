"""Where a dataset's images live, independent of the dataset kind (Flickr8k / Karpathy)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from captioner.config import Config
from captioner.data.download import raw_layout
from captioner.data.karpathy import SourceFiles
from captioner.data.manifest import list_images

KARPATHY_DIR = "karpathy"


def image_root(cfg: Config) -> Path:
    """Folder that manifest ``file_name`` values are relative to."""
    return raw_layout(cfg).image_dir if cfg.dataset.kind == "flickr8k" else cfg.paths.raw_dir


def karpathy_sources(cfg: Config) -> list[SourceFiles]:
    """Resolve the configured Karpathy sources to concrete files."""
    raw = cfg.paths.raw_dir
    out = []
    for src in cfg.dataset.sources:
        root = Path(src.image_root)
        out.append(
            SourceFiles(
                name=src.name,
                json_path=raw / KARPATHY_DIR / src.json_file,
                image_root=root if root.is_absolute() else raw / root,
            )
        )
    return out


def feature_entries(cfg: Config, manifest: pd.DataFrame | None = None) -> list[tuple[str, Path]]:
    """``(image_id, path)`` in feature-row order.

    Flickr8k: every image in the archive, sorted (ADR-006). Karpathy datasets: every image in
    the manifest, ordered by ``img_feature_idx``.
    """
    if cfg.dataset.kind == "flickr8k":
        image_dir = raw_layout(cfg).image_dir
        return [(n.removesuffix(".jpg"), image_dir / n) for n in list_images(image_dir)]
    if manifest is None:
        from captioner.data.manifest import load_manifest

        manifest = load_manifest(cfg.manifest_file)
    root = image_root(cfg)
    images = manifest.drop_duplicates("image_id").sort_values("img_feature_idx")
    return [
        (str(i), Path(f) if Path(f).is_absolute() else root / str(f))
        for i, f in zip(images["image_id"], images["file_name"], strict=True)
    ]

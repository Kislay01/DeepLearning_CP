"""Stage `extract`: frozen ResNet-50 features for every image → HDF5 (database.md §3.3)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from captioner.config import Config
from captioner.data.sources import feature_entries
from captioner.data.transforms import load_image, transform_for
from captioner.models.encoder import build_encoder
from captioner.utils.device import get_device
from captioner.utils.io import bytes_digest, git_sha

logger = logging.getLogger(__name__)

FeatureKind = Literal["pooled", "spatial"]
FEATURE_KINDS: tuple[FeatureKind, ...] = ("pooled", "spatial")


class FeatureFileError(ValueError):
    """A feature file is missing, malformed or does not match the image list."""


class ImageFolderDataset(Dataset[torch.Tensor]):
    """Images in a fixed order, preprocessed with the shared ``load_image``."""

    def __init__(self, paths: Sequence[Path], encoder_name: str = "resnet50") -> None:
        self.paths = list(paths)
        self.transform = transform_for(encoder_name)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return load_image(self.paths[i], self.transform)


def feature_paths(cfg: Config) -> dict[FeatureKind, Path]:
    """Paths of the pooled and spatial feature files for a config."""
    base = cfg.features_dir
    name = cfg.encoder.name
    return {"pooled": base / f"{name}_pooled.h5", "spatial": base / f"{name}_spatial.h5"}


def image_list_sha256(names: Sequence[str]) -> str:
    """Hash of the ordered image list a feature file was built from."""
    return bytes_digest("\n".join(names).encode("utf-8"))


def _create(
    path: Path, kind: FeatureKind, names: Sequence[str], cfg: Config, regions: int
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = h5py.File(path, "w")
    c = cfg.model.encoder_dim
    if kind == "pooled":
        fh.create_dataset("features", shape=(len(names), c), dtype="float16")
    else:
        fh.create_dataset(
            "features",
            shape=(len(names), regions, c),
            dtype="float16",
            chunks=(1, regions, c),
            compression="lzf",
        )
    fh.create_dataset("image_ids", data=list(names), dtype=h5py.string_dtype())
    fh.attrs.update(
        {
            "encoder": cfg.encoder.name,
            "weights": cfg.encoder.weights or "random",
            "input": cfg.encoder.image_size,
            "created_at": datetime.now(UTC).isoformat(),
            "code_sha": git_sha(),
            "image_list_sha256": image_list_sha256(names),
            "kind": kind,
        }
    )
    return fh


def extract_features(cfg: Config, device: torch.device | None = None) -> dict[FeatureKind, Path]:
    """Encode every image in the raw image folder and write both HDF5 files.

    Files are written to ``*.h5.tmp`` and renamed when complete, so an interrupted run never
    leaves a file that looks valid.

    Args:
        cfg: Resolved config (paths, encoder weights, batch size).
        device: Compute device; defaults to ``get_device()``.

    Returns:
        ``{"pooled": path, "spatial": path}``.
    """
    device = device or get_device()
    entries = feature_entries(cfg)
    names = [image_id for image_id, _ in entries]
    encoder = build_encoder(cfg.encoder.name, cfg.encoder.weights).to(device).eval()
    loader = DataLoader(
        ImageFolderDataset([path for _, path in entries], cfg.encoder.name),
        batch_size=cfg.encoder.batch_size,
        num_workers=cfg.encoder.num_workers,
        shuffle=False,
    )
    paths = feature_paths(cfg)
    tmp = {k: p.with_suffix(".h5.tmp") for k, p in paths.items()}
    files = {k: _create(tmp[k], k, names, cfg, cfg.model.num_regions) for k in FEATURE_KINDS}
    try:
        start = 0
        with torch.inference_mode():
            for images in loader:
                spatial = encoder(images.to(device))  # (B, L, C)
                end = start + spatial.shape[0]
                files["spatial"]["features"][start:end] = spatial.half().cpu().numpy()
                files["pooled"]["features"][start:end] = spatial.mean(dim=1).half().cpu().numpy()
                start = end
                logger.info("Encoded %d / %d images", end, len(names))
    finally:
        for fh in files.values():
            fh.close()
    for kind in FEATURE_KINDS:
        tmp[kind].replace(paths[kind])
    return paths


def validate_feature_file(path: Path, names: Sequence[str], kind: FeatureKind) -> None:
    """Check shape, dtype and image order of a feature file.

    Raises:
        FeatureFileError: Describing the first mismatch.
    """
    if not path.exists():
        raise FeatureFileError(f"{path} not found (run `captioner extract`)")
    with h5py.File(path, "r") as fh:
        feats = fh["features"]
        expected_ndim = 2 if kind == "pooled" else 3
        if feats.ndim != expected_ndim or feats.shape[0] != len(names):
            raise FeatureFileError(f"{path.name}: shape {feats.shape} for {len(names)} images")
        if feats.dtype != np.float16:
            raise FeatureFileError(f"{path.name}: dtype {feats.dtype}, expected float16")
        if fh.attrs.get("image_list_sha256") != image_list_sha256(names):
            raise FeatureFileError(f"{path.name}: built from a different image list")

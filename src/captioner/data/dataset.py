"""Datasets over cached features (and over images for encoder fine-tuning).

HDF5 handles are opened lazily inside each DataLoader worker because ``h5py`` handles are not
fork-safe (database.md §3.3). Batches are sorted by caption length (descending) and trimmed to
the longest caption, which is what ``pack_padded_sequence``-style decoders expect.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from captioner.data.text import tokenize
from captioner.data.transforms import load_image
from captioner.data.vocab import Vocabulary


@dataclass(frozen=True)
class CaptionTable:
    """All captions of one split, grouped per image.

    Attributes:
        image_ids: Image ids in a stable order (sorted).
        feature_idx: Row of each image in the feature files, ``(N,)``.
        captions: Encoded, padded captions ``(N, K, max_len)`` int64.
        lengths: Encoded lengths incl. ``<start>``/``<end>``, ``(N, K)``.
        references: Raw reference captions per image (used for scoring).
    """

    image_ids: tuple[str, ...]
    feature_idx: np.ndarray
    captions: np.ndarray
    lengths: np.ndarray
    references: tuple[tuple[str, ...], ...]

    def __len__(self) -> int:
        return len(self.image_ids)


def build_caption_table(df: pd.DataFrame, split: str, vocab: Vocabulary) -> CaptionTable:
    """Group a split's manifest rows by image and encode every caption."""
    rows = df[df["split"] == split].sort_values(["image_id", "caption_idx"])
    if rows.empty:
        raise ValueError(f"No rows for split {split!r}")
    groups = rows.groupby("image_id", sort=True)
    image_ids, feat_idx, caps, lens, refs = [], [], [], [], []
    for image_id, g in groups:
        tokens = [vocab.prepare_tokens(tokenize(str(c))) for c in g["caption_clean"]]
        image_ids.append(str(image_id))
        feat_idx.append(int(g["img_feature_idx"].iloc[0]))
        caps.append([vocab.encode(t) for t in tokens])
        lens.append([min(len(t) + 2, vocab.max_len) for t in tokens])
        refs.append(tuple(str(c) for c in g["caption_raw"]))
    return CaptionTable(
        image_ids=tuple(image_ids),
        feature_idx=np.asarray(feat_idx, dtype=np.int64),
        captions=np.asarray(caps, dtype=np.int64),
        lengths=np.asarray(lens, dtype=np.int64),
        references=tuple(refs),
    )


class FeatureStore:
    """Read-only, lazily opened view of one HDF5 feature file."""

    def __init__(
        self, path: Path, expected_ids: Sequence[str] | None = None, index: np.ndarray | None = None
    ) -> None:
        self.path = path
        self._fh: h5py.File | None = None
        if expected_ids is not None and index is not None:
            with h5py.File(path, "r") as fh:
                stored = fh["image_ids"].asstr()[:]
            if [stored[i] for i in index] != list(expected_ids):
                raise ValueError(
                    f"{path.name}: image ids do not match the manifest's feature index"
                )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None  # open handles must not cross process boundaries
        return state

    def _file(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def get(self, row: int) -> torch.Tensor:
        """Features for one row as float32: ``(C,)`` or ``(L, C)``."""
        return torch.from_numpy(np.asarray(self._file()["features"][row], dtype=np.float32))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class CaptionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]):
    """Training samples ``(features, caption_ids, length, image_index)``.

    With ``captions_per_image=1`` an epoch visits each image once with one random caption of its
    five (ADR-004); with ``5`` it visits every (image, caption) pair.
    """

    def __init__(
        self, table: CaptionTable, store: FeatureStore, captions_per_image: int = 1
    ) -> None:
        self.table = table
        self.store = store
        self.k = captions_per_image
        self.n_caps = table.captions.shape[1]

    def __len__(self) -> int:
        return len(self.table) * (1 if self.k == 1 else self.n_caps)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.k == 1:
            img, cap = i, int(torch.randint(self.n_caps, (1,)).item())
        else:
            img, cap = divmod(i, self.n_caps)
        feats = self.store.get(int(self.table.feature_idx[img]))
        ids = torch.from_numpy(self.table.captions[img, cap])
        return feats, ids, int(self.table.lengths[img, cap]), img


class ImageFeatureDataset(Dataset[tuple[torch.Tensor, int]]):
    """One sample per image ``(features, image_index)`` for decoding/evaluation."""

    def __init__(self, table: CaptionTable, store: FeatureStore) -> None:
        self.table = table
        self.store = store

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.store.get(int(self.table.feature_idx[i])), i


class ImageCaptionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]):
    """Like ``CaptionDataset`` but yields preprocessed images (encoder fine-tuning, FR-12)."""

    def __init__(
        self,
        table: CaptionTable,
        image_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        captions_per_image: int = 1,
    ) -> None:
        self.table = table
        self.image_dir = image_dir
        self.transform = transform
        self.k = captions_per_image
        self.n_caps = table.captions.shape[1]

    def __len__(self) -> int:
        return len(self.table) * (1 if self.k == 1 else self.n_caps)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.k == 1:
            img, cap = i, int(torch.randint(self.n_caps, (1,)).item())
        else:
            img, cap = divmod(i, self.n_caps)
        image = load_image(self.image_dir / f"{self.table.image_ids[img]}.jpg", self.transform)
        ids = torch.from_numpy(self.table.captions[img, cap])
        return image, ids, int(self.table.lengths[img, cap]), img


class ImageTableDataset(Dataset[tuple[torch.Tensor, int]]):
    """One preprocessed image per table entry ``(image, image_index)`` (fine-tuning eval)."""

    def __init__(self, table: CaptionTable, image_dir: Path) -> None:
        self.table = table
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return load_image(self.image_dir / f"{self.table.image_ids[i]}.jpg"), i


@dataclass(frozen=True)
class Batch:
    """A collated training batch, sorted by caption length (descending)."""

    feats: torch.Tensor  # (B, C) | (B, L, C) | (B, 3, H, W)
    captions: torch.Tensor  # (B, T) int64, T = longest caption in the batch
    lengths: torch.Tensor  # (B,) int64
    image_index: torch.Tensor  # (B,) int64

    def to(self, device: torch.device) -> Batch:
        return Batch(
            self.feats.to(device, non_blocking=True),
            self.captions.to(device, non_blocking=True),
            self.lengths.to(device),
            self.image_index,
        )


def collate_captions(samples: list[tuple[torch.Tensor, torch.Tensor, int, int]]) -> Batch:
    """Stack samples, sort by length (desc) and trim padding to the longest caption."""
    samples = sorted(samples, key=lambda s: s[2], reverse=True)
    feats = torch.stack([s[0] for s in samples])
    lengths = torch.tensor([s[2] for s in samples], dtype=torch.long)
    captions = torch.stack([s[1] for s in samples])[:, : int(lengths.max())]  # (B, T)
    index = torch.tensor([s[3] for s in samples], dtype=torch.long)
    return Batch(feats, captions, lengths, index)


def collate_images(samples: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``(features, image_index)`` samples."""
    return torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples])


def open_feature_store(path: Path, table: CaptionTable) -> FeatureStore:
    """Open a feature file and verify it lines up with ``table`` (ids at ``feature_idx``)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run `captioner extract`)")
    return FeatureStore(path, expected_ids=table.image_ids, index=table.feature_idx)

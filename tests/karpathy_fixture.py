"""Build a tiny COCO-style Karpathy dataset (zips + split JSON) from the mini Flickr8k images."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from captioner.config import Config
from captioner.utils.io import file_digest
from tests.conftest import MINI, make_config

SPLITS = ["train"] * 8 + ["restval"] * 4 + ["val"] * 4 + ["test"] * 4


def build(tmp_path: Path, n_extra_caps: int = 2) -> tuple[Config, dict[str, Any]]:
    images = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))
    captions: dict[str, list[str]] = {}
    for line in (MINI / "Flickr8k_text" / "Flickr8k.token.txt").read_text().splitlines():
        key, cap = line.split("\t", 1)
        captions.setdefault(key.rsplit("#", 1)[0], []).append(cap)
    entries, train_zip, val_zip = [], {}, {}
    for i, (img, split) in enumerate(zip(images, SPLITS, strict=True)):
        folder = "val2014" if split in ("val", "test") else "train2014"
        fname = f"COCO_{folder}_{i:012d}.jpg"
        (val_zip if folder == "val2014" else train_zip)[f"{folder}/{fname}"] = img.read_bytes()
        caps = captions[img.name] + [f"extra caption number {j}" for j in range(n_extra_caps)]
        entries.append(
            {"filepath": folder, "filename": fname, "split": split, "cocoid": i,
             "sentences": [{"raw": c, "tokens": c.lower().split()} for c in caps]}
        )  # fmt: skip
    entries.append(  # too few captions → dropped
        {"filepath": "train2014", "filename": "COCO_train2014_999.jpg", "split": "train",
         "sentences": [{"raw": "only one"}]}
    )  # fmt: skip
    src = tmp_path / "src"
    src.mkdir()
    zips = {}
    for name, members in (
        ("caption_datasets.zip", {"dataset_coco.json": json.dumps({"images": entries}).encode()}),
        ("train2014.zip", train_zip),
        ("val2014.zip", val_zip),
    ):
        with zipfile.ZipFile(src / name, "w") as zf:
            for member, data in members.items():
                zf.writestr(member, data)
        zips[name] = src / name
    overrides = {
        "name": "minicoco",
        "dataset": {
            "name": "coco",
            "kind": "karpathy",
            "karpathy_url": zips["caption_datasets.zip"].as_uri(),
            "karpathy_md5": file_digest(zips["caption_datasets.zip"], "md5"),
            "sources": [
                {
                    "name": "coco",
                    "json_file": "dataset_coco.json",
                    "image_root": "coco",
                    "archives": [
                        {"url": zips["train2014.zip"].as_uri(), "md5": None, "dest": "coco"},
                        {"url": zips["val2014.zip"].as_uri(), "md5": None, "dest": "coco"},
                    ],
                }
            ],
        },
    }
    return make_config(tmp_path / "work", overrides), {"zips": zips, "entries": entries}

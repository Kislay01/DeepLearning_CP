from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.data.download import ChecksumError, download_dataset, download_tofu
from captioner.data.karpathy import SourceFiles, build_karpathy_manifest
from captioner.data.manifest import ManifestError, check_invariants
from captioner.data.prepare import prepare_data
from captioner.data.sources import feature_entries, image_root, karpathy_sources
from tests.karpathy_fixture import build


def test_download_and_manifest(tmp_path: Path) -> None:
    cfg, _ = build(tmp_path)
    download_dataset(cfg)
    raw = cfg.paths.raw_dir
    assert (raw / "karpathy" / "dataset_coco.json").exists()
    assert len(list((raw / "coco" / "train2014").glob("*.jpg"))) == 12
    ledger = json.loads((raw / "checksums.json").read_text())
    assert set(ledger) == {"train2014.zip", "val2014.zip"}  # TOFU-recorded MD5s
    prepared = prepare_data(cfg)
    df = prepared.manifest
    check_invariants(df, expected_images={"train": 12, "val": 4, "test": 4})
    assert df["image_id"].str.startswith("coco_").all()
    assert not df["caption_raw"].str.startswith("extra caption").any()  # truncated to 5
    assert "coco_COCO_train2014_999" not in set(df["image_id"])  # < 5 captions → dropped
    assert cfg.manifest_file.parent.name == "coco"
    assert cfg.vocab_file.name.startswith("vocab_coco")
    entries = feature_entries(cfg, df)
    assert len(entries) == 20 and all(p.exists() for _, p in entries)
    assert [e[0] for e in entries] == sorted(e[0] for e in entries)
    assert image_root(cfg) == raw
    download_dataset(cfg)  # idempotent


def test_tofu_ledger_rejects_changed_archive(tmp_path: Path) -> None:
    src = tmp_path / "a.zip"
    src.write_bytes(b"first")
    ledger = tmp_path / "ledger.json"
    dest = download_tofu(src.as_uri(), tmp_path / "dl" / "a.zip", None, ledger)
    assert json.loads(ledger.read_text())["a.zip"]
    dest.write_bytes(b"tampered")
    with pytest.raises(ChecksumError):
        download_tofu(src.as_uri(), dest, None, ledger)


def test_combined_sources_and_errors(tmp_path: Path) -> None:
    cfg, _ = build(tmp_path)
    download_dataset(cfg)
    (src,) = karpathy_sources(cfg)
    twin = SourceFiles("flickr30k", src.json_path, src.image_root)
    with pytest.raises(ManifestError, match="same file"):
        build_karpathy_manifest([src, twin], cfg.paths.raw_dir, validate_images=False)
    next((src.image_root / "val2014").glob("*.jpg")).unlink()
    with pytest.raises(ManifestError, match="missing"):
        build_karpathy_manifest([src], cfg.paths.raw_dir)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"images": []}))
    with pytest.raises(ManifestError, match="No usable"):
        build_karpathy_manifest([SourceFiles("coco", empty, src.image_root)], cfg.paths.raw_dir)

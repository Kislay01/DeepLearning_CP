from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from captioner.config import Config
from captioner.data.download import raw_layout
from captioner.data.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    build_manifest,
    check_invariants,
    list_images,
    load_manifest,
    parse_token_file,
    save_manifest,
)
from tests.conftest import MINI_COUNTS

ORPHAN = "2258277193_586949ec62.jpg.1"


def _build(cfg: Config) -> pd.DataFrame:
    lay = raw_layout(cfg)
    ds = cfg.dataset
    return build_manifest(
        lay.image_dir,
        lay.text_dir / ds.token_file,
        {s: lay.text_dir / f for s, f in ds.split_files.items()},
    )


def test_mini_manifest_invariants(mini_cfg: Config) -> None:
    df = _build(mini_cfg)
    assert tuple(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 20 * 5
    check_invariants(df, expected_images=MINI_COUNTS)
    assert df["caption_idx"].between(0, 4).all()
    assert (df["n_tokens"] == df["caption_clean"].str.split().str.len()).all()
    assert (df["width"] > 0).all() and (df["height"] > 0).all()
    images = list_images(raw_layout(mini_cfg).image_dir)
    for row in df.drop_duplicates("image_id").itertuples():
        assert images[row.img_feature_idx] == row.file_name


def test_regression_D8_orphan_caption_id_dropped(mini_cfg: Config) -> None:
    token_file = raw_layout(mini_cfg).text_dir / mini_cfg.dataset.token_file
    assert ORPHAN in parse_token_file(token_file)  # the quirk is present in the raw file
    df = _build(mini_cfg)
    assert ORPHAN not in set(df["file_name"])
    assert not df["image_id"].str.contains(r"\.jpg").any()


def test_unsplit_images_are_ignored(mini_cfg: Config) -> None:
    lay = raw_layout(mini_cfg)
    Image.new("RGB", (10, 10)).save(lay.image_dir / "0000000000_unsplit.jpg")
    df = _build(mini_cfg)
    assert "0000000000_unsplit" not in set(df["image_id"])
    # The extra image shifts feature indices (it sorts first) but the mapping stays consistent.
    assert df["img_feature_idx"].min() == 1


def test_save_load_roundtrip(mini_cfg: Config, tmp_path: Path) -> None:
    df = _build(mini_cfg)
    save_manifest(df, tmp_path / "m.parquet")
    back = load_manifest(tmp_path / "m.parquet")
    pd.testing.assert_frame_equal(back, df)
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "missing.parquet")


def test_missing_split_image_rejected(mini_cfg: Config) -> None:
    lay = raw_layout(mini_cfg)
    next(lay.image_dir.glob("*.jpg")).unlink()
    with pytest.raises(ManifestError, match="missing"):
        _build(mini_cfg)


def test_unreadable_image_rejected(mini_cfg: Config) -> None:
    lay = raw_layout(mini_cfg)
    next(lay.image_dir.glob("*.jpg")).write_bytes(b"not an image")
    with pytest.raises(ManifestError, match="Unreadable"):
        _build(mini_cfg)


def test_wrong_caption_count_and_malformed_lines(mini_cfg: Config) -> None:
    lay = raw_layout(mini_cfg)
    token = lay.text_dir / mini_cfg.dataset.token_file
    lines = token.read_text().splitlines()
    token.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ManifestError, match="expected captions"):
        _build(mini_cfg)
    token.write_text("no tab or hash here\n")
    with pytest.raises(ManifestError, match="malformed"):
        parse_token_file(token)


def test_empty_cleaned_caption_rejected(mini_cfg: Config) -> None:
    lay = raw_layout(mini_cfg)
    token = lay.text_dir / mini_cfg.dataset.token_file
    lines = token.read_text().splitlines()
    first = next(i for i, ln in enumerate(lines) if not ln.startswith(ORPHAN.split(".")[0]))
    key = lines[first].split("\t")[0]
    lines[first] = f"{key}\t1234 !!"
    token.write_text("\n".join(lines) + "\n")
    with pytest.raises(ManifestError, match="empty"):
        _build(mini_cfg)


def test_overlapping_splits_rejected(mini_cfg: Config) -> None:
    lay = raw_layout(mini_cfg)
    ds = mini_cfg.dataset
    train = (lay.text_dir / ds.split_files["train"]).read_text()
    (lay.text_dir / ds.split_files["val"]).write_text(train)
    with pytest.raises(ManifestError, match="overlap"):
        _build(mini_cfg)


def test_missing_image_dir(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        list_images(tmp_path / "nope")


def test_check_invariants_reports_violations(mini_cfg: Config) -> None:
    df = _build(mini_cfg)
    with pytest.raises(ManifestError, match="expected 6000"):
        check_invariants(df, expected_images={"train": 6000, "val": 4, "test": 4})
    bad = df.drop(index=0)
    with pytest.raises(ManifestError, match="captions"):
        check_invariants(bad)
    dup = df.copy()
    dup.loc[dup["split"] == "val", "img_feature_idx"] = 0
    with pytest.raises(ManifestError, match="one-to-one"):
        check_invariants(dup)
    orphan = df.copy()
    orphan["file_name"] = orphan["file_name"].astype(str) + ".1"
    orphan["caption_clean"] = ""
    with pytest.raises(ManifestError, match="orphan"):
        check_invariants(orphan)
    overlap = df.copy()
    overlap.loc[overlap["split"] == "val", "image_id"] = str(df["image_id"].iloc[0])
    with pytest.raises(ManifestError, match="overlaps"):
        check_invariants(overlap)
    with pytest.raises(ManifestError, match="columns"):
        check_invariants(df.rename(columns={"width": "w"}))


def test_fixture_is_small() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "mini_flickr"
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert total < 1_000_000
    assert len(list((root / "Flicker8k_Dataset").glob("*.jpg"))) == 20

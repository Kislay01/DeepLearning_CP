from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from captioner.data.download import (
    ChecksumError,
    UnsafeArchiveError,
    download_file,
    download_flickr8k,
    raw_layout,
    safe_extract,
)
from captioner.utils.io import file_digest
from tests.conftest import MINI, make_config


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_download_file_verifies_md5(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    md5 = file_digest(src, "md5")
    dest = download_file(src.as_uri(), tmp_path / "out" / "f.bin", md5)
    assert dest.read_bytes() == b"hello"
    assert download_file(src.as_uri(), dest, md5) == dest  # cached
    with pytest.raises(ChecksumError, match="mismatch"):
        download_file(src.as_uri(), tmp_path / "out" / "g.bin", "0" * 32)
    assert not (tmp_path / "out" / "g.bin").exists()
    assert not (tmp_path / "out" / "g.bin.part").exists()
    with pytest.raises(ChecksumError, match="exists"):
        download_file(src.as_uri(), dest, "0" * 32)


def test_safe_extract_skips_macos_metadata(tmp_path: Path) -> None:
    z = _zip(
        tmp_path / "a.zip",
        {
            "Flicker8k_Dataset/x.jpg": b"1",
            "__MACOSX/Flicker8k_Dataset/._x.jpg": b"junk",
            "Flicker8k_Dataset/._y.jpg": b"junk",
        },
    )
    n = safe_extract(z, tmp_path / "out")
    assert n == 1
    assert (tmp_path / "out" / "Flicker8k_Dataset" / "x.jpg").read_bytes() == b"1"
    assert not (tmp_path / "out" / "__MACOSX").exists()


@pytest.mark.parametrize(
    "evil", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt", "C:/evil.txt"]
)
def test_safe_extract_rejects_zip_slip(tmp_path: Path, evil: str) -> None:
    z = _zip(tmp_path / "evil.zip", {"ok.txt": b"ok", evil: b"pwned"})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out" / "ok.txt").exists()  # nothing written before validation


def test_download_flickr8k_end_to_end_with_local_archives(tmp_path: Path) -> None:
    images = {
        f"Flicker8k_Dataset/{p.name}": p.read_bytes()
        for p in (MINI / "Flicker8k_Dataset").iterdir()
    }
    images["__MACOSX/Flicker8k_Dataset/._junk.jpg"] = b"junk"
    texts = {p.name: p.read_bytes() for p in (MINI / "Flickr8k_text").iterdir()}
    img_zip = _zip(tmp_path / "img.zip", images)
    txt_zip = _zip(tmp_path / "txt.zip", texts)
    cfg = make_config(
        tmp_path,
        {
            "dataset": {
                "images_url": img_zip.as_uri(),
                "images_md5": file_digest(img_zip, "md5"),
                "text_url": txt_zip.as_uri(),
                "text_md5": file_digest(txt_zip, "md5"),
            }
        },
    )
    layout = download_flickr8k(cfg)
    assert layout == raw_layout(cfg)
    assert len(list(layout.image_dir.glob("*.jpg"))) == 20
    assert (layout.text_dir / "Flickr8k.token.txt").exists()
    again = download_flickr8k(cfg)  # idempotent: markers short-circuit extraction
    assert again == layout

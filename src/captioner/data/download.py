"""Download, verify (MD5) and safely extract the Flickr8k archives (FR-1)."""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from captioner.config import Config
from captioner.utils.io import file_digest

logger = logging.getLogger(__name__)

_SKIP_PREFIXES = ("__MACOSX/",)
_CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    """A downloaded file does not match its pinned checksum."""


class UnsafeArchiveError(RuntimeError):
    """An archive member would be written outside the destination (zip-slip)."""


@dataclass(frozen=True)
class RawLayout:
    """Where the extracted Flickr8k files live."""

    image_dir: Path
    text_dir: Path


def download_file(url: str, dest: Path, md5: str, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` unless a file with the right MD5 already exists.

    The file is streamed to ``dest.part`` and only renamed after the MD5 matches.

    Args:
        url: Source URL (``https://`` or ``file://``).
        dest: Target path.
        md5: Expected hex MD5; the function refuses to continue on mismatch.
        timeout: Socket timeout in seconds.

    Returns:
        ``dest``.

    Raises:
        ChecksumError: If the downloaded (or existing) file has a different MD5.
    """
    if dest.exists():
        if file_digest(dest, "md5") == md5:
            logger.info("%s already present, MD5 ok", dest.name)
            return dest
        raise ChecksumError(f"{dest} exists but its MD5 differs from {md5}; delete it and retry")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp, part.open("wb") as out:
        shutil.copyfileobj(resp, out, _CHUNK)
    actual = file_digest(part, "md5")
    if actual != md5:
        part.unlink()
        raise ChecksumError(f"MD5 mismatch for {url}: expected {md5}, got {actual}")
    part.replace(dest)
    return dest


def _member_target(dest: Path, name: str) -> Path | None:
    """Resolve where a zip member goes, or ``None`` to skip it. Rejects path traversal."""
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and pure.parts[0].endswith(":")):
        raise UnsafeArchiveError(f"Refusing unsafe archive member: {name!r}")
    if name.startswith(_SKIP_PREFIXES) or pure.name.startswith("._"):
        return None
    target = (dest / Path(*pure.parts)).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise UnsafeArchiveError(f"Refusing archive member outside destination: {name!r}")
    return target


def safe_extract(zip_path: Path, dest: Path) -> int:
    """Extract a zip archive, skipping macOS metadata and rejecting zip-slip members.

    All members are validated before anything is written.

    Args:
        zip_path: Archive to extract.
        dest: Destination directory.

    Returns:
        Number of files written.

    Raises:
        UnsafeArchiveError: If any member has an absolute path or ``..`` component.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        plan = [(info, _member_target(dest, info.filename)) for info in zf.infolist()]
        for info, target in plan:
            if target is None or info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
            written += 1
    return written


def raw_layout(cfg: Config) -> RawLayout:
    """Paths of the extracted image and text folders for a config."""
    raw = cfg.paths.raw_dir
    return RawLayout(
        image_dir=raw / cfg.dataset.image_dir_name, text_dir=raw / cfg.dataset.text_dir_name
    )


def download_flickr8k(cfg: Config) -> RawLayout:
    """Download both Flickr8k archives, verify MD5s and extract them (idempotent).

    Args:
        cfg: Resolved configuration (URLs, MD5s and ``paths.data_dir``).

    Returns:
        The extracted layout.
    """
    ds = cfg.dataset
    raw = cfg.paths.raw_dir
    layout = raw_layout(cfg)
    text_zip = download_file(ds.text_url, raw / "Flickr8k_text.zip", ds.text_md5)
    image_zip = download_file(ds.images_url, raw / "Flickr8k_Dataset.zip", ds.images_md5)
    for zip_path, dest, md5 in (
        (text_zip, layout.text_dir, ds.text_md5),
        (image_zip, raw, ds.images_md5),
    ):
        marker = raw / f".{zip_path.stem}.extracted"
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == md5:
            continue
        n = safe_extract(zip_path, dest)
        marker.write_text(md5, encoding="utf-8")
        logger.info("Extracted %d files from %s", n, zip_path.name)
    return layout


def download_tofu(
    url: str, dest: Path, md5: str | None, ledger: Path, timeout: float = 60.0
) -> Path:
    """Download with a pinned MD5, or — when upstream publishes none — trust on first use.

    Without ``md5`` the hash of the first download is written to ``ledger`` (JSON) and every
    later run must match it, so a changed or corrupted archive is still rejected (ADR-023).
    """
    if md5 is not None:
        return download_file(url, dest, md5, timeout)
    known: dict[str, str] = (
        json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {}
    )
    if dest.name in known:
        return download_file(url, dest, known[dest.name], timeout)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        logger.info("Downloading %s (no published checksum: recording it)", url)
        with urllib.request.urlopen(url, timeout=timeout) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, _CHUNK)
        part.replace(dest)
    known[dest.name] = file_digest(dest, "md5")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(known, indent=2) + "\n", encoding="utf-8")
    logger.warning(
        "Recorded MD5 %s for %s in %s; pin it in the config", known[dest.name], dest.name, ledger
    )
    return dest


def _extract_once(archive: Path, dest: Path, raw: Path) -> None:
    md5 = file_digest(archive, "md5")
    marker = raw / f".{archive.stem}.extracted"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == md5:
        return
    n = safe_extract(archive, dest)
    marker.write_text(md5, encoding="utf-8")
    logger.info("Extracted %d files from %s", n, archive.name)


def download_karpathy(cfg: Config) -> Path:
    """Download the Karpathy split files (MD5-pinned) and every source's image archives."""
    raw = cfg.paths.raw_dir
    ds = cfg.dataset
    ledger = raw / "checksums.json"
    split_zip = download_file(
        ds.karpathy_url, raw / "archives" / "caption_datasets.zip", ds.karpathy_md5
    )
    _extract_once(split_zip, raw / "karpathy", raw)
    for src in ds.sources:
        for arc in src.archives:
            name = arc.url.rsplit("/", 1)[-1]
            archive = download_tofu(arc.url, raw / "archives" / name, arc.md5, ledger)
            _extract_once(archive, raw / (arc.dest or src.name), raw)
    return raw


def download_dataset(cfg: Config) -> Path:
    """Download whatever ``cfg.dataset`` describes; returns the folder holding the images."""
    if cfg.dataset.kind == "flickr8k":
        return download_flickr8k(cfg).image_dir
    return download_karpathy(cfg)

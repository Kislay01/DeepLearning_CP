"""Verify the prepared Flickr8k data against the numbers in docs/database.md.

Checks: raw archive MD5s, manifest invariants (40,000 rows, 6000/1000/1000 images, disjoint
splits, 5 captions each, orphan id dropped, one feature row per image), vocab = 2,529 words +
4 specials built from train.

Usage: uv run python scripts/verify_data.py [--config configs/base.yaml] [--skip-md5]
Exit code 0 when everything matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from captioner.config import Config, load_config
from captioner.data.download import raw_layout
from captioner.data.manifest import ManifestError, check_invariants, list_images
from captioner.data.prepare import load_prepared
from captioner.utils.io import file_digest

EXPECTED_IMAGES = {"train": 6000, "val": 1000, "test": 1000}
EXPECTED_ROWS = 40_000
EXPECTED_ALL_IMAGES = 8091
EXPECTED_VOCAB_WORDS = 2529

console = Console()


def _check(ok: bool, label: str, failures: list[str]) -> None:
    console.print(f"[{'green' if ok else 'red'}]{'PASS' if ok else 'FAIL'}[/] {label}")
    if not ok:
        failures.append(label)


def verify_archives(cfg: Config, failures: list[str]) -> None:
    raw = cfg.paths.raw_dir
    for name, md5 in (
        ("Flickr8k_Dataset.zip", cfg.dataset.images_md5),
        ("Flickr8k_text.zip", cfg.dataset.text_md5),
    ):
        path = raw / name
        _check(path.exists() and file_digest(path, "md5") == md5, f"{name} MD5 == {md5}", failures)


def verify_prepared(cfg: Config, failures: list[str]) -> None:
    prepared = load_prepared(cfg)
    df, vocab = prepared.manifest, prepared.vocab
    _check(len(df) == EXPECTED_ROWS, f"manifest rows == {EXPECTED_ROWS} (got {len(df)})", failures)
    try:
        check_invariants(df, expected_images=EXPECTED_IMAGES)
        _check(True, "manifest invariants (database.md §3.1)", failures)
    except ManifestError as exc:
        _check(False, f"manifest invariants: {exc}", failures)
    n_images = len(list_images(raw_layout(cfg).image_dir))
    _check(
        n_images == EXPECTED_ALL_IMAGES,
        f"images on disk == {EXPECTED_ALL_IMAGES} (got {n_images})",
        failures,
    )
    _check(
        vocab.n_words == EXPECTED_VOCAB_WORDS and vocab.min_freq == 5,
        f"vocab == {EXPECTED_VOCAB_WORDS} + 4 specials at min_freq 5 (got {vocab.n_words})",
        failures,
    )
    _check(vocab.built_from == "train", "vocab built from train split", failures)
    _check(vocab["a"] != 3 and vocab.freqs.get("a", 0) > 40_000, "article 'a' kept (D2)", failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-md5", action="store_true", help="skip re-hashing the 1.1 GB zip")
    args = parser.parse_args()
    cfg = load_config(args.config)
    failures: list[str] = []
    if not args.skip_md5:
        verify_archives(cfg, failures)
    verify_prepared(cfg, failures)
    console.print(
        f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

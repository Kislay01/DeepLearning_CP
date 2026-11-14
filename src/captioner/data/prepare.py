"""Stage `prepare`: raw files → validated manifest → train-only vocabulary."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from captioner.config import Config
from captioner.data.download import raw_layout
from captioner.data.manifest import (
    build_manifest,
    check_invariants,
    load_manifest,
    manifest_sha256,
    save_manifest,
)
from captioner.data.text import tokenize
from captioner.data.vocab import VocabError, Vocabulary, build_vocab

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    """Outputs of the prepare stage."""

    manifest: pd.DataFrame
    vocab: Vocabulary
    manifest_sha256: str


def build_vocab_from_manifest(df: pd.DataFrame, min_freq: int, max_len: int) -> Vocabulary:
    """Build the vocabulary from the **train** rows of a manifest only (fixes D5)."""
    train = df.loc[df["split"] == "train", "caption_clean"]
    return build_vocab(
        (tokenize(str(c)) for c in train),
        min_freq=min_freq,
        max_len=max_len,
        sha256_manifest=manifest_sha256(df),
        built_from="train",
    )


def prepare_data(cfg: Config) -> PreparedData:
    """Build and save ``manifest.parquet`` and ``vocab.json`` from the extracted raw data."""
    layout = raw_layout(cfg)
    ds = cfg.dataset
    df = build_manifest(
        image_dir=layout.image_dir,
        token_file=layout.text_dir / ds.token_file,
        split_files={s: layout.text_dir / f for s, f in ds.split_files.items()},
        n_captions=ds.captions_per_image,
    )
    check_invariants(df, n_captions=ds.captions_per_image)
    sha = save_manifest(df, cfg.paths.manifest_path)
    vocab = build_vocab_from_manifest(df, cfg.vocab.min_freq, cfg.vocab.max_len)
    vocab.save(cfg.paths.vocab_path)
    logger.info(
        "Manifest: %d rows, %d images; vocab: %d words + 4 specials",
        len(df),
        df["image_id"].nunique(),
        vocab.n_words,
    )
    return PreparedData(manifest=df, vocab=vocab, manifest_sha256=sha)


def load_prepared(cfg: Config) -> PreparedData:
    """Load manifest + vocab and check the vocab was built from this exact manifest."""
    df = load_manifest(cfg.paths.manifest_path)
    vocab = Vocabulary.load(cfg.paths.vocab_path)
    sha = manifest_sha256(df)
    if vocab.sha256_manifest != sha:
        raise VocabError(
            "vocab.json was built from a different manifest; re-run `captioner prepare`"
        )
    return PreparedData(manifest=df, vocab=vocab, manifest_sha256=sha)

"""Vocabulary with reserved special tokens, built from the train split only.

This module is the single home of the special-token constants (code_style.md §3.9).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captioner.utils.io import bytes_digest

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3
SPECIALS: dict[str, int] = {
    PAD_TOKEN: PAD_IDX,
    START_TOKEN: START_IDX,
    END_TOKEN: END_IDX,
    UNK_TOKEN: UNK_IDX,
}
SPECIAL_IDS = frozenset(SPECIALS.values())
VOCAB_FORMAT_VERSION = 1


class VocabError(ValueError):
    """Raised for malformed vocabulary files or mismatched inputs."""


@dataclass(frozen=True)
class Vocabulary:
    """Token ↔ id mapping.

    Attributes:
        itos: Id → token; the first four entries are the special tokens.
        freqs: Train-split frequency of every kept word.
        min_freq: Frequency threshold used to build it.
        max_len: Maximum encoded length including ``<start>`` and ``<end>``.
        sha256_manifest: Hash of the manifest the vocabulary was built from.
        built_from: Split name the counts come from.
    """

    itos: tuple[str, ...]
    freqs: dict[str, int]
    min_freq: int
    max_len: int
    sha256_manifest: str = ""
    built_from: str = "train"
    stoi: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if tuple(self.itos[: len(SPECIALS)]) != tuple(SPECIALS):
            raise VocabError(f"First tokens must be {list(SPECIALS)}, got {self.itos[:4]}")
        if len(set(self.itos)) != len(self.itos):
            raise VocabError("Duplicate tokens in vocabulary")
        object.__setattr__(self, "stoi", {tok: i for i, tok in enumerate(self.itos)})

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: object) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    @property
    def n_words(self) -> int:
        """Number of real words (excluding the special tokens)."""
        return len(self.itos) - len(SPECIALS)

    def encode(self, tokens: Sequence[str], pad: bool = True) -> list[int]:
        """Map words to ids wrapped in ``<start> … <end>``.

        Captions longer than ``max_len`` are truncated, keeping ``<end>`` as the final id.

        Args:
            tokens: Words of a cleaned caption.
            pad: Right-pad with ``<pad>`` up to ``max_len``.

        Returns:
            List of ids of length ``max_len`` (padded) or ``len(tokens) + 2`` (capped).
        """
        body = [self[t] for t in tokens][: self.max_len - 2]
        ids = [START_IDX, *body, END_IDX]
        if pad:
            ids += [PAD_IDX] * (self.max_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Map ids back to words, stopping at ``<end>`` and dropping special tokens."""
        words: list[str] = []
        for i in ids:
            if i == END_IDX:
                break
            if i in SPECIAL_IDS:
                continue
            words.append(self.itos[i])
        return words

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (database.md §3.2)."""
        return {
            "version": VOCAB_FORMAT_VERSION,
            "built_from": self.built_from,
            "min_freq": self.min_freq,
            "max_len": self.max_len,
            "specials": dict(SPECIALS),
            "itos": list(self.itos),
            "freqs": dict(self.freqs),
            "sha256_manifest": self.sha256_manifest,
        }

    def to_json(self) -> str:
        """Canonical JSON text."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n"

    def sha256(self) -> str:
        """SHA-256 of the canonical JSON text (stored in checkpoints and runs)."""
        return bytes_digest(self.to_json().encode("utf-8"))

    def save(self, path: Path) -> None:
        """Write ``vocab.json``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        """Validate and build from the JSON representation."""
        if data.get("version") != VOCAB_FORMAT_VERSION:
            raise VocabError(f"Unsupported vocab version: {data.get('version')}")
        if data.get("specials") != SPECIALS:
            raise VocabError(f"Special tokens must be {SPECIALS}")
        return cls(
            itos=tuple(data["itos"]),
            freqs={str(k): int(v) for k, v in data["freqs"].items()},
            min_freq=int(data["min_freq"]),
            max_len=int(data["max_len"]),
            sha256_manifest=str(data.get("sha256_manifest", "")),
            built_from=str(data.get("built_from", "train")),
        )

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        """Read ``vocab.json``."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VocabError(f"{path} is not valid JSON") from exc
        return cls.from_dict(data)


def build_vocab(
    captions: Iterable[Sequence[str]],
    min_freq: int,
    max_len: int,
    sha256_manifest: str = "",
    built_from: str = "train",
) -> Vocabulary:
    """Count words and keep those with frequency ≥ ``min_freq``.

    Words are ordered by descending frequency, ties broken alphabetically, so the result is
    deterministic. Pass **train** captions only (fixes reference D5).

    Args:
        captions: Tokenised, cleaned captions.
        min_freq: Minimum count for a word to get its own id.
        max_len: Maximum encoded length including ``<start>``/``<end>``.
        sha256_manifest: Hash of the source manifest, stored for integrity checks.
        built_from: Name of the split the counts come from.

    Returns:
        The vocabulary.
    """
    counts: Counter[str] = Counter()
    for tokens in captions:
        counts.update(tokens)
    for special in SPECIALS:
        counts.pop(special, None)
    kept = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
    return Vocabulary(
        itos=(*SPECIALS, *kept),
        freqs={w: counts[w] for w in kept},
        min_freq=min_freq,
        max_len=max_len,
        sha256_manifest=sha256_manifest,
        built_from=built_from,
    )

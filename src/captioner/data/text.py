"""Caption text normalisation (pure functions, no I/O).

Rules (architecture.md §4): lowercase → replace every non ``[a-z]`` character with a space →
collapse whitespace. One-letter words such as the article "a" are kept (fixes reference D2).
"""

from __future__ import annotations

import re

_NON_ALPHA = re.compile(r"[^a-z]+")


def clean_caption(text: str) -> str:
    """Normalise a raw caption.

    Args:
        text: Raw caption, e.g. ``"A dog's ball - 2 times!"``.

    Returns:
        Cleaned caption, e.g. ``"a dog s ball times"``. May be empty if the input had no letters.
    """
    return " ".join(_NON_ALPHA.sub(" ", text.lower()).split())


def tokenize(clean: str) -> list[str]:
    """Split a cleaned caption into word tokens."""
    return clean.split()


def detokenize(tokens: list[str]) -> str:
    """Turn word tokens into a display sentence: capitalised, single trailing period.

    Args:
        tokens: Word tokens without special symbols.

    Returns:
        ``"A dog runs through the grass."`` style text, or ``""`` for no tokens.
    """
    if not tokens:
        return ""
    sentence = " ".join(tokens)
    return sentence[0].upper() + sentence[1:] + "."

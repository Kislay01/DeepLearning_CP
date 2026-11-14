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


# Function words that may legitimately repeat inside one caption ("a dog in a park").
def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


FUNCTION_WORDS = _wordset(
    """a an the is are was were be been being in on of at with and or to its it his her their
    they them this that these those while for by from into through up down out over under
    some front behind near next another other each two three one has have"""
)

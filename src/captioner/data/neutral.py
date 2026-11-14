"""Gender-neutral caption language (ADR-021).

Captioning models learn to *guess* gender from appearance (Hendricks et al., 2018, "Women also
snowboard"), and Flickr8k captions say "man"/"woman"/"boy"/"girl" in most sentences. Gender is not
visually verifiable, so captions here describe people neutrally. The same mapping is used to
rewrite training targets, evaluation references, and — as a safety net — the output of models
trained on raw captions.

Age is kept only at the coarse, visually grounded child/adult level ("child", "person").
"""

from __future__ import annotations

import re
from collections.abc import Sequence

NEUTRAL_MAP: dict[str, str] = {
    # adults
    "man": "person", "woman": "person", "guy": "person", "lady": "person",
    "gentleman": "person", "gal": "person", "male": "person", "female": "person",
    "men": "people", "women": "people", "guys": "people", "ladies": "people",
    "gentlemen": "people", "males": "people", "females": "people",
    # children
    "boy": "child", "girl": "child", "boys": "children", "girls": "children",
    "schoolboy": "student", "schoolgirl": "student",
    # relations and roles
    "mother": "parent", "father": "parent", "mom": "parent", "dad": "parent",
    "mothers": "parents", "fathers": "parents", "son": "child", "daughter": "child",
    "sons": "children", "daughters": "children", "brother": "sibling", "sister": "sibling",
    "brothers": "siblings", "sisters": "siblings", "husband": "partner", "wife": "partner",
    "boyfriend": "partner", "girlfriend": "partner", "bride": "newlywed", "groom": "newlywed",
    "policeman": "officer", "policemen": "officers", "fireman": "firefighter",
    "firemen": "firefighters", "businessman": "businessperson", "businessmen": "businesspeople",
    "cameraman": "photographer", "fisherman": "fisher", "fishermen": "fishers",
    "sportsman": "athlete", "cowboy": "rider", "cowboys": "riders", "cowgirl": "rider",
    # pronouns
    "he": "they", "she": "they", "him": "them", "his": "their", "her": "their",
    "hers": "theirs", "himself": "themselves", "herself": "themselves",
}  # fmt: skip

GENDERED_WORDS = frozenset(NEUTRAL_MAP)
MALE_WORDS = frozenset(
    [
        "man",
        "men",
        "guy",
        "guys",
        "gentleman",
        "gentlemen",
        "male",
        "males",
        "boy",
        "boys",
        "schoolboy",
        "father",
        "fathers",
        "dad",
        "son",
        "sons",
        "brother",
        "brothers",
        "husband",
        "boyfriend",
        "groom",
        "policeman",
        "policemen",
        "fireman",
        "firemen",
        "businessman",
        "businessmen",
        "cameraman",
        "fisherman",
        "fishermen",
        "sportsman",
        "cowboy",
        "cowboys",
        "he",
        "him",
        "his",
        "himself",
    ]
)
FEMALE_WORDS = GENDERED_WORDS - MALE_WORDS - {"they", "them", "their"}

_AGREEMENT = (
    (re.compile(r"\bthey is\b"), "they are"),
    (re.compile(r"\bthey was\b"), "they were"),
    (re.compile(r"\bthey has\b"), "they have"),
    (re.compile(r"\bthey does\b"), "they do"),
    (re.compile(r"\ba person and a person\b"), "two people"),
    (re.compile(r"\ba child and a child\b"), "two children"),
)


def neutralize_tokens(tokens: Sequence[str]) -> list[str]:
    """Map gendered words to neutral ones and fix the resulting agreement errors.

    Args:
        tokens: Lower-case word tokens (output of ``tokenize(clean_caption(...))``).

    Returns:
        Neutral tokens, e.g. ``["a", "man", "and", "a", "woman"] → ["two", "people"]``.
    """
    text = " ".join(NEUTRAL_MAP.get(t, t) for t in tokens)
    for pattern, repl in _AGREEMENT:
        text = pattern.sub(repl, text)
    return text.split()


_ALIGNED_FIXES = {
    ("they", "is"): "are",
    ("they", "was"): "were",
    ("they", "has"): "have",
    ("they", "does"): "do",
}


def neutralize_aligned(tokens: Sequence[str]) -> list[str]:
    """One-to-one neutral rewrite (same length), so per-word attention maps stay aligned.

    Used on the output of models trained on raw captions; phrase merges such as
    "a person and a person" → "two people" are only applied to training text.
    """
    out = [NEUTRAL_MAP.get(t, t) for t in tokens]
    for i in range(1, len(out)):
        out[i] = _ALIGNED_FIXES.get((out[i - 1], out[i]), out[i])
    return out


def neutralize_text(text: str) -> str:
    """Neutralise a cleaned (lower-case, space-separated) caption."""
    return " ".join(neutralize_tokens(text.split()))


def gender_of(tokens: Sequence[str]) -> str | None:
    """``"male"``/``"female"`` if only one gender's words occur, ``"mixed"`` if both, else None."""
    words = set(tokens)
    male, female = bool(words & MALE_WORDS), bool(words & FEMALE_WORDS)
    if male and female:
        return "mixed"
    return "male" if male else "female" if female else None

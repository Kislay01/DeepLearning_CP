"""Automatic, keyword-based failure taxonomy (testing.md §4, qualitative analysis).

These are **heuristics**: a category is flagged when the prediction uses a word of that kind that
no reference uses. They approximate a manual audit and are labelled as such in reports.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from captioner.data.text import clean_caption, tokenize


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


COLOURS = _words(
    """black white red blue green yellow brown orange pink purple gray grey tan blond blonde"""
)
COUNTS = _words("""two three four five six several many group couple crowd one""")
GENDER_AGE = _words(
    """man men woman women boy boys girl girls lady ladies guy guys child children kid kids
    baby toddler teenager old young elderly"""
)
GENERIC = _words(
    """a an the is are in on of at with and man woman person people standing sitting dog two
    outside near something"""
)
STOPWORDS = _words(
    """a an the is are in on of at with and to its his her their while for by from into through
    up down out over under some front behind"""
)
CATEGORIES = (
    "wrong_object",
    "wrong_count",
    "wrong_colour",
    "hallucinated_object",
    "gender_age_assumption",
    "generic_caption",
)


def _content(words: Sequence[str]) -> list[str]:
    return [w for w in words if w not in STOPWORDS]


def _stem(word: str) -> str:
    """Crude plural/3rd-person folding so "dogs"/"runs" match "dog"/"run"."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def classify(prediction: str, references: Sequence[str]) -> set[str]:
    """Return the failure categories triggered by one prediction."""
    pred = tokenize(clean_caption(prediction))
    ref_raw = {w for r in references for w in tokenize(clean_caption(r))}
    ref_words = ref_raw | {_stem(w) for w in ref_raw}
    flags: set[str] = set()
    content = _content(pred)
    if not content:
        return {"generic_caption"}
    nouns = [w for w in content if w not in COLOURS | COUNTS]
    if nouns and _stem(nouns[0]) not in ref_words:
        flags.add("wrong_object")
    if any(w in COLOURS and w not in ref_words for w in pred):
        flags.add("wrong_colour")
    if any(w in COUNTS and w not in ref_words for w in pred):
        flags.add("wrong_count")
    if any(w in GENDER_AGE and w not in ref_words for w in pred):
        flags.add("gender_age_assumption")
    other = [w for w in nouns[1:] if w not in GENDER_AGE]
    if any(_stem(w) not in ref_words for w in other):
        flags.add("hallucinated_object")
    if all(w in GENERIC for w in pred):
        flags.add("generic_caption")
    return flags


def taxonomy_counts(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> dict[str, int]:
    """Count each category over the given predictions (an image can hit several)."""
    counts: Counter[str] = Counter({c: 0 for c in CATEGORIES})
    for pred, refs in zip(predictions, references, strict=True):
        counts.update(classify(pred, refs))
    return {c: counts[c] for c in CATEGORIES}


def diversity(predictions: Sequence[str]) -> dict[str, float]:
    """Unique-caption share, vocabulary used and mean length (generic-caption collapse check)."""
    words = [tokenize(clean_caption(p)) for p in predictions]
    n = max(1, len(predictions))
    repeated = sum(len(c := [w for w in ws if w not in STOPWORDS]) != len(set(c)) for ws in words)
    return {
        "n_captions": float(len(predictions)),
        "repeated_word_rate": repeated / n,
        "unique_caption_ratio": len(set(predictions)) / n,
        "vocab_used": float(len({w for ws in words for w in ws})),
        "mean_length": sum(len(ws) for ws in words) / n,
    }

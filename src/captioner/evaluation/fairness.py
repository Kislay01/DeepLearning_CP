"""Gender-bias metrics for captions (ADR-021), after Burns/Hendricks et al. (2018).

* ``gender_word_rate`` — share of predictions that use any gendered word.
* ``gender_error_rate`` — among images whose references agree on one gender, the share of
  predictions that use the *other* gender's words.
* ``gender_guess_rate`` — among images whose references disagree or are all neutral, the share of
  predictions that still assert a gender (an unsupported guess).
"""

from __future__ import annotations

from collections.abc import Sequence

from captioner.data.neutral import GENDERED_WORDS, gender_of
from captioner.data.text import clean_caption, tokenize


def _ref_gender(references: Sequence[str]) -> str | None:
    genders = {gender_of(tokenize(clean_caption(r))) for r in references}
    genders.discard(None)
    if genders == {"male"}:
        return "male"
    if genders == {"female"}:
        return "female"
    return None


def gender_metrics(
    predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Compute the three rates over tokenised predictions and raw references."""
    n = max(1, len(predictions))
    used = sum(bool(set(p) & GENDERED_WORDS) for p in predictions)
    agreed = wrong = unclear = guessed = 0
    for pred, refs in zip(predictions, references, strict=True):
        truth, said = _ref_gender(refs), gender_of(pred)
        if truth is None:
            unclear += 1
            guessed += said in ("male", "female")
        else:
            agreed += 1
            wrong += said not in (None, truth)
    return {
        "gender_word_rate": used / n,
        "gender_error_rate": wrong / agreed if agreed else 0.0,
        "gender_guess_rate": guessed / unclear if unclear else 0.0,
        "n_reference_gender_agreed": float(agreed),
    }

"""Caption metrics.

* ``corpus_bleu4`` — fast NLTK corpus BLEU-4 used for early stopping during training.
* ``coco_scores`` — literature-standard ``pycocoevalcap`` metrics (BLEU-1..4, METEOR, ROUGE-L,
  CIDEr) on PTB-tokenised raw references (testing.md §4).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from nltk.translate.bleu_score import corpus_bleu

from captioner.data.text import clean_caption, tokenize


def reference_tokens(raw_refs: Sequence[str]) -> list[list[str]]:
    """Lowercase + strip punctuation for NLTK BLEU (keeps "a")."""
    return [tokenize(clean_caption(r)) for r in raw_refs]


def corpus_bleu4(
    hypotheses: Sequence[Sequence[str]], references: Sequence[Sequence[Sequence[str]]]
) -> float:
    """Corpus-level BLEU-4 (uniform weights, no smoothing).

    Args:
        hypotheses: Tokenised predictions, one per image.
        references: Tokenised references, several per image.

    Returns:
        BLEU-4 in ``[0, 1]``; ``0.0`` when there is no 4-gram overlap.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # NLTK warns when an n-gram order has 0 hits
        score = corpus_bleu(
            [list(map(list, refs)) for refs in references],
            [list(h) for h in hypotheses],
            weights=(0.25, 0.25, 0.25, 0.25),
        )
    return float(score) if score > 1e-12 else 0.0

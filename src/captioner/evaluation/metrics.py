"""Caption metrics.

* ``corpus_bleu4`` — fast NLTK corpus BLEU-4 used for early stopping during training.
* ``coco_scores`` — literature-standard ``pycocoevalcap`` metrics (BLEU-1..4, METEOR, ROUGE-L,
  CIDEr) on PTB-tokenised raw references (testing.md §4).
"""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

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


METRIC_KEYS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr")


@dataclass(frozen=True)
class CocoResult:
    """Corpus scores plus per-image BLEU-4 / CIDEr.

    Attributes:
        corpus: ``METRIC_KEYS`` → score (``METEOR`` is ``None`` without Java).
        bleu4_per_image: Sentence BLEU-4 per image.
        cider_per_image: Sentence CIDEr per image.
        tokenizer: ``"ptb"`` (Stanford PTB tokenizer, needs Java) or ``"simple"`` fallback.
        java: Whether Java was available.
    """

    corpus: dict[str, float | None]
    bleu4_per_image: list[float]
    cider_per_image: list[float]
    tokenizer: str
    java: bool


def java_available() -> bool:
    """True if a working ``java`` binary is on PATH (needed for PTB tokenizer and METEOR)."""
    java = shutil.which("java")
    if java is None:
        return False
    try:
        subprocess.run([java, "-version"], capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _simple_tokenize(data: dict[str, list[str]]) -> dict[str, list[str]]:
    return {k: [" ".join(tokenize(clean_caption(s))) for s in v] for k, v in data.items()}


def _ptb_tokenize(data: dict[str, list[str]]) -> dict[str, list[str]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    with contextlib.redirect_stdout(io.StringIO()):
        out: dict[str, list[str]] = PTBTokenizer().tokenize(
            {k: [{"caption": s} for s in v] for k, v in data.items()}
        )
    return out


def coco_scores(
    hypotheses: Sequence[str], references: Sequence[Sequence[str]], use_java: bool | None = None
) -> CocoResult:
    """Score captions with ``pycocoevalcap`` (BLEU-1..4, METEOR, ROUGE-L, CIDEr).

    References and hypotheses are lowercased and PTB-tokenised (testing.md §4). Pass the **raw**
    references. Without Java the tokenizer falls back to ``clean_caption`` and METEOR is skipped.

    Args:
        hypotheses: One predicted sentence per image.
        references: Raw reference sentences per image.
        use_java: Force (``True``/``False``) or auto-detect (``None``) Java.

    Returns:
        ``CocoResult``.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("need the same, non-zero number of hypotheses and reference sets")
    java = java_available() if use_java is None else use_java
    tok = _ptb_tokenize if java else _simple_tokenize
    keys = [str(i) for i in range(len(hypotheses))]
    gts = tok({k: [r.lower() for r in refs] for k, refs in zip(keys, references, strict=True)})
    res = tok({k: [h.lower()] for k, h in zip(keys, hypotheses, strict=True)})
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, bleu_img = Bleu(4).compute_score(gts, res)
        rouge, _ = Rouge().compute_score(gts, res)
        cider, cider_img = Cider().compute_score(gts, res)
        meteor = _meteor(gts, res) if java else None
    corpus: dict[str, float | None] = {f"BLEU-{n + 1}": float(bleu[n]) for n in range(4)}
    corpus.update({"METEOR": meteor, "ROUGE-L": float(rouge), "CIDEr": float(cider)})
    return CocoResult(
        corpus=corpus,
        bleu4_per_image=[float(x) for x in bleu_img[3]],
        cider_per_image=[float(x) for x in cider_img],
        tokenizer="ptb" if java else "simple",
        java=java,
    )


def _meteor(gts: dict[str, list[str]], res: dict[str, list[str]]) -> float:
    from pycocoevalcap.meteor.meteor import Meteor

    scorer = Meteor()  # spawns a Java subprocess; its __del__ terminates it
    score, _ = scorer.compute_score(gts, res)
    del scorer
    return float(score)

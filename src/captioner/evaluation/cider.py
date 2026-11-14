"""Fast, corpus-reusable CIDEr-D (Vedantam et al., 2015), identical to ``pycocoevalcap``.

``pycocoevalcap.Cider`` recomputes document frequencies from whatever batch it is given and needs
string round-trips. Training needs CIDEr thousands of times per epoch — as a model-selection
metric and as the reward of self-critical sequence training (ADR-026) — against a *fixed* corpus,
so this implementation fixes the document frequencies once and caches every reference vector.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

Ngram = tuple[str, ...]
_Vec = tuple[list[dict[Ngram, float]], list[float], int]


def ngram_counts(tokens: Sequence[str], n: int = 4) -> Counter[Ngram]:
    """All 1..n-gram counts of a token list."""
    counts: Counter[Ngram] = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            counts[tuple(tokens[i : i + k])] += 1
    return counts


class CiderD:
    """CIDEr-D with document frequencies from a fixed reference corpus.

    Args:
        corpus: References per image (tokenised); defines the IDF statistics.
        n: Maximum n-gram order.
        sigma: Gaussian length-penalty width.
    """

    def __init__(
        self, corpus: Sequence[Sequence[Sequence[str]]], n: int = 4, sigma: float = 6.0
    ) -> None:
        self.n = n
        self.sigma = sigma
        self.df: defaultdict[Ngram, float] = defaultdict(float)
        for refs in corpus:
            for ngram in {g for ref in refs for g in ngram_counts(ref, n)}:
                self.df[ngram] += 1
        self.ref_len = math.log(float(max(1, len(corpus))))
        self._cache: dict[int, list[_Vec]] = {}
        self._corpus = corpus

    def _vec(self, counts: Counter[Ngram]) -> _Vec:
        vec: list[dict[Ngram, float]] = [{} for _ in range(self.n)]
        norm = [0.0] * self.n
        length = 0
        for ngram, tf in counts.items():
            k = len(ngram) - 1
            value = float(tf) * (self.ref_len - math.log(max(1.0, self.df.get(ngram, 0.0))))
            vec[k][ngram] = value
            norm[k] += value * value
            if k == 1:
                length += tf
        return vec, [math.sqrt(x) for x in norm], length

    def _sim(self, hyp: _Vec, ref: _Vec) -> float:
        (vh, nh, lh), (vr, nr, lr) = hyp, ref
        penalty = math.exp(-((lh - lr) ** 2) / (2 * self.sigma**2))
        total = 0.0
        for k in range(self.n):
            val = sum(min(v, vr[k].get(g, 0.0)) * vr[k].get(g, 0.0) for g, v in vh[k].items())
            if nh[k] and nr[k]:
                val /= nh[k] * nr[k]
            total += val * penalty
        return total

    def _ref_vecs(self, refs: Sequence[Sequence[str]], key: int | None) -> list[_Vec]:
        if key is not None and key in self._cache:
            return self._cache[key]
        vecs = [self._vec(ngram_counts(r, self.n)) for r in refs]
        if key is not None:
            self._cache[key] = vecs
        return vecs

    def score(
        self, hyp: Sequence[str], refs: Sequence[Sequence[str]], key: int | None = None
    ) -> float:
        """CIDEr-D of one hypothesis. ``key`` caches the reference vectors (e.g. image index)."""
        hv = self._vec(ngram_counts(hyp, self.n))
        ref_vecs = self._ref_vecs(refs, key)
        if not ref_vecs:
            return 0.0
        return 10.0 * sum(self._sim(hv, rv) for rv in ref_vecs) / (self.n * len(ref_vecs))

    def score_corpus_item(self, hyp: Sequence[str], index: int) -> float:
        """Score against the references of corpus item ``index`` (cached)."""
        return self.score(hyp, self._corpus[index], key=index)

    def corpus_score(self, hyps: Sequence[Sequence[str]]) -> float:
        """Mean CIDEr-D of one hypothesis per corpus item."""
        if not hyps:
            return 0.0
        return sum(self.score_corpus_item(h, i) for i, h in enumerate(hyps)) / len(hyps)

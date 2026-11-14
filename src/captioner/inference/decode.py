"""Model-agnostic decoding: greedy and batched beam search (FR-5).

Both only call ``model.init_state`` and ``model.step``. ``<pad>``, ``<start>`` and ``<unk>`` are
never emitted; decoding stops at ``<end>`` or after ``max_len`` generated tokens (``<end>``
included).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from captioner.data.vocab import END_IDX, PAD_IDX, SPECIAL_IDS, START_IDX, UNK_IDX
from captioner.models.captioner import CaptionModel, DecoderState

_BLOCKED = (PAD_IDX, START_IDX, UNK_IDX)
NEG_INF = float("-inf")


@dataclass(frozen=True)
class Hypothesis:
    """One decoded caption.

    Attributes:
        ids: Generated word ids, without ``<start>``/``<end>``.
        logprob: Sum of token log-probabilities (``<end>`` included when emitted).
        score: Length-normalised score used for ranking (``logprob / n**alpha``).
        attention: ``(len(ids), L)`` attention weights per word, or ``None``.
        finished: ``True`` if ``<end>`` was emitted before the length cap.
    """

    ids: tuple[int, ...]
    logprob: float
    score: float
    attention: torch.Tensor | None
    finished: bool

    @property
    def mean_logprob(self) -> float:
        """Average log-probability per generated token (confidence hint)."""
        return self.logprob / max(1, len(self.ids) + int(self.finished))


def length_penalty(n_tokens: torch.Tensor | int, alpha: float) -> torch.Tensor | float:
    """``n ** alpha`` normaliser; ``alpha = 0`` disables normalisation."""
    if isinstance(n_tokens, int):
        return float(float(max(n_tokens, 1)) ** alpha)
    penalty: torch.Tensor = n_tokens.clamp(min=1).float() ** alpha
    return penalty


def _block(logp: torch.Tensor) -> torch.Tensor:
    logp = logp.clone()
    logp[:, list(_BLOCKED)] = NEG_INF
    return logp


@dataclass(frozen=True)
class RepeatPolicy:
    """Anti-hallucination constraints on repetition (ADR-022).

    Attributes:
        ngram: Ban any n-gram that already occurs in the hypothesis (0 disables).
        content_mask: ``(V,)`` bool, True for content words that may appear only once
            (``None`` disables), e.g. no "a dog and a dog".
    """

    ngram: int = 0
    content_mask: torch.Tensor | None = None


def build_repeat_policy(itos: tuple[str, ...], ngram: int, block_words: bool) -> RepeatPolicy:
    """Policy for a vocabulary: content words are everything except specials and function words."""
    from captioner.data.text import FUNCTION_WORDS

    mask = None
    if block_words:
        mask = torch.tensor(
            [i not in SPECIAL_IDS and w not in FUNCTION_WORDS for i, w in enumerate(itos)]
        )
    return RepeatPolicy(ngram=ngram, content_mask=mask)


def _repeat_penalty(
    logp: torch.Tensor, history: torch.Tensor, policy: RepeatPolicy | None
) -> torch.Tensor:
    """Set log-probs of tokens that would repeat a banned n-gram or content word to -inf."""
    if policy is None or history.shape[1] == 0:
        return logp
    penalty = torch.zeros_like(logp)
    n = policy.ngram
    if n >= 2 and history.shape[1] >= n - 1:
        prefix = history[:, history.shape[1] - (n - 1) :]  # (R, n-1)
        if history.shape[1] >= n:
            windows = history.unfold(1, n, 1)  # (R, W, n)
            match = (windows[:, :, :-1] == prefix.unsqueeze(1)).all(-1)  # (R, W)
            vals = torch.where(match, NEG_INF, 0.0).to(logp.dtype)
            penalty.scatter_reduce_(1, windows[:, :, -1], vals, reduce="amin")
    if policy.content_mask is not None:
        mask = policy.content_mask.to(logp.device)
        seen = torch.zeros_like(logp, dtype=torch.bool).scatter_(1, history, True)  # (R, V)
        penalty = penalty.masked_fill(seen & mask.unsqueeze(0), NEG_INF)
    return logp + penalty


@torch.inference_mode()
def greedy_decode(
    model: CaptionModel,
    feats: torch.Tensor,
    max_len: int,
    length_alpha: float = 0.0,
    repeat: RepeatPolicy | None = None,
) -> list[Hypothesis]:
    """Pick the most likely token at every step.

    Args:
        model: Any ``CaptionModel``.
        feats: Encoder output for B images.
        max_len: Hard cap on generated tokens including ``<end>``.
        length_alpha: Only used to fill ``Hypothesis.score``.
        repeat: Optional repetition constraints.

    Returns:
        One hypothesis per image.
    """
    b = feats.shape[0]
    state = model.init_state(feats)
    tokens = torch.full((b,), START_IDX, dtype=torch.long, device=feats.device)
    out_ids = torch.full((b, max_len), PAD_IDX, dtype=torch.long)
    logprob = torch.zeros(b)
    lengths = torch.zeros(b, dtype=torch.long)
    done = torch.zeros(b, dtype=torch.bool)
    alphas: list[torch.Tensor] = []
    hist = torch.empty((b, 0), dtype=torch.long, device=feats.device)
    for t in range(max_len):
        logp, state, alpha = model.step(tokens, state)
        best_lp, tokens = _repeat_penalty(_block(logp.float()), hist, repeat).max(dim=-1)  # (B,)
        hist = torch.cat([hist, tokens.unsqueeze(1)], dim=1)
        active = ~done
        out_ids[active, t] = tokens[active].cpu()
        logprob[active] += best_lp[active].float().cpu()
        lengths[active] += 1
        if alpha is not None:
            alphas.append(alpha.float().cpu())
        done |= (tokens == END_IDX).cpu()
        if bool(done.all()):
            break
    return [_to_hyp(out_ids[i], float(logprob[i]), alphas, i, length_alpha) for i in range(b)]


def _to_hyp(
    ids: torch.Tensor, logprob: float, alphas: list[torch.Tensor], row: int, alpha: float
) -> Hypothesis:
    seq = ids.tolist()
    finished = END_IDX in seq
    words = seq[: seq.index(END_IDX)] if finished else [i for i in seq if i != PAD_IDX]
    att = torch.stack([a[row] for a in alphas[: len(words)]]) if alphas and words else None
    n = len(words) + int(finished)
    return Hypothesis(
        ids=tuple(words),
        logprob=logprob,
        score=logprob / float(length_penalty(n, alpha)),
        attention=att,
        finished=finished,
    )


def _expand(state: DecoderState, k: int) -> DecoderState:
    return {key: v.repeat_interleave(k, dim=0) for key, v in state.items()}


def _reorder(state: DecoderState, index: torch.Tensor) -> DecoderState:
    return {key: v.index_select(0, index) for key, v in state.items()}


@torch.inference_mode()
def beam_search(
    model: CaptionModel,
    feats: torch.Tensor,
    beam: int,
    max_len: int,
    length_alpha: float = 0.7,
    repeat: RepeatPolicy | None = None,
) -> list[Hypothesis]:
    """Batched beam search with length normalisation.

    Beams are pruned on cumulative log-probability; the final caption per image is the beam
    with the best ``logprob / n**length_alpha`` (``n`` = generated tokens incl. ``<end>``).
    Finished beams stay in the beam with a frozen score, so ``beam=1`` reduces to greedy.

    Args:
        model: Any model implementing ``init_state`` and ``step``.
        feats: Encoder output, ``(B, L, C)`` or ``(B, C)``.
        beam: Beam width; 1 == greedy.
        max_len: Hard cap on generated tokens including ``<end>``.
        length_alpha: Length-normalisation exponent (0 disables it).
        repeat: Optional repetition constraints (n-gram / content-word blocking).

    Returns:
        One best hypothesis per image, specials removed.
    """
    b, k, dev = feats.shape[0], beam, feats.device
    state = _expand(model.init_state(feats), k)
    scores = torch.full((b, k), NEG_INF, device=dev)
    scores[:, 0] = 0.0  # only one live beam at t=0 so the k beams don't start identical
    tokens = torch.full((b * k,), START_IDX, dtype=torch.long, device=dev)
    history = torch.empty((b * k, 0), dtype=torch.long, device=dev)
    att_hist: torch.Tensor | None = None
    finished = torch.zeros(b * k, dtype=torch.bool, device=dev)
    lengths = torch.zeros(b * k, dtype=torch.long, device=dev)
    for _ in range(max_len):
        logp, state, alpha = model.step(tokens, state)
        logp = _repeat_penalty(_block(logp.float()), history, repeat)  # (B*k, V)
        vocab_size = logp.shape[1]
        logp[finished] = NEG_INF
        logp[finished, PAD_IDX] = 0.0  # finished beams carry over unchanged
        cand = (scores.view(-1, 1) + logp).view(b, k * vocab_size)  # (B, k*V)
        scores, flat = cand.topk(k, dim=1)  # (B, k)
        src = (torch.arange(b, device=dev).unsqueeze(1) * k + flat // vocab_size).view(-1)
        tokens = (flat % vocab_size).view(-1)  # (B*k,)
        state = _reorder(state, src)
        was_finished = finished.index_select(0, src)
        history = torch.cat([history.index_select(0, src), tokens.unsqueeze(1)], dim=1)
        if alpha is not None:
            step_att = alpha.float().index_select(0, src).unsqueeze(1)  # (B*k, 1, L)
            att_hist = (
                step_att
                if att_hist is None
                else torch.cat([att_hist.index_select(0, src), step_att], 1)
            )
        lengths = lengths.index_select(0, src) + (~was_finished).long()
        finished = was_finished | (tokens == END_IDX)
        if bool(finished.all()):
            break
    norm = scores.view(-1) / length_penalty(lengths, length_alpha)  # (B*k,)
    best = norm.view(b, k).argmax(dim=1) + torch.arange(b, device=dev) * k  # (B,)
    return [
        _beam_hyp(
            history[i],
            float(scores.view(-1)[i]),
            float(norm[i]),
            att_hist,
            int(i),
            bool(finished[i]),
        )
        for i in best.tolist()
    ]


def _beam_hyp(
    seq: torch.Tensor,
    logprob: float,
    score: float,
    att: torch.Tensor | None,
    row: int,
    finished: bool,
) -> Hypothesis:
    ids = seq.tolist()
    words = ids[: ids.index(END_IDX)] if END_IDX in ids else [i for i in ids if i != PAD_IDX]
    attention = att[row, : len(words)].cpu() if att is not None and words else None
    return Hypothesis(tuple(words), logprob, score, attention, finished)


def decode(
    model: CaptionModel,
    feats: torch.Tensor,
    beam: int,
    max_len: int,
    length_alpha: float = 0.7,
    repeat: RepeatPolicy | None = None,
) -> list[Hypothesis]:
    """Greedy when ``beam == 1`` (scores still length-normalised), beam search otherwise."""
    if beam == 1:
        return greedy_decode(model, feats, max_len, length_alpha, repeat)
    return beam_search(model, feats, beam, max_len, length_alpha, repeat)

"""Model-agnostic decoding: greedy and batched beam search (FR-5).

Both only call ``model.init_state`` and ``model.step``. ``<pad>``, ``<start>`` and ``<unk>`` are
never emitted; decoding stops at ``<end>`` or after ``max_len`` generated tokens (``<end>``
included).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from captioner.data.vocab import END_IDX, PAD_IDX, START_IDX, UNK_IDX
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


@torch.inference_mode()
def greedy_decode(
    model: CaptionModel, feats: torch.Tensor, max_len: int, length_alpha: float = 0.0
) -> list[Hypothesis]:
    """Pick the most likely token at every step.

    Args:
        model: Any ``CaptionModel``.
        feats: Encoder output for B images.
        max_len: Hard cap on generated tokens including ``<end>``.
        length_alpha: Only used to fill ``Hypothesis.score``.

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
    for t in range(max_len):
        logp, state, alpha = model.step(tokens, state)
        best_lp, tokens = _block(logp).max(dim=-1)  # (B,), (B,)
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


__all__ = ["DecoderState", "Hypothesis", "greedy_decode", "length_penalty"]

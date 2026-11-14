"""The ``CaptionModel`` interface, shared loss, and model factory (architecture.md §5.4).

Decoding only ever calls ``init_state`` and ``step``, so greedy and beam search are written once
for every model. ``DecoderState`` is a dict of tensors whose first dimension is the batch, so beam
search can reorder any model's state with ``index_select``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from captioner.config import ModelConfig
from captioner.data.vocab import PAD_IDX

DecoderState = dict[str, torch.Tensor]
FeatureKind = Literal["pooled", "spatial"]


@dataclass(frozen=True)
class ModelOutput:
    """Teacher-forced predictions for ``captions[:, 1:]``.

    Attributes:
        logits: ``(B, T-1, V)``.
        alphas: Attention weights ``(B, T-1, L)`` (zero on padded steps) or ``None``.
    """

    logits: torch.Tensor
    alphas: torch.Tensor | None = None


@dataclass(frozen=True)
class LossOutput:
    """Total optimised loss and its parts (``reg`` is 0 for models without attention)."""

    total: torch.Tensor
    ce: torch.Tensor
    reg: torch.Tensor


@runtime_checkable
class CaptionModel(Protocol):
    """Interface every decoder implements."""

    feature_kind: FeatureKind

    def forward(
        self, feats: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Teacher-forced pass. ``captions`` is ``(B, T)`` starting with ``<start>``."""
        ...

    def init_state(self, feats: torch.Tensor) -> DecoderState:
        """Decoder state before the first token."""
        ...

    def step(
        self, prev_tokens: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        """One decoding step: log-probs ``(B, V)``, new state, optional attention ``(B, L)``."""
        ...

    def __call__(
        self, feats: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput: ...


def step_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    """Boolean ``(B, steps)`` mask of valid prediction steps (``length - 1`` per caption)."""
    pos = torch.arange(steps, device=lengths.device).unsqueeze(0)  # (1, T-1)
    return pos < (lengths - 1).unsqueeze(1)  # (B, T-1)


def caption_loss(
    output: ModelOutput, captions: torch.Tensor, lengths: torch.Tensor, alpha_c: float
) -> LossOutput:
    """Cross-entropy over real tokens plus the doubly-stochastic attention regulariser.

    The regulariser is ``alpha_c * mean_b mean_i (1 - sum_t alpha_{t,i})^2`` (ADR-010).

    Args:
        output: Model output for ``captions``.
        captions: ``(B, T)`` ids; targets are ``captions[:, 1:]``.
        lengths: ``(B,)`` caption lengths incl. ``<start>``/``<end>``.
        alpha_c: Regulariser weight λ.

    Returns:
        ``LossOutput``.
    """
    targets = captions[:, 1:]  # (B, T-1)
    logits = output.logits
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD_IDX
    )
    reg = torch.zeros((), device=ce.device)
    if output.alphas is not None and alpha_c > 0:
        coverage = output.alphas.sum(dim=1)  # (B, L)
        reg = alpha_c * ((1.0 - coverage) ** 2).mean()
    return LossOutput(total=ce + reg, ce=ce, reg=reg)


def ensure_pooled(feats: torch.Tensor) -> torch.Tensor:
    """Accept ``(B, C)`` or ``(B, L, C)`` features; return ``(B, C)``."""
    return feats.mean(dim=1) if feats.dim() == 3 else feats


def build_model(cfg: ModelConfig, vocab_size: int) -> nn.Module:
    """Instantiate the decoder named by ``cfg.kind``."""
    if cfg.kind == "merge":
        from captioner.models.merge import MergeModel

        return MergeModel(cfg, vocab_size)
    from captioner.models.attention import AttentionModel

    return AttentionModel(cfg, vocab_size)


def feature_kind_for(cfg: ModelConfig) -> FeatureKind:
    """Which cached feature file a model kind trains on."""
    return "pooled" if cfg.kind == "merge" else "spatial"


def count_trainable(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ImageCaptioner(nn.Module):
    """Encoder + decoder operating on images (fine-tuning stage 2 and serving)."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_kind: FeatureKind = "spatial"

    def forward(
        self, images: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        out: ModelOutput = self.decoder(self.encoder(images), captions, lengths)
        return out

    def init_state(self, images: torch.Tensor) -> DecoderState:
        return cast(CaptionModel, self.decoder).init_state(self.encoder(images))

    def step(
        self, prev_tokens: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        return cast(CaptionModel, self.decoder).step(prev_tokens, state)

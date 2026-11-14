"""M1 — merge baseline (bug-fixed re-implementation of the reference model, architecture.md §5.2).

image (C) ─ Dropout ─ Linear(C→D) ─ ReLU ──────────────┐
                                                       add ─ Linear(D→D) ─ ReLU ─ Linear(D→V)
caption ─ Embedding(V,E,pad) ─ Dropout ─ LSTM(D) ──────┘

Trained as a sequence model: the LSTM runs over the whole shifted caption and the image
projection is added at every step (instead of the reference's exploded prefix samples).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from captioner.config import ModelConfig
from captioner.data.vocab import PAD_IDX
from captioner.models.captioner import DecoderState, FeatureKind, ModelOutput, ensure_pooled


class MergeModel(nn.Module):
    """Merge-architecture decoder over pooled image features."""

    feature_kind: FeatureKind = "pooled"

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.img_drop = nn.Dropout(cfg.dropout)
        self.img_proj = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.decoder_dim, batch_first=True)
        self.fc1 = nn.Linear(cfg.decoder_dim, cfg.decoder_dim)
        self.fc2 = nn.Linear(cfg.decoder_dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embed.weight[PAD_IDX].zero_()
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)

    def _image(self, feats: torch.Tensor) -> torch.Tensor:
        img: torch.Tensor = F.relu(self.img_proj(self.img_drop(ensure_pooled(feats))))  # (B, D)
        return img

    def _head(self, img: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.fc2(F.relu(self.fc1(img + hidden)))
        return logits

    def forward(
        self, feats: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Teacher-forced logits ``(B, T-1, V)`` for ``captions[:, 1:]``.

        Args:
            feats: ``(B, C)`` pooled (or ``(B, L, C)``, pooled here) features.
            captions: ``(B, T)`` ids starting with ``<start>``.
            lengths: ``(B,)`` lengths; padded positions are ignored by the loss.
        """
        img = self._image(feats)  # (B, D)
        emb = self.emb_drop(self.embed(captions[:, :-1]))  # (B, T-1, E)
        hidden, _ = self.lstm(emb)  # (B, T-1, D)
        return ModelOutput(logits=self._head(img.unsqueeze(1), hidden))

    def init_state(self, feats: torch.Tensor) -> DecoderState:
        img = self._image(feats)
        zeros = img.new_zeros(img.shape[0], self.lstm.hidden_size)
        return {"img": img, "h": zeros, "c": zeros.clone()}

    def step(
        self, prev_tokens: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        """One step: ``prev_tokens`` ``(B,)`` → log-probs ``(B, V)``."""
        emb = self.emb_drop(self.embed(prev_tokens)).unsqueeze(1)  # (B, 1, E)
        out, (h, c) = self.lstm(emb, (state["h"].unsqueeze(0), state["c"].unsqueeze(0)))
        logits = self._head(state["img"], out.squeeze(1))  # (B, V)
        new_state = {"img": state["img"], "h": h.squeeze(0), "c": c.squeeze(0)}
        return F.log_softmax(logits, dim=-1), new_state, None

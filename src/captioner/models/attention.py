"""M2 — soft-attention LSTM decoder ("Show, Attend and Tell"), architecture.md §5.3.

Per step t:  e_t = w·tanh(W_a a + W_h h_{t-1});  α_t = softmax(e_t);
z_t = σ(f_β(h_{t-1})) ⊙ Σ_i α_{t,i} a_i;  h_t, c_t = LSTMCell([E(y_{t-1}); z_t]);
p_t = softmax(W_o Dropout(h_t)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from captioner.config import ModelConfig
from captioner.data.vocab import PAD_IDX
from captioner.models.captioner import DecoderState, FeatureKind, ModelOutput, step_mask


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention over image regions."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, attention_dim)
        self.dec_proj = nn.Linear(decoder_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(
        self, feats: torch.Tensor, feats_proj: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context ``(B, C)`` and weights ``(B, L)``.

        Args:
            feats: ``(B, L, C)`` region features.
            feats_proj: ``enc_proj(feats)`` ``(B, L, A)``, computed once per sequence.
            h: ``(B, D)`` previous hidden state.
        """
        e = self.score(torch.tanh(feats_proj + self.dec_proj(h).unsqueeze(1))).squeeze(2)  # (B, L)
        alpha = F.softmax(e, dim=1)  # (B, L)
        context = (alpha.unsqueeze(2) * feats).sum(dim=1)  # (B, C)
        return context, alpha


class AttentionModel(nn.Module):
    """Soft-attention LSTM decoder over spatial features ``(B, 49, 2048)``."""

    feature_kind: FeatureKind = "spatial"

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        c, d, e = cfg.encoder_dim, cfg.decoder_dim, cfg.embed_dim
        self.attention = AdditiveAttention(c, d, cfg.attention_dim)
        self.embed = nn.Embedding(vocab_size, e, padding_idx=PAD_IDX)
        self.init_h = nn.Linear(c, d)
        self.init_c = nn.Linear(c, d)
        self.f_beta = nn.Linear(d, c)
        self.cell = nn.LSTMCell(e + c, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(d, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embed.weight[PAD_IDX].zero_()
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    def init_state(self, feats: torch.Tensor) -> DecoderState:
        """Initial ``h``/``c`` from the mean feature; cache projected features."""
        mean = feats.mean(dim=1)  # (B, C)
        return {
            "h": self.init_h(mean),
            "c": self.init_c(mean),
            "feats": feats,
            "feats_proj": self.attention.enc_proj(feats),  # (B, L, A)
        }

    def _advance(
        self, emb: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor]:
        context, alpha = self.attention(state["feats"], state["feats_proj"], state["h"])
        gate = torch.sigmoid(self.f_beta(state["h"]))  # (B, C)
        h, c = self.cell(torch.cat([emb, gate * context], dim=1), (state["h"], state["c"]))
        logits = self.fc(self.drop(h))  # (B, V)
        return logits, {**state, "h": h, "c": c}, alpha

    def forward(
        self, feats: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Teacher-forced logits ``(B, T-1, V)`` and attention ``(B, T-1, L)``.

        Finished sequences keep running but their outputs are masked: logits are ignored by the
        loss via ``<pad>`` targets and their attention weights are zeroed.
        """
        steps = captions.shape[1] - 1
        state = self.init_state(feats)
        emb = self.drop(self.embed(captions[:, :-1]))  # (B, T-1, E)
        logits, alphas = [], []
        for t in range(steps):
            step_logits, state, alpha = self._advance(emb[:, t], state)
            logits.append(step_logits)
            alphas.append(alpha)
        mask = step_mask(lengths, steps)  # (B, T-1)
        alpha_seq = torch.stack(alphas, dim=1) * mask.unsqueeze(2)  # (B, T-1, L)
        return ModelOutput(logits=torch.stack(logits, dim=1), alphas=alpha_seq)

    def step(
        self, prev_tokens: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        """One step: tokens ``(B,)`` → log-probs ``(B, V)``, state, attention ``(B, L)``."""
        logits, new_state, alpha = self._advance(self.drop(self.embed(prev_tokens)), state)
        return F.log_softmax(logits, dim=-1), new_state, alpha

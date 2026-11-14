"""M3 — Transformer decoder over the 7×7 grid features (FR-14, ADR-027).

grid (B, L, C) ─ Linear + LayerNorm + region positions ─ [Transformer encoder × E] ─┐
tokens ─ tied Embedding + positions ─ [pre-norm decoder layer × N] ─ LayerNorm ─ tied Linear
                                       (masked self-attn, cross-attn on the grid, GELU FFN)

Cross-attention weights of the last layer (averaged over heads) are returned as ``alphas``, so
heat-maps work exactly as for M2. ``step()`` re-runs the decoder on the cached prefix, so greedy
and beam search are shared with the LSTM models unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from captioner.config import ModelConfig
from captioner.data.vocab import PAD_IDX
from captioner.models.captioner import DecoderState, FeatureKind, ModelOutput, step_mask


class DecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer that also returns cross-attention weights."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal: torch.Tensor,
        pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        sa, _ = self.self_attn(h, h, h, attn_mask=causal, key_padding_mask=pad, need_weights=False)
        x = x + self.drop(sa)
        h = self.norm2(x)
        ca, weights = self.cross_attn(
            h, memory, memory, need_weights=True, average_attn_weights=True
        )
        x = x + self.drop(ca)
        x = x + self.drop(self.ff(self.norm3(x)))
        return x, weights  # (B, T, D), (B, T, L)


class TransformerCaptioner(nn.Module):
    """Transformer caption decoder implementing the ``CaptionModel`` protocol."""

    feature_kind: FeatureKind = "spatial"

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        d, heads = cfg.decoder_dim, cfg.transformer_heads
        self.feat_proj = nn.Sequential(nn.Linear(cfg.encoder_dim, d), nn.LayerNorm(d))
        self.region_pos = nn.Parameter(torch.zeros(1, cfg.num_regions, d))
        self.grid_encoder: nn.Module = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d, heads, cfg.transformer_ff, cfg.dropout, activation="gelu",
                    batch_first=True, norm_first=True,
                ),
                cfg.transformer_enc_layers,
                enable_nested_tensor=False,
            )
            if cfg.transformer_enc_layers > 0
            else nn.Identity()
        )  # fmt: skip
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD_IDX)
        self.pos = nn.Embedding(cfg.max_positions, d)
        self.layers = nn.ModuleList(
            DecoderLayer(d, heads, cfg.transformer_ff, cfg.dropout)
            for _ in range(cfg.transformer_layers)
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = d**0.5
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.embed.embedding_dim**-0.5)
        with torch.no_grad():
            self.embed.weight[PAD_IDX].zero_()
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.region_pos, std=0.02)
        self.out.weight = self.embed.weight  # weight tying
        nn.init.zeros_(self.out.bias)

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """``(B, L, C)`` grid features → memory ``(B, L, D)``."""
        memory: torch.Tensor = self.grid_encoder(self.drop(self.feat_proj(feats) + self.region_pos))
        return memory

    def _decode(
        self, memory: torch.Tensor, tokens: torch.Tensor, pad: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = tokens.shape[1]
        positions = torch.arange(t, device=tokens.device)
        x = self.drop(self.embed(tokens) * self.scale + self.pos(positions))  # (B, T, D)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tokens.device), diagonal=1)
        weights = memory.new_zeros(tokens.shape[0], t, memory.shape[1])
        for layer in self.layers:
            x, weights = layer(x, memory, causal, pad)
        return self.out(self.norm(x)), weights  # (B, T, V), (B, T, L)

    def forward(
        self, feats: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> ModelOutput:
        """Teacher-forced logits ``(B, T-1, V)`` and cross-attention ``(B, T-1, L)``."""
        inputs = captions[:, :-1]
        valid = step_mask(lengths, inputs.shape[1])  # (B, T-1)
        logits, alphas = self._decode(self.encode(feats), inputs, pad=~valid)
        return ModelOutput(logits=logits, alphas=alphas * valid.unsqueeze(2))

    def init_state(self, feats: torch.Tensor) -> DecoderState:
        memory = self.encode(feats)
        return {"memory": memory, "tokens": memory.new_zeros(memory.shape[0], 0, dtype=torch.long)}

    def step(
        self, prev_tokens: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        """Append ``prev_tokens`` to the prefix and predict the next token."""
        tokens = torch.cat([state["tokens"], prev_tokens.unsqueeze(1)], dim=1)  # (B, t)
        logits, alphas = self._decode(state["memory"], tokens)
        return F.log_softmax(logits[:, -1], dim=-1), {**state, "tokens": tokens}, alphas[:, -1]

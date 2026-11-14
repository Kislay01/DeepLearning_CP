"""ADR-027 / FR-14 — Transformer decoder implements the CaptionModel protocol."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from captioner.config import ModelConfig
from captioner.data.vocab import END_IDX, PAD_IDX, START_IDX
from captioner.inference.decode import beam_search, greedy_decode
from captioner.models.captioner import (
    CaptionModel,
    build_model,
    caption_loss,
    count_trainable,
    feature_kind_for,
)
from captioner.models.transformer import TransformerCaptioner

V = 30
CFG = ModelConfig.model_validate(
    {"kind": "transformer", "decoder_dim": 32, "encoder_dim": 16, "transformer_layers": 2,
     "transformer_heads": 4, "transformer_ff": 64, "transformer_enc_layers": 1, "dropout": 0.1}
)  # fmt: skip


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    lengths = torch.tensor([7, 5, 4])
    caps = torch.full((3, 7), PAD_IDX, dtype=torch.long)
    for i, n in enumerate(lengths.tolist()):
        caps[i, 0], caps[i, n - 1] = START_IDX, END_IDX
        caps[i, 1 : n - 1] = torch.randint(4, V, (n - 2,))
    return torch.randn(3, 49, 16), caps, lengths


def test_protocol_shapes_and_alphas() -> None:
    model = build_model(CFG, V).eval()  # attention dropout off so weights sum to 1
    assert isinstance(model, TransformerCaptioner) and isinstance(model, CaptionModel)
    assert feature_kind_for(CFG) == "spatial"
    feats, caps, lengths = _batch()
    out = model(feats, caps, lengths)
    assert (
        out.logits.shape == (3, 6, V) and out.alphas is not None and out.alphas.shape == (3, 6, 49)
    )
    valid = torch.arange(6).unsqueeze(0) < (lengths - 1).unsqueeze(1)
    torch.testing.assert_close(out.alphas[valid].sum(-1), torch.ones(int(valid.sum())))
    assert (out.alphas[~valid] == 0).all()
    assert model.out.weight is model.embed.weight  # tied


def test_padding_invariance_and_step_consistency() -> None:
    model = build_model(CFG, V).eval()
    feats, caps, lengths = _batch()
    padded = torch.cat([caps, torch.full((3, 3), PAD_IDX, dtype=torch.long)], 1)
    with torch.no_grad():
        a = caption_loss(model(feats, caps, lengths), caps, lengths, 0.0)
        b = caption_loss(model(feats, padded, lengths), padded, lengths, 0.0)
        torch.testing.assert_close(a.ce, b.ce)
        out = model(feats, caps, lengths)
        state = model.init_state(feats)
        for t in range(caps.shape[1] - 1):
            logp, state, alpha = model.step(caps[:, t], state)
            valid = t < lengths - 1
            torch.testing.assert_close(
                logp[valid], F.log_softmax(out.logits[valid, t], -1), atol=1e-5, rtol=1e-5
            )
            assert alpha is not None
            torch.testing.assert_close(alpha[valid], out.alphas[valid, t], atol=1e-5, rtol=1e-5)  # type: ignore[index]


def test_gradients_and_decoding() -> None:
    model = build_model(CFG, V).train()
    feats, caps, lengths = _batch()
    caption_loss(model(feats, caps, lengths), caps, lengths, 0.0, 0.1).total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
    model.eval()
    g = greedy_decode(model, feats, 10)  # type: ignore[arg-type]
    b1 = beam_search(model, feats, 1, 10)  # type: ignore[arg-type]
    assert [h.ids for h in g] == [h.ids for h in b1]
    batched = beam_search(model, feats, 3, 10)  # type: ignore[arg-type]
    single = [beam_search(model, feats[i : i + 1], 3, 10)[0] for i in range(3)]  # type: ignore[arg-type]
    assert [h.ids for h in batched] == [h.ids for h in single]
    full = TransformerCaptioner(ModelConfig(kind="transformer", decoder_dim=512, dropout=0.1), 2533)
    assert 10e6 < count_trainable(full) < 20e6
    no_enc = build_model(CFG.model_copy(update={"transformer_enc_layers": 0}), V)
    assert no_enc(feats, caps, lengths).logits.shape == (3, 6, V)

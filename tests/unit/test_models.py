from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from captioner.config import ModelConfig
from captioner.data.vocab import END_IDX, PAD_IDX, START_IDX
from captioner.models.attention import AttentionModel
from captioner.models.captioner import (
    CaptionModel,
    ImageCaptioner,
    build_model,
    caption_loss,
    count_trainable,
    feature_kind_for,
    step_mask,
)
from captioner.models.encoder import ResNetEncoder
from captioner.models.merge import MergeModel

V = 30
SMALL = {"embed_dim": 8, "decoder_dim": 12, "attention_dim": 10, "encoder_dim": 16, "dropout": 0.0}


def _cfg(kind: str, **kw: object) -> ModelConfig:
    return ModelConfig.model_validate({"kind": kind, **SMALL, **kw})


def _batch(
    b: int = 3, regions: int = 49, c: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    lengths = torch.tensor([7, 5, 4])[:b]
    t = int(lengths.max())
    caps = torch.full((b, t), PAD_IDX, dtype=torch.long)
    for i, n in enumerate(lengths.tolist()):
        caps[i, 0] = START_IDX
        caps[i, 1 : n - 1] = torch.randint(4, V, (n - 2,))
        caps[i, n - 1] = END_IDX
    return torch.randn(b, regions, c), caps, lengths


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_forward_shapes(kind: str) -> None:
    model = build_model(_cfg(kind), V)
    assert isinstance(model, CaptionModel)
    feats, caps, lengths = _batch()
    out = model(feats, caps, lengths)
    assert out.logits.shape == (3, caps.shape[1] - 1, V)
    if kind == "attention":
        assert out.alphas is not None and out.alphas.shape == (3, caps.shape[1] - 1, 49)
        mask = step_mask(lengths, caps.shape[1] - 1)
        sums = out.alphas.sum(-1)
        torch.testing.assert_close(sums[mask], torch.ones(int(mask.sum())))
        assert (sums[~mask] == 0).all()
    else:
        assert out.alphas is None
        # M1 also accepts pooled features directly.
        assert model(feats.mean(1), caps, lengths).logits.shape == out.logits.shape


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_loss_ignores_padding(kind: str) -> None:
    model = build_model(_cfg(kind), V).eval()
    feats, caps, lengths = _batch()
    padded = torch.cat([caps, torch.full((3, 4), PAD_IDX, dtype=torch.long)], dim=1)
    a = caption_loss(model(feats, caps, lengths), caps, lengths, alpha_c=1.0)
    b = caption_loss(model(feats, padded, lengths), padded, lengths, alpha_c=1.0)
    torch.testing.assert_close(a.ce, b.ce)
    torch.testing.assert_close(a.reg, b.reg)
    torch.testing.assert_close(a.total, a.ce + a.reg)


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_step_matches_forward_under_teacher_forcing(kind: str) -> None:
    model = build_model(_cfg(kind, dropout=0.5), V).eval()
    feats, caps, lengths = _batch()
    with torch.no_grad():
        out = model(feats, caps, lengths)
        state = model.init_state(feats)
        for t in range(caps.shape[1] - 1):
            logp, state, alpha = model.step(caps[:, t], state)
            torch.testing.assert_close(
                logp, F.log_softmax(out.logits[:, t], -1), atol=1e-5, rtol=1e-5
            )
            if kind == "attention":
                assert alpha is not None and out.alphas is not None
                valid = t < lengths - 1
                torch.testing.assert_close(alpha[valid], out.alphas[valid, t], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_gradients_reach_every_trainable_parameter(kind: str) -> None:
    model = build_model(_cfg(kind), V).train()
    feats, caps, lengths = _batch()
    caption_loss(model(feats, caps, lengths), caps, lengths, alpha_c=1.0).total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        if name != "embed.weight":
            assert p.grad.abs().sum() > 0, name
    assert model.embed.weight.grad is not None
    assert model.embed.weight.grad[PAD_IDX].abs().sum() == 0  # padding row never updates


def test_attention_regulariser() -> None:
    model = AttentionModel(_cfg("attention"), V)
    feats, caps, lengths = _batch()
    out = model(feats, caps, lengths)
    on = caption_loss(out, caps, lengths, alpha_c=1.0)
    off = caption_loss(out, caps, lengths, alpha_c=0.0)
    assert on.reg > 0 and off.reg == 0
    assert out.alphas is not None
    expected = ((1 - out.alphas.sum(1)) ** 2).mean()
    torch.testing.assert_close(on.reg, expected)


def test_encoder_frozen_in_stage1_and_image_captioner_wrapper() -> None:
    cfg = _cfg("attention", encoder_dim=2048)
    model = ImageCaptioner(ResNetEncoder(weights=None), AttentionModel(cfg, V))
    images = torch.randn(2, 3, 224, 224)
    _, caps, lengths = _batch(b=2)
    out = model(images, caps, lengths)
    caption_loss(out, caps, lengths, 1.0).total.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert count_trainable(model) == count_trainable(model.decoder)
    state = model.init_state(images)
    logp, _, alpha = model.step(caps[:, 0], state)
    assert logp.shape == (2, V) and alpha is not None and alpha.shape == (2, 49)


def test_parameter_counts_match_architecture() -> None:
    """Full-size M2 with the Flickr8k vocabulary (2,533 tokens); see ADR-011."""
    vocab = 2533
    cfg = ModelConfig()  # architecture.md §5.3 defaults
    m2 = AttentionModel(cfg, vocab)
    c, d, e, a = 2048, 512, 256, 512
    expected = (
        vocab * e  # embedding
        + (c * a + a)
        + (d * a + a)
        + (a + 1)  # attention
        + 2 * (c * d + d)  # init_h, init_c
        + (d * c + c)  # f_beta
        + 4 * d * (e + c)
        + 4 * d * d
        + 8 * d  # LSTMCell
        + (d * vocab + vocab)  # output layer
    )
    assert count_trainable(m2) == expected
    assert 12.0e6 < expected < 12.5e6
    m1 = MergeModel(ModelConfig(kind="merge", decoder_dim=256), vocab)
    m1_expected = (
        (c * 256 + 256)  # image projection
        + vocab * 256  # embedding
        + 4 * 256 * (256 + 256)
        + 8 * 256  # LSTM
        + (256 * 256 + 256)  # fc1
        + (256 * vocab + vocab)  # fc2
    )
    assert count_trainable(m1) == m1_expected == 2_416_101
    assert feature_kind_for(cfg) == "spatial"
    assert feature_kind_for(ModelConfig(kind="merge")) == "pooled"

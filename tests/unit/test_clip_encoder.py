from __future__ import annotations

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from captioner.config import Config
from captioner.data.transforms import CLIP_MEAN, CLIP_STD, load_image, transform_for
from captioner.models.encoder import ClipEncoder, ResNetEncoder, build_encoder


def test_clip_encoder_tokens_and_frozen() -> None:
    enc = build_encoder("clip_vit_b32", None)
    assert isinstance(enc, ClipEncoder)
    assert not any(p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.visual.training
    with torch.inference_mode():
        out = enc(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 49, 768)
    assert ClipEncoder.pool(out).shape == (2, 768)


def test_build_encoder_choices() -> None:
    assert isinstance(build_encoder("resnet50", None), ResNetEncoder)
    with pytest.raises(ValueError, match="only implemented for resnet50"):
        build_encoder("clip_vit_b32", None, train_layer4=True)
    with pytest.raises(ValueError, match="Unknown"):
        build_encoder("vgg", None)


def test_clip_preprocessing_profile() -> None:
    white = load_image(Image.new("RGB", (300, 260), (255, 255, 255)), transform_for("clip_vit_b32"))
    assert white.shape == (3, 224, 224)
    for c in range(3):
        assert white[c].mean().item() == pytest.approx((1 - CLIP_MEAN[c]) / CLIP_STD[c], abs=1e-4)


def test_config_rejects_mismatched_encoder_dim(mini_cfg: Config) -> None:
    with pytest.raises(ValidationError, match="encoder_dim"):
        mini_cfg.with_overrides({"encoder": {"name": "clip_vit_b32"}})
    ok = mini_cfg.with_overrides(
        {"encoder": {"name": "clip_vit_b32"}, "model": {"encoder_dim": 768}}
    )
    assert ok.encoder.name == "clip_vit_b32"

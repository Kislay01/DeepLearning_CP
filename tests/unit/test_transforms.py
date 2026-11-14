from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import captioner.data.dataset as dataset_mod
import captioner.features.extract as extract_mod
from captioner.data import transforms
from captioner.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_train_augment,
    load_image,
    to_rgb,
)


def _encode(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _exif_rotated_jpeg() -> bytes:
    img = Image.new("RGB", (300, 200), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    return _encode(img, "JPEG", exif=exif.tobytes())


CASES = {
    "rgb_jpeg": lambda: _encode(Image.new("RGB", (320, 240), (10, 20, 30)), "JPEG"),
    "grayscale": lambda: _encode(Image.new("L", (300, 260), 128), "PNG"),
    "gray_alpha": lambda: _encode(Image.new("LA", (300, 260), (128, 100)), "PNG"),
    "palette": lambda: _encode(Image.new("RGB", (300, 260), (0, 128, 255)).convert("P"), "PNG"),
    "palette_transparency": lambda: _encode(Image.new("P", (300, 260), 3), "PNG", transparency=3),
    "rgba": lambda: _encode(Image.new("RGBA", (300, 260), (255, 0, 0, 0)), "PNG"),
    "cmyk": lambda: _encode(Image.new("CMYK", (300, 260), (0, 255, 255, 0)), "JPEG"),
    "16bit_png": lambda: _encode(
        Image.fromarray(np.full((260, 300), 40000, dtype=np.uint16)), "PNG"
    ),
    "exif_rotated": _exif_rotated_jpeg,
    "tiny_1x1": lambda: _encode(Image.new("RGB", (1, 1), (5, 5, 5)), "PNG"),
    "panorama": lambda: _encode(Image.new("RGB", (8000, 100), (9, 9, 9)), "JPEG"),
    "webp": lambda: _encode(Image.new("RGB", (300, 260), (1, 2, 3)), "WEBP"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_regression_D6_every_mode_yields_normalised_tensor(name: str) -> None:
    out = load_image(CASES[name]())
    assert out.dtype == torch.float32
    assert out.shape == (3, 224, 224)
    assert torch.isfinite(out).all()


def test_normalisation_values() -> None:
    white = load_image(Image.new("RGB", (256, 256), (255, 255, 255)))
    for c in range(3):
        expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert white[c].mean().item() == pytest.approx(expected, abs=1e-5)


def test_transparent_pixels_become_white_and_16bit_scales() -> None:
    rgb = to_rgb(Image.new("RGBA", (4, 4), (255, 0, 0, 0)))
    assert rgb.mode == "RGB" and rgb.getpixel((0, 0)) == (255, 255, 255)
    gray16 = to_rgb(Image.fromarray(np.full((4, 4), 65535, dtype=np.uint16)))
    assert gray16.getpixel((0, 0))[0] >= 254


def test_exif_orientation_applied() -> None:
    with Image.open(io.BytesIO(_exif_rotated_jpeg())) as img:
        assert to_rgb(img).size == (200, 300)


def test_path_bytes_and_pil_inputs_agree(tmp_path: Path) -> None:
    img = Image.new("RGB", (300, 260), (40, 80, 120))
    path = tmp_path / "x.png"
    img.save(path)
    a = load_image(path)
    b = load_image(path.read_bytes())
    c = load_image(img)
    d = load_image(str(path))
    assert torch.equal(a, b) and torch.equal(a, c) and torch.equal(a, d)


def test_train_augment_shape() -> None:
    out = load_image(Image.new("RGB", (300, 260)), build_train_augment())
    assert out.shape == (3, 224, 224)


def test_regression_D6_training_and_serving_share_one_function() -> None:
    # Feature extraction and the fine-tuning dataset call the very same callable.
    assert extract_mod.load_image is transforms.load_image
    assert dataset_mod.load_image is transforms.load_image

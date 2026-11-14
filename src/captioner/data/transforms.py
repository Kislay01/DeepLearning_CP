"""The single image preprocessing function shared by training and serving (architecture.md §4).

Fixes reference defect D6: every image — grayscale, palette, RGBA, CMYK, 16-bit, EXIF-rotated —
goes through the same ``load_image`` call whether it comes from the dataset or an upload.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE = 256
CROP = 224

ImageSource = Path | str | bytes | Image.Image

_HIGH_BIT_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})


def to_rgb(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert any Pillow mode to 8-bit RGB."""
    img = ImageOps.exif_transpose(img)
    if img.mode in _HIGH_BIT_MODES:
        # 16/32-bit grayscale: scale to 8 bit before converting (Pillow would clip otherwise).
        scale = 1 / 256 if img.mode.startswith("I;16") or img.mode == "I" else 255.0
        img = img.convert("F").point(lambda v: v * scale).convert("L")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    return img.convert("RGB")


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class PreprocessProfile:
    """Geometry and normalisation an encoder was pre-trained with."""

    resize: int = RESIZE
    crop: int = CROP
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "bilinear"


PROFILES = {
    "resnet50": PreprocessProfile(),
    "clip_vit_b32": PreprocessProfile(224, 224, CLIP_MEAN, CLIP_STD, "bicubic"),
}


def build_transform(
    resize: int = RESIZE, crop: int = CROP, profile: PreprocessProfile | None = None
) -> Callable[[Image.Image], torch.Tensor]:
    """Deterministic eval/serve transform: resize shorter side, center-crop, normalise."""
    p = profile or PreprocessProfile(resize, crop)
    mode = (
        T.InterpolationMode.BICUBIC
        if p.interpolation == "bicubic"
        else T.InterpolationMode.BILINEAR
    )
    return cast(
        Callable[[Image.Image], torch.Tensor],
        T.Compose(
            [
                T.Resize(p.resize, interpolation=mode, antialias=True),
                T.CenterCrop(p.crop),
                T.ToTensor(),
                T.Normalize(p.mean, p.std),
            ]
        ),
    )


def transform_for(encoder_name: str) -> Callable[[Image.Image], torch.Tensor]:
    """The serve/extract transform matching an encoder (same function for train and serve)."""
    return build_transform(profile=PROFILES[encoder_name])


def build_train_augment(crop: int = CROP) -> Callable[[Image.Image], torch.Tensor]:
    """Augmentation used only when fine-tuning the encoder (random-resized-crop + h-flip)."""
    return cast(
        Callable[[Image.Image], torch.Tensor],
        T.Compose(
            [
                T.RandomResizedCrop(crop, scale=(0.8, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    )


_DEFAULT_TRANSFORM = build_transform()


def open_image(src: ImageSource) -> Image.Image:
    """Open a path, raw bytes or PIL image and return a fully loaded RGB image."""
    if isinstance(src, Image.Image):
        return to_rgb(src)
    with Image.open(io.BytesIO(src) if isinstance(src, bytes) else Path(src)) as img:
        img.load()
        return to_rgb(img)


def load_image(
    src: ImageSource,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Load and preprocess one image.

    Args:
        src: File path, encoded bytes or a PIL image.
        transform: Override (e.g. the fine-tuning augmentation); defaults to resize 256 →
            center-crop 224 → ImageNet normalisation.

    Returns:
        ``float32`` tensor of shape ``(3, 224, 224)``.
    """
    return (transform or _DEFAULT_TRANSFORM)(open_image(src))

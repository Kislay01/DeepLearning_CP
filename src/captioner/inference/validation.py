"""Upload validation for untrusted images (security.md §2).

Order matters: size check on raw bytes → header-only checks (format allow-list, dimensions,
pixel budget) → ``verify()`` → reopen and decode with decompression-bomb warnings as errors →
EXIF transpose, RGB, metadata dropped. Nothing touches the disk.
"""

from __future__ import annotations

import io
import warnings

from PIL import Image, UnidentifiedImageError

from captioner.config import AppConfig
from captioner.data.transforms import to_rgb

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
GENERIC_MESSAGE = "That file isn't an image we can read. Try a JPG or PNG under 10 MB."


class InvalidImageError(ValueError):
    """An upload was rejected. ``user_message`` is safe to show in the UI."""

    def __init__(self, user_message: str, reason: str) -> None:
        super().__init__(f"{reason}: {user_message}")
        self.user_message = user_message
        self.reason = reason


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _check_header(img: Image.Image, cfg: AppConfig) -> None:
    if img.format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            "Only JPG, PNG and WebP images are supported. Try a JPG or PNG under 10 MB.",
            f"format {img.format}",
        )
    width, height = img.size
    if width < 1 or height < 1:
        raise InvalidImageError(GENERIC_MESSAGE, "zero-size image")
    if width > cfg.max_side_px or height > cfg.max_side_px or width * height > cfg.max_image_pixels:
        raise InvalidImageError(
            f"That image is too large ({width}×{height} px). "
            f"Try one under {cfg.max_side_px} px per side.",
            "dimensions",
        )
    if getattr(img, "n_frames", 1) > 1:
        raise InvalidImageError(
            "Animated images aren't supported. Try a still JPG or PNG.", "animated"
        )


def validate_upload(data: bytes, cfg: AppConfig) -> Image.Image:
    """Validate and normalise uploaded bytes.

    Args:
        data: Raw upload.
        cfg: Limits (``max_upload_mb``, ``max_image_pixels``, ``max_side_px``).

    Returns:
        A new RGB image with EXIF orientation applied and no metadata.

    Raises:
        InvalidImageError: With a user-facing message for every rejection.
    """
    if not data:
        raise InvalidImageError("The file is empty. Try a JPG or PNG under 10 MB.", "empty")
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise InvalidImageError(
            f"That file is larger than {cfg.max_upload_mb:g} MB. Try a smaller image.", "too large"
        )
    Image.MAX_IMAGE_PIXELS = cfg.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with _open(data) as probe:
                _check_header(probe, cfg)
                probe.verify()
            with _open(data) as img:
                img.load()
                rgb = to_rgb(img)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise InvalidImageError(GENERIC_MESSAGE, type(exc).__name__) from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError(
            "That image is too large to process safely.", "decompression bomb"
        ) from exc
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    return clean

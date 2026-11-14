"""testing.md §5 — robustness & security of upload handling."""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from captioner.config import AppConfig
from captioner.inference.validation import InvalidImageError, validate_upload

CFG = AppConfig()


def _enc(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _png_header_only(width: int, height: int) -> bytes:
    """A PNG whose IHDR claims huge dimensions (decompression-bomb shape)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + chunk(b"IEND", b"")
    )


JPEG = _enc(Image.new("RGB", (64, 48), (200, 10, 10)), "JPEG")

REJECT = {
    "zero_bytes": b"",
    "truncated_jpeg": JPEG[: len(JPEG) // 2],
    "random_bytes": np.random.default_rng(0).bytes(4096),
    "png_bomb_50k": _png_header_only(50_000, 50_000),
    "too_wide": _png_header_only(9_000, 10),
    "eleven_mb": JPEG + b"\x00" * (11 * 1024 * 1024),
    "svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
    "gif": _enc(Image.new("P", (10, 10)), "GIF"),
    "tiff": _enc(Image.new("RGB", (10, 10)), "TIFF"),
    "bmp": _enc(Image.new("RGB", (10, 10)), "BMP"),
    "animated_webp": _enc(
        Image.new("RGB", (10, 10), (0, 0, 0)),
        "WEBP",
        save_all=True,
        append_images=[Image.new("RGB", (10, 10), (255, 255, 255))],
    ),
}


@pytest.mark.parametrize("name", sorted(REJECT))
def test_rejected_inputs_raise_clean_error(name: str) -> None:
    with pytest.raises(InvalidImageError) as info:
        validate_upload(REJECT[name], CFG)
    assert info.value.user_message and "Traceback" not in info.value.user_message


def test_heic_is_rejected() -> None:
    # Pillow cannot identify HEIC without a plugin → rejected like any unknown format.
    heic_header = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64
    with pytest.raises(InvalidImageError):
        validate_upload(heic_header, CFG)


def _exif_rotated() -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x8825] = {1: "N"}  # GPS IFD stub — must not survive
    return _enc(Image.new("RGB", (300, 200), (1, 2, 3)), "JPEG", exif=exif.tobytes())


ACCEPT = {
    "tiny_1x1": (_enc(Image.new("RGB", (1, 1)), "PNG"), (1, 1)),
    "panorama_8000x100": (_enc(Image.new("RGB", (8000, 100)), "JPEG"), (8000, 100)),
    "grayscale": (_enc(Image.new("L", (50, 40)), "PNG"), (50, 40)),
    "rgba": (_enc(Image.new("RGBA", (50, 40)), "PNG"), (50, 40)),
    "cmyk": (_enc(Image.new("CMYK", (50, 40)), "JPEG"), (50, 40)),
    "webp": (_enc(Image.new("RGB", (50, 40)), "WEBP"), (50, 40)),
    "exif_rotated": (_exif_rotated(), (200, 300)),
}


@pytest.mark.parametrize("name", sorted(ACCEPT))
def test_accepted_inputs_normalised(name: str) -> None:
    data, size = ACCEPT[name]
    img = validate_upload(data, CFG)
    assert img.mode == "RGB" and img.size == size
    assert not img.info and not img.getexif()


def test_size_limit_is_configurable() -> None:
    with pytest.raises(InvalidImageError, match="too large"):
        validate_upload(JPEG, AppConfig(max_upload_mb=0.0001))

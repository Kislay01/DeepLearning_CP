"""Attention heat-maps (FR-10): overlays on the model's 224 px view plus a text summary."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from captioner.data.transforms import CROP, RESIZE, open_image

_ROWS = ("upper", "middle", "lower")
_COLS = ("left", "center", "right")


def attention_grid(alpha: torch.Tensor | np.ndarray, side: int = 7) -> np.ndarray:
    """Reshape a ``(L,)`` attention vector (L = side²) to ``(side, side)``."""
    arr = alpha.detach().cpu().numpy() if isinstance(alpha, torch.Tensor) else np.asarray(alpha)
    return arr.reshape(side, side).astype(np.float32)


def model_view(image: Image.Image) -> Image.Image:
    """The 224×224 crop the encoder sees (same geometry as ``load_image``)."""
    img = open_image(image)
    scale = RESIZE / min(img.size)
    w, h = max(CROP, round(img.width * scale)), max(CROP, round(img.height * scale))
    img = img.resize((w, h), Image.Resampling.BILINEAR)
    left, top = (w - CROP) // 2, (h - CROP) // 2
    return img.crop((left, top, left + CROP, top + CROP))


def overlay(image: Image.Image, grid: np.ndarray, opacity: float = 0.55) -> Image.Image:
    """Blend an upsampled attention grid (``(7, 7)``) over the model's view of ``image``."""
    view = model_view(image)
    t = torch.from_numpy(grid)[None, None]  # (1, 1, 7, 7)
    up = F.interpolate(t, size=(CROP, CROP), mode="bilinear", align_corners=False)[0, 0].numpy()
    span = float(up.max() - up.min())
    norm = (up - up.min()) / span if span > 0 else np.zeros_like(up)
    heat = (colormaps["inferno"](norm)[..., :3] * 255).astype(np.uint8)
    return Image.blend(view, Image.fromarray(heat), opacity)


def region_name(grid: np.ndarray) -> str:
    """Coarse location of the attention peak, e.g. ``"lower-left"`` or ``"center"``."""
    side = grid.shape[0]
    row, col = np.unravel_index(int(np.argmax(grid)), grid.shape)
    r, c = _ROWS[min(2, row * 3 // side)], _COLS[min(2, col * 3 // side)]
    if r == "middle" and c == "center":
        return "center"
    return f"{r}-{c}" if r != "middle" else f"middle-{c}"


def attention_summary(words: list[str], grids: list[np.ndarray]) -> str:
    """Text equivalent of the heat-maps, e.g. ``"'dog' → lower-left; 'grass' → lower-center"``."""
    return "; ".join(f"'{w}' → {region_name(g)}" for w, g in zip(words, grids, strict=False))

"""CLIPScore and RefCLIPScore (Hessel et al., EMNLP 2021) — ADR-028.

n-gram metrics only reward overlap with five human sentences; they can't tell whether a caption
is *true of the image*. CLIPScore measures image–caption compatibility with CLIP and correlates
better with human judgements, and it penalises hallucinated objects that happen to share n-grams.

* ``CLIP-S(c, v)    = 2.5 · max(cos(E_v, E_c), 0)``  (caption prefixed "A photo depicts")
* ``RefCLIP-S(c, R, v) = H-Mean(CLIP-S(c, v), max(max_r cos(E_c, E_r), 0))``
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from captioner.data.transforms import load_image, transform_for

PROMPT = "A photo depicts "
W = 2.5


def clip_s(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Row-wise CLIP-S for L2-normalised ``(N, D)`` embeddings."""
    return W * (image_emb * text_emb).sum(-1).clamp(min=0)


def ref_clip_s(
    clip: torch.Tensor, text_emb: torch.Tensor, ref_embs: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Harmonic mean of CLIP-S and the best caption–reference cosine (clamped at 0)."""
    best = torch.stack(
        [(r @ t).max().clamp(min=0) for t, r in zip(text_emb, ref_embs, strict=True)]
    )
    return torch.where(
        clip + best > 0, 2 * clip * best / (clip + best).clamp(min=1e-12), torch.zeros_like(clip)
    )


class ClipScorer:
    """Embeds images and captions with a frozen CLIP ViT-B/32 (separate from the captioner)."""

    def __init__(self, weights: str | None = "openai", device: torch.device | None = None) -> None:
        import open_clip

        self.device = device or torch.device("cpu")
        self.model: Any = (
            open_clip.create_model("ViT-B-32", pretrained=weights).to(self.device).eval()
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.transform = transform_for("clip_vit_b32")

    @torch.inference_mode()
    def embed_images(self, paths: Sequence[Path], batch_size: int = 64) -> torch.Tensor:
        out = []
        for i in range(0, len(paths), batch_size):
            imgs = torch.stack([load_image(p, self.transform) for p in paths[i : i + batch_size]])
            out.append(F.normalize(self.model.encode_image(imgs.to(self.device)).float(), dim=-1))
        return torch.cat(out).cpu()

    @torch.inference_mode()
    def embed_texts(self, texts: Sequence[str], batch_size: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(list(texts[i : i + batch_size])).to(self.device)
            out.append(F.normalize(self.model.encode_text(tokens).float(), dim=-1))
        return torch.cat(out).cpu()

    def score(
        self, image_embs: torch.Tensor, captions: Sequence[str], references: Sequence[Sequence[str]]
    ) -> dict[str, Any]:
        """Corpus means and per-image values of CLIP-S and RefCLIP-S."""
        text = self.embed_texts([PROMPT + c for c in captions])
        flat = [r for refs in references for r in refs]
        ref_flat = self.embed_texts([PROMPT + r for r in flat])
        ref_embs, start = [], 0
        for refs in references:
            ref_embs.append(ref_flat[start : start + len(refs)])
            start += len(refs)
        cs = clip_s(image_embs, text)
        rcs = ref_clip_s(cs, text, ref_embs)
        return {
            "CLIP-S": float(cs.mean()),
            "RefCLIP-S": float(rcs.mean()),
            "clip_s_per_image": cs.tolist(),
        }

"""``Predictor``: image → caption with optional attention, for the CLI and the app (FR-8, FR-9).

Holds encoder + decoder + vocabulary, runs under ``torch.inference_mode()`` and never mutates its
state after construction, so one instance can serve concurrent read-only requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from captioner.config import Config
from captioner.data.neutral import neutralize_aligned
from captioner.data.text import detokenize
from captioner.data.transforms import ImageSource, load_image
from captioner.data.vocab import Vocabulary
from captioner.inference.decode import build_repeat_policy, decode
from captioner.inference.heatmap import attention_grid
from captioner.inference.reliability import Reliability
from captioner.models.captioner import CaptionModel
from captioner.models.encoder import ResNetEncoder
from captioner.training.checkpoint import load_run
from captioner.utils.device import get_device


@dataclass(frozen=True)
class Caption:
    """A generated caption.

    Attributes:
        text: Display sentence (capitalised, trailing period).
        tokens: Words without special tokens.
        score: Length-normalised log-probability used for ranking.
        mean_logprob: Average log-probability per token (confidence hint).
        attention: ``(T, 7, 7)`` attention per word (M2), else ``None``.
        low_confidence: ``mean_logprob`` below the calibrated (or configured) threshold.
        out_of_domain: The photo looks unlike the training images (ADR-022).
        ood_score: Distance to the training distribution, or ``None`` without calibration.
        beam: Beam width used.
        latency_s: Wall-clock seconds for preprocessing + encoding + decoding.
    """

    text: str
    tokens: tuple[str, ...]
    score: float
    mean_logprob: float
    attention: np.ndarray | None
    low_confidence: bool
    beam: int
    latency_s: float
    out_of_domain: bool = False
    ood_score: float | None = None

    @property
    def reliable(self) -> bool:
        return not (self.low_confidence or self.out_of_domain)


class Predictor:
    """Caption images with a trained model.

    Args:
        decoder: Trained ``CaptionModel``.
        vocab: Vocabulary it was trained with.
        cfg: Resolved run config (decode + app settings).
        encoder: ResNet-50 encoder; built from ``cfg.encoder.weights`` when omitted.
        device: Compute device.
    """

    def __init__(
        self,
        decoder: nn.Module,
        vocab: Vocabulary,
        cfg: Config,
        encoder: nn.Module | None = None,
        device: torch.device | None = None,
        reliability: Reliability | None = None,
    ) -> None:
        self.device = device or get_device()
        self.cfg = cfg
        self.vocab = vocab
        self.encoder = (encoder or ResNetEncoder(cfg.encoder.weights)).to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.reliability = reliability
        self.repeat = build_repeat_policy(
            vocab.itos, cfg.decode.no_repeat_ngram, cfg.decode.block_repeated_words
        )

    @classmethod
    def from_run(
        cls, run_dir: Path, which: str = "best", device: torch.device | None = None
    ) -> Predictor:
        """Load a training run folder (``best.pt`` + ``vocab.json``)."""
        run = load_run(run_dir, which)
        encoder = ResNetEncoder(run.cfg.encoder.weights)
        if run.encoder_state:
            encoder.load_state_dict(run.encoder_state)
        rel_path = run_dir / "reliability.safetensors"
        reliability = Reliability.load(rel_path) if rel_path.exists() else None
        return cls(run.model, run.vocab, run.cfg, encoder, device, reliability)

    @property
    def has_attention(self) -> bool:
        return self.cfg.model.kind == "attention"

    @torch.inference_mode()
    def caption(self, src: ImageSource, beam: int | None = None) -> Caption:
        """Caption one image (path, bytes or PIL image) with the shared ``load_image``."""
        start = time.perf_counter()
        beam = beam or self.cfg.decode.beam
        image = load_image(src).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        feats = self.encoder(image)  # (1, L, C)
        ood_score = (
            float(self.reliability.ood_score(feats.mean(dim=1))[0]) if self.reliability else None
        )
        model: CaptionModel = self.decoder  # type: ignore[assignment]
        d = self.cfg.decode
        (hyp,) = decode(model, feats, beam, d.max_len, d.length_alpha, self.repeat)
        words = self.vocab.decode(hyp.ids)
        if self.cfg.decode.gender_neutral_output:
            words = neutralize_aligned(words)
        side = round(self.cfg.model.num_regions**0.5)
        attention = (
            np.stack([attention_grid(a, side) for a in hyp.attention])
            if hyp.attention is not None
            else None
        )
        return Caption(
            text=detokenize(words),
            tokens=tuple(words),
            score=hyp.score,
            mean_logprob=hyp.mean_logprob,
            attention=attention,
            low_confidence=hyp.mean_logprob < self._confidence_threshold,
            beam=beam,
            latency_s=time.perf_counter() - start,
            out_of_domain=bool(
                self.reliability
                and ood_score is not None
                and ood_score > self.reliability.ood_threshold
            ),
            ood_score=ood_score,
        )

    @property
    def _confidence_threshold(self) -> float:
        if self.reliability is not None:
            return self.reliability.confidence_threshold
        return self.cfg.app.low_confidence_logprob

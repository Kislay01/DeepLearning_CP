"""Training/validation primitives: one optimisation step, one epoch, validation decoding."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from captioner.data.dataset import Batch, CaptionTable
from captioner.data.vocab import Vocabulary
from captioner.evaluation.metrics import corpus_bleu4, reference_tokens
from captioner.inference.decode import Hypothesis, greedy_decode
from captioner.models.captioner import CaptionModel, LossOutput, caption_loss

Decoder = Callable[[CaptionModel, torch.Tensor], list[Hypothesis]]


@dataclass
class Amp:
    """Mixed-precision settings: bf16 when supported, otherwise fp16 with a grad scaler."""

    device: torch.device
    enabled: bool
    dtype: torch.dtype = torch.float32
    scaler: torch.amp.GradScaler = field(init=False)

    def __post_init__(self) -> None:
        self.enabled = self.enabled and self.device.type == "cuda"
        if self.enabled:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = self.enabled and self.dtype == torch.float16
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=use_scaler)

    def autocast(self) -> contextlib.AbstractContextManager[None]:
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)


def train_step(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    amp: Amp,
    grad_clip: float,
    alpha_c: float,
) -> LossOutput:
    """Forward, loss, backward, clip (global norm) and optimizer step for one batch."""
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast():
        out = model(batch.feats, batch.captions, batch.lengths)
        loss = caption_loss(out, batch.captions, batch.lengths, alpha_c)
    scaled: torch.Tensor = amp.scaler.scale(loss.total)
    scaled.backward()  # type: ignore[no-untyped-call]
    amp.scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
    amp.scaler.step(optimizer)
    amp.scaler.update()
    return LossOutput(loss.total.detach(), loss.ce.detach(), loss.reg.detach())


@dataclass
class EpochStats:
    """Mean losses over the steps of one epoch."""

    loss: float
    ce: float
    reg: float
    steps: int
    step_losses: list[float]

    @property
    def perplexity(self) -> float:
        return math.exp(min(self.ce, 50.0))


def _limited(loader: DataLoader[Batch], max_steps: int | None) -> Iterator[Batch]:
    for i, batch in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            return
        yield batch


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    amp: Amp,
    grad_clip: float,
    alpha_c: float,
    max_steps: int | None = None,
) -> EpochStats:
    """Run one training epoch (``model.train()`` is set here)."""
    model.train()
    totals, ces, regs = [], [], []
    for batch in _limited(loader, max_steps):
        loss = train_step(model, batch.to(amp.device), optimizer, amp, grad_clip, alpha_c)
        totals.append(float(loss.total))
        ces.append(float(loss.ce))
        regs.append(float(loss.reg))
    n = max(1, len(totals))
    return EpochStats(sum(totals) / n, sum(ces) / n, sum(regs) / n, len(totals), totals)


@torch.inference_mode()
def validation_loss(
    model: nn.Module, loader: DataLoader[Batch], amp: Amp, alpha_c: float
) -> EpochStats:
    """Teacher-forced loss on held-out captions (``model.eval()`` is set here)."""
    model.eval()
    totals, ces, regs = [], [], []
    for batch in loader:
        batch = batch.to(amp.device)
        with amp.autocast():
            out = model(batch.feats, batch.captions, batch.lengths)
            loss = caption_loss(out, batch.captions, batch.lengths, alpha_c)
        totals.append(float(loss.total))
        ces.append(float(loss.ce))
        regs.append(float(loss.reg))
    n = max(1, len(totals))
    return EpochStats(sum(totals) / n, sum(ces) / n, sum(regs) / n, len(totals), totals)


def greedy_decoder(max_len: int, length_alpha: float = 0.0) -> Decoder:
    """Decoder callable for ``decode_split`` (beam width 1)."""
    return lambda model, feats: greedy_decode(model, feats, max_len, length_alpha)


@torch.inference_mode()
def decode_split(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    decoder: Decoder,
) -> list[Hypothesis]:
    """Decode every image of an ``ImageFeatureDataset`` loader, in dataset order."""
    model.eval()
    hyps: dict[int, Hypothesis] = {}
    for feats, index in loader:
        for i, hyp in zip(index.tolist(), decoder(model, feats.to(device)), strict=True):  # type: ignore[arg-type]
            hyps[i] = hyp
    return [hyps[i] for i in range(len(hyps))]


def bleu4_for(hyps: list[Hypothesis], table: CaptionTable, vocab: Vocabulary) -> float:
    """NLTK corpus BLEU-4 of decoded hypotheses against the table's raw references."""
    return corpus_bleu4(
        [vocab.decode(h.ids) for h in hyps],
        [reference_tokens(refs) for refs in table.references],
    )

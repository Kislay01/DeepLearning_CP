"""Optimiser, learning-rate schedule and weight averaging (ADR-025)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from captioner.config import Config
from captioner.models.captioner import ImageCaptioner

_NO_DECAY = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d)


def _decay_split(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Weights of Linear/LSTM/attention layers decay; biases, norms and embeddings don't."""
    no_decay_ids = {
        id(p)
        for m in module.modules()
        if isinstance(m, _NO_DECAY)
        for p in m.parameters(recurse=False)
    }
    decay: list[nn.Parameter] = []
    rest: list[nn.Parameter] = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (rest if name.endswith("bias") or id(p) in no_decay_ids else decay).append(p)
    return decay, rest


def _groups(module: nn.Module, lr: float, wd: float, adamw: bool) -> list[dict[str, Any]]:
    if not adamw:
        return [
            {
                "params": [p for p in module.parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": wd,
            }
        ]
    decay, rest = _decay_split(module)
    return [
        {"params": decay, "lr": lr, "weight_decay": wd},
        {"params": rest, "lr": lr, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Adam (blueprint) or AdamW; a fine-tuned encoder gets its own, smaller learning rate."""
    t = cfg.train
    adamw = t.optimizer == "adamw"
    if isinstance(model, ImageCaptioner):
        groups = _groups(model.decoder, t.lr, t.weight_decay, adamw)
        groups += _groups(model.encoder, t.encoder_lr, t.weight_decay, adamw)
    else:
        groups = _groups(model, t.lr, t.weight_decay, adamw)
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups) if adamw else torch.optim.Adam(groups)


class Schedule:
    """Per-epoch ReduceLROnPlateau (blueprint) or per-step cosine decay; both with linear warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: Config, total_steps: int) -> None:
        t = cfg.train
        self.optimizer = optimizer
        self.kind = t.scheduler
        self.warmup = t.warmup_steps
        self.total = max(total_steps, self.warmup + 1)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0
        self.plateau = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=t.lr_factor, patience=t.lr_patience
            )
            if self.kind == "plateau"
            else None
        )
        self._apply(self._factor(0))

    def _factor(self, step: int) -> float:
        if step < self.warmup:
            return (step + 1) / self.warmup
        if self.kind == "plateau":
            return 1.0
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _apply(self, factor: float) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * factor

    def on_step(self) -> None:
        """Call after every optimiser step."""
        self.step_count += 1
        if self.kind == "cosine" or self.step_count <= self.warmup:
            self._apply(self._factor(self.step_count))

    def on_epoch(self, metric: float) -> None:
        """Call after validation with the selection metric."""
        if self.plateau is not None and self.step_count >= self.warmup:
            self.plateau.step(metric)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"step_count": self.step_count, "kind": self.kind}
        if self.plateau is not None:
            state["plateau"] = self.plateau.state_dict()
        return state


class Ema:
    """Exponential moving average of the weights; evaluation and checkpoints use the average."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = model
        self.avg: AveragedModel | None = (
            AveragedModel(
                model,
                multi_avg_fn=get_ema_multi_avg_fn(decay),  # type: ignore[no-untyped-call]
                use_buffers=True,
            )
            if decay > 0
            else None
        )

    def update(self) -> None:
        if self.avg is not None:
            self.avg.update_parameters(self.model)

    @property
    def eval_model(self) -> nn.Module:
        """The weights to validate, decode and save."""
        return self.avg.module if self.avg is not None else self.model

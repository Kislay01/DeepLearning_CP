"""Frozen ResNet-50 encoder producing spatial (49 x 2048) and pooled (2048) features."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class ResNetEncoder(nn.Module):
    """ResNet-50 with ``avgpool`` and ``fc`` removed (architecture.md §5.1).

    Args:
        weights: torchvision weight name (``"IMAGENET1K_V2"``) or ``None`` for random init.
        train_layer4: Unfreeze the last residual stage (fine-tuning stage 2, FR-12).
    """

    def __init__(self, weights: str | None = "IMAGENET1K_V2", train_layer4: bool = False) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights[weights] if weights else None)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.train_layer4 = train_layer4
        self._set_trainable()

    def _set_trainable(self) -> None:
        for p in self.body.parameters():
            p.requires_grad = False
        if self.train_layer4:
            for p in self.body[-1].parameters():  # layer4
                p.requires_grad = True

    def train(self, mode: bool = True) -> ResNetEncoder:
        """Frozen stages always stay in eval mode so BatchNorm statistics never change."""
        super().train(mode)
        self.body.eval()
        if mode and self.train_layer4:
            self.body[-1].train()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            images: ``(B, 3, 224, 224)`` normalised batch.

        Returns:
            Spatial features ``(B, 49, 2048)``.
        """
        fmap = self.body(images)  # (B, C, 7, 7)
        spatial: torch.Tensor = fmap.flatten(2).transpose(1, 2)  # (B, L, C)
        return spatial

    @staticmethod
    def pool(spatial: torch.Tensor) -> torch.Tensor:
        """Mean over regions: ``(B, L, C) → (B, C)``."""
        return spatial.mean(dim=1)

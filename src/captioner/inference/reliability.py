"""Knowing when not to trust a caption (ADR-022).

Two calibrated signals, both fitted on held-out data after training:

* **Out-of-domain score** — 1 − max cosine similarity between an image's pooled encoder feature
  and k-means centroids of the training images. The threshold is the ``ood_quantile`` of the
  validation scores, so ~1 % of in-domain photos are flagged and screenshots, drawings, documents
  or scenes unlike the training data stand out.
* **Confidence** — mean token log-probability of the decoded caption; the threshold is the
  ``confidence_quantile`` of validation captions (replaces the fixed config constant).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class Reliability:
    """Fitted reliability model.

    Attributes:
        centroids: ``(K, C)`` L2-normalised training-feature centroids.
        ood_threshold: Scores above this are out of domain.
        confidence_threshold: Mean log-probs below this are low confidence.
    """

    centroids: torch.Tensor
    ood_threshold: float
    confidence_threshold: float

    def ood_score(self, pooled: torch.Tensor) -> torch.Tensor:
        """``(B, C)`` pooled features → ``(B,)`` distance to the training distribution."""
        sims = F.normalize(pooled.float(), dim=-1) @ self.centroids.to(pooled.device).T  # (B, K)
        return 1.0 - sims.max(dim=1).values

    def is_out_of_domain(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.ood_score(pooled) > self.ood_threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        thresholds = torch.tensor([self.ood_threshold, self.confidence_threshold])
        save_file(
            {"centroids": self.centroids.contiguous(), "thresholds": thresholds},
            str(path),
            metadata={"format": "captioner-reliability-v1"},
        )

    @classmethod
    def load(cls, path: Path) -> Reliability:
        data = load_file(str(path))
        ood, conf = data["thresholds"].tolist()
        return cls(
            centroids=data["centroids"], ood_threshold=float(ood), confidence_threshold=float(conf)
        )


def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    """Cosine k-means. ``x`` is ``(N, C)``; returns ``(min(k, N), C)`` unit centroids."""
    x = F.normalize(x.float(), dim=-1)
    k = min(k, x.shape[0])
    gen = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(x.shape[0], generator=gen)[:k]].clone()
    for _ in range(iters):
        assign = (x @ centroids.T).argmax(dim=1)  # (N,)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).unsqueeze(1)
        empty = counts.squeeze(1) == 0
        centroids = torch.where(empty.unsqueeze(1), centroids, sums / counts.clamp(min=1))
        centroids = F.normalize(centroids, dim=-1)
    return centroids


def fit_reliability(
    train_pooled: torch.Tensor,
    val_pooled: torch.Tensor,
    val_mean_logprobs: torch.Tensor,
    k: int = 64,
    ood_quantile: float = 0.99,
    confidence_quantile: float = 0.10,
    seed: int = 0,
) -> Reliability:
    """Fit centroids on train features and both thresholds on validation data."""
    centroids = spherical_kmeans(train_pooled, k, seed=seed)
    partial = Reliability(centroids, 0.0, 0.0)
    ood = float(torch.quantile(partial.ood_score(val_pooled), ood_quantile))
    conf = float(torch.quantile(val_mean_logprobs.float(), confidence_quantile))
    return Reliability(centroids, ood, conf)

"""Early stopping on a metric that should increase (val BLEU-4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track the best value and count epochs without improvement.

    Attributes:
        patience: Epochs without improvement tolerated before stopping.
        min_delta: Minimum increase that counts as an improvement.
    """

    patience: int
    min_delta: float = 0.0
    best: float = float("-inf")
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        """Record a new value; return ``True`` if it is a new best."""
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

"""Experiment tracking: MLflow (local file store) + TensorBoard scalars."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import TracebackType

# The local ./mlruns file store is in maintenance mode in MLflow 3 and must be opted into (ADR-012).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class RunTracker:
    """Log one training run to MLflow and TensorBoard.

    Args:
        mlruns_dir: MLflow file-store root (``./mlruns``).
        experiment: MLflow experiment name.
        run_name: Human-readable run id (also the artifacts folder name).
        tb_dir: TensorBoard log directory.
    """

    def __init__(self, mlruns_dir: Path, experiment: str, run_name: str, tb_dir: Path) -> None:
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run(run_name=run_name)
        self._tb = SummaryWriter(log_dir=str(tb_dir))

    @property
    def run_id(self) -> str:
        return str(self._run.info.run_id)

    def log_params(self, params: dict[str, str]) -> None:
        mlflow.log_params(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)
        for key, value in metrics.items():
            self._tb.add_scalar(key, value, step)

    def log_text(self, text: str, name: str, step: int) -> None:
        mlflow.log_text(text, f"samples/{name}")
        self._tb.add_text(name, text.replace("\n", "  \n"), step)

    def log_artifact(self, path: Path) -> None:
        if path.exists():
            mlflow.log_artifact(str(path))

    def close(self, status: str = "FINISHED") -> None:
        self._tb.close()
        mlflow.end_run(status=status)

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close("FAILED" if exc_type else "FINISHED")

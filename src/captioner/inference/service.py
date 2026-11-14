"""Serving helpers used by the Streamlit app: model discovery, time-boxed inference, heat-maps.

The app module only lays out widgets; everything that touches models lives here (code_style §9).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from captioner import __version__
from captioner.data.transforms import PROFILES
from captioner.inference.heatmap import attention_summary, overlay
from captioner.inference.predictor import Caption, Predictor
from captioner.inference.release import ReleaseError, fetch_release, load_release, load_release_spec
from captioner.training.checkpoint import CheckpointError, latest_run

logger = logging.getLogger(__name__)

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="captioner-infer")


class ModelNotLoadedError(RuntimeError):
    """No trained model or release could be found."""


class InferenceTimeoutError(RuntimeError):
    """Captioning exceeded its time budget."""


@dataclass(frozen=True)
class ModelSource:
    """Where the serving model comes from."""

    kind: Literal["release", "run"]
    path: Path


def resolve_model_source(artifacts_dir: Path | None = None) -> ModelSource:
    """Pick the model to serve.

    Order: ``CAPTIONER_RELEASE_DIR`` → ``CAPTIONER_RUN_DIR`` → ``<artifacts>/release/<version>``
    → the pinned release in ``configs/release.yaml`` (downloaded + SHA-256 verified into
    ``CAPTIONER_CACHE_DIR``) → newest run under ``<artifacts>/runs``.

    Raises:
        ModelNotLoadedError: If nothing is found.
    """
    if env := os.environ.get("CAPTIONER_RELEASE_DIR"):
        return ModelSource("release", Path(env))
    if env := os.environ.get("CAPTIONER_RUN_DIR"):
        return ModelSource("run", Path(env))
    root = artifacts_dir or Path(os.environ.get("CAPTIONER_ARTIFACTS_DIR", "artifacts"))
    release = root / "release" / __version__
    if (release / "model.safetensors").exists():
        return ModelSource("release", release)
    if (published := _published_release()) is not None:
        return ModelSource("release", published)
    try:
        return ModelSource("run", latest_run(root / "runs"))
    except CheckpointError as exc:
        raise ModelNotLoadedError(
            "No trained model found. Train one (make train) or set CAPTIONER_RELEASE_DIR."
        ) from exc


def load_predictor(source: ModelSource) -> Predictor:
    """Instantiate the predictor (the app wraps this in ``st.cache_resource``)."""
    logger.info("Loading %s model from %s", source.kind, source.path)
    if source.kind == "release":
        return load_release(source.path)
    return Predictor.from_run(source.path)


def caption_with_timeout(
    predictor: Predictor, image: Image.Image, beam: int, timeout_s: float
) -> Caption:
    """Run ``predictor.caption`` with a wall-clock budget (security.md §2.7)."""
    future = _POOL.submit(predictor.caption, image, beam)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise InferenceTimeoutError(f"Captioning took longer than {timeout_s:g} s") from exc


@dataclass(frozen=True)
class Heatmaps:
    """Per-word overlays plus their text equivalent."""

    images: list[Image.Image]
    words: list[str]
    summary: str


def build_heatmaps(
    image: Image.Image, caption: Caption, max_words: int, encoder_name: str = "resnet50"
) -> Heatmaps | None:
    """Overlays for up to ``max_words`` words, or ``None`` for models without attention."""
    if caption.attention is None or not caption.tokens:
        return None
    words = list(caption.tokens[:max_words])
    grids = list(caption.attention[:max_words])
    profile = PROFILES[encoder_name]
    return Heatmaps(
        [overlay(image, g, profile=profile) for g in grids], words, attention_summary(words, grids)
    )


def _published_release() -> Path | None:
    spec_path = Path(os.environ.get("CAPTIONER_RELEASE_CONFIG", "configs/release.yaml"))
    if not spec_path.exists():
        return None
    spec = load_release_spec(spec_path)
    if not spec.is_published:
        return None
    cache = Path(os.environ.get("CAPTIONER_CACHE_DIR", Path.home() / ".cache" / "captioner"))
    try:
        return fetch_release(spec, cache)
    except (ReleaseError, OSError) as exc:
        raise ModelNotLoadedError(f"Could not fetch the pinned model release: {exc}") from exc

"""NFR-1 — CPU latency, single image, beam 3, full-size M2 + ResNet-50: p95 ≤ 2 s after warm-up."""

from __future__ import annotations

import statistics
import time
from typing import Any

import pytest
import torch
from PIL import Image

from captioner.config import Config
from captioner.data.vocab import build_vocab
from captioner.inference.predictor import Predictor
from captioner.models.attention import AttentionModel
from captioner.models.encoder import ResNetEncoder
from tests.conftest import MINI

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]


def test_cpu_latency_beam3_p95_under_2s(prepared_mini: Config, benchmark: Any) -> None:
    torch.manual_seed(0)
    cfg = prepared_mini.with_overrides(
        {"model": {"embed_dim": 256, "decoder_dim": 512, "attention_dim": 512, "dropout": 0.5}}
    )
    vocab = build_vocab([[f"w{i}"] for i in range(2529)], min_freq=1, max_len=25)  # V = 2,533
    predictor = Predictor(
        AttentionModel(cfg.model, len(vocab)), vocab, cfg, ResNetEncoder(None), torch.device("cpu")
    )
    with Image.open(sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0]) as img:
        image = img.convert("RGB").resize((500, 375))
    for _ in range(2):  # warm-up
        predictor.caption(image, beam=3)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        predictor.caption(image, beam=3)
        times.append(time.perf_counter() - start)
    p95 = statistics.quantiles(times, n=20)[-1]
    benchmark.extra_info["p95_s"] = p95
    benchmark.pedantic(predictor.caption, args=(image,), kwargs={"beam": 3}, rounds=3, iterations=1)
    assert p95 <= 2.0, f"p95 latency {p95:.2f} s"

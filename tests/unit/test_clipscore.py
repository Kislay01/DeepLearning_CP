"""ADR-028 — CLIPScore / RefCLIPScore formulas and scorer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from captioner.evaluation.clipscore import ClipScorer, W, clip_s, ref_clip_s
from tests.conftest import MINI


def test_formulas() -> None:
    img = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)
    txt = F.normalize(torch.tensor([[1.0, 1.0], [-1.0, 0.0]]), dim=-1)
    cs = clip_s(img, txt)
    assert cs[0].item() == pytest.approx(W * 2**-0.5) and cs[1].item() == 0.0  # negative clamped
    refs = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    rcs = ref_clip_s(cs, txt, refs)
    best0 = 2**-0.5
    assert rcs[0].item() == pytest.approx(2 * cs[0].item() * best0 / (cs[0].item() + best0))
    assert rcs[1].item() == 0.0


def test_scorer_with_random_weights() -> None:
    torch.manual_seed(0)
    scorer = ClipScorer(weights=None)
    paths = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[:3]
    imgs = scorer.embed_images(paths, batch_size=2)
    assert imgs.shape == (3, 512)
    torch.testing.assert_close(imgs.norm(dim=-1), torch.ones(3))
    out = scorer.score(
        imgs, ["a dog runs", "two people", "a child"], [["a dog"], ["people", "two"], ["kid"]]
    )
    assert 0.0 <= out["CLIP-S"] <= W and 0.0 <= out["RefCLIP-S"] <= W
    assert len(out["clip_s_per_image"]) == 3


def test_evaluation_reports_clipscore(prepared_mini, trained_runs: dict[str, Path]) -> None:  # type: ignore[no-untyped-def]
    from captioner.evaluation.evaluate import evaluate_run

    cfg = prepared_mini.with_overrides(
        {"eval": {"clipscore": True, "clipscore_weights": None, "tag": "clip"}}
    )
    res = evaluate_run(cfg, trained_runs["merge"], beams=[1])
    assert {"CLIP-S", "RefCLIP-S"} <= set(res.metrics["results"]["beam_1"])

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from captioner.inference.decode import (
    RepeatPolicy,
    beam_search,
    build_repeat_policy,
    decode,
    greedy_decode,
)
from captioner.inference.predictor import Caption, Predictor
from captioner.inference.release import export_run, load_release, verify_release
from captioner.inference.reliability import Reliability, fit_reliability, spherical_kmeans
from tests.conftest import MINI
from tests.unit.test_decode import START_IDX, StubModel, X, Y, _feats

IMAGE = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0]

# Markov chain that loves to loop: x → y → x → y …, <end> unlikely.
LOOP = {START_IDX: {X: 0.9, Y: 0.1}, X: {Y: 0.9, 2: 0.1}, Y: {X: 0.9, 2: 0.1}}


def test_without_policy_the_loop_repeats() -> None:
    (h,) = greedy_decode(StubModel(LOOP), _feats(), max_len=8)  # type: ignore[arg-type]
    assert h.ids[:4] == (X, Y, X, Y)


@pytest.mark.parametrize("beam", [1, 3])
def test_regression_ngram_and_word_repeats_blocked(beam: int) -> None:
    model = StubModel(LOOP)
    (h,) = decode(model, _feats(), beam, 8, 0.7, RepeatPolicy(ngram=2))  # type: ignore[arg-type]
    pairs = list(zip(h.ids, h.ids[1:], strict=False))
    assert len(pairs) == len(set(pairs))
    mask = torch.zeros(6, dtype=torch.bool)
    mask[[X, Y]] = True
    (w,) = decode(model, _feats(), beam, 8, 0.7, RepeatPolicy(content_mask=mask))  # type: ignore[arg-type]
    assert w.ids.count(X) <= 1 and w.ids.count(Y) <= 1 and w.finished


def test_build_repeat_policy_marks_content_words() -> None:
    itos = ("<pad>", "<start>", "<end>", "<unk>", "a", "dog", "in", "park")
    pol = build_repeat_policy(itos, 3, True)
    assert pol.ngram == 3 and pol.content_mask is not None
    assert pol.content_mask.tolist() == [False, False, False, False, False, True, False, True]
    assert build_repeat_policy(itos, 0, False).content_mask is None


def test_batched_beam_with_policy_equals_single() -> None:
    from captioner.config import ModelConfig
    from captioner.models.captioner import build_model

    torch.manual_seed(1)
    model = build_model(
        ModelConfig(kind="merge", embed_dim=8, decoder_dim=8, encoder_dim=6), 30
    ).eval()
    feats = torch.randn(3, 6)
    pol = RepeatPolicy(ngram=2, content_mask=torch.arange(30) >= 4)
    batched = beam_search(model, feats, 3, 10, 0.7, pol)  # type: ignore[arg-type]
    single = [beam_search(model, feats[i : i + 1], 3, 10, 0.7, pol)[0] for i in range(3)]  # type: ignore[arg-type]
    assert [h.ids for h in batched] == [h.ids for h in single]
    for h in batched:
        assert len(set(h.ids)) == len(h.ids)


def test_kmeans_and_fit_thresholds(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(0)
    centers = torch.nn.functional.normalize(torch.randn(4, 16, generator=gen), dim=-1)
    train = torch.cat([c + 0.05 * torch.randn(200, 16, generator=gen) for c in centers])
    val = torch.cat([c + 0.05 * torch.randn(50, 16, generator=gen) for c in centers])
    cents = spherical_kmeans(train, 4)
    assert cents.shape == (4, 16)
    torch.testing.assert_close(cents.norm(dim=1), torch.ones(4))
    rel = fit_reliability(
        train, val, torch.linspace(-3, 0, 200), k=4, ood_quantile=0.99, confidence_quantile=0.1
    )
    assert (rel.ood_score(val) > rel.ood_threshold).float().mean() <= 0.02
    far = -centers.sum(0, keepdim=True)  # opposite direction from every cluster
    assert bool(rel.is_out_of_domain(far)[0])
    assert rel.confidence_threshold == pytest.approx(
        torch.quantile(torch.linspace(-3, 0, 200), 0.1).item()
    )
    rel.save(tmp_path / "r.safetensors")
    back = Reliability.load(tmp_path / "r.safetensors")
    torch.testing.assert_close(back.centroids, rel.centroids)
    assert back.ood_threshold == pytest.approx(rel.ood_threshold)
    assert spherical_kmeans(train[:3], 10).shape == (3, 16)


def test_training_calibrates_and_predictor_flags_unfamiliar_photos(
    trained_runs: dict[str, Path], tmp_path: Path
) -> None:
    run = trained_runs["attention"]
    assert (run / "reliability.safetensors").exists()
    torch.manual_seed(7)
    pred = Predictor.from_run(run)
    assert pred.reliability is not None
    cap = pred.caption(IMAGE, beam=2)
    assert cap.ood_score is not None
    # Force everything to look unfamiliar: a centroid no real feature can match.
    pred.reliability = Reliability(
        -torch.ones(1, pred.reliability.centroids.shape[1]) / 45.25, -1.0, 0.0
    )
    flagged = pred.caption(IMAGE, beam=2)
    assert flagged.out_of_domain and not flagged.reliable
    sums = export_run(run, tmp_path / "rel")
    assert "reliability.safetensors" in sums
    assert "reliability.safetensors" in json.loads(
        (tmp_path / "rel" / "checksums.json").read_text()
    )
    verify_release(tmp_path / "rel")
    assert load_release(tmp_path / "rel").reliability is not None


def test_spoken_text_hedges_unreliable_captions() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
    import components as ui

    base = dict(
        text="A dog runs.",
        tokens=("a", "dog", "runs"),
        score=-1.0,
        mean_logprob=-0.5,
        attention=None,
        beam=3,
        latency_s=0.1,
    )
    assert ui.spoken_text(Caption(low_confidence=False, **base)) == "A dog runs."  # type: ignore[arg-type]
    assert ui.spoken_text(Caption(low_confidence=True, **base)).startswith("Possibly")  # type: ignore[arg-type]
    unusual = Caption(low_confidence=False, out_of_domain=True, ood_score=0.9, **base)  # type: ignore[arg-type]
    assert "unsure" in ui.spoken_text(unusual)

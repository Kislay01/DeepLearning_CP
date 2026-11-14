from __future__ import annotations

import math

import pytest
import torch

from captioner.config import ModelConfig
from captioner.data.vocab import END_IDX, PAD_IDX, START_IDX, UNK_IDX
from captioner.inference.decode import (
    Hypothesis,
    beam_search,
    decode,
    greedy_decode,
    length_penalty,
)
from captioner.models.captioner import DecoderState, build_model

X, Y = 4, 5
V = 6


class StubModel:
    """Log-probs depend only on the previous token (a hand-set Markov chain)."""

    feature_kind = "pooled"

    def __init__(self, table: dict[int, dict[int, float]], floor: float = 1e-6) -> None:
        self.rows = torch.full((V, V), math.log(floor))
        for prev, nxt in table.items():
            for tok, p in nxt.items():
                self.rows[prev, tok] = math.log(p)

    def init_state(self, feats: torch.Tensor) -> DecoderState:
        return {"dummy": torch.zeros(feats.shape[0], 1)}

    def step(
        self, prev: torch.Tensor, state: DecoderState
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        return self.rows[prev], state, None


# START→x .6 / y .4;  x→y .55 / END .45;  y→END .9 / x .1
CHAIN = {
    START_IDX: {X: 0.6, Y: 0.4},
    X: {Y: 0.55, END_IDX: 0.45},
    Y: {END_IDX: 0.9, X: 0.1},
}


def _feats(b: int = 1) -> torch.Tensor:
    return torch.zeros(b, 3)


def test_greedy_follows_argmax() -> None:
    (hyp,) = greedy_decode(StubModel(CHAIN), _feats(), max_len=10)  # type: ignore[arg-type]
    assert hyp.ids == (X, Y) and hyp.finished
    assert hyp.logprob == pytest.approx(math.log(0.6 * 0.55 * 0.9), rel=1e-5)


def test_beam_finds_known_best_and_length_norm_changes_ranking() -> None:
    model = StubModel(CHAIN)
    (raw,) = beam_search(model, _feats(), beam=2, max_len=10, length_alpha=0.0)  # type: ignore[arg-type]
    assert raw.ids == (Y,)  # p = .4*.9 = .36 beats .297
    assert raw.logprob == pytest.approx(math.log(0.36), rel=1e-5)
    (norm,) = beam_search(model, _feats(), beam=2, max_len=10, length_alpha=0.7)  # type: ignore[arg-type]
    assert norm.ids == (X, Y)  # log(.297)/3^.7 > log(.36)/2^.7
    assert norm.score == pytest.approx(math.log(0.297) / 3**0.7, rel=1e-5)


def test_beam_width_one_equals_greedy_on_stub() -> None:
    model = StubModel(CHAIN)
    (g,) = greedy_decode(model, _feats(), 10)  # type: ignore[arg-type]
    (b,) = beam_search(model, _feats(), 1, 10)  # type: ignore[arg-type]
    assert g.ids == b.ids and g.logprob == pytest.approx(b.logprob)


def test_never_emits_specials_and_respects_max_len() -> None:
    # Specials are the most likely tokens and <end> is nearly impossible.
    chain = {
        p: {PAD_IDX: 0.3, START_IDX: 0.3, UNK_IDX: 0.2, X: 0.15, Y: 0.05} for p in (START_IDX, X, Y)
    }
    model = StubModel(chain, floor=1e-9)
    for beam in (1, 3):
        (hyp,) = decode(model, _feats(), beam=beam, max_len=7)  # type: ignore[arg-type]
        assert len(hyp.ids) == 7 and not hyp.finished
        assert not {PAD_IDX, START_IDX, UNK_IDX, END_IDX} & set(hyp.ids)


def test_stops_at_end_immediately() -> None:
    model = StubModel({START_IDX: {END_IDX: 0.99, X: 0.01}})
    for beam in (1, 2):
        (hyp,) = decode(model, _feats(), beam=beam, max_len=5)  # type: ignore[arg-type]
        assert hyp.ids == () and hyp.finished and hyp.attention is None
        assert hyp.mean_logprob == pytest.approx(math.log(0.99), rel=1e-4)


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_beam_one_equals_greedy_and_batched_equals_single(kind: str) -> None:
    torch.manual_seed(3)
    cfg = ModelConfig.model_validate(
        {
            "kind": kind,
            "embed_dim": 8,
            "decoder_dim": 16,
            "attention_dim": 8,
            "encoder_dim": 12,
            "dropout": 0.3,
        }
    )
    model = build_model(cfg, 40).eval()
    feats = torch.randn(4, 49, 12)
    greedy = greedy_decode(model, feats, 12)  # type: ignore[arg-type]
    beam1 = beam_search(model, feats, 1, 12)  # type: ignore[arg-type]
    assert [h.ids for h in greedy] == [h.ids for h in beam1]
    batched = beam_search(model, feats, 3, 12, 0.7)  # type: ignore[arg-type]
    single = [beam_search(model, feats[i : i + 1], 3, 12, 0.7)[0] for i in range(4)]  # type: ignore[arg-type]
    assert [h.ids for h in batched] == [h.ids for h in single]
    for a, b in zip(batched, single, strict=True):
        assert a.score == pytest.approx(b.score, abs=1e-5)
        assert all(len(h.ids) <= 12 for h in (a, b))
    if kind == "attention":
        h = next(h for h in batched if h.ids)
        assert h.attention is not None and h.attention.shape == (len(h.ids), 49)
        torch.testing.assert_close(h.attention.sum(-1), torch.ones(len(h.ids)))
    else:
        assert all(h.attention is None for h in batched)


def test_length_penalty_and_hypothesis() -> None:
    assert length_penalty(4, 0.0) == 1.0
    assert length_penalty(0, 0.7) == 1.0
    assert torch.allclose(length_penalty(torch.tensor([1, 8]), 1.0), torch.tensor([1.0, 8.0]))  # type: ignore[arg-type]
    h = Hypothesis((4, 5), -2.0, -1.0, None, True)
    assert h.mean_logprob == pytest.approx(-2.0 / 3)

from __future__ import annotations

from pathlib import Path

import pytest

from captioner.config import Config
from captioner.data.neutral import (
    FEMALE_WORDS,
    GENDERED_WORDS,
    MALE_WORDS,
    gender_of,
    neutralize_aligned,
    neutralize_text,
    neutralize_tokens,
)
from captioner.data.prepare import load_prepared, prepare_data
from captioner.evaluation.fairness import gender_metrics
from captioner.inference.predictor import Predictor


@pytest.mark.parametrize(
    ("raw", "neutral"),
    [
        ("a man rides a bike", "a person rides a bike"),
        ("two women walk with their dogs", "two people walk with their dogs"),
        ("a girl in her red dress", "a child in their red dress"),
        ("she is smiling", "they are smiling"),
        ("a man and a woman sit on a bench", "two people sit on a bench"),
        ("a boy and a girl play", "two children play"),
        ("a policeman on a horse", "a officer on a horse"),
        ("a dog runs", "a dog runs"),
    ],
)
def test_neutralize_text(raw: str, neutral: str) -> None:
    assert neutralize_text(raw) == neutral
    assert not set(neutralize_text(raw).split()) & GENDERED_WORDS


def test_aligned_rewrite_keeps_length() -> None:
    words = ["a", "man", "and", "a", "woman", "he", "is", "happy"]
    out = neutralize_aligned(words)
    assert len(out) == len(words)
    assert out == ["a", "person", "and", "a", "person", "they", "are", "happy"]


def test_gender_lexicon_partition() -> None:
    assert MALE_WORDS | FEMALE_WORDS <= GENDERED_WORDS
    assert not MALE_WORDS & FEMALE_WORDS
    assert gender_of(["a", "man"]) == "male" and gender_of(["a", "girl"]) == "female"
    assert gender_of(["a", "man", "and", "a", "woman"]) == "mixed"
    assert gender_of(["a", "dog"]) is None
    assert neutralize_tokens([]) == []


def test_gender_metrics() -> None:
    refs = [
        ["A man rides a bike.", "A guy on a bicycle."],  # agreed: male
        ["A person surfs.", "Someone on a wave."],  # neutral → any gender is a guess
        ["A woman walks.", "A lady walking."],  # agreed: female
    ]
    preds = [["a", "woman", "rides"], ["a", "man", "surfs"], ["a", "person", "walks"]]
    m = gender_metrics(preds, refs)
    assert m["gender_word_rate"] == pytest.approx(2 / 3)
    assert m["gender_error_rate"] == pytest.approx(1 / 2)
    assert m["gender_guess_rate"] == pytest.approx(1.0)
    assert m["n_reference_gender_agreed"] == 2
    neutral = gender_metrics([["a", "person"]] * 3, refs)
    assert (
        neutral["gender_word_rate"]
        == neutral["gender_error_rate"]
        == neutral["gender_guess_rate"]
        == 0
    )


def test_neutral_vocab_variant(mini_cfg: Config) -> None:
    cfg = mini_cfg.with_overrides({"vocab": {"gender_neutral": True}})
    assert cfg.vocab_file.name == "vocab_mf1_neutral.json"
    assert mini_cfg.vocab_file.name == "vocab_mf1.json"
    prepared = prepare_data(cfg)
    assert prepared.vocab.text_transform == "gender_neutral"
    assert not set(prepared.vocab.itos) & GENDERED_WORDS
    assert "person" in prepared.vocab or "people" in prepared.vocab or "child" in prepared.vocab
    assert load_prepared(cfg).vocab == prepared.vocab
    from captioner.data.dataset import build_caption_table

    table = build_caption_table(prepared.manifest, "train", prepared.vocab)
    words = {prepared.vocab.itos[i] for i in table.captions.ravel()}
    assert not words & GENDERED_WORDS


def test_served_captions_are_gender_neutral(trained_runs: dict[str, Path]) -> None:
    import torch

    torch.manual_seed(7)
    pred = Predictor.from_run(trained_runs["merge"])
    assert pred.cfg.decode.gender_neutral_output
    from tests.conftest import MINI

    for img in sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[:5]:
        assert not set(pred.caption(img, beam=2).tokens) & GENDERED_WORDS

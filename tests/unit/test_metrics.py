from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.evaluation.metrics import METRIC_KEYS, coco_scores, java_available

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "coco_golden.json").read_text()
)

needs_java = pytest.mark.skipif(not java_available(), reason="Java is required for PTB/METEOR")


@needs_java
def test_wrapper_reproduces_pycocoevalcap_golden() -> None:
    res = coco_scores(GOLDEN["hypotheses"], GOLDEN["references"])
    assert res.java and res.tokenizer == "ptb"
    for key in METRIC_KEYS:
        assert res.corpus[key] == pytest.approx(GOLDEN["scores"][key], rel=1e-6), key
    assert res.bleu4_per_image == pytest.approx(GOLDEN["bleu4_per_image"], rel=1e-6)
    assert res.cider_per_image == pytest.approx(GOLDEN["cider_per_image"], rel=1e-6)


def test_identical_and_disjoint_bleu_without_java() -> None:
    refs = [["a dog runs on the grass"], ["two kids play in the sand"]]
    same = coco_scores(
        ["a dog runs on the grass", "two kids play in the sand"], refs, use_java=False
    )
    assert same.corpus["BLEU-4"] == pytest.approx(1.0, abs=1e-6)
    assert same.corpus["METEOR"] is None and same.tokenizer == "simple" and not same.java
    other = coco_scores(
        ["red car parked by trees", "old man reads newspaper today"], refs, use_java=False
    )
    assert other.corpus["BLEU-4"] == pytest.approx(0.0, abs=1e-6)
    assert other.corpus["CIDEr"] == pytest.approx(0.0, abs=1e-6)


def test_coco_scores_input_validation() -> None:
    with pytest.raises(ValueError):
        coco_scores([], [])
    with pytest.raises(ValueError):
        coco_scores(["a"], [])


def test_java_detection_without_java(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert java_available() is False

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import captioner.inference.predictor as predictor_mod
from captioner.data import transforms
from captioner.inference.predictor import Predictor
from captioner.inference.release import ReleaseError, export_run, load_release, verify_release
from tests.conftest import MINI

IMAGE = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0]


def _predictor(run_dir: Path) -> Predictor:
    torch.manual_seed(7)  # random-init encoder (no pretrained weights in CI)
    return Predictor.from_run(run_dir)


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_predictor_caption(trained_runs: dict[str, Path], kind: str) -> None:
    p = _predictor(trained_runs[kind])
    cap = p.caption(IMAGE.read_bytes(), beam=2)
    assert cap.beam == 2 and cap.latency_s > 0
    assert cap.text == "" or (cap.text[0].isupper() and cap.text.endswith("."))
    assert all(not t.startswith("<") for t in cap.tokens)
    if kind == "attention":
        assert p.has_attention
        assert cap.attention is None or cap.attention.shape == (len(cap.tokens), 7, 7)
    else:
        assert cap.attention is None and not p.has_attention
    assert p.caption(IMAGE).tokens == p.caption(str(IMAGE)).tokens


def test_regression_D6_predictor_uses_shared_preprocessing() -> None:
    assert predictor_mod.load_image is transforms.load_image


@pytest.mark.parametrize("kind", ["merge", "attention"])
def test_export_roundtrip_identical_predictions(
    trained_runs: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    run_dir = trained_runs[kind]
    sums = export_run(run_dir, tmp_path / "rel")
    assert set(sums) == {
        "model.safetensors",
        "vocab.json",
        "config.yaml",
        "model_card.md",
        "reliability.safetensors",
    }
    assert json.loads((tmp_path / "rel" / "checksums.json").read_text()) == sums
    card = (tmp_path / "rel" / "model_card.md").read_text()
    assert "Not yet evaluated" in card and "non-commercial" in card
    assert all(
        k.startswith("decoder.") for k in load_file(str(tmp_path / "rel" / "model.safetensors"))
    )
    a = _predictor(run_dir)
    torch.manual_seed(7)
    b = load_release(tmp_path / "rel")
    for beam in (1, 3):
        ca, cb = a.caption(IMAGE, beam=beam), b.caption(IMAGE, beam=beam)
        assert ca.tokens == cb.tokens and ca.score == pytest.approx(cb.score)


def test_verify_release_detects_tampering(trained_runs: dict[str, Path], tmp_path: Path) -> None:
    rel = tmp_path / "rel"
    export_run(trained_runs["merge"], rel)
    verify_release(rel)
    with (rel / "vocab.json").open("a") as fh:
        fh.write(" ")
    with pytest.raises(ReleaseError, match="SHA-256"):
        verify_release(rel)
    (rel / "checksums.json").unlink()
    with pytest.raises(ReleaseError, match="missing"):
        load_release(rel)

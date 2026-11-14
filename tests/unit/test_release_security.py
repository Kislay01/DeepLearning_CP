"""Release download verification, security/accessibility gates, and regressions D4, D7, D12."""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from captioner.data.vocab import UNK_IDX, build_vocab
from captioner.inference import service
from captioner.inference.decode import beam_search, greedy_decode
from captioner.inference.release import (
    ReleaseError,
    ReleaseSpec,
    export_run,
    fetch_release,
    load_release,
    load_release_spec,
)
from tests.conftest import ROOT
from tests.unit.test_decode import CHAIN, StubModel, _feats

STREAMLIT = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())


def _published(trained_runs: dict[str, Path], tmp_path: Path) -> ReleaseSpec:
    sums = export_run(trained_runs["attention"], tmp_path / "upload")
    return ReleaseSpec(version="9.9.9", base_url=(tmp_path / "upload").as_uri(), files=sums)


def test_fetch_release_verifies_sha256(trained_runs: dict[str, Path], tmp_path: Path) -> None:
    spec = _published(trained_runs, tmp_path)
    dest = fetch_release(spec, tmp_path / "cache")
    assert load_release(dest).vocab is not None
    assert fetch_release(spec, tmp_path / "cache") == dest  # cached files reused
    (dest / "vocab.json").write_text("tampered")
    assert fetch_release(spec, tmp_path / "cache") == dest  # re-downloaded because hash differed
    bad = spec.model_copy(update={"files": {**spec.files, "model.safetensors": "0" * 64}})
    with pytest.raises(ReleaseError, match="mismatch"):
        fetch_release(bad, tmp_path / "cache2")
    assert not list((tmp_path / "cache2").rglob("*.safetensors"))
    with pytest.raises(ReleaseError, match="no published"):
        fetch_release(ReleaseSpec(version="1"), tmp_path / "c3")


def test_app_downloads_pinned_release_at_startup(
    trained_runs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _published(trained_runs, tmp_path)
    cfg_file = tmp_path / "release.yaml"
    cfg_file.write_text(yaml.safe_dump(spec.model_dump()))
    for var in ("CAPTIONER_RELEASE_DIR", "CAPTIONER_RUN_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTIONER_RELEASE_CONFIG", str(cfg_file))
    monkeypatch.setenv("CAPTIONER_CACHE_DIR", str(tmp_path / "cache"))
    source = service.resolve_model_source(tmp_path / "no-artifacts")
    assert source.kind == "release" and (source.path / "model.safetensors").exists()
    cfg_file.write_text(
        yaml.safe_dump({**spec.model_dump(), "files": {**spec.files, "vocab.json": "0" * 64}})
    )
    monkeypatch.setenv("CAPTIONER_CACHE_DIR", str(tmp_path / "cache-b"))
    with pytest.raises(service.ModelNotLoadedError, match="pinned"):
        service.resolve_model_source(tmp_path / "no-artifacts")


def test_repo_release_spec_is_valid_and_unpublished() -> None:
    spec = load_release_spec(ROOT / "configs" / "release.yaml")
    assert spec.version and not spec.is_published  # nothing is published until a real run exists


# --- security.md §9 / §6 -------------------------------------------------------------------


def test_streamlit_hardening() -> None:
    assert STREAMLIT["server"]["maxUploadSize"] == 10
    assert STREAMLIT["server"]["enableXsrfProtection"] is True
    assert STREAMLIT["client"]["showErrorDetails"] == "none"


def test_no_unsafe_html_or_print_in_app_and_library() -> None:
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "app").rglob("*.py")]:
        text = path.read_text()
        assert "unsafe_allow_html" not in text, path
        assert not re.search(r"^\s*print\(", text, re.MULTILINE), path
        assert not re.search(r"except\s*:", text), path


# --- design_system.md §6: contrast ≥ 4.5:1 in both themes ------------------------------------


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_contrast_at_least_4_5(theme: str) -> None:
    t = STREAMLIT["theme"][theme]
    for fg in ("textColor", "primaryColor", "greenColor", "orangeColor", "redColor"):
        for bg in ("backgroundColor", "secondaryBackgroundColor"):
            ratio = _contrast(t[fg], t[bg])
            assert ratio >= 4.5, f"{theme} {fg} on {bg}: {ratio:.2f}"


# --- named regressions for reference defects ------------------------------------------------


def test_regression_D4_min_freq_threshold_maps_rare_words_to_unk() -> None:
    caps = [["a", "dog"]] * 5 + [["a", "zebra"]]
    vocab = build_vocab(caps, min_freq=5, max_len=10)
    assert vocab["zebra"] == UNK_IDX and vocab["dog"] != UNK_IDX
    assert vocab.n_words == 2  # singletons do not get their own softmax column


def test_regression_D7_beam_search_finds_higher_probability_caption() -> None:
    model = StubModel(CHAIN)
    (greedy,) = greedy_decode(model, _feats(), 10)  # type: ignore[arg-type]
    (beam,) = beam_search(model, _feats(), 2, 10, length_alpha=0.0)  # type: ignore[arg-type]
    assert beam.logprob > greedy.logprob
    assert math.exp(beam.logprob) == pytest.approx(0.36, rel=1e-4)


def test_regression_D12_reproducible_and_portable() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = " ".join(pyproject["project"]["dependencies"])
    assert "sklearn" not in deps.replace("scikit-learn", "")
    assert (ROOT / "uv.lock").exists()
    lock = (ROOT / "uv.lock").read_text()
    for pkg in ("torch", "torchvision", "streamlit", "gtts", "pycocoevalcap"):
        assert re.search(rf'^name = "{pkg}"\n^version = "', lock, re.MULTILINE), pkg
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "app").rglob("*.py")]:
        text = path.read_text()
        assert not re.search(r"[A-Za-z]:\\\\", text), f"Windows path in {path}"
        assert "/home/" not in text and "fit_generator" not in text and "layers.merge" not in text

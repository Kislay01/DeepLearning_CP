"""testing.md §6 — Streamlit AppTest with TTS mocked."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from captioner.inference import service
from captioner.tts import speak
from tests.conftest import MINI, ROOT

APP = str(ROOT / "app" / "streamlit_app.py")
IMAGE = sorted((MINI / "Flicker8k_Dataset").glob("*.jpg"))[0].read_bytes()
FAKE_MP3 = b"ID3fake-mp3-bytes"


@pytest.fixture
def app_env(
    monkeypatch: pytest.MonkeyPatch, trained_runs: dict[str, Path]
) -> Iterator[dict[str, int]]:
    st.cache_resource.clear()
    monkeypatch.setenv("CAPTIONER_RUN_DIR", str(trained_runs["attention"]))
    calls = {"load": 0, "tts": 0}
    real_load = service.load_predictor

    def counting_load(source: service.ModelSource) -> object:
        calls["load"] += 1
        return real_load(source)

    def fake_tts(text: str, **_kw: object) -> speak.Speech:
        calls["tts"] += 1
        return speak.Speech(FAKE_MP3, "audio/mp3", "fake")

    monkeypatch.setattr(service, "load_predictor", counting_load)
    monkeypatch.setattr(speak, "synthesize", fake_tts)
    yield calls
    st.cache_resource.clear()


def _run_with_upload(data: bytes, name: str = "photo.jpg") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.file_uploader[0].set_value((name, data, "image/jpeg"))
    at.run()
    return at


def test_upload_shows_caption_and_audio(app_env: dict[str, int]) -> None:
    at = _run_with_upload(IMAGE)
    assert not at.exception
    assert at.title[0].value == "Image Caption Generator"
    assert any(m.value.startswith("## ") for m in at.markdown)
    assert at.success or at.warning
    assert len(at.get("audio")) == 1
    assert app_env["tts"] == 1
    assert at.code and at.code[0].value.endswith(".")


def test_tts_failure_still_shows_caption(
    app_env: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(text: str, **_kw: object) -> speak.Speech:
        raise speak.TTSError("offline")

    monkeypatch.setattr(speak, "synthesize", broken)
    at = _run_with_upload(IMAGE)
    assert not at.exception
    assert any(m.value.startswith("## ") for m in at.markdown)
    assert any("Audio unavailable" in i.value for i in at.info)
    assert len(at.get("audio")) == 0


def test_invalid_file_shows_friendly_error(app_env: dict[str, int]) -> None:
    at = _run_with_upload(b"definitely not an image", "fake.jpg")
    assert not at.exception
    assert any("isn't an image we can read" in e.value for e in at.error)
    assert not any(m.value.startswith("## ") for m in at.markdown)


def test_regression_D10_predictor_loaded_once(app_env: dict[str, int]) -> None:
    at = _run_with_upload(IMAGE)
    for _ in range(3):
        at.run()
    at2 = AppTest.from_file(APP, default_timeout=120)
    at2.run()
    assert app_env["load"] == 1


def test_regression_D9_audio_is_in_memory_and_autoplay_off(
    app_env: dict[str, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    at = _run_with_upload(IMAGE)
    audio = at.get("audio")[0]
    assert audio.proto.autoplay is False
    assert not list(tmp_path.rglob("*.mp3"))  # nothing written to a shared path


def test_heatmaps_and_about_present(app_env: dict[str, int]) -> None:
    at = _run_with_upload(IMAGE)
    labels = [e.label for e in at.expander]
    assert "See where the model looked (heat-maps)" in labels
    assert "About & limitations" in labels
    about = " ".join(m.value for m in at.markdown)
    assert "nothing leaves it" in about and "Google" not in about


def test_missing_model_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    st.cache_resource.clear()
    monkeypatch.delenv("CAPTIONER_RUN_DIR", raising=False)
    monkeypatch.delenv("CAPTIONER_RELEASE_DIR", raising=False)
    monkeypatch.setenv("CAPTIONER_ARTIFACTS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No trained model found" in e.value for e in at.error)
    assert at.button[0].disabled
    st.cache_resource.clear()

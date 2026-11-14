"""ADR-020 — the pinned Piper voice downloads, verifies and speaks locally (needs GitHub access)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from captioner.config import TTSConfig
from captioner.tts import speak

pytestmark = pytest.mark.integration


def test_real_piper_voice_speaks_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPTIONER_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CAPTIONER_VOICE_DIR", raising=False)
    speech = speak.synthesize(
        "A black dog is running through the grass.", engine="piper", cfg=TTSConfig()
    )
    assert speech.engine == "piper"
    with wave.open(io.BytesIO(speech.audio)) as wav:
        seconds = wav.getnframes() / wav.getframerate()
    assert 1.0 < seconds < 6.0

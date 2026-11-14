from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from captioner.config import TTSConfig
from captioner.inference import service
from captioner.inference.predictor import Caption
from captioner.tts import speak


class _FakeGTTS:
    fail: bool = False
    last_timeout: float | None = None

    def __init__(self, text: str, lang: str, timeout: float) -> None:
        _FakeGTTS.last_timeout = timeout
        self.text = text

    def write_to_fp(self, fp: io.BytesIO) -> None:
        if _FakeGTTS.fail:
            from gtts.tts import gTTSError

            raise gTTSError("network down")
        fp.write(b"ID3" + self.text.encode())


@pytest.fixture
def fake_gtts(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGTTS]:
    import gtts

    _FakeGTTS.fail = False
    monkeypatch.setattr(gtts, "gTTS", _FakeGTTS)
    return _FakeGTTS


ONLINE = TTSConfig(allow_online=True)


def test_gtts_returns_in_memory_mp3(fake_gtts: type[_FakeGTTS]) -> None:
    speech = speak.synthesize("A dog runs.", engine="gtts", timeout_s=5.0, cfg=ONLINE)
    assert speech.audio == b"ID3A dog runs." and speech.mime == "audio/mp3"
    assert fake_gtts.last_timeout == 5.0


def test_tts_failures_raise_tts_error(
    fake_gtts: type[_FakeGTTS], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(speak.TTSError, match="disabled"):
        speak.synthesize("A dog runs.", engine="gtts")  # online speech is off by default
    fake_gtts.fail = True
    with pytest.raises(speak.TTSError):
        speak.synthesize("A dog runs.", engine="gtts", cfg=ONLINE)
    monkeypatch.setattr(speak, "offline_available", lambda: False)
    monkeypatch.setattr(speak, "piper_available", lambda: False)
    with pytest.raises(speak.TTSError, match="Piper"):
        speak.synthesize("A dog runs.", engine="auto")
    with pytest.raises(speak.TTSError, match="pyttsx3"):
        speak.synthesize("A dog runs.", engine="offline")
    with pytest.raises(speak.TTSError, match="Nothing"):
        speak.synthesize("   ")


def test_auto_falls_back_to_offline(
    fake_gtts: type[_FakeGTTS], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_piper(text: str, cfg: TTSConfig) -> speak.Speech:
        raise speak.TTSError("no voice")

    monkeypatch.setattr(speak, "_piper", no_piper)
    monkeypatch.setattr(speak, "offline_available", lambda: True)
    monkeypatch.setattr(
        speak, "_pyttsx3", lambda text: speak.Speech(b"RIFF", "audio/wav", "pyttsx3")
    )
    assert speak.synthesize("A dog.", engine="auto").engine == "pyttsx3"


def test_resolve_model_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("CAPTIONER_RELEASE_DIR", "CAPTIONER_RUN_DIR"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(service.ModelNotLoadedError):
        service.resolve_model_source(tmp_path)
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"x")
    assert service.resolve_model_source(tmp_path) == service.ModelSource("run", run)
    from captioner import __version__

    rel = tmp_path / "release" / __version__
    rel.mkdir(parents=True)
    (rel / "model.safetensors").write_bytes(b"x")
    assert service.resolve_model_source(tmp_path).kind == "release"
    monkeypatch.setenv("CAPTIONER_RUN_DIR", "/some/run")
    assert service.resolve_model_source(tmp_path) == service.ModelSource("run", Path("/some/run"))
    monkeypatch.setenv("CAPTIONER_RELEASE_DIR", "/some/rel")
    assert service.resolve_model_source(tmp_path).kind == "release"


class _SlowPredictor:
    def caption(self, image: Image.Image, beam: int) -> str:
        time.sleep(0.5)
        return "late"


def test_caption_timeout() -> None:
    with pytest.raises(service.InferenceTimeoutError):
        service.caption_with_timeout(_SlowPredictor(), Image.new("RGB", (4, 4)), 3, 0.05)  # type: ignore[arg-type]


def test_build_heatmaps() -> None:
    img = Image.new("RGB", (300, 300))
    att = np.random.default_rng(0).random((3, 7, 7)).astype(np.float32)
    cap = Caption("A dog runs.", ("a", "dog", "runs"), -1.0, -0.5, att, False, 3, 0.1)
    maps = service.build_heatmaps(img, cap, max_words=2)
    assert maps is not None and maps.words == ["a", "dog"] and len(maps.images) == 2
    assert "'dog' →" in maps.summary
    no_att = Caption("A dog.", ("a", "dog"), -1.0, -0.5, None, False, 3, 0.1)
    assert service.build_heatmaps(img, no_att, 12) is None


def test_load_predictor_from_run(trained_runs: dict[str, Path]) -> None:
    pred = service.load_predictor(service.ModelSource("run", trained_runs["merge"]))
    assert not pred.has_attention


def _voice_archive(tmp_path: Path, model: bytes, config: bytes, extra: str | None = None) -> Path:
    import tarfile

    arc = tmp_path / "voice.tar.gz"
    with tarfile.open(arc, "w:gz") as tar:
        for name, data in (("v/m.onnx", model), ("v/m.onnx.json", config)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra:
            info = tarfile.TarInfo(extra)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    return arc


def _voice_cfg(arc: Path, model: bytes, config: bytes) -> TTSConfig:
    from captioner.utils.io import bytes_digest, file_digest

    return TTSConfig(
        voice_url=arc.as_uri(),
        voice_sha256=file_digest(arc),
        model_file="m.onnx",
        model_sha256=bytes_digest(model),
        config_file="m.onnx.json",
        config_sha256=bytes_digest(config),
    )


def test_ensure_voice_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPTIONER_VOICE_DIR", raising=False)
    arc = _voice_archive(tmp_path, b"MODEL", b"{}")
    cfg = _voice_cfg(arc, b"MODEL", b"{}")
    model, config = speak.ensure_voice(cfg)
    assert model.read_bytes() == b"MODEL" and config.read_bytes() == b"{}"
    arc.unlink()  # cached: no second download needed
    assert speak.ensure_voice(cfg) == (model, config)
    bad = cfg.model_copy(update={"voice_sha256": "0" * 64, "model_sha256": "1" * 64})
    with pytest.raises(speak.TTSError):
        speak.ensure_voice(bad)


def test_ensure_voice_rejects_tampered_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONER_VOICE_DIR", str(tmp_path / "voice"))
    arc = _voice_archive(tmp_path, b"MODEL", b"{}")
    cfg = _voice_cfg(arc, b"OTHER", b"{}")  # archive hash ok, model hash pinned to other bytes
    with pytest.raises(speak.TTSError, match="verification"):
        speak.ensure_voice(cfg)
    assert not (tmp_path / "voice" / "m.onnx").exists()
    with pytest.raises(speak.TTSError, match="download failed"):
        speak.ensure_voice(
            cfg.model_copy(update={"voice_url": (tmp_path / "missing.tgz").as_uri()})
        )


def test_piper_engine_with_stub_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import wave

    class StubVoice:
        def synthesize_wav(self, text: str, wav: wave.Wave_write) -> None:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\x00\x01" * 100)

    monkeypatch.setattr(speak, "ensure_voice", lambda cfg: (tmp_path / "m", tmp_path / "c"))
    monkeypatch.setattr(speak, "_load_piper", lambda m, c: StubVoice())
    speech = speak.synthesize("A dog runs.", engine="piper")
    assert speech.engine == "piper" and speech.mime == "audio/wav"
    assert speech.audio[:4] == b"RIFF"

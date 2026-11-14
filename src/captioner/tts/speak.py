"""Text-to-speech → in-memory audio bytes for the browser (fixes reference D9).

Privacy-first (ADR-020):

* ``piper`` (default) — local neural voice (Piper, ONNX). The voice is downloaded once from a
  pinned URL, SHA-256-verified, cached, and run in-process. Nothing leaves the server.
* ``offline`` — eSpeak through ``pyttsx3`` (optional extra), also local.
* ``gtts`` — Google Translate's unofficial endpoint. Sends the caption text (never the image) to
  Google; only allowed when ``tts.allow_online`` is true.

Failures raise ``TTSError`` so the UI can fall back to text. Audio is never played on the server.
"""

from __future__ import annotations

import functools
import importlib.util
import io
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from captioner.config import TTSConfig
from captioner.utils.io import file_digest

logger = logging.getLogger(__name__)

Engine = Literal["piper", "offline", "gtts", "auto"]


class TTSError(RuntimeError):
    """Speech synthesis failed or no engine is available."""


@dataclass(frozen=True)
class Speech:
    """Synthesised audio."""

    audio: bytes
    mime: str
    engine: str


def offline_available() -> bool:
    """True if the optional ``pyttsx3`` engine is installed."""
    return importlib.util.find_spec("pyttsx3") is not None


def piper_available() -> bool:
    """True if the ``piper-tts`` runtime is installed."""
    return importlib.util.find_spec("piper") is not None


# --- Piper voice: pinned download + verification -------------------------------------------


def voice_dir(cfg: TTSConfig) -> Path:
    """Folder holding the verified voice files (``CAPTIONER_VOICE_DIR`` or the user cache)."""
    if env := os.environ.get("CAPTIONER_VOICE_DIR"):
        return Path(env)
    cache = Path(os.environ.get("CAPTIONER_CACHE_DIR", Path.home() / ".cache" / "captioner"))
    return cache / "voices" / cfg.voice_sha256[:16]


def _verified(path: Path, sha256: str) -> bool:
    return path.exists() and file_digest(path) == sha256


def _extract_voice(archive: Path, dest: Path, wanted: set[str]) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            if Path(m.name).name not in wanted or not m.isfile():
                continue
            if Path(m.name).is_absolute() or ".." in Path(m.name).parts:
                raise TTSError(f"Unsafe path in voice archive: {m.name!r}")
            src = tar.extractfile(m)
            if src is None:
                raise TTSError(f"Cannot read {m.name} from voice archive")
            with src, (dest / Path(m.name).name).open("wb") as out:
                shutil.copyfileobj(src, out)


def _download(url: str, dest: Path, timeout_s: float) -> None:
    logger.info("Downloading voice %s", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
    except OSError as exc:
        raise TTSError(f"Voice download failed: {exc}") from exc


def ensure_voice(cfg: TTSConfig, timeout_s: float = 120.0) -> tuple[Path, Path]:
    """Return verified ``(model.onnx, config.json)``, downloading the pinned voice if needed.

    Raises:
        TTSError: On download failure or any SHA-256 mismatch (files are never used unverified).
    """
    dest = voice_dir(cfg)
    model, config = dest / cfg.model_file, dest / cfg.config_file
    if _verified(model, cfg.model_sha256) and _verified(config, cfg.config_sha256):
        return model, config
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="captioner-voice-") as tmp:
        archive = Path(tmp) / "voice.tar.gz"
        _download(cfg.voice_url, archive, timeout_s)
        if file_digest(archive) != cfg.voice_sha256:
            raise TTSError("Voice archive SHA-256 mismatch; refusing to use it")
        _extract_voice(archive, dest, {cfg.model_file, cfg.config_file})
    if not (_verified(model, cfg.model_sha256) and _verified(config, cfg.config_sha256)):
        for p in (model, config):
            p.unlink(missing_ok=True)
        raise TTSError("Voice files failed SHA-256 verification")
    return model, config


@functools.lru_cache(maxsize=2)
def _load_piper(model: str, config: str) -> Any:
    from piper import PiperVoice

    return PiperVoice.load(model, config_path=config)


def _piper(text: str, cfg: TTSConfig) -> Speech:
    if not piper_available():
        raise TTSError("Piper is not installed (uv sync installs piper-tts by default)")
    model, config = ensure_voice(cfg)
    voice = _load_piper(str(model), str(config))
    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wav:
            voice.synthesize_wav(text, wav)
    except (RuntimeError, OSError, ValueError) as exc:
        raise TTSError(f"Piper failed: {exc}") from exc
    if len(buf.getvalue()) <= 44:  # WAV header only
        raise TTSError("Piper produced no audio")
    return Speech(buf.getvalue(), "audio/wav", "piper")


# --- other engines -------------------------------------------------------------------------


def _gtts(text: str, lang: str, timeout_s: float) -> Speech:
    from gtts import gTTS
    from gtts.tts import gTTSError

    buf = io.BytesIO()
    try:
        gTTS(text=text, lang=lang, timeout=timeout_s).write_to_fp(buf)
    except (gTTSError, OSError, ValueError, AssertionError) as exc:
        raise TTSError(f"gTTS failed: {exc}") from exc
    if not buf.getvalue():
        raise TTSError("gTTS returned no audio")
    return Speech(buf.getvalue(), "audio/mp3", "gtts")


def _pyttsx3(text: str) -> Speech:
    if not offline_available():
        raise TTSError(
            "eSpeak speech needs the optional `pyttsx3` package (uv sync --extra offline-tts)"
        )
    import pyttsx3

    # pyttsx3 can only render to a file: use a private temp dir that is removed immediately.
    with tempfile.TemporaryDirectory(prefix="captioner-tts-") as tmp:
        out = Path(tmp) / "speech.wav"
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, str(out))
            engine.runAndWait()
        except (RuntimeError, OSError) as exc:
            raise TTSError(f"pyttsx3 failed: {exc}") from exc
        if not out.exists() or out.stat().st_size == 0:
            raise TTSError("pyttsx3 produced no audio")
        return Speech(out.read_bytes(), "audio/wav", "pyttsx3")


def synthesize(
    text: str,
    engine: Engine = "auto",
    timeout_s: float = 5.0,
    lang: str = "en",
    cfg: TTSConfig | None = None,
) -> Speech:
    """Turn a caption into audio bytes.

    Args:
        text: Caption to speak.
        engine: ``"piper"`` (local neural), ``"offline"`` (eSpeak), ``"gtts"`` (online, needs
            ``cfg.allow_online``) or ``"auto"`` (Piper, then eSpeak — local engines only).
        timeout_s: Network timeout for gTTS; no retries in the request path.
        lang: Language code (gTTS only; the Piper voice is English).
        cfg: Voice and online settings; defaults to ``TTSConfig()``.

    Returns:
        ``Speech`` with the audio bytes and MIME type.

    Raises:
        TTSError: If no allowed engine could produce audio.
    """
    cfg = cfg or TTSConfig()
    if not text.strip():
        raise TTSError("Nothing to speak")
    if engine == "gtts":
        if not cfg.allow_online:
            raise TTSError("Online speech is disabled (tts.allow_online: false)")
        return _gtts(text, lang, timeout_s)
    if engine == "offline":
        return _pyttsx3(text)
    try:
        return _piper(text, cfg)
    except TTSError:
        if engine == "piper" or not offline_available():
            raise
        logger.info("Piper unavailable, falling back to eSpeak")
        return _pyttsx3(text)

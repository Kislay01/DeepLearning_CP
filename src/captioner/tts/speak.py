"""Text-to-speech → in-memory audio bytes for the browser (fixes reference D9).

``gTTS`` (online) sends **only the caption text** to Google's TTS endpoint; ``pyttsx3`` (offline,
optional extra) speaks locally. Failures raise ``TTSError`` so the UI can fall back to text.
Audio is never played on the server and never written to a shared path.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Engine = Literal["gtts", "offline", "auto"]


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
            "Offline speech needs the optional `pyttsx3` package (uv sync --extra offline-tts)"
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
    text: str, engine: Engine = "auto", timeout_s: float = 5.0, lang: str = "en"
) -> Speech:
    """Turn a caption into audio bytes.

    Args:
        text: Caption to speak (only this string is sent to the online service).
        engine: ``"gtts"`` (online), ``"offline"`` (pyttsx3) or ``"auto"`` (gTTS, then offline).
        timeout_s: Network timeout for gTTS; no retries in the request path.
        lang: Language code.

    Returns:
        ``Speech`` with the audio bytes and MIME type.

    Raises:
        TTSError: If no engine could produce audio.
    """
    if not text.strip():
        raise TTSError("Nothing to speak")
    if engine == "offline":
        return _pyttsx3(text)
    try:
        return _gtts(text, lang, timeout_s)
    except TTSError:
        if engine == "gtts" or not offline_available():
            raise
        logger.info("gTTS unavailable, falling back to offline speech")
        return _pyttsx3(text)

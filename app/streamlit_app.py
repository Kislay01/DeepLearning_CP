"""Streamlit demo: upload → caption → speech → heat-maps. Layout and callbacks only.

Run with ``make app`` (``streamlit run app/streamlit_app.py``).
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import components as ui

from captioner.config import AppConfig
from captioner.inference import service
from captioner.inference.predictor import Predictor
from captioner.inference.validation import InvalidImageError, validate_upload
from captioner.tts import speak

SPEECH_MODES = {"Online voice": "gtts", "Offline voice": "offline", "Off": "off"}


@st.cache_resource(show_spinner="Loading the model…")
def get_predictor() -> Predictor:
    """Loaded once per process and shared by every session (NFR-2, fixes D10)."""
    return service.load_predictor(service.resolve_model_source())


def _options(predictor: Predictor | None) -> tuple[int, bool, str]:
    state = st.session_state
    default_beam = predictor.cfg.decode.beam if predictor else 3
    beam = int(state.get("beam", default_beam))
    return (
        beam,
        bool(state.get("autoplay", False)),
        SPEECH_MODES[state.get("speech", "Online voice")],
    )


def _render_options(predictor: Predictor | None) -> None:
    with st.expander("Options: beam width, autoplay, speech"):
        default_beam = predictor.cfg.decode.beam if predictor else 3
        choices = [1, 3, 5]
        index = choices.index(default_beam) if default_beam in choices else 1
        st.radio("Beam width", choices, index=index, key="beam", horizontal=True)
        st.toggle("Play audio automatically", value=False, key="autoplay")
        modes = list(SPEECH_MODES) if speak.offline_available() else ["Online voice", "Off"]
        st.radio("Speech", modes, key="speech", horizontal=True)


def _speak(text: str, mode: str, cfg: AppConfig) -> tuple[bytes | None, str, str | None]:
    if mode == "off" or not text:
        return None, "audio/mp3", None
    try:
        speech = speak.synthesize(text, engine=mode, timeout_s=cfg.tts_timeout_s)  # type: ignore[arg-type]
    except speak.TTSError:
        return None, "audio/mp3", ui.AUDIO_UNAVAILABLE
    return speech.audio, speech.mime, None


def _describe(data: bytes, predictor: Predictor, beam: int, mode: str) -> ui.Result | None:
    cfg = predictor.cfg.app
    with st.status("Describing…", expanded=False) as status:
        try:
            image = validate_upload(data, cfg)
            caption = service.caption_with_timeout(predictor, image, beam, cfg.inference_timeout_s)
        except InvalidImageError as exc:
            status.update(label="Couldn't read that file", state="error")
            st.error(f"✖ {exc.user_message}")
            return None
        except service.InferenceTimeoutError:
            status.update(label="Took too long", state="error")
            st.error("✖ Describing this image took too long. Please try again.")
            return None
        audio, mime, audio_error = _speak(caption.text, mode, cfg)
        heatmaps = service.build_heatmaps(image, caption, cfg.max_heatmap_words)
        status.update(label="Done", state="complete")
    return ui.Result(image, caption, audio, mime, audio_error, heatmaps)


def main() -> None:
    st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")
    ui.header()
    try:
        predictor: Predictor | None = get_predictor()
    except service.ModelNotLoadedError as exc:
        predictor = None
        st.error(f"✖ {exc}")
    beam, autoplay, mode = _options(predictor)

    upload = st.file_uploader(ui.UPLOAD_LABEL, type=ui.ALLOWED_TYPES, key="upload")
    camera = st.camera_input("Take a photo") if st.toggle("Use camera", key="use_camera") else None
    source = camera or upload
    clicked = st.button(
        "Describe image", type="primary", disabled=source is None or predictor is None
    )

    if source is not None and predictor is not None:
        data = source.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if clicked or st.session_state.get("last_digest") != digest:
            now = time.monotonic()
            if now - st.session_state.get("last_request", -1e9) < predictor.cfg.app.cooldown_s:
                st.warning("⚠ Please wait a moment before describing another image.")
            else:
                st.session_state["last_request"] = now
                st.session_state["last_digest"] = digest
                st.session_state["result"] = _describe(data, predictor, beam, mode)
    elif source is None:
        st.session_state.pop("result", None)
        st.session_state.pop("last_digest", None)

    result: ui.Result | None = st.session_state.get("result")
    if result is not None:
        ui.caption_card(result, autoplay)
        if predictor is not None and predictor.has_attention:
            ui.heatmap_grid(result.heatmaps)
    _render_options(predictor)
    ui.about()


main()

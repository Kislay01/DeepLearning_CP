"""Presentational components for the caption demo (docs/design_system.md §4).

Only layout lives here; models, validation and speech are in ``captioner.*``.
No raw HTML is ever rendered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import streamlit as st
from PIL import Image

from captioner.inference.predictor import Caption
from captioner.inference.service import Heatmaps

ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
UPLOAD_LABEL = "Choose a photo (JPG, PNG or WebP, up to 10 MB)"
AUDIO_UNAVAILABLE = "Audio unavailable, caption shown above."
LIMITATIONS = (
    "Trained on 8,000 Flickr photos of people and animals. It may misidentify objects, "
    "and it can guess gender or age wrongly. Not for safety-critical use."
)
PRIVACY = (
    "Your photo is processed in memory on this server and is never stored or sent anywhere. "
    "Speech is generated on this server too, so nothing leaves it."
)
PRIVACY_ONLINE = (
    " If you pick the online voice, only the caption text is sent to Google's text-to-speech "
    "service to create the audio."
)
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<])")


def escape_markdown(text: str) -> str:
    """Backslash-escape Markdown control characters so a caption renders as plain text."""
    return _MD_SPECIAL.sub(r"\\\1", text)


@dataclass(frozen=True)
class Result:
    """Everything needed to (re)render one captioning result."""

    image: Image.Image
    caption: Caption
    audio: bytes | None
    audio_mime: str
    audio_error: str | None
    heatmaps: Heatmaps | None


def header() -> None:
    st.title("Image Caption Generator")
    st.write("Describe a photo in words and speech.")


def caption_card(result: Result, autoplay: bool) -> None:
    """Image preview (alt = caption) next to the caption, status line and audio player."""
    cap = result.caption
    text = cap.text or "No description could be generated."
    left, right = st.columns([1, 1.2])
    with left:
        st.image(result.image, caption=text, alt=text, width="stretch")
    with right:
        st.markdown(f"## {escape_markdown(text)}")
        if cap.low_confidence:
            st.warning("⚠ Low confidence — this description may be inaccurate.")
        else:
            st.success(f"✓ Caption ready · {cap.latency_s:.1f} s")
        if result.audio is not None:
            st.audio(
                result.audio,
                format=result.audio_mime,
                autoplay=autoplay,
                alt=f"Spoken caption: {text}",
            )
        elif result.audio_error:
            st.info(f"ℹ {result.audio_error}")
        st.caption(
            f"Generated automatically and may be inaccurate · beam {cap.beam} · "
            f"mean log-prob {cap.mean_logprob:.2f}"
        )
        st.code(text, language=None)


def heatmap_grid(heatmaps: Heatmaps | None) -> None:
    """Per-word attention overlays with a text summary (M2 only)."""
    with st.expander("See where the model looked (heat-maps)"):
        if heatmaps is None:
            st.write("Heat-maps are available for the attention model only.")
            return
        st.write(f"Text summary: {heatmaps.summary}")
        st.image(
            heatmaps.images,
            caption=heatmaps.words,
            alt=[f"Where the model looked for '{w}'" for w in heatmaps.words],
            width=150,
        )


def about(online_speech: bool = False) -> None:
    with st.expander("About & limitations"):
        st.write(LIMITATIONS)
        st.write(PRIVACY + (PRIVACY_ONLINE if online_speech else ""))
        st.caption(
            "Model: ResNet-50 encoder + LSTM decoder with soft attention. Data: Flickr8k "
            "(Hodosh, Young & Hockenmaier, 2013), non-commercial research and education use only."
        )

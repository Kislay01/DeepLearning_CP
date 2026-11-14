# Design System — Caption Demo UI

Scope: the single-page Streamlit demo. The main audience includes **blind and low-vision users**, so accessibility comes first and decoration second. Target: WCAG 2.2 AA.

---

## 1. Principles

1. **Caption first.** The generated sentence is the largest, highest-contrast element on screen and is announced by screen readers automatically.
2. **One task per screen.** Upload → result. No sidebars needed for the core flow; advanced options are behind an expander.
3. **Every visual has a text equivalent.** Image preview uses the caption as alt text. Attention heat-maps have a text summary.
4. **Never autoplay audio by default.** Autoplay is a user toggle (off by default) because it clashes with screen-reader speech.
5. **Honest output.** Show a confidence hint and the "may be inaccurate" note. Never present a caption as fact.

## 2. Tokens

### Colour (`.streamlit/config.toml`)

| Token | Light | Dark | Use |
|---|---|---|---|
| `primaryColor` | `#1D4ED8` | `#60A5FA` | Buttons, focus ring, links |
| `backgroundColor` | `#FFFFFF` | `#0B1220` | Page |
| `secondaryBackgroundColor` | `#F1F5F9` | `#1E293B` | Cards, upload zone |
| `textColor` | `#0F172A` | `#F1F5F9` | Body + caption |
| Success | `#15803D` | `#4ADE80` | "Caption ready" |
| Warning | `#B45309` | `#FBBF24` | Low confidence |
| Error | `#B91C1C` | `#F87171` | Invalid file |

All text/background pairs above are ≥ 4.5:1. Never use colour alone to carry meaning: pair it with an icon and a word ("⚠ Low confidence").

### Typography

| Role | Size | Weight | Notes |
|---|---|---|---|
| Page title | 32 px | 700 | `st.title` |
| **Caption** | 28–32 px | 600 | `st.markdown("## …")`, max 60ch line length |
| Section heading | 20 px | 600 | `st.subheader` |
| Body | 16 px | 400 | minimum size, line-height 1.5 |
| Meta (scores, timing) | 14 px | 400 | `st.caption` |

Font: system sans (`font = "sans serif"` in config). No custom web fonts, which keeps the page fast.

### Spacing & shape
4 px base grid (4/8/16/24/32). Cards: 12 px radius, 16–24 px padding. Touch targets ≥ 44 × 44 px.

## 3. Layout

```
┌───────────────────────────────────────────────┐
│ Image Caption Generator                        │  title
│ Describe a photo in words and speech.          │  one-line purpose
├───────────────────────────────────────────────┤
│ [ Drop an image or Browse ]  JPG/PNG/WebP ≤10MB│  upload card
│ ( ) Use camera  (st.camera_input, optional)    │
├───────────────────────────────────────────────┤
│  ┌──────────────┐   ## "A dog runs through     │  result row
│  │ image preview│      the grass."             │  (stacks on mobile)
│  └──────────────┘   ▶ audio player             │
│                     ✓ Caption ready · 0.8 s    │
├───────────────────────────────────────────────┤
│ ▸ Options: beam width 1/3/5 · autoplay · model │  expander
│ ▸ See where the model looked (heat-maps)       │  expander, M2 only
│ ▸ About & limitations                          │  expander
└───────────────────────────────────────────────┘
```

Desktop: `st.columns([1, 1.2])`. Below ~640 px Streamlit stacks the columns automatically. Keep reading order image → caption → audio.

## 4. Components

| Component | Streamlit primitive | States | Rules |
|---|---|---|---|
| Upload card | `st.file_uploader(type=["jpg","jpeg","png","webp"])` | empty, invalid, loaded | Label is a visible text label, not placeholder-only. Errors via `st.error` with fix hint |
| Generate button | `st.button("Describe image", type="primary")` | disabled until upload | Also triggered automatically on upload (button kept for re-run with new options) |
| Progress | `st.status("Describing…")` | running/complete/error | Never more than one spinner on screen |
| Caption card | `st.markdown` heading | ready, low-confidence | Ends with a period. Copy button via `st.code(text, language=None)` |
| Audio | `st.audio(mp3_bytes, format="audio/mp3", autoplay=pref)` | ready, unavailable | If TTS fails: "Audio unavailable, caption shown above" (non-blocking) |
| Heat-map grid | `st.image` list with captions = word | — | Max 12 words. Text summary e.g. "'dog' → lower-left" |
| Toasts | `st.toast` | — | Only for non-critical info |

## 5. Content style

- Captions: sentence case, single sentence, trailing period, no "Image of…" prefix.
- Messages: say what happened and what to do. "That file isn't an image we can read. Try a JPG or PNG under 10 MB."
- Limitations text (always available): trained on 8k Flickr photos of people and animals; may misidentify objects, gender or age; not for safety-critical use.

## 6. Accessibility checklist (release gate)

- [ ] Full flow works with keyboard only (Tab / Enter / Space)
- [ ] Visible focus ring on every control (Streamlit default, kept un-overridden)
- [ ] Caption announced by NVDA / VoiceOver after generation
- [ ] Image preview alt text = generated caption
- [ ] Contrast ≥ 4.5:1 in both themes (checked with WebAIM contrast checker)
- [ ] 200% browser zoom without horizontal scroll
- [ ] Audio never autoplays unless the user enabled it
- [ ] No information conveyed by colour alone

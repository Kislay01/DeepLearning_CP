# Security

Threat surface is small: there is no authentication, no database and no stored user data. The real risks are (1) untrusted image uploads, (2) loading untrusted model/artifact files, (3) a third-party TTS service, (4) supply chain, and (5) dataset licensing. Several of these were live problems in the reference repo.

---

## 1. Threat model

| Asset | Threat | Likelihood | Impact |
|---|---|---|---|
| App process | Malicious/oversized image → decompression bomb, parser bug, OOM | Medium | DoS |
| App process | Malicious pickle / checkpoint → arbitrary code execution | Medium (if weights are pulled from the internet) | Critical |
| User privacy | Caption text sent to Google via gTTS; images logged | Medium | Privacy leak |
| Availability | Request flooding on a free host | Medium | DoS |
| Build | Typo-squatted / compromised dependency | Low | Critical |
| Legal | Dataset images used commercially | Low | Licence breach |

## 2. Upload handling (`captioner.inference.validation`)

1. **Size limit:** reject > 10 MB before decoding (`server.maxUploadSize = 10` in `.streamlit/config.toml` plus an explicit `len(bytes)` check).
2. **Allow-list by content, not extension:** decode with Pillow and require `img.format in {"JPEG","PNG","WEBP"}`. Reject everything else, including SVG and GIF animations.
3. **Decompression bombs:** set `Image.MAX_IMAGE_PIXELS = 40_000_000` and treat `Image.DecompressionBombWarning` as an error. Also reject width or height > 8,000 px.
4. **Verify then reload:** `img.verify()` on one handle, then reopen to decode (Pillow requirement).
5. **Normalise:** `exif_transpose`, convert to RGB, drop all metadata (EXIF GPS is never read or logged).
6. **In-memory only:** never write uploads to disk. The reference wrote `description.mp3` to a shared path, which leaks between concurrent users.
7. **Timeout:** inference wrapped with a 10 s budget. On timeout, return an error message.

## 3. Model & artifact loading

- **No pickle anywhere.** The reference loads `tokenizer.p` and `features.p` with `pickle.load`, and a malicious pickle executes code on load.
  - Vocab → JSON. Features → HDF5. Release weights → `safetensors`.
  - Training checkpoints → `torch.load(path, map_location="cpu", weights_only=True)`.
- **Pinned and verified downloads:** the app downloads weights from a pinned HF Hub revision / GitHub Release tag and checks a SHA-256 stored in `configs/release.yaml` before loading.
- Dataset zips verified by MD5 (`database.md` §2.1). The downloader extracts only expected paths (guards against zip-slip: reject members with absolute paths or `..`).

## 4. Third-party services (TTS)

- `gTTS` uses Google Translate's undocumented TTS endpoint. Its own docs say it is not affiliated with Google and can break without notice.
- What leaves the server: **only the caption text**, never the image. State this in the UI's "About" expander.
- Provide a "Speak offline" option (`pyttsx3`/eSpeak) and an "audio off" option. If TTS fails, show the caption without audio, don't crash.
- Timeout 5 s on the TTS call. No retries in the request path.

## 5. Secrets & configuration

- No secrets are needed for the core app. HF tokens (for private model repos) and Kaggle API keys come only from environment variables or platform secret stores (`st.secrets`, HF Space secrets).
- `.env`, `kaggle.json`, `*.token` are in `.gitignore`. A `gitleaks` pre-commit hook blocks accidental commits.

## 6. Web app hardening

- Streamlit: `server.enableXsrfProtection = true` (default, keep it), no `unsafe_allow_html=True`, `client.showErrorDetails = false` in production so stack traces aren't shown to users.
- Captions are rendered as plain text/markdown with no raw HTML. The model's vocabulary is closed, but escaping still applies.
- Rate limiting: per-session cooldown (e.g. 1 request / 2 s) via `st.session_state`. Rely on the host's limits for IP-level throttling.
- Run as a non-root user in the container. Read-only filesystem except the model cache dir.

## 7. Supply chain

- Dependencies pinned in `uv.lock` with hashes. `pip-audit` (or `uv pip audit`) runs in CI weekly and on every PR.
- Install PyTorch from the official index only.
- Dependabot/Renovate for update PRs. Never `pip install` from untrusted Git URLs in production.
- Note: the reference `requirements.txt` listed `sklearn`, a deprecated placeholder package. Use real package names only.

## 8. Privacy & responsible AI

- No image, caption or IP logging by default. Metrics logs hold only latency, model version and error codes.
- Model card lists known failure modes: gender/age assumptions from appearance, Western-centric scenes, people/dogs over-represented in Flickr8k. Do not market the demo as an accessibility product without user testing.
- The demo is non-commercial, which Flickr8k/Flickr30k licensing requires (images are provided for research/education only).

## 9. Security checklist (release gate)

- [ ] No `pickle.load` / `torch.load` without `weights_only=True` (grep in CI)
- [ ] Upload fuzz tests pass (`testing.md` §5)
- [ ] Release artifact SHA-256 verified at start-up
- [ ] `pip-audit` clean or exceptions documented
- [ ] `gitleaks` clean
- [ ] Error pages show no stack traces
- [ ] "About" states what data leaves the server (caption text → TTS)

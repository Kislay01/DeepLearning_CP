# PRD — Image Caption Generator with Speech Output

| Field | Value |
|---|---|
| Codename | `captioner` |
| Type | Deep learning course project (CNN encoder + LSTM decoder), rebuilt from scratch |
| Reference repo | `vishwajit2005/DL-CP` (incomplete; used only as a requirements source) |
| Dataset | Flickr8k (official 6000 / 1000 / 1000 split) — see `database.md` §2 |
| Status | Draft v1.0 — 2026-11-14 |

---

## 1. Problem

Given a photo, produce one fluent English sentence that describes it, and read that sentence aloud. This combines computer vision (what is in the image) and natural language generation (how to say it). Spoken output makes the system useful as an **assistive tool for visually impaired users**, which is also why the reference project added text-to-speech.

## 2. What the reference repo actually is (analysis summary)

The repo is a clone of a widely circulated tutorial project ("Image Caption Generator with CNN & LSTM", originally deployed on Heroku):

- `training_caption_generator.ipynb`: parses `Flickr8k.token.txt`, cleans captions, extracts **Xception** global-pooled features (2048-d) for all images into `features.p`, fits a Keras `Tokenizer`, and trains a **merge model**: `Dense(256)` on image features + `Embedding(256) → LSTM(256)` on the partial caption, combined with `add`, then `Dense(256) → softmax(vocab)`. Trained for 10 epochs; the last checkpoint is `models/model_9.h5` (Keras 2.1.6, vocab 7,577, max length 32).
- `app.py`: Streamlit upload → greedy decoding → `gTTS` → `playsound`.
- Committed artifacts: the full 1.1 GB image set, `features.p` (67 MB pickle), `tokenizer.p` (pickle), `model_9.h5` (60 MB).

### Why it is "incomplete"/broken — and what the rebuild must fix

| # | Defect in reference | Consequence | Requirement in rebuild |
|---|---|---|---|
| D1 | No evaluation at all (no BLEU/METEOR/CIDEr), dev split never used | Quality is unknown; no model selection | FR-6, FR-7 |
| D2 | Cleaning drops all 1-character tokens (`len(word)>1`) | The article **"a"** (most frequent word, ~46.8k train occurrences) is deleted, so outputs are ungrammatical ("dog runs through grass") | FR-2 |
| D3 | `<start>`/`<end>` are stripped to `start`/`end` by Keras `Tokenizer` default filters | Collide with real words "start"/"end"; app uses a string-slicing hack `in_text[6:-3]` | FR-3 (reserved special tokens) |
| D4 | No `<unk>`, no frequency threshold → vocab 7,577 with many singletons | Large, noisy softmax; rare words never learned | FR-3 (min_freq = 5 → 2,529 words) |
| D5 | `max_length` computed over **all** captions incl. test | Minor test leakage | FR-3 (fit on train only) |
| D6 | Training feature extraction does not handle grayscale/RGBA; inference does a different hack | Train/serve skew; crashes on some uploads | FR-1 (single shared preprocessing) |
| D7 | Greedy decoding only | Lower caption quality | FR-5 (beam search) |
| D8 | Orphan caption id `2258277193_586949ec62.jpg.1` (no image) | 8,092 keys vs 8,091 images | FR-1 (validated manifest) |
| D9 | `playsound` plays audio **on the server**; writes a shared `description.mp3` | Remote users hear nothing; concurrent users overwrite each other | FR-9 (in-memory audio to browser) |
| D10 | Model + tokenizer re-loaded on every button click | Slow, memory spikes | NFR-2 (cached loading) |
| D11 | Pickle artifacts; data and weights committed to git | Code-execution risk; 2.4 GB repo | `security.md`, `database.md` |
| D12 | Hard-coded Windows paths, deprecated APIs (`fit_generator`, `layers.merge`), `sklearn` pip name, unpinned deps, dead Heroku deployment | Not reproducible | NFR-4, NFR-5 |

## 3. Goals and non-goals

### Goals
1. Reproducible end-to-end pipeline: download → verify → preprocess → extract features → train → evaluate → serve.
2. Two models, compared fairly on the same test split:
   - **M1 Baseline** — merge model (faithful, bug-fixed re-implementation of the reference idea).
   - **M2 Main** — CNN + **soft-attention** LSTM decoder ("Show, Attend and Tell").
3. Standard evaluation (BLEU-1..4, METEOR, ROUGE-L, CIDEr) on the 1,000-image test split.
4. A web demo that captions an uploaded image, shows an attention heat-map, and plays the caption as speech in the browser.

### Non-goals
- Training a CNN from scratch (encoder is ImageNet-pretrained).
- Large vision-language models (BLIP, GPT-style decoders) — mentioned only as a stretch comparison.
- Multilingual captions, video captioning, user accounts, mobile apps.
- Commercial use (Flickr8k images are licensed for non-commercial research/education only).

## 4. Users

| Persona | Need |
|---|---|
| Visually impaired user | Upload/take a photo, hear a short accurate description; keyboard-only and screen-reader friendly UI |
| Evaluator (faculty/examiner) | See metrics table, baseline-vs-attention comparison, attention visualisations, reproducible commands |
| Developer (you) | Clear modules, configs, tests, fast iteration on cached features |

## 5. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Script downloads Flickr8k from the pinned source, verifies MD5, builds a validated manifest (image exists, readable, RGB-converted, 5 captions) and drops the orphan id | Must |
| FR-2 | Caption cleaning: lowercase, strip non-alphabetic characters, collapse whitespace; **keep "a"** and all real words | Must |
| FR-3 | Vocabulary built on **train split only**, `min_freq = 5`, reserved `<pad>=0, <start>=1, <end>=2, <unk>=3`, saved as JSON; max caption length 25 tokens incl. specials (covers p99 = 22 words) | Must |
| FR-4 | Frozen ResNet-50 feature extraction cached to HDF5: pooled `2048` (M1) and spatial `7×7×2048` (M2) | Must |
| FR-5 | Decoding: greedy and beam search (`k` configurable, default 3), length normalisation, max length guard | Must |
| FR-6 | Training with validation every epoch, early stopping on val BLEU-4, best-checkpoint saving | Must |
| FR-7 | Evaluation script producing metrics JSON + per-image predictions CSV for test split | Must |
| FR-8 | CLI: `captioner predict --image path.jpg [--beam 3]` | Must |
| FR-9 | Streamlit app: upload JPG/PNG/WebP ≤ 10 MB → caption → in-browser audio player (autoplay optional) → attention overlay toggle | Must |
| FR-10 | Attention visualisation: per-word heat-map over the image (M2 only) | Should |
| FR-11 | Model card (data, metrics, limitations, biases) generated with each release | Should |
| FR-12 | Encoder fine-tuning stage (unfreeze last ResNet block) | Could |
| FR-13 | Flickr30k / COCO training config for scale-up | Could |
| FR-14 | Stretch comparison with a Transformer decoder (M3) | Won't (this release) |

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Inference latency (CPU, single image, beam 3) ≤ 2 s p95 after warm-up |
| NFR-2 | Model/vocab loaded once per process (`st.cache_resource`) |
| NFR-3 | Full M2 training on a single free Colab/Kaggle T4 GPU in < 2 h using cached features |
| NFR-4 | Reproducibility: fixed seeds, pinned dependencies (`uv.lock` / `requirements.lock`), config snapshot saved with every run |
| NFR-5 | Runs on Linux/macOS/Windows with no hard-coded paths (all paths from config/env) |
| NFR-6 | Accessibility: WCAG 2.2 AA for the demo UI (see `design_system.md`) |
| NFR-7 | No pickle for artifacts; weights in `safetensors` or `torch.save` loaded with `weights_only=True` |

## 7. Success metrics (Flickr8k test, 1,000 images × 5 references)

Reference points from the literature: on Flickr8k, Show-Attend-Tell reports BLEU-4 19.5 / METEOR 18.93 (soft attention), versus 17.7 / 17.31 for the Google NIC baseline.

| Metric | M1 baseline target | M2 attention target | Stretch (M2 + fine-tune) |
|---|---|---|---|
| BLEU-1 | ≥ 0.55 | ≥ 0.60 | ≥ 0.65 |
| BLEU-4 | ≥ 0.15 | ≥ 0.19 | ≥ 0.21 |
| METEOR | ≥ 0.17 | ≥ 0.19 | ≥ 0.20 |
| CIDEr | ≥ 0.40 | ≥ 0.50 | ≥ 0.55 |

Product metrics: 0 crashes on the 200-image robustness set (grayscale, RGBA, CMYK, tiny, huge, EXIF-rotated); latency per NFR-1.

## 8. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| 1 | Data pipeline | Manifest + vocab built, EDA notebook, data tests green |
| 2 | Feature cache | HDF5 for all 8,091 images, shape tests green |
| 3 | M1 baseline | Trains, early-stops, metrics JSON produced |
| 4 | M2 attention | Beats M1 on BLEU-4 and CIDEr; attention maps render |
| 5 | Beam search + eval | Greedy vs beam table in report |
| 6 | App + TTS | Demo works end-to-end in browser, accessibility checklist passed |
| 7 | Hardening | Security checklist, CI green, model card, deployment |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Small dataset → overfitting | Dropout, early stopping, frozen encoder, weight decay; optional Flickr30k |
| Java missing → METEOR/SPICE fail | `pycocoevalcap` needs Java 1.8+; document, provide NLTK BLEU fallback in training loop |
| gTTS is unofficial and network-dependent | Optional offline TTS fallback (`pyttsx3`), UI degrades to text-only |
| Bias in captions (gender/age guesses) | Model card limitations; evaluate qualitative failure cases |

## 10. Open questions
- Framework choice is fixed to **PyTorch** in `architecture.md`. If the course mandates TensorFlow/Keras, the architecture maps 1:1 (Keras 3 functional API); only `code_style.md` tooling notes change.

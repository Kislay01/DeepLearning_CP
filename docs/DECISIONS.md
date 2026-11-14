# Architecture Decision Records

Format: context → decision → consequence. Numbered in the order they were taken.

---

## ADR-001 — PyTorch wheels come from PyPI, not `download.pytorch.org`

- **Context:** `security.md` §7 says "install PyTorch from the official index". The build sandbox used to create this repo cannot reach `download.pytorch.org` or `huggingface.co` (egress policy returns 403), so `uv lock` cannot resolve against the PyTorch CPU index. PyPI's `torch`/`torchvision` wheels are published by the PyTorch team themselves.
- **Decision:** lock `torch`/`torchvision` from PyPI (still the official PyTorch release artifacts, pinned with hashes in `uv.lock`).
- **Consequence:** Linux installs pull the CUDA build plus `nvidia-*` wheels (≈3 GB); CI caches them. To use the smaller CPU wheels, add a `[[tool.uv.index]]` for `https://download.pytorch.org/whl/cpu` with `[tool.uv.sources] torch = { index = "pytorch-cpu" }` and re-run `uv lock` from a machine that can reach it. The Dockerfile (M7) does exactly that at image build time.

## ADR-002 — Mini fixture images are re-encoded to fit the 1 MB budget

- **Context:** `testing.md` §1 asks for 20 real Flickr8k images "total < 1 MB". The originals average ~120 KB (20 images ≈ 2.4 MB), and `check-added-large-files` caps single files at 1 MB.
- **Decision:** the fixture builder copies 20 real images (12 train / 4 val / 4 test, chosen deterministically) re-encoded as JPEG with the shorter side 256 px and quality 70, plus their 100 real captions and split lists in the original file formats.
- **Consequence:** the fixtures exercise the real file formats and parsing quirks. Pixel content is downscaled, which doesn't matter for software tests.

## ADR-003 — `utils/device.py` added to the repository layout

- **Context:** `code_style.md` §4 requires `utils.device.get_device()`; `architecture.md` §9 lists only `seed.py · io.py · logging.py` under `utils/`.
- **Decision:** add `utils/device.py`. Precedence puts `architecture.md` above `code_style.md`, but the two don't conflict: the layout lists the minimum set of modules and doesn't forbid others.
- **Consequence:** none beyond one extra module.

## ADR-004 — Training batches are 64 images × 1 random caption per epoch

- **Context:** `architecture.md` §6 allows "64 (images × 1 random caption of 5 each epoch) or 32 × all 5 captions".
- **Decision:** default `train.batch_size: 64`, `train.captions_per_image: 1`. Option `5` iterates all 5 captions per image per epoch.
- **Consequence:** an epoch is 6,000 samples. Both modes are covered by tests.

## ADR-005 — Manifest stored as Parquet requires `pandas` + `pyarrow`

- **Context:** `database.md` §3.1 fixes `manifest.parquet`; the stack list in the master prompt doesn't name a Parquet library.
- **Decision:** add `pandas` and `pyarrow` as dependencies.
- **Consequence:** two more locked dependencies; no new services.

## ADR-006 — `img_feature_idx` indexes all 8,091 images, not just the 8,000 in splits

- **Context:** `database.md` §3.3 stores features for 8,091 images, while the manifest (§3.1) has rows only for the 8,000 split images.
- **Decision:** features are extracted for the sorted list of every image present in the archive (8,091). `img_feature_idx` is that sorted position. The feature files store `image_ids` so the mapping is checked on load.
- **Consequence:** the 91 unsplit images cost ~16 MB of spatial features; in exchange the index is stable whatever split is used.

## ADR-007 — Overfit-one-batch threshold applies to cross-entropy only

- **Context:** `testing.md` §3.2 requires "train loss < 0.5 within 300 steps". For M2 the training loss includes the doubly-stochastic regulariser, which isn't driven to 0 by memorising captions.
- **Decision:** the threshold is checked on the cross-entropy term; the regulariser is still part of the optimised loss.
- **Consequence:** the test detects learning bugs without depending on the attention-spread penalty.

## ADR-008 — MLflow via `mlflow-skinny`

- **Context:** MLflow is required for local tracking (`./mlruns` file store). The full `mlflow` package pulls a server stack (Flask, SQLAlchemy, Alembic, …) that the project doesn't use.
- **Decision:** depend on `mlflow-skinny`, the official lightweight distribution, which includes the tracking client and file store.
- **Consequence:** `mlflow ui` needs `pip install mlflow` (or `uvx mlflow ui --backend-store-uri mlruns`) to browse runs.

## ADR-009 — ResNet-50 pretrained weights not available in the build sandbox

- **Context:** torchvision downloads `IMAGENET1K_V2` weights from `download.pytorch.org`, which the build sandbox can't reach. There is no GPU here either.
- **Decision:** code paths accept `encoder.weights: null` (random init) for tests; the default configs keep `IMAGENET1K_V2`. Full feature extraction, training and evaluation are run through `notebooks/colab_train.ipynb` / `make data features train eval` on a machine with network access (and ideally a GPU).
- **Consequence:** result tables in this repo say `not yet run` until someone runs those commands; nothing is estimated.

## ADR-010 — Doubly-stochastic regulariser is averaged over regions and batch

- **Context:** `architecture.md` §5.3 writes the regulariser as `λ · Σ_i (1 − Σ_t α_{t,i})²` with `λ = 1.0`. Summed literally over the 49 regions, the term's floor for a 12-word caption is about `(49 − 12)² / 49 ≈ 28`, roughly 10× the cross-entropy, so λ = 1 would let the penalty dominate training. The widely used PyTorch implementation of Show-Attend-Tell, which reproduces the paper's Flickr8k numbers with λ = 1, averages over regions and batch.
- **Decision:** `reg = alpha_c · mean_b mean_i (1 − Σ_t α_{t,i})²`, with `alpha_c = 1.0` from config. Attention weights on padded steps are zeroed before the sum over t.
- **Consequence:** the term equals the doc formula scaled by 1/L. To use the literal sum, set `model.alpha_c: 49.0`. Ablation A1 (`alpha_c: 0`) is unaffected.

## ADR-011 — M2 has 12.18 M trainable parameters, not "≈ 8–9 M"

- **Context:** `architecture.md` §5.3 estimates ≈ 8–9 M trainable parameters for M2 with V ≈ 2.5 k. Counting the layers it specifies (E = 256, D = 512, A = 512, C = 2048, V = 2,533) gives 12,180,198. Most of the gap is the `LSTMCell` input `[E(y); z_t]`, which includes the 2,048-d context: `4·512·(256+2048)` ≈ 4.7 M weights alone.
- **Decision:** keep the documented dimensions and treat the doc's count as an estimate error. `test_parameter_counts_match_architecture` asserts the exact analytic count for M2 (12,180,198) and M1 (2,416,101).
- **Consequence:** still fits easily on a T4 (≈ 49 MB fp32 weights). Checkpoints will be ~50 MB rather than the ~35 MB guessed in `database.md` §1; the release file stays out of git either way.

## ADR-012 — MLflow file store needs an explicit opt-in

- **Context:** `architecture.md` §1 fixes MLflow with a local `./mlruns` store. MLflow 3.x (locked: 3.18) puts the file store in maintenance mode and raises unless `MLFLOW_ALLOW_FILE_STORE=true`. The SQLite backend it recommends needs SQLAlchemy/Alembic, which `mlflow-skinny` doesn't ship (ADR-008).
- **Decision:** `captioner.training.tracking` sets `MLFLOW_ALLOW_FILE_STORE=true` (and silences the agent hint) with `os.environ.setdefault` before importing MLflow, so users can still override both.
- **Consequence:** runs land in `./mlruns` as documented. Browse them with `uvx mlflow ui --backend-store-uri mlruns` (with `MLFLOW_ALLOW_FILE_STORE=true`).

## ADR-013 — Overfit-one-batch test setup

- **Context:** `testing.md` §3.2 fixes the criterion (8 images, loss < 0.5 within 300 steps, exact greedy reproduction) but not the model size. CI has no pretrained weights (ADR-009), so cached fixture features come from a random-init ResNet-50; they are ~40× larger in magnitude than ImageNet features and very similar across images. Measured: with `decoder_dim=128` M2 crosses CE < 0.5 at step 216–280 for 3 of 4 seeds and misses for 1; with `embed_dim=128, decoder_dim=256` it crosses at step 118–131 on every seed tried.
- **Decision:** the test rescales the batch features to unit standard deviation and uses `embed_dim=128, decoder_dim=256, attention_dim=64, dropout=0` with the full regulariser (λ = 1). Training continues after the 0.5 threshold (up to 300 steps) before checking exact greedy reproduction. Fixture features are extracted under a fixed seed so the test is deterministic.
- **Consequence:** the test is a stable bug detector. Production hyper-parameters are unaffected.

## ADR-014 — Failure taxonomy is computed with keyword heuristics

- **Context:** `testing.md` §4 asks for failure-category counts (wrong object/count/colour, hallucinated object, gender/age assumption, generic caption) over 100 test images, without saying how they are labelled.
- **Decision:** `captioner.evaluation.taxonomy` flags a category when the prediction uses a word of that kind (colour, number, gender/age term, first content word, other content words) that none of the 5 references uses, after crude plural folding. "Generic" = every word is in a small generic list. Reports label the counts "keyword heuristics".
- **Consequence:** counts are reproducible and cheap but approximate. A manual audit is still recommended before drawing conclusions; `samples.html` shows the cases.

## ADR-015 — Upload validation and `export` land in M5 (pulled forward from M6/M7)

- **Context:** M5 requires a `predict` and `export` CLI. `security.md` §2 validation (M6) must sit in front of every untrusted image, and `export` (M7) is needed for `captioner export`.
- **Decision:** implement `inference/validation.py` and `inference/release.py` (safetensors export, model card, checksums, verified loading) in M5. M6 wires validation into the app; M7 adds the start-up downloader, Dockerfile and release checklist.
- **Consequence:** M5's CLI is complete. No placeholder commands.

## ADR-016 — Decoding never emits `<unk>`; `max_len` counts generated tokens

- **Context:** `testing.md` §2 requires that decoding never emits `<pad>`/`<start>`. An `<unk>` in a spoken caption is meaningless to a listener.
- **Decision:** `<pad>`, `<start>` and `<unk>` log-probs are set to −∞ at every step. `decode.max_len` (25) caps generated tokens including `<end>` (code_style.md §6). Length normalisation is `logprob / n^α` with `n` = generated tokens incl. `<end>`, α = 0.7.
- **Consequence:** captions consist of real words only. Training targets still contain `<unk>`, so the model learns its probability mass; it's just never selected.

## ADR-017 — Offline speech (pyttsx3) renders through a private temp file

- **Context:** `security.md` §2.6 says uploads are never written to disk, and the reference wrote a shared `description.mp3` (D9). `pyttsx3` can only synthesise to a file path.
- **Decision:** gTTS (default) stays fully in memory. The optional offline engine writes to a `tempfile.TemporaryDirectory()` that is unique per request and deleted before the function returns; only caption text is involved, never the image. `pyttsx3` is an optional extra (`uv sync --extra offline-tts`) and the "Offline voice" option only appears when it is installed.
- **Consequence:** no shared paths and no cross-user leakage. Audio still plays in the user's browser via `st.audio`.

## ADR-018 — Serving logic lives in `captioner.inference.service`

- **Context:** `code_style.md` §9 keeps ML out of `app/streamlit_app.py`. The app also needs model discovery, a 10 s inference budget (security.md §2.7) and heat-map construction.
- **Decision:** `service.py` provides `resolve_model_source` (env `CAPTIONER_RELEASE_DIR` → `CAPTIONER_RUN_DIR` → `artifacts/release/<version>` → latest run), `load_predictor` (wrapped in `st.cache_resource` by the app), `caption_with_timeout` (thread pool + `future.result(timeout)`), and `build_heatmaps`.
- **Consequence:** the app is layout plus callbacks. Serving behaviour is unit-tested without Streamlit.

## ADR-019 — Docker image installs CPU torch wheels of the locked versions

- **Context:** ADR-001 locks PyPI torch, which pulls about 3 GB of CUDA libraries. The serving image is CPU-only (architecture.md §10).
- **Decision:** the Dockerfile exports `uv.lock` to requirements, installs the **same** `torch`/`torchvision` versions from the official CPU index (`download.pytorch.org/whl/cpu`), installs the rest of the locked requirements, and pre-downloads the IMAGENET1K_V2 weights into `TORCH_HOME` at build time so the container runs with a read-only filesystem. It runs as uid 10001 with a health check. CI builds the image, checks the uid, and probes `/_stcore/health` with `--read-only`.
- **Consequence:** small CPU image with no runtime downloads apart from the pinned model release, which is SHA-256-verified.

---

# Post-blueprint improvements

ADR-020 onward go beyond the original seven documents to remove the limitations listed after the
first release. Where they differ from `security.md`, `architecture.md` or `design_system.md`,
these ADRs win.

## ADR-020 — Speech is generated locally by default (Piper neural TTS); gTTS is opt-in only

- **Context:** gTTS uses Google Translate's unofficial endpoint. It can break without notice and it sends every caption to a third party, a privacy cost for the visually impaired users the demo targets. eSpeak runs locally but sounds robotic.
- **Decision:** the default engine is **Piper** (`piper-tts`, ONNX, CPU), using the `en-US lessac medium` voice. The voice archive is pinned by URL and SHA-256, and each extracted file is pinned by SHA-256. It is downloaded once to `CAPTIONER_VOICE_DIR` or the user cache; the Docker image bakes it in at build time. Synthesis runs in-process into a `BytesIO` WAV; measured 0.19 s for a 9-word caption on the 2-core sandbox. Fallback order is Piper → eSpeak (optional extra). gTTS is reachable only when `tts.allow_online: true`, which defaults to **false**; the UI then labels it "sends caption text to Google".
- **Consequence:** in the default configuration, no image or text leaves the server, and speech keeps working offline. CI checks this by running Piper inside the container with `--network none`. The Lessac voice dataset licence is for non-commercial use, which matches the Flickr8k restriction.

## ADR-021 — Gender-neutral captions: training option, serving guarantee, measured bias

- **Context:** Flickr8k captions name gender in most sentences, so a model learns to guess gender from appearance and is often wrong. Hendricks et al. (2018) document this as a known captioning failure. Gender isn't visually verifiable, and a confidently wrong guess read aloud to a blind user is worse than neutral wording.
- **Decision:**
  1. `captioner.data.neutral` maps gendered words to neutral ones (man/woman → person, boy/girl → child, his/her → their, mother/father → parent, policeman → officer, …) and fixes agreement ("they is" → "they are", "a person and a person" → "two people"). Coarse, visually grounded age (child vs person) is kept.
  2. `vocab.gender_neutral: true` trains on neutral targets. The vocabulary records `text_transform`, so tables, validation BLEU and evaluation apply the same transform. `configs/attention_neutral.yaml` is the recommended demo model. `base.yaml` keeps the blueprint pipeline (false) so the documented vocab numbers still reproduce.
  3. `decode.gender_neutral_output: true` (default) rewrites the output of **any** model one-to-one, keeping attention maps aligned. Served captions never contain gendered words.
  4. Evaluation reports metrics against neutralised references (primary) and raw references (literature comparison), plus `fairness_model` (what the network itself says: gender word rate, gender error rate when references agree, gender guess rate when they don't) and `fairness_served`.
- **Consequence:** the gender-guessing failure is removed from user-facing output and is measured rather than hidden. BLEU against raw references drops slightly for neutral models; both numbers are reported.

## ADR-022 — Reliability: repetition blocking, calibrated confidence, out-of-domain detection

- **Context:** small captioning models hallucinate, often by looping ("a dog and a dog") or by forcing a familiar template onto unfamiliar photos (screenshots, drawings, food, cities). A fixed log-prob threshold (`app.low_confidence_logprob`) isn't calibrated to any particular model.
- **Decision:**
  1. Decoding bans repeated trigrams (`decode.no_repeat_ngram: 3`) and repeated content words (`decode.block_repeated_words`). Both are vectorised over beams. Training-time validation, evaluation and serving use the same policy.
  2. After training, `calibrate` fits spherical k-means (64 centroids) on train pooled features. The **out-of-domain** threshold is set at the 99th percentile of validation distances, and the **low-confidence** threshold at the 10th percentile of validation mean log-probs. Both are saved in `reliability.safetensors` (run folder and release) and loaded by `Predictor`.
  3. `Caption.out_of_domain` / `low_confidence` drive the UI: a warning with an icon and words, plus hedged *spoken* wording ("Possibly: …", "This photo looks unusual for me, so I'm unsure…"), so blind users hear the uncertainty too.
  4. Evaluation checks the calibration on test (`metrics.reliability`): flagged rate, and CIDEr of flagged vs unflagged captions. Diversity now includes `repeated_word_rate`.
- **Consequence:** the model can't be made to know everything, but it now says when it is likely to be wrong, and the check that this signal works is part of every evaluation.

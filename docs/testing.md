# Testing & Evaluation

Two kinds of correctness matter here: **software correctness** (the pipeline does what we think) and **model quality** (captions are good). The reference repo had neither: no tests and no metrics. This doc defines both.

---

## 1. Test pyramid

| Layer | Tool | Runs | Time budget |
|---|---|---|---|
| Unit | `pytest` | every commit (CI, CPU) | < 30 s |
| Integration (tiny data) | `pytest -m integration` | every PR (CI, CPU) | < 3 min |
| Model quality (full eval) | `captioner evaluate` | per experiment / release (GPU) | ~5 min |
| App / E2E | `streamlit.testing.v1.AppTest` + manual a11y | per release | < 2 min + manual |

Fixtures: `tests/fixtures/mini_flickr/` holds **20 real Flickr8k images** (12 train / 4 val / 4 test) with their 100 captions, plus synthetic edge-case images (§5). Total < 1 MB, committed.

## 2. Unit tests

### Data (`tests/unit/test_text.py`, `test_vocab.py`, `test_manifest.py`)
- `clean_caption("A dog's ball - 2 times!") == "a dog s ball times"`, and **"a" is preserved** (regression for reference defect D2).
- Special tokens are reserved: `vocab["<pad>"] == 0`, `<start>`/`<end>`/`<unk>` == 1/2/3, and the words "start"/"end" get normal ids distinct from specials (regression D3).
- Vocab built from train only: a word that appears only in val/test maps to `<unk>` (regression D5).
- `min_freq` respected. Vocab JSON round-trips. Hash changes when the manifest changes.
- Encode → decode round-trip drops specials and pads to `max_len`. Over-long captions are truncated with `<end>` kept.
- Manifest: orphan id `2258277193_586949ec62.jpg.1` excluded. Splits disjoint. 5 captions per image. Counts match `database.md` §3.1 (run on full data in `verify_data.py`, on mini data in tests).

### Preprocessing (`test_transforms.py`)
- Grayscale, RGBA, palette (P), CMYK, 16-bit PNG, EXIF-rotated JPEG → all yield `float32 (3,224,224)` with ImageNet normalisation (regression D6).
- Training and serving use the **same** function (assert identity of the callable used by `Dataset` and `Predictor`).

### Models (`test_models.py`)
- Forward shapes: M1 `(B,T,V)`, M2 `(B,T,V)` + alphas `(B,T,49)`. Alphas sum to 1 over regions.
- Masking: loss ignores `<pad>` (loss equal for a batch with and without extra padding).
- `step()` output matches the equivalent slice of `forward()` under teacher forcing (consistency between training and decoding paths).
- Gradients flow to every trainable param. Encoder params are frozen in stage 1.

### Decoding (`test_decode.py`)
- Beam width 1 == greedy output.
- Beam search with a stub model whose log-probs are hand-set returns the known best sequence. Length normalisation changes the ranking as expected.
- Stops at `<end>`, never exceeds `max_len`, never emits `<pad>`/`<start>`.
- Batched beam search == per-image beam search.

### Metrics (`test_metrics.py`)
- Identical candidate/reference → BLEU-4 = 1.0. Disjoint → 0.
- Our wrapper reproduces `pycocoevalcap` on a fixed 5-image toy set (golden JSON).

## 3. Integration tests (`tests/integration/`, marker `integration`)

1. **Pipeline smoke:** on `mini_flickr`, run `manifest → vocab → features → train 2 epochs → evaluate` via the CLI. Asserts the artifacts exist and the metrics JSON has all keys.
2. **Overfit one batch:** M1 and M2 on 8 images must reach train loss < 0.5 within 300 steps and reproduce a training caption exactly with greedy decoding. This is the main bug detector for DL code.
3. **Determinism:** two runs with seed 42 on CPU produce identical loss for the first 20 steps.
4. **Checkpoint round-trip:** save → load (`weights_only=True`) → identical predictions. Export `safetensors` → `Predictor` → identical predictions.
5. **No pickle:** grep `src/` and `app/` for `pickle` and for `torch.load(` without `weights_only=True`. Fail if found.

## 4. Model evaluation protocol

### Data
Flickr8k **official test split**: 1,000 images × 5 references. Validation split (1,000) for early stopping and hyper-parameter choice only. The test split is touched **once per final model**.

### Metrics
`pycocoevalcap` (requires Java 1.8+): BLEU-1..4, METEOR, ROUGE-L, CIDEr. SPICE optional (downloads Stanford CoreNLP). During training, use fast `nltk` corpus BLEU-4 on val for early stopping.

Pre-processing for scoring: references and hypotheses are both lowercased and passed through the PTB tokenizer bundled with `pycocoevalcap`. Use the **raw** references (with "a"), not the cleaned ones.

### Experiments to report

| ID | Model | Decoding | Purpose |
|---|---|---|---|
| E1 | M1 merge | greedy | Reference-equivalent baseline |
| E2 | M1 merge | beam 3 | Effect of decoding |
| E3 | M2 attention | greedy | Effect of attention |
| E4 | M2 attention | beam 3 / 5 | Main result |
| E5 | M2 + encoder fine-tune | beam 3 | Optional stretch |
| A1 | M2 without doubly-stochastic reg. | beam 3 | Ablation |
| A2 | M2 vocab min_freq 1 vs 5 | beam 3 | Ablation |

Report mean ± std over **3 seeds** for E1, E4. Pass/fail thresholds are in `prd.md` §7. For context: Show-Attend-Tell reports Flickr8k BLEU-4 19.5 (soft attention) and 21.3 (hard attention), and Google NIC 17.7.

### Qualitative analysis (`reports/<run>/samples.html`)
- 25 random test images: prediction vs 5 references.
- 10 best and 10 worst by sentence CIDEr.
- Attention maps for 10 images.
- Failure taxonomy: wrong object, wrong count, wrong colour, hallucinated object, gender/age assumption, generic caption ("a man is standing"). Count each category over 100 test images.
- Diversity: % unique captions across the test set, and vocabulary actually used (attention models often collapse to generic captions; track it).

## 5. Robustness & security tests (`tests/unit/test_validation.py`)

Each must give a clean, user-facing error or a valid caption, never a crash:

| Input | Expected |
|---|---|
| 0-byte file, truncated JPEG, random bytes renamed `.jpg` | `InvalidImageError` |
| PNG decompression bomb (e.g. 50k × 50k header) | rejected before full decode |
| 11 MB file | rejected (size) |
| SVG, GIF, TIFF, HEIC | rejected (format) |
| 1×1 px, 8,000×100 px panorama | caption returned |
| Grayscale / RGBA / CMYK / EXIF-rotated | caption returned |

Plus a 200-image robustness set (random Flickr8k test images re-encoded into the above formats) run before release: 0 crashes required.

## 6. App tests

- `AppTest.from_file("app/streamlit_app.py")`: upload a fixture → caption element present, audio element present. TTS mocked to return fixed bytes. TTS raising an error → caption still shown plus "audio unavailable" message.
- Predictor loaded exactly once across 3 reruns (cache test, regression D10).
- Latency benchmark (`pytest-benchmark`, CPU, beam 3): p95 ≤ 2 s (NFR-1).
- Manual: accessibility checklist in `design_system.md` §6.

## 7. CI (GitHub Actions)

```
on: [push, pull_request]
jobs:
  lint:        ruff format --check · ruff check · mypy src
  unit:        pytest -m "not integration" --cov=captioner --cov-fail-under=80
  integration: pytest -m integration          (CPU, mini_flickr)
  security:    pip-audit · gitleaks · no-pickle grep
```

Coverage target: ≥ 80% on `src/captioner` (excluding `cli.py` and training entry points). GPU evaluation runs manually and attaches `metrics.json` to the PR for any model change.

## 8. Definition of done (per feature)

Tests written first for data/decoding logic. CI green. For model changes: val metrics logged in MLflow, and the comparison table updated in `reports/README.md`.

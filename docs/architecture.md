# Architecture — Image Caption Generator

Companion docs: `prd.md` (what/why), `database.md` (data & artifacts), `testing.md` (evaluation).

---

## 1. Technology decisions

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Stable wheels for all deps |
| DL framework | **PyTorch 2.x + torchvision** | Explicit training loop makes attention, beam search and teacher forcing easy to write and debug; Keras-equivalent design is possible if mandated |
| Encoder | ResNet-50, `IMAGENET1K_V2` weights, frozen (stage 1) | Gives both pooled 2048-d and spatial 7×7×2048 maps from one model; the reference used Xception (pooled only, no spatial attention) |
| Decoders | M1 merge-LSTM (baseline), M2 soft-attention LSTM (main) | Baseline reproduces the reference idea; M2 is the standard upgrade |
| Feature cache | HDF5 via `h5py` | Random access by index, no pickle |
| Config | YAML + typed dataclasses (`pydantic` v2 models) | One file per experiment, validated |
| Tracking | MLflow (local `./mlruns`) + TensorBoard scalars | Run comparison, artifact logging |
| Metrics | `pycocoevalcap` (BLEU, METEOR, ROUGE-L, CIDEr); `nltk` BLEU during training | Literature-standard metrics |
| UI | Streamlit | Same as reference; fastest to a usable demo |
| TTS | `gTTS` (online) → bytes → `st.audio`; optional `pyttsx3` offline fallback | Audio plays in the user's browser, not on the server |
| Packaging | `pyproject.toml`, `uv` lockfile | Reproducible installs |
| Deploy | Hugging Face Spaces (Streamlit) or Streamlit Community Cloud; weights pulled from HF Hub / GitHub Release at start-up | Heroku free tier no longer exists; weights stay out of git |

## 2. System context

```
┌──────────────┐  upload image   ┌────────────────────────────┐
│ User browser │ ──────────────► │ Streamlit app (app/)       │
│ (screen-     │ ◄────────────── │  • validate & preprocess   │
│  reader ok)  │ caption + audio │  • Predictor (cached)      │
└──────────────┘  + heat-map     │  • TTS → mp3 bytes         │
                                 └──────────┬─────────────────┘
                                            │ imports
                                 ┌──────────▼─────────────────┐
                                 │ captioner package (src/)   │
                                 │ data · models · inference  │
                                 │ training · evaluation · tts│
                                 └──────────┬─────────────────┘
                                            │ reads
                    ┌───────────────────────▼───────────────────────┐
                    │ artifacts/: model.safetensors, vocab.json,    │
                    │ config.yaml, model_card.md  (from HF Hub)     │
                    └───────────────────────────────────────────────┘
```

## 3. Offline pipeline (training side)

```
download ─► verify(MD5) ─► build manifest ─► clean captions ─► build vocab (train only)
   │                            │                                   │
   ▼                            ▼                                   ▼
data/raw/                 data/processed/manifest.parquet     artifacts/vocab.json
   │
   └─► extract features (frozen ResNet-50) ─► data/features/resnet50.h5
                                                   │
                       train M1 / M2 (cached features, teacher forcing)
                                                   │
                       validate each epoch (val BLEU-4, beam=1) → early stop
                                                   │
                       evaluate best ckpt on test (beam 1/3/5) → reports/
                                                   │
                       export model.safetensors + vocab + config + model card
```

Each arrow is one CLI command (`make data`, `make features`, `make train CONFIG=...`, `make eval RUN=...`, `make export RUN=...`), so any stage can be re-run independently.

## 4. Preprocessing (shared by training and serving)

One function, `captioner.data.transforms.load_image(path_or_bytes) -> Tensor[3,224,224]`, used everywhere:

1. Open with Pillow, `ImageOps.exif_transpose` (fix phone rotation).
2. Convert to `RGB` (handles L, LA, P, RGBA, CMYK).
3. Resize shorter side to 256, center-crop 224.
4. `ToTensor` + ImageNet mean/std normalisation.

Training-time augmentation (only when fine-tuning the encoder): random-resized-crop (scale 0.8–1.0) and horizontal flip. Horizontal flip can contradict the rare captions that say "left"/"right"; this is accepted and noted in the model card.

Text pipeline (`captioner.data.text`):

```
raw  : "A black dog is running through the grass ."
clean: "a black dog is running through the grass"
ids  : [<start>, a, black, dog, is, running, through, the, grass, <end>, <pad>, ...]
```

Rules: lowercase → replace non `[a-z]` with space → split. Words with train frequency < 5 → `<unk>`. Max 25 tokens incl. specials (train p99 = 22 words; longer captions truncated, < 1%).

## 5. Model architecture

### 5.1 Encoder (shared)

`torchvision.models.resnet50` with `fc` and `avgpool` removed.

| Output | Shape | Used by |
|---|---|---|
| Spatial map | `B × 2048 × 7 × 7` → reshape `B × 49 × 2048` | M2 |
| Pooled vector | mean over 49 → `B × 2048` | M1 |

Stage 1: encoder frozen, features pre-computed once (8,091 images ≈ 1.6 GB float16 spatial, 33 MB pooled). Stage 2 (optional): unfreeze `layer4`, train end-to-end from images with LR 1e-4 for the encoder.

### 5.2 M1 — Merge baseline (bug-fixed reference)

```
image 2048 ─ Dropout(0.5) ─ Linear(2048→256) ─ ReLU ──┐
                                                     add ─ Linear(256→256) ─ ReLU ─ Linear(256→V)
seq[≤25] ─ Embedding(V,256,pad=0) ─ Dropout(0.5) ─ LSTM(256) ─ last hidden ──┘
```

Implementation difference from the reference: instead of exploding every caption into N prefix samples (the reference generator), train it as a standard sequence model — run the LSTM over the whole shifted caption and add the image projection at every step. Same inductive bias, ~10× faster, and it works with the shared loss/decoder code.

### 5.3 M2 — Soft-attention LSTM ("Show, Attend and Tell")

Hyper-parameters (config `attention.yaml`): `embed_dim=256`, `decoder_dim=512`, `attention_dim=512`, `encoder_dim=2048`, `dropout=0.5`.

Per time step *t*:

```
a  = features (B×49×2048)
h0, c0 = Linear(mean(a)) , Linear(mean(a))                # init from image
e_t   = w · tanh(W_a a + W_h h_{t-1})                      # B×49  (additive attention)
α_t   = softmax(e_t)                                        # attention weights
z_t   = σ(f_β(h_{t-1})) ⊙ Σ_i α_{t,i} a_i                  # gated context, B×2048
h_t,c_t = LSTMCell([E(y_{t-1}); z_t], (h_{t-1}, c_{t-1}))
p_t   = softmax(Linear(Dropout(h_t)))                       # B×V
```

Loss = cross-entropy (ignore `<pad>`) + doubly-stochastic attention regulariser `λ · Σ_i (1 − Σ_t α_{t,i})²` with `λ = 1.0`. Training uses teacher forcing over sorted-by-length batches (`pack_padded_sequence` or a step loop masking finished sequences).

Parameter count ≈ 8–9 M trainable (V ≈ 2.5k), fits easily in a T4.

### 5.4 Interfaces

```python
class CaptionModel(nn.Module, Protocol):
    def forward(self, feats: Tensor, captions: Tensor, lengths: Tensor) -> ModelOutput: ...
    def init_state(self, feats: Tensor) -> DecoderState: ...
    def step(self, prev_tokens: Tensor, state: DecoderState) -> tuple[Tensor, DecoderState, Tensor | None]:
        """Returns log-probs (B×V), new state, optional attention (B×49)."""
```

`step()` is the only thing decoding needs, so greedy and beam search are written once and work for M1 and M2.

## 6. Training

| Setting | Value |
|---|---|
| Optimiser | Adam, LR 4e-4 (decoder), weight decay 0 (M2) / 1e-5 (M1) |
| Batch | 64 (images × 1 random caption of 5 each epoch) or 32 × all 5 captions |
| Epochs | max 30, early stop patience 5 on val BLEU-4 |
| LR schedule | ReduceLROnPlateau ×0.5 after 2 stagnant epochs |
| Grad clip | 5.0 (global norm) |
| Precision | AMP (bf16/fp16) on GPU |
| Seed | 42 (Python, NumPy, torch, dataloader workers) |
| Logging | loss, perplexity, val BLEU-4, LR, 8 sample captions per epoch |
| Checkpoint | `best.pt` (by val BLEU-4) + `last.pt`; config snapshot + git SHA + vocab hash |

## 7. Inference

```
bytes ─► validate (security.md) ─► load_image ─► encoder ─► decoder.step loop
       ─► beam search (k=3, len-norm α=0.7, max 25) ─► ids→words (stop at <end>, drop specials)
       ─► detokenise + capitalise + "." ─► Caption{ text, tokens, attention[T×7×7], score }
```

`Predictor` holds encoder + decoder + vocab, is created once (`@st.cache_resource`), runs under `torch.inference_mode()`, and is thread-safe for read-only use.

## 8. Serving (Streamlit app)

```
app/
├── streamlit_app.py      # layout + callbacks only (no ML code)
├── components.py         # upload card, caption card, audio, heat-map
└── theme (.streamlit/config.toml)
```

Flow: upload → size/type check → `Predictor.caption()` → render caption (large text, `aria-live`) → `tts.synthesize(text) -> bytes` → `st.audio(bytes, format="audio/mp3", autoplay=user_pref)` → optional heat-map grid (one small overlay per word). Nothing is written to disk per request.

## 9. Repository layout

```
image-captioner/
├── configs/            base.yaml · merge_baseline.yaml · attention.yaml · attention_finetune.yaml
├── scripts/            download_flickr8k.py · verify_data.py
├── src/captioner/
│   ├── config.py
│   ├── data/           download.py · manifest.py · text.py · vocab.py · dataset.py · transforms.py
│   ├── features/       extract.py
│   ├── models/         encoder.py · merge.py · attention.py · captioner.py
│   ├── training/       loop.py · train.py · early_stopping.py
│   ├── inference/      decode.py (greedy, beam) · predictor.py
│   ├── evaluation/     metrics.py · evaluate.py · report.py
│   ├── tts/            speak.py
│   ├── cli.py          # `captioner` entry point (Typer)
│   └── utils/          seed.py · io.py · logging.py
├── app/                streamlit_app.py · components.py
├── tests/              unit/ · integration/ · fixtures/
├── notebooks/          01_eda.ipynb · 02_error_analysis.ipynb   (exploration only)
├── data/  artifacts/  mlruns/  reports/        (all git-ignored)
├── pyproject.toml · uv.lock · Makefile · README.md · LICENSE
└── docs/               the seven documents
```

## 10. Deployment

1. `make export RUN=<id>` → `artifacts/release/` (weights, vocab, config, model card).
2. Upload to a HF Hub model repo (or GitHub Release).
3. Space/Cloud app downloads the release on first start (pinned revision + SHA-256 check), caches it, serves.
4. CPU-only image; target cold start < 60 s.

## 11. Extension points
- Swap encoder (EfficientNet, ViT) via `encoder.name` in config — features re-extracted, decoder unchanged except `encoder_dim`/grid size.
- Flickr30k / COCO: same manifest schema, different downloader (see `database.md` §2.4).
- M3 Transformer decoder: implement `CaptionModel.step()`; decoding/eval code is reused.

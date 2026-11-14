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

## ADR-011 — M2 has 12.2 M trainable parameters, not "≈ 8–9 M"

- **Context:** `architecture.md` §5.3 estimates ≈ 8–9 M trainable parameters for M2 with V ≈ 2.5 k. Counting the layers it specifies (E = 256, D = 512, A = 512, C = 2048, V = 2,533) gives 12,229,862. Most of the gap is the `LSTMCell` input `[E(y); z_t]`, which includes the 2,048-d context: `4·512·(256+2048)` ≈ 4.7 M weights alone.
- **Decision:** keep the documented dimensions and treat the doc's count as an estimate error. `test_parameter_counts_match_architecture` asserts the exact analytic count for M2 (12,229,862) and M1 (2,416,101).
- **Consequence:** still fits easily on a T4 (≈ 49 MB fp32 weights). Checkpoints will be ~50 MB rather than the ~35 MB guessed in `database.md` §1; the release file stays out of git either way.

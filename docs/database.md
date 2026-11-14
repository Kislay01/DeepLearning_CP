# Data & Artifact Storage

This project needs **no relational database**: there are no users, accounts or transactional writes, and the app keeps nothing between requests. Its "database" is the dataset, the derived training files, and the model artifacts. This document fixes where they come from, their schemas, and how they are versioned.

---

## 1. Storage map

| Layer | Path (git-ignored) | Format | Size (approx.) | Rebuildable? |
|---|---|---|---|---|
| Raw downloads | `data/raw/` | `Flickr8k_Dataset.zip`, `Flickr8k_text.zip` | 1.12 GB + 2.3 MB | Re-download |
| Extracted | `data/raw/Flicker8k_Dataset/*.jpg`, `data/raw/Flickr8k_text/*.txt` | JPEG, TSV | ~1.1 GB | From zips |
| Manifest | `data/processed/manifest.parquet` | Parquet | < 5 MB | From raw |
| Vocabulary | `artifacts/vocab.json` | JSON | ~60 KB | From manifest |
| Features | `data/features/resnet50_{pooled,spatial}.h5` | HDF5 (float16) | 33 MB / ~1.6 GB | From images |
| Runs | `mlruns/` | MLflow file store | small | No (experiment history) |
| Checkpoints | `artifacts/runs/<run_id>/{best,last}.pt` | PyTorch state dicts | ~35 MB each | Re-train |
| Release | `artifacts/release/` → HF Hub | `model.safetensors`, `vocab.json`, `config.yaml`, `model_card.md` | ~35 MB | Re-export |
| Reports | `reports/<run_id>/` | `metrics.json`, `predictions.csv`, figures | small | Re-evaluate |

Nothing under `data/`, `artifacts/`, `mlruns/` or `reports/` is committed. The original repo committed 2.4 GB of data, pickles and weights; that is explicitly not repeated.

---

## 2. Dataset source

### 2.1 Primary: Flickr8k (Hodosh, Young & Hockenmaier, JAIR 2013)

8,000 Flickr photos of people and animals doing everyday things, each with **5 crowd-sourced English captions**, plus an official train/dev/test split. This is the dataset the reference project uses, the one the Show-Attend-Tell paper reports on, and the right size for a single-GPU course project.

**Exact download source (pinned)** — the original UIUC request form is no longer served; the canonical mirror used by Machine Learning Mastery, R `torchvision`, and Keras examples is Jason Brownlee's GitHub release:

| File | URL | Bytes | MD5 (verified 2026-11-14) |
|---|---|---|---|
| Images | `https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip` | 1,115,419,746 | `f18a1e2920de5bd84dae7cf08ec78978` |
| Text | `https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip` | 2,340,801 | `bf6c1abcb8e4a833b7f922104de18627` |

The downloader must refuse to continue if either MD5 differs.

**Contents of `Flickr8k_text.zip`** (all used):

| File | Lines | Use |
|---|---|---|
| `Flickr8k.token.txt` | 40,460 | `<image>.jpg#<0-4>\t<caption>` — raw captions (**source of truth**) |
| `Flickr_8k.trainImages.txt` | 6,000 | Train split |
| `Flickr_8k.devImages.txt` | 1,000 | Validation split |
| `Flickr_8k.testImages.txt` | 1,000 | Test split |
| `Flickr8k.lemma.token.txt` | 40,460 | Lemmatised captions (not used) |
| `ExpertAnnotations.txt`, `CrowdFlowerAnnotations.txt` | 5,822 / 47,830 | Human image–caption relevance judgments (optional: human-correlation analysis) |

**Known quirks (handle in `manifest.py`):**
- Image folder inside the zip is spelled `Flicker8k_Dataset/` (sic), alongside macOS `__MACOSX/` metadata entries that must be skipped.
- The zip has **8,091 images**; the token file references 8,092 ids. The extra id `2258277193_586949ec62.jpg.1` has captions but no image → drop it (40,455 usable captions).
- 6,000 + 1,000 + 1,000 = 8,000 split images; the remaining 91 images are in no split and are ignored.

**Licence:** images belong to their Flickr owners and are provided for **non-commercial research and educational use**; cite Hodosh et al. (2013). The deployed demo must not be monetised.

### 2.2 Alternative mirrors (same data, different packaging)

| Source | Notes |
|---|---|
| Kaggle `adityajn105/flickr8k` | `captions.txt` (CSV `image,caption`) + `Images/`. **No split files** — combine with the official split lists above. Download: `kaggle datasets download -d adityajn105/flickr8k`. Good on Kaggle notebooks. |
| Academic Torrents `9dea07ba660a722ae1008c4c8afdd303b6f6e53b` | Same two zips (1.12 GB + 2.34 MB). |
| `torchvision.datasets.Flickr8k` | Loader only, no download; expects the files above. |

### 2.3 Karpathy split (for comparing with published numbers)

Most papers use Karpathy's pre-tokenised splits from `caption_datasets.zip` (`dataset_flickr8k.json`, also `dataset_flickr30k.json`, `dataset_coco.json`), at `https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip` (MD5 `4fa8c08369d22fe16e41dc124bd1adc2`). A copy is on the HF Hub at `gpantaz/karpathy_split`. For Flickr8k its split sizes are 6,000/1,000/1,000 images, the same sizes as the official split (check image-level overlap before mixing the two). Use the official split lists as the primary split. The Karpathy JSON is an optional cross-check (`configs/base.yaml: split_source: official | karpathy`).

### 2.4 Scale-up datasets (optional, same schema)

| Dataset | Size | Access |
|---|---|---|
| Flickr30k | 31,783 images, 158,915 captions | Request form on `shannon.cs.illinois.edu/DenotationGraph/`; non-commercial; splits from Karpathy JSON |
| MS-COCO 2014 | 82,783 train + 40,504 val images, 5 captions each | `images.cocodataset.org` (train2014/val2014 + annotations), Karpathy split |

---

## 3. Schemas

### 3.1 `manifest.parquet` (one row per caption)

| Column | Type | Constraint |
|---|---|---|
| `image_id` | string | e.g. `1000268201_693b08cb0e` (no extension), FK → image file |
| `file_name` | string | `<image_id>.jpg`, file must exist |
| `split` | category | `train` \| `val` \| `test` |
| `caption_idx` | int8 | 0–4 |
| `caption_raw` | string | original text |
| `caption_clean` | string | output of `text.clean()`; non-empty |
| `n_tokens` | int16 | words in `caption_clean` |
| `width`, `height` | int32 | from validation pass |
| `img_feature_idx` | int32 | row index into the feature HDF5 files |

Invariants (asserted in `verify_data.py` and tests): 8,000 split images × 5 = 40,000 rows; splits disjoint; every `image_id` maps to exactly one `img_feature_idx`.

### 3.2 `vocab.json`

```json
{
  "version": 1,
  "built_from": "train",
  "min_freq": 5,
  "max_len": 25,
  "specials": {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3},
  "itos": ["<pad>", "<start>", "<end>", "<unk>", "a", "in", "the", "..."],
  "freqs": {"a": 46784, "in": 14094, "...": 0},
  "sha256_manifest": "…"
}
```

Expected size with the official train split: 2,529 words + 4 specials = **2,533** (measured on the train captions; 7,333 distinct words before the threshold).

### 3.3 Feature store (HDF5)

```
resnet50_pooled.h5
  /features   float16 [8091, 2048]
  /image_ids  utf-8   [8091]
  attrs: encoder="resnet50", weights="IMAGENET1K_V2", input=224, created_at, code_sha

resnet50_spatial.h5
  /features   float16 [8091, 49, 2048]   chunks=(1,49,2048), lzf compression
  /image_ids  utf-8   [8091]
```

The `Dataset` opens the file lazily per worker (`h5py` handles are not fork-safe).

### 3.4 Run record (MLflow)

Params: full flattened config. Metrics per epoch: `train/loss`, `val/loss`, `val/bleu4`, `lr`. Artifacts: `config.yaml`, `vocab.json`, `best.pt`, `metrics_test.json`, `predictions_test.csv`, `samples.html`. Tags: `git_sha`, `dataset_md5`, `vocab_sha256`, `model=merge|attention`.

### 3.5 `predictions.csv`

`image_id, split, beam, prediction, ref_0..ref_4, bleu4_sentence, cider_sentence`

---

## 4. Versioning and integrity

- Raw data: MD5 pinned in `configs/base.yaml`; downloader verifies.
- Derived files store the SHA-256 of their input (`manifest → vocab → features`). Each stage refuses to run on a mismatched input.
- Optional: DVC on `data/processed` and `data/features` if you want remote caching. Not required.
- Releases are immutable: `artifacts/release/<semver>/` + HF Hub revision tag.

## 5. Runtime data (serving)

The app stores nothing: uploaded bytes and generated audio live in memory for one request. There are no logs of images or captions by default. If usage analytics are ever added, log only latency and error codes, never image content (see `security.md`).

# Results

Every number in this file must be copied from a `reports/<run>/metrics.json` produced by
`captioner evaluate`. Nothing here is estimated. Rows say **not yet run** until that file exists.

Protocol (testing.md §4): official **test** split of each dataset (Flickr8k 1,000 images, COCO
Karpathy 5,000 images), PTB-tokenised, `pycocoevalcap` BLEU-1..4 / METEOR / ROUGE-L / CIDEr. The test
split is touched once per final model; model selection uses val BLEU-4 only. Models with
gender-neutral output are scored against neutralised references (`results.*`) and against the raw
references (`results.*.raw_references`, for comparison with the literature).

## Caption quality

| ID | Model | Data | Decoding | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Source |
|---|---|---|---|---|---|---|---|---|---|
| E1 | M1 merge | Flickr8k | greedy | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E2 | M1 merge | Flickr8k | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E3 | M2 attention | Flickr8k | greedy | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E4 | M2 attention | Flickr8k | beam 3 / 5 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E5 | M2 + encoder fine-tune | Flickr8k | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E6 | M2 attention, neutral captions | Flickr8k | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E7 | M2 attention, ResNet-50 | COCO | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E8 | **M2 attention, CLIP ViT-B/32 (recommended)** | COCO | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| A1 | M2 without doubly-stochastic reg. | Flickr8k | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| A2 | M2 vocab min_freq 1 | Flickr8k | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| A3 | M2 without repetition blocking | Flickr8k | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |

E1 and E4 should be reported as mean ± std over 3 seeds (42, 1, 2).

## Bias, hallucination and reliability (main beam)

| ID | gender word rate (model) | gender error rate (model) | gender guess rate (model) | gender word rate (served) | repeated-word rate | flagged rate | CIDEr flagged / unflagged | Source |
|---|---|---|---|---|---|---|---|---|
| E4 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E6 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E8 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | — |

Keys in `metrics.json`: `results.beam_N.fairness_model`, `fairness_served`,
`diversity.repeated_word_rate`, and the top-level `reliability` block. Calibration works if the
CIDEr of flagged captions is clearly below that of unflagged ones.

Why empty: the repository was built in a sandbox without a GPU and without access to the
pretrained encoder weights (docs/DECISIONS.md ADR-009). Run the commands below on Colab/Kaggle
(`notebooks/colab_train.ipynb`) or any machine with a GPU, then fill the tables from the JSON files.

## Commands

```bash
make setup

# Flickr8k (E1–E6, A1–A3)
make data     CONFIG=configs/attention.yaml
make features CONFIG=configs/attention.yaml
make train CONFIG=configs/merge_baseline.yaml   && make eval CONFIG=configs/merge_baseline.yaml RUN=<run>   # E1/E2
make train CONFIG=configs/attention.yaml        && make eval CONFIG=configs/attention.yaml RUN=<run>        # E3/E4
uv run captioner train -c configs/attention.yaml --set seed=1                                              # seeds
uv run captioner train -c configs/attention_finetune.yaml --set train.init_from=artifacts/runs/<run>/best.pt  # E5
uv run captioner prepare -c configs/attention_neutral.yaml && make train CONFIG=configs/attention_neutral.yaml  # E6
uv run captioner train -c configs/attention.yaml --set name=attention_noreg --set model.alpha_c=0             # A1
uv run captioner prepare -c configs/attention.yaml --set vocab.min_freq=1
uv run captioner train   -c configs/attention.yaml --set name=attention_minfreq1 --set vocab.min_freq=1      # A2
uv run captioner evaluate -c configs/attention.yaml --run <E4 run> --set eval.tag=norepeat \
    --set decode.no_repeat_ngram=0 --set decode.block_repeated_words=false                                # A3

# COCO (E7, E8) — ≈ 40–60 GB disk
make data CONFIG=configs/coco_clip.yaml          # downloads COCO 2014 + Karpathy splits
make features CONFIG=configs/coco_attention.yaml && make train CONFIG=configs/coco_attention.yaml && make eval CONFIG=configs/coco_attention.yaml   # E7
make features CONFIG=configs/coco_clip.yaml      && make train CONFIG=configs/coco_clip.yaml      && make eval CONFIG=configs/coco_clip.yaml        # E8
```

`make eval` writes `reports/<run>/metrics.json` (all beams), `predictions.csv` and `samples.html`
(best/worst/random, attention maps, keyword-heuristic failure taxonomy, diversity). Commit only
`metrics.json`, `predictions.csv` and figures under 1 MB each.

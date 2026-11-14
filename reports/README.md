# Results

Every number in this file must be copied from a `reports/<run>/metrics.json` produced by
`captioner evaluate`. Nothing here is estimated. Rows say **not yet run** until that file exists.

Protocol (testing.md §4): Flickr8k official **test** split, 1,000 images × 5 raw references,
PTB-tokenised, `pycocoevalcap` BLEU-1..4 / METEOR / ROUGE-L / CIDEr. The test split is touched
once per final model; model selection uses val BLEU-4 only.

## Experiment table

| ID | Model | Decoding | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Source |
|---|---|---|---|---|---|---|---|---|
| E1 | M1 merge | greedy | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E2 | M1 merge | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E3 | M2 attention | greedy | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E4 | M2 attention | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E4 | M2 attention | beam 5 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| E5 | M2 + encoder fine-tune | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| A1 | M2 without doubly-stochastic reg. | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |
| A2 | M2, vocab min_freq 1 | beam 3 | not yet run | not yet run | not yet run | not yet run | not yet run | — |

E1 and E4 should be reported as mean ± std over 3 seeds (42, 1, 2).

Why empty: the repository was built in a sandbox without a GPU and without access to the
ImageNet ResNet-50 weights (docs/DECISIONS.md ADR-009). Run the commands below on Colab/Kaggle
(`notebooks/colab_train.ipynb`) or any machine with a GPU, then fill the table from the JSON files.

## Commands

```bash
make setup
make data   CONFIG=configs/base.yaml          # download + MD5 + manifest + vocab
make features CONFIG=configs/base.yaml        # ResNet-50 features → data/features/*.h5

# E1/E2 — merge baseline (greedy + beam 3 come from the same evaluation)
make train CONFIG=configs/merge_baseline.yaml
make eval  CONFIG=configs/merge_baseline.yaml RUN=<merge run folder>

# E3/E4 — attention (beam 1/3/5)
make train CONFIG=configs/attention.yaml
make eval  CONFIG=configs/attention.yaml RUN=<attention run folder>

# extra seeds for E1/E4
uv run captioner train -c configs/attention.yaml --set seed=1
uv run captioner train -c configs/attention.yaml --set seed=2

# E5 — fine-tune ResNet layer4 starting from the best M2 checkpoint
uv run captioner train -c configs/attention_finetune.yaml \
    --set train.init_from=artifacts/runs/<attention run>/best.pt

# A1 — no doubly-stochastic regulariser
uv run captioner train -c configs/attention.yaml --set name=attention_noreg --set model.alpha_c=0

# A2 — vocabulary min_freq 1 (separate artifacts dir so the main vocab is untouched)
export CAPTIONER_ARTIFACTS_DIR=artifacts_minfreq1
uv run captioner prepare -c configs/attention.yaml --set vocab.min_freq=1
uv run captioner train   -c configs/attention.yaml --set name=attention_minfreq1 --set vocab.min_freq=1
```

`make eval` writes `reports/<run>/metrics.json` (all beams), `predictions.csv` and `samples.html`
(best/worst/random, attention maps, keyword-heuristic failure taxonomy, diversity). Commit only
`metrics.json`, `predictions.csv` and figures under 1 MB each.

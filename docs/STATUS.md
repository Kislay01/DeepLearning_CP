# Status

Last updated: M2 (preprocessing & features).

| Milestone | Branch | State | Notes |
|---|---|---|---|
| M0 Scaffold | `feat/scaffold` | done | pyproject + uv.lock, package layout, Makefile, pre-commit, CI, base config, docs |
| M1 Data pipeline | `feat/data` | done | Full data verified: MD5s match, 40,000 rows, 6000/1000/1000, orphan dropped, vocab 2,529 + 4 (7,333 distinct train words), train p99 = 22 words |
| M2 Preprocessing & features | `feat/features` | done (code + tests) | Verified on mini data with random-init ResNet-50. **Full 8,091-image extraction not run here**: IMAGENET1K_V2 weights are unreachable from the sandbox (ADR-009). Run `make features` on Colab/Kaggle. |
| M3 Models | `feat/models` | not started | |
| M4 Training | `feat/training` | not started | |
| M5 Decoding & evaluation | `feat/eval` | not started | |
| M6 App & TTS | `feat/app` | not started | |
| M7 Hardening & release | `feat/release` | not started | |

## Environment limits (build sandbox)

- CPU only (2 cores, 7 GB RAM), no GPU.
- Reachable: GitHub (Flickr8k zips, MD5s verified), PyPI.
- Blocked by egress policy: `download.pytorch.org` (ResNet-50 weights, CPU wheel index) and `huggingface.co`. See ADR-001 and ADR-009.
- No working GitHub API token: PRs can't be opened, so milestones are merged with `--no-ff` and the PR summary goes in the merge commit.

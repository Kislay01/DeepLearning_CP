# Status

Last updated: post-blueprint improvements (ADR-020–024), 2026-11-14.

## Milestones

| Milestone | Branch | State | Notes |
|---|---|---|---|
| M0 Scaffold | `feat/scaffold` | done | pyproject + uv.lock, package layout, Makefile, pre-commit, CI, base config, docs |
| M1 Data pipeline | `feat/data` | done | Full data verified: MD5s match, 40,000 rows, 6000/1000/1000, orphan dropped, vocab 2,529 + 4 (7,333 distinct train words), train p99 = 22 words |
| M2 Preprocessing & features | `feat/features` | done (code + tests) | Verified on mini data with random-init ResNet-50. **Full 8,091-image extraction not run here**: IMAGENET1K_V2 weights are unreachable from the sandbox (ADR-009). Run `make features` on Colab/Kaggle. |
| M3 Models | `feat/models` | done | M1 merge (2.42 M params) and M2 soft attention (12.18 M params, ADR-011), shared `step()` interface, loss with masked attention regulariser (ADR-010) |
| M4 Training | `feat/training` | done (code + tests) | Loop verified on mini data (CPU): overfit-one-batch, determinism, checkpoint round-trip, CLI smoke, fine-tune stage. **Full-data training not run here** (no GPU, no pretrained features, ADR-009). |
| M5 Decoding & evaluation | `feat/eval` | done (code + tests) | Batched beam search + length norm, pycocoevalcap wrapper (matches golden), metrics.json / predictions.csv / samples.html, full CLI, Colab notebook. **Test-set metrics: not yet run.** |
| M6 App & TTS | `feat/app` | done (code + tests) | Streamlit UI per design_system.md, cached predictor (D10), in-memory audio with autoplay off (D9), heat-maps, AppTest suite, CPU latency p95 ≈ 0.11 s |
| M7 Hardening & release | `feat/release` | done (code + tests) | safetensors export + model card + checksums, start-up downloader with SHA-256 verification, non-root CPU Dockerfile (built and health-checked in CI), HF Space config, TRACEABILITY, README |

## Post-blueprint improvements (limitations addressed)

| Branch | State | What |
|---|---|---|
| `feat/local-tts` | done | Piper local neural voice by default, pinned and SHA-256-verified; gTTS opt-in only; CI speaks with `--network none` (ADR-020) |
| `feat/fairness` | done | Gender-neutral training option, neutral served output, gender word/error/guess metrics (ADR-021) |
| `feat/reliability` | done | Repetition blocking; calibrated out-of-domain detector and confidence threshold shipped with each run/release; UI warnings and hedged speech; calibration validated in evaluation (ADR-022) |
| `feat/datasets` | done | MS-COCO / Flickr30k / combined via Karpathy splits, per-dataset paths, trust-on-first-use checksums for COCO (ADR-023, FR-13) |
| `feat/clip-encoder` | done | CLIP ViT-B/32 patch-token encoder, per-encoder preprocessing; `configs/coco_clip.yaml` is the recommended model (ADR-024) |

All of these are implemented and tested on CPU with mini fixtures. Their effect on caption quality
and bias is **not yet measured**: that needs the GPU runs listed below.

## Blocked / not yet run

| Item | Why | What is needed |
|---|---|---|
| Full feature extraction (FR-4 on 8,091 images) | Sandbox cannot reach `download.pytorch.org` (ResNet-50 IMAGENET1K_V2 weights) | Any machine with internet: `make features` |
| Training M1/M2 on full data, NFR-3 timing | No GPU in the sandbox | `notebooks/colab_train.ipynb` on a T4, or `make train CONFIG=…` |
| COCO download, features and the recommended `coco_clip` model | Sandbox can't reach `images.cocodataset.org`, the Karpathy split host, or the open_clip/HF weight hosts | `notebooks/colab_train.ipynb` with `CONFIG=configs/coco_clip.yaml` |
| Test-set metrics (E1–E8, A1–A3), fairness and reliability tables, sample outputs | Depend on the items above | `make eval`, then copy numbers from `reports/<run>/metrics.json` into `README.md` / `reports/README.md` |
| Published model release (`configs/release.yaml` `base_url`/`files`) | No trained model exists yet | `make export`, upload `artifacts/release/0.1.0/*`, paste `release.yaml` |
| Manual accessibility checks (below) | Need a person with a browser and screen reader | Run `make app` and tick the items |

## Test summary

- 215 tests (unit, app and integration), all passing locally and in CI. Integration covers the Flickr8k, COCO-format and CLIP pipelines and real Piper speech.
- Coverage on `src/captioner`: 98 % for the full suite, 92 % unit-only; the CI gate is 80 %.
- D1–D12 each have a passing `test_regression_D*` test; see `docs/TRACEABILITY.md`.

## Security checklist (security.md §9)

- [x] No `pickle.load` / `torch.load` without `weights_only=True`: `scripts/check_no_pickle.py` runs in pre-commit, in CI `security`, and in `test_regression_D11_no_pickle_in_source`.
- [x] Upload fuzz tests pass: `tests/unit/test_validation.py` covers the full testing.md §5 table (0-byte, truncated, random bytes, 50k×50k PNG bomb, 11 MB, SVG/GIF/TIFF/HEIC/BMP/animated rejected; 1×1, 8000×100, grayscale/RGBA/CMYK/EXIF accepted).
- [x] Release artifact SHA-256 verified at start-up: `fetch_release` + `verify_release` (`test_fetch_release_verifies_sha256`, `test_app_downloads_pinned_release_at_startup`, `test_verify_release_detects_tampering`).
- [x] `pip-audit` clean: CI `security` job, every push plus a weekly schedule.
- [x] `gitleaks` clean: CI `security` job (full history) and pre-commit hook.
- [x] Error pages show no stack traces: `client.showErrorDetails = "none"` (`test_streamlit_hardening`); the app catches validation, timeout, TTS and missing-model errors and shows plain messages.
- [x] "About" states what data leaves the server: `test_heatmaps_and_about_present`.
- [ ] The 200-image re-encoded Flickr8k robustness set (testing.md §5) needs the full dataset plus a trained model. Run it before a public release; the synthetic cases above already pass.

## Accessibility checklist (design_system.md §6)

- [ ] Full flow works with keyboard only (Tab / Enter / Space): manual, pending. All controls are native Streamlit widgets.
- [ ] Visible focus ring on every control: manual, pending. The Streamlit default is kept (no CSS overrides exist).
- [ ] Caption announced by NVDA / VoiceOver after generation: manual, pending. The caption is an `h2`, and the success/warning line is an alert.
- [x] Image preview alt text = generated caption: `st.image(..., alt=text)`. Heat-maps and audio also carry alt text.
- [x] Contrast ≥ 4.5:1 in both themes: `test_theme_contrast_at_least_4_5` computes WCAG ratios for every token pair.
- [ ] 200 % browser zoom without horizontal scroll: manual, pending. The layout uses `st.columns`, which stack on narrow viewports.
- [x] Audio never autoplays unless the user enabled it: the toggle defaults to off (`test_regression_D9_audio_is_in_memory_and_autoplay_off`).
- [x] No information conveyed by colour alone: every status message carries an icon and words ("✓ Caption ready", "⚠ Low confidence", "✖ …").

## Environment notes (build sandbox)

- CPU only (2 cores, 7 GB RAM), no GPU.
- Reachable: GitHub (Flickr8k zips, MD5s verified) and PyPI.
- Blocked by the egress policy: `download.pytorch.org` and `huggingface.co` (ADR-001, ADR-009).
- No working GitHub API token, so PRs couldn't be opened. Each milestone was merged into `main` with `--no-ff`, and the PR summary is in the merge commit.

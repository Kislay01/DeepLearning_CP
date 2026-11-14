# Traceability: requirement → module → test

Tests are under `tests/unit/` (u), `tests/integration/` (i, marker `integration`) and `tests/app/` (a).
"Full-data" notes mark what was verified on the real Flickr8k files rather than the mini fixture.

## Functional requirements (prd.md §5)

| ID | Priority | Module(s) | Tests |
|---|---|---|---|
| FR-1 download + MD5 + validated manifest, orphan dropped | Must | `data/download.py`, `data/manifest.py`, `data/prepare.py`, `scripts/verify_data.py` | u `test_download_file_verifies_md5`, `test_safe_extract_rejects_zip_slip`, `test_safe_extract_skips_macos_metadata`, `test_download_flickr8k_end_to_end_with_local_archives`, `test_mini_manifest_invariants`, `test_regression_D8_orphan_caption_id_dropped`, `test_unsplit_images_are_ignored`, `test_unreadable_image_rejected`, `test_missing_split_image_rejected`, `test_overlapping_splits_rejected`. Full-data: `verify_data.py` all PASS (MD5s, 40,000 rows, 6000/1000/1000, 8,091 images) |
| FR-2 caption cleaning, keep "a" | Must | `data/text.py` | u `test_clean_caption_documented_example`, `test_regression_D2_keeps_article_a`, `test_clean_caption_rules` |
| FR-3 train-only vocab, min_freq 5, specials 0–3, JSON, max_len 25 | Must | `data/vocab.py`, `data/prepare.py` | u `test_regression_D3_reserved_special_tokens`, `test_regression_D4_min_freq_threshold_maps_rare_words_to_unk`, `test_regression_D5_vocab_built_from_train_only`, `test_min_freq_and_ordering`, `test_encode_decode_roundtrip_and_padding`, `test_overlong_caption_truncated_keeping_end`, `test_json_roundtrip_and_hash`, `test_vocab_hash_tracks_manifest`. Full-data: 2,529 + 4 |
| FR-4 frozen ResNet-50 features → HDF5 (pooled + spatial) | Must | `models/encoder.py`, `features/extract.py`, `data/dataset.py` | u `test_encoder_shapes_and_freezing`, `test_feature_files_schema`, `test_validate_feature_file_errors`, `test_feature_store_rejects_misaligned_ids`, `test_caption_table_and_datasets`, `test_collate_sorts_and_trims`, `test_dataloader_with_workers_is_reproducible` |
| FR-5 greedy + beam (k, length norm, max len) | Must | `inference/decode.py` | u `test_greedy_follows_argmax`, `test_beam_finds_known_best_and_length_norm_changes_ranking`, `test_beam_width_one_equals_greedy_on_stub`, `test_beam_one_equals_greedy_and_batched_equals_single`, `test_never_emits_specials_and_respects_max_len`, `test_stops_at_end_immediately`, `test_regression_D7_beam_search_finds_higher_probability_caption` |
| FR-6 val each epoch, early stop on val BLEU-4, best checkpoint | Must | `training/train.py`, `training/loop.py`, `training/early_stopping.py`, `training/checkpoint.py` | u `test_early_stopping`, `test_run_training_writes_run_artifacts`, `test_early_stopping_ends_training`; i `test_overfit_one_batch`, `test_determinism_first_20_steps`, `test_checkpoint_roundtrip_identical_predictions`, `test_pipeline_smoke_via_cli` |
| FR-7 evaluation → metrics JSON + predictions CSV | Must | `evaluation/evaluate.py`, `evaluation/metrics.py`, `evaluation/report.py` | u `test_wrapper_reproduces_pycocoevalcap_golden`, `test_identical_and_disjoint_bleu_without_java`; i `test_regression_D1_evaluation_reports` |
| FR-8 `captioner predict --image x.jpg --beam 3` | Must | `cli.py`, `inference/predictor.py` | i `test_full_cli_pipeline`; u `test_predictor_caption` |
| FR-9 Streamlit upload ≤ 10 MB → caption → browser audio → attention toggle | Must | `app/streamlit_app.py`, `app/components.py`, `inference/service.py`, `inference/validation.py`, `tts/speak.py` | a `test_upload_shows_caption_and_audio`, `test_tts_failure_still_shows_caption`, `test_invalid_file_shows_friendly_error`, `test_heatmaps_and_about_present`, `test_missing_model_message`; u `test_rejected_inputs_raise_clean_error`, `test_accepted_inputs_normalised` |
| FR-10 per-word attention heat-map (M2) | Should | `inference/heatmap.py`, `inference/service.py`, `evaluation/report.py` | u `test_heatmap_helpers`, `test_build_heatmaps`; a `test_heatmaps_and_about_present`; i `test_regression_D1_evaluation_reports` (attention maps in samples.html) |
| FR-11 model card with each release | Should | `inference/release.py` | u `test_export_roundtrip_identical_predictions`; i `test_full_cli_pipeline` |
| FR-12 encoder fine-tuning (layer4) | Could | `models/encoder.py`, `training/train.py`, `configs/attention_finetune.yaml` | u `test_encoder_layer4_finetune_unfreezes_only_layer4`, `test_init_from_and_optimizer_groups`; i `test_finetune_stage_runs_from_images` |
| FR-13 Flickr30k / COCO config | Could | not implemented (stretch) | — |
| FR-14 Transformer decoder (M3) | Won't | not implemented (stretch) | — |

## Non-functional requirements (prd.md §6)

| ID | Module(s) | Evidence |
|---|---|---|
| NFR-1 CPU latency ≤ 2 s p95 (beam 3) | `inference/predictor.py` | a `test_cpu_latency_beam3_p95_under_2s` (≈ 0.11 s on the 2-core sandbox, random weights) |
| NFR-2 model loaded once | `app/streamlit_app.py` (`st.cache_resource`) | a `test_regression_D10_predictor_loaded_once` |
| NFR-3 M2 training < 2 h on a T4 | `training/` | **not yet run** (no GPU in the sandbox) — run `notebooks/colab_train.ipynb` |
| NFR-4 seeds, pinned deps, config snapshot | `utils/seed.py`, `uv.lock`, `training/train.py` | u `test_seed_everything_is_reproducible`, `test_regression_D12_reproducible_and_portable`, `test_run_training_writes_run_artifacts`; i `test_determinism_first_20_steps` |
| NFR-5 no hard-coded paths | `config.py` (`CAPTIONER_*` env) | u `test_env_overrides_paths`, `test_regression_D12_reproducible_and_portable` |
| NFR-6 WCAG 2.2 AA | `.streamlit/config.toml`, `app/` | u `test_theme_contrast_at_least_4_5`; a `test_regression_D9_audio_is_in_memory_and_autoplay_off`; manual checklist in `STATUS.md` |
| NFR-7 no pickle; safetensors / `weights_only=True` | `training/checkpoint.py`, `inference/release.py` | i `test_regression_D11_no_pickle_in_source`; u `test_checker_flags_pickle_and_unsafe_torch_load`, `test_checkpoint_errors`; CI `security` job |

## Reference defects (prd.md §2)

| Defect | Fix | Regression test |
|---|---|---|
| D1 no evaluation | `evaluation/`, val BLEU-4 model selection | `test_regression_D1_evaluation_reports` |
| D2 "a" deleted | `text.clean_caption` | `test_regression_D2_keeps_article_a` |
| D3 `<start>`/`<end>` collide with words | reserved specials in `vocab.py` | `test_regression_D3_reserved_special_tokens` |
| D4 no `<unk>` / threshold | `build_vocab(min_freq=5)` | `test_regression_D4_min_freq_threshold_maps_rare_words_to_unk` |
| D5 max length from all splits | vocab from train only | `test_regression_D5_vocab_built_from_train_only` |
| D6 train/serve preprocessing skew | single `transforms.load_image` | `test_regression_D6_every_mode_yields_normalised_tensor`, `test_regression_D6_training_and_serving_share_one_function`, `test_regression_D6_predictor_uses_shared_preprocessing` |
| D7 greedy only | `decode.beam_search` | `test_regression_D7_beam_search_finds_higher_probability_caption` |
| D8 orphan caption id | `manifest.build_manifest` | `test_regression_D8_orphan_caption_id_dropped` |
| D9 server-side audio, shared mp3 | in-memory TTS → `st.audio` | `test_regression_D9_audio_is_in_memory_and_autoplay_off` |
| D10 model reloaded per click | `st.cache_resource` | `test_regression_D10_predictor_loaded_once` |
| D11 pickles + committed data | JSON/HDF5/safetensors, large-file hook | `test_regression_D11_no_pickle_in_source` |
| D12 non-reproducible setup | uv.lock, env paths, modern APIs | `test_regression_D12_reproducible_and_portable` |

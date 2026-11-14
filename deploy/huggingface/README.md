---
title: Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: CNN + attention LSTM image captions read aloud (Flickr8k, non-commercial)
---

# Image Caption Generator — Hugging Face Space

This Space runs the repository's `Dockerfile` (CPU, non-root). Deployment steps:

1. Train, evaluate and export a model (`make train eval export`), then upload
   `artifacts/release/<version>/*` to a GitHub Release or a HF model repo.
2. Copy `artifacts/release/<version>/release.yaml` into `configs/release.yaml` and set `base_url`
   to the folder URL of the uploaded files. The app downloads them on first start and refuses to
   load any file whose SHA-256 differs.
3. Create a Docker Space and push the repository contents to it, using this file as the Space's
   `README.md` (the front matter above configures the Space).
4. Optional Space secrets: `HF_TOKEN` only if the model repo is private.

Flickr8k images are licensed for non-commercial research and education only: do not monetise
this Space.

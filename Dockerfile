# CPU-only serving image for the Streamlit demo (architecture.md §10, security.md §6).
# Build:  docker build -t captioner .
# Run:    docker run --rm -p 8501:8501 --read-only --tmpfs /tmp \
#           -v captioner-cache:/home/app/.cache captioner
# The model is not baked in: the app downloads the release pinned in configs/release.yaml and
# verifies its SHA-256, or loads a mounted folder via CAPTIONER_RELEASE_DIR.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    CAPTIONER_DEVICE=cpu \
    CAPTIONER_CACHE_DIR=/home/app/.cache/captioner \
    CAPTIONER_VOICE_DIR=/opt/voices \
    PORT=8501

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./

# Everything comes from uv.lock; torch/torchvision are swapped for the CPU wheels of the same
# versions from the official PyTorch index (ADR-001) so the image carries no CUDA libraries.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes > /tmp/all.txt \
 && grep -vE '^(torch|torchvision|triton|nvidia-)' /tmp/all.txt > /tmp/req.txt \
 && TORCH="$(grep -E '^torch==' /tmp/all.txt | cut -d' ' -f1)" \
 && TV="$(grep -E '^torchvision==' /tmp/all.txt | cut -d' ' -f1)" \
 && pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH" "$TV" \
 && pip install -r /tmp/req.txt \
 && rm /tmp/all.txt /tmp/req.txt

COPY src ./src
COPY configs ./configs
COPY app ./app
COPY .streamlit ./.streamlit
RUN pip install --no-deps . \
 && python -c "from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)" \
 && python -c "from captioner.config import TTSConfig; from captioner.tts.speak import ensure_voice; ensure_voice(TTSConfig())" \
 && chmod -R a+rX /opt/torch /opt/voices

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /home/app/.cache/captioner \
 && chown -R app:app /home/app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]

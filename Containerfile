FROM cgr.dev/chainguard/wolfi-base:latest

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    NEMO_CACHE_DIR=/cache/nemo \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PATH="/app/venv/bin:$PATH"

# Install Python 3.12 and native system libraries on Wolfi base
RUN apk update && apk add --no-cache \
        build-base \
        cmake \
        pkgconf \
        git \
        ffmpeg \
        libsndfile \
        libsndfile-dev \
        python-3.12 \
        python-3.12-dev \
        py3.12-pip

WORKDIR /app

# Create isolated virtual environment using Python 3.12
RUN python3.12 -m venv /app/venv \
    && /app/venv/bin/pip install --no-cache-dir --retries 10 --timeout 120 --upgrade pip "setuptools>=61.0" wheel cython packaging

# Install Python dependencies using pip
COPY requirements.txt /app/requirements.txt
RUN /app/venv/bin/pip install --no-cache-dir --retries 10 --timeout 120 -r /app/requirements.txt

# Overlay lhotse@main (for lhotse.indexing) and NeMo@main (for RoPE Sortformer support)
# directly from GitHub and verify import + RoPE support at build time.
RUN SITE_PKGS=$(/app/venv/bin/python -c "import site; print(site.getsitepackages()[0])") \
    && git clone --depth 1 https://github.com/lhotse-speech/lhotse.git /tmp/lhotse \
    && cp -rf /tmp/lhotse/lhotse "$SITE_PKGS/" \
    && rm -rf /tmp/lhotse \
    && git clone --depth 1 https://github.com/NVIDIA/NeMo.git /tmp/NeMo \
    && cp -rf /tmp/NeMo/nemo "$SITE_PKGS/" \
    && rm -rf /tmp/NeMo \
    && /app/venv/bin/python -c "import pathlib, nemo.collections.asr as nemo_asr; from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel; from nemo.collections.asr.modules import transformer_encoder as te; assert 'rope' in pathlib.Path(te.__file__).read_text(); print('Verified NeMo ASR + Sortformer + RoPE TransformerEncoder!')"

# Bake both NVIDIA models into /app/models/huggingface so the image runs out-of-the-box without cold-start downloads
ENV HF_HOME=/app/models/huggingface
RUN mkdir -p /app/models/huggingface \
    && /app/venv/bin/python -c "\
from huggingface_hub import hf_hub_download; \
p1 = hf_hub_download(repo_id='nvidia/Nemotron-3-Diarization', filename='Nemotron-3-Diarization.nemo'); \
print('Baked diarization model:', p1); \
p2 = hf_hub_download(repo_id='nvidia/canary-1b-v2', filename='canary-1b-v2.nemo'); \
print('Baked ASR model:', p2)"

LABEL org.opencontainers.image.licenses="Apache-2.0 AND OpenMDW-1.1 AND CC-BY-4.0"

ENV ASR_LANGUAGE=de \
    ASR_BATCH_SIZE=1 \
    ASR_MAX_CHUNK_SEC=30 \
    DIARIZATION_PRESET=offline_high_accuracy \
    PRELOAD_MODELS=false

# Copy application source and model license notices
COPY pipeline.py app.py pyproject.toml README.md MODEL_LICENSES.md /app/
COPY tests /app/tests

# Prepare writable cache directory and switch to nonroot user
RUN (id -u nonroot >/dev/null 2>&1 || adduser -D -u 65532 nonroot) \
    && mkdir -p /cache/huggingface /cache/torch /cache/nemo \
    && chown -R nonroot:nonroot /app /cache \
    && chmod -R 777 /cache

USER nonroot

EXPOSE 7860

ENTRYPOINT ["/app/venv/bin/python", "/app/app.py"]

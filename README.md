# NVIDIA Nemotron-3-Diarization + Canary-1B-v2 Demo

An interactive Web UI (Gradio) and CLI tool that combines **end-to-end speaker diarization** with **language-locked multilingual speech recognition and word-level timestamps**:

- **Speaker Diarization (`who spoke when`)**: [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) — ~100M-parameter Streaming Sortformer model supporting up to **8 speakers** and **overlapping speech** at 10ms frame resolution.
- **Automatic Speech Recognition (`what was said`)**: [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) — 1B-parameter FastConformer Encoder + Transformer Decoder model supporting **25 European languages** with explicit language locking (`source_lang == target_lang`), native punctuation & capitalization (`pnc='yes'`), and **forced-aligned word-level timestamps**.

---

## Features

- **Web UI (File Upload & Microphone Recording)**: Upload `.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a` files, or record live from your microphone.
- **Language-Locked Transcription**: Explicitly locks `source_lang` and `target_lang` (default: `de` for German; supports 25 European languages) so English loanwords or mixed-language phrases are transcribed verbatim in the spoken language rather than translated.
- **Long-Audio Memory Safety**:
  - Streams audio conversion to 16 kHz mono PCM directly on disk via `ffmpeg` (`preprocess_audio`) without loading multi-gigabyte uncompressed waveforms into RAM.
  - Automatically diarizes long recordings (> 8 minutes) in bounded 8-minute windows with a 20-second overlap and speaker-permutation stitching (`stitch_diarization_window`), avoiding unbounded CPU `flex_attention` memory growth.
  - Splits audio at silence gaps between speaker segments (`compute_asr_chunk_windows`) and streams chunk slices directly from disk with configurable **ASR Batch Size** (default `1` on CPU, scalable up to `16` on GPU).
- **Word-to-Speaker Temporal Alignment**: Matches every word's `[start, end]` span from `canary-1b-v2` against active speaker intervals from `Nemotron-3-Diarization` via maximum temporal overlap, sentence-boundary turn smoothing, and configurable nearest-speaker fallback.
- **Multi-Track Speaker Timeline**: High-contrast dark-themed swimlane chart showing each speaker's activity intervals and overlapping speech across the audio duration.
- **Sortformer Context Presets**: Switch between **Offline High Accuracy** (`30.4s` window), **Streaming Balanced** (`~2.5s` latency), and **Streaming Low Latency** (`~0.32s` latency).
- **Rich Exports**: View and download **Markdown**, **SRT Subtitles (`.srt`)**, **NIST RTTM (`.rttm`)**, and **Full JSON (`.json`)**.

---

## Quick Start

### 1. Install Dependencies

Using **`uv`** (recommended):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using standard **`pip`**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

*(Note: Ensure `ffmpeg` and `libsndfile` are installed on your system for multi-format audio decoding.)*

### 2. Launch the Interactive Web UI

```bash
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

#### CLI Flags & Environment Variables

| CLI Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--language` | `ASR_LANGUAGE` | `de` | Spoken language ISO code (`de`, `en`, `fr`, `es`, `it`, `nl`, `pl`, `pt`, `uk`, etc.) |
| `--batch-size` | `ASR_BATCH_SIZE` | `1` | Number of audio chunks transcribed per pass (`1` minimizes CPU RAM; `4`–`8` speeds up GPU inference) |
| `--max-chunk-sec` | `ASR_MAX_CHUNK_SEC` | `30.0` | Maximum ASR chunk duration (seconds) before splitting at silence gaps |
| `--preset` | — | `offline_high_accuracy` | Sortformer preset (`offline_high_accuracy`, `streaming_balanced`, `streaming_low_latency`) |
| `--preload` | — | `false` | Load both Hugging Face models into memory at startup instead of on first request |
| `--device` | — | auto | Torch device override (`cuda` or `cpu`) |
| `--host` | `GRADIO_SERVER_NAME` | `127.0.0.1` | Gradio server bind address |
| `--port` | `GRADIO_SERVER_PORT` | `7860` | Gradio server port |

### 3. Run in CLI Batch Mode

Process an audio file directly from the terminal and export `.md`, `.srt`, `.rttm`, and `.json` files:

```bash
python app.py \
  --audio path/to/conversation.wav \
  --language de \
  --batch-size 1 \
  --output-dir ./outputs
```

---

## Running with Podman

### 1. Build the Container Image

```bash
podman build -t nemo-diarization-demo -f Containerfile .
```

### 2. Run the Web UI Container

Both `nvidia/Nemotron-3-Diarization` and `nvidia/canary-1b-v2` are baked directly into `/app/models/huggingface` inside the image during build (see [`MODEL_LICENSES.md`](./MODEL_LICENSES.md)), so no volume mount or runtime download is required:

```bash
# CPU mode (default ASR_BATCH_SIZE=1)
podman run --rm -it \
  -p 7860:7860 \
  nemo-diarization-demo

# NVIDIA GPU mode (via CDI, with higher batch size for faster throughput)
podman run --rm -it \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -e ASR_BATCH_SIZE=4 \
  -p 7860:7860 \
  nemo-diarization-demo
```

Then open **http://127.0.0.1:7860** in your browser.

### 3. Run CLI Batch Mode with Podman

```bash
podman run --rm -it \
  -v "$(pwd)":/data:Z \
  nemo-diarization-demo \
  --audio /data/conversation.wav \
  --language de \
  --batch-size 1 \
  --output-dir /data/outputs
```

---

## Kubernetes Deployment (Helm)

A Helm chart is included in [`charts/speaker-attributed-asr-demo`](./charts/speaker-attributed-asr-demo). By default, it deploys `ghcr.io/balex42/speaker-attributed-asr-demo:latest` with the models baked into `/app/models/huggingface` (`persistence.enabled: false`).

### 1. Install on CPU Cluster

```bash
helm upgrade --install asr-demo ./charts/speaker-attributed-asr-demo \
  --namespace asr-demo \
  --create-namespace
```

Access the UI locally via port-forward:

```bash
kubectl -n asr-demo port-forward svc/asr-demo-speaker-attributed-asr-demo 7860:7860
```

Then open **http://127.0.0.1:7860**.

### 2. Install with NVIDIA GPU, Custom Batch Size & Preloaded Models

```bash
helm upgrade --install asr-demo ./charts/speaker-attributed-asr-demo \
  --namespace asr-demo \
  --create-namespace \
  --set gpu.enabled=true \
  --set gpu.count=1 \
  --set app.batchSize=4 \
  --set app.language=de \
  --set app.preloadModels=true
```

---

## Project Structure

- `.github/workflows/container.yml`: GitHub Actions workflow that runs unit tests and builds/pushes the container image to `ghcr.io/balex42/speaker-attributed-asr-demo`.
- `charts/speaker-attributed-asr-demo/`: Helm chart (`Deployment`, `Service`, `PersistentVolumeClaim`, `Ingress`) for Kubernetes deployment.
- `Containerfile` & `.containerignore`: Container build definition based on `cgr.dev/chainguard/wolfi-base:latest` with baked Hugging Face models.
- `MODEL_LICENSES.md`: Attribution and license details for `nvidia/Nemotron-3-Diarization` (NVIDIA Open Model License) and `nvidia/canary-1b-v2` (CC-BY-4.0).
- `pipeline.py`: Core `DiarizationASRPipeline` class, streaming `ffmpeg` 16 kHz mono audio preprocessing, windowed Sortformer diarization with speaker-permutation stitching, silence-aware Canary chunked/batched transcription, word-to-speaker overlap alignment, and `.md` / `.srt` / `.rttm` / `.json` exporters.
- `app.py`: Gradio Web UI (with dark-themed HTML swimlane timeline, speaker analytics table, batch size / chunking sliders, and export tabs) and CLI entrypoint.
- `tests/test_alignment.py`: Unit tests for segment parsing, windowed speaker stitching, silence-gap chunking, word-to-speaker alignment, overlap detection, and export formatting.

# NVIDIA Nemotron-3-Diarization + Parakeet-TDT-0.6B-v3 Demo

An interactive Web UI (Gradio) and CLI tool that combines **end-to-end speaker diarization** with **word-timestamped multilingual speech recognition**:

- **Speaker Diarization (`who spoke when`)**: [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) — ~100M-parameter Streaming Sortformer model supporting up to **8 speakers** and **overlapping speech** at 10ms frame resolution.
- **Automatic Speech Recognition (`what was said`)**: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — 600M-parameter FastConformer-TDT model supporting **25 European languages** with native punctuation, capitalization, and **word-level timestamps**.

---

## Features

- **Web UI (File Upload & Microphone Recording)**: Upload `.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a` files, or record live from your microphone.
- **Multi-Track Speaker Timeline**: Visual swimlane chart showing each speaker's activity intervals and overlapping speech across the audio duration.
- **Word-to-Speaker Temporal Alignment**: Matches every word's `[start, end]` span from `parakeet-tdt-0.6b-v3` against active speaker intervals from `Nemotron-3-Diarization` via maximum temporal overlap + configurable nearest-speaker fallback.
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

*(Note: Ensure `ffmpeg` or `libsndfile` is installed on your system for multi-format audio decoding.)*

### 2. Launch the Interactive Web UI

```bash
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.

Optional flags:
- `--preload`: Download and load both Hugging Face models into memory on startup instead of on first inference.
- `--device cuda` / `--device cpu`: Explicitly select inference device.
- `--host 0.0.0.0 --port 7860 --share`: Expose the server on your LAN or generate a public Gradio link.

### 3. Run in CLI Batch Mode

You can also process an audio file directly from the terminal and export `.md`, `.srt`, `.rttm`, and `.json` files:

```bash
python app.py --audio path/to/conversation.wav --output-dir ./outputs
```

---

## Running with Podman

### 1. Build the Container Image

```bash
podman build -t nemo-diarization-demo -f Containerfile .
```

### 2. Run the Web UI Container

Use a named Podman volume (`nemo-cache:/cache`)—Podman creates it automatically so you don't need to pre-create a host directory, and the downloaded `nvidia/Nemotron-3-Diarization` and `nvidia/parakeet-tdt-0.6b-v3` weights persist across runs:

```bash
# CPU mode
podman run --rm -it \
  -p 7860:7860 \
  -v nemo-cache:/cache \
  nemo-diarization-demo

# NVIDIA GPU mode (via CDI)
podman run --rm -it \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -p 7860:7860 \
  -v nemo-cache:/cache \
  nemo-diarization-demo
```

Then open **http://127.0.0.1:7860** in your browser.

### 3. Run CLI Batch Mode with Podman

```bash
podman run --rm -it \
  -v nemo-cache:/cache \
  -v "$(pwd)":/data:Z \
  nemo-diarization-demo \
  --audio /data/conversation.wav --output-dir /data/outputs
```

---

## Project Structure

- `Containerfile` & `.containerignore`: Container build definition for Podman / Docker.
- `pipeline.py`: Core `DiarizationASRPipeline` class, 16 kHz mono audio preprocessing, Sortformer diarization segment parser, Parakeet word-timestamp extractor, word-to-speaker overlap alignment algorithm, and `.md` / `.srt` / `.rttm` / `.json` exporters.
- `app.py`: Gradio Web UI (with HTML swimlane timeline, speaker analytics table, and export tabs) and CLI entrypoint.
- `tests/test_alignment.py`: Unit tests for segment parsing, word-to-speaker alignment, overlap detection, and export formatting.

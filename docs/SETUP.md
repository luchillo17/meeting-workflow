# Setup

**Windows, Linux, macOS**. GPU transcription: NVIDIA CUDA on Windows/Linux. macOS: CPU Whisper (`whisper.device: cpu` in `config.yaml`).

## Prerequisites

| Tool                             | Purpose                        |
| -------------------------------- | ------------------------------ |
| [uv](https://docs.astral.sh/uv/) | Python 3.12 + dependencies     |
| [ffmpeg](https://ffmpeg.org/)    | Audio extract + frame capture  |
| [Ollama](https://ollama.com/)    | Vision + structured extraction |

**GPU (optional):** NVIDIA + CUDA 12 for faster-whisper on Windows/Linux. Runtime libs via `pyproject.toml` (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`).

### By platform

**Windows (winget)**

```powershell
winget install astral-sh.uv
winget install Gyan.FFmpeg
winget install Ollama.Ollama
```

**macOS (Homebrew)**

```bash
brew install uv ffmpeg ollama
```

**Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install ffmpeg   # Debian/Ubuntu; or distro equivalent
# ollama: https://ollama.com/download
```

## Quick start

Project root:

```bash
python3 scripts/install.py   # uv sync + setup
# or:
uv sync
uv run meeting-workflow setup
```

`.env` (created by `setup` if missing):

```env
RECORDING_PATH=/path/to/recording.mp4
OUTPUT_DIR=output
```

Ollama models:

```bash
uv run meeting-workflow setup --pull-models
```

Verify:

```bash
uv run meeting-workflow check
```

Run:

```bash
uv run meeting-workflow process
uv run meeting-workflow process --force
uv run meeting-workflow process --file "/path/to/recording.mp4"
```

Or: `uv run python run.py process --force`

## v1 status

| Stage                                | Status            |
| ------------------------------------ | ----------------- |
| Transcript (ffmpeg + faster-whisper) | **Real**          |
| Visual Capture (ffmpeg + Ollama VL)  | **Real**          |
| Structured Extraction + Summary      | **Real** (Ollama) |

All artifacts real: `transcript.txt`, `frames/`, `visual_content.json`, `extraction.json`, `summary.md`.

## macOS

No NVIDIA CUDA. `config.yaml`:

```yaml
whisper:
  device: cpu
  compute_type: int8
```

First run downloads Whisper model (~3 GB for `large-v3`).

## Troubleshooting

[TROUBLESHOOTING.md](TROUBLESHOOTING.md)

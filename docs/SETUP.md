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
# optional speaker diarization (pyannote + CUDA PyTorch):
uv sync --extra diarization
uv run meeting-workflow setup
```

`.env` (created by `setup` if missing):

```env
RECORDING_PATH=/path/to/recording.mp4
OUTPUT_DIR=output
# HF_TOKEN=...   # required when diarization.enabled is true (see below)
```

### Speaker diarization (optional)

When `diarization.enabled: true` in `config.yaml` (default in repo):

1. Accept model terms: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Create a Hugging Face access token and set `HF_TOKEN` in `.env`
3. Install extras: `uv sync --extra diarization` (installs CUDA PyTorch on Windows/Linux)
4. Re-transcribe pilots with `process --force` (extract-only does not add speakers)

Batch runs print step-based progress (`Progress 12% | step 1/84 | stage 1/4 Transcript | …`) plus a Rich bar when stdout is a TTY. When piping to a log file, use the plain lines only. Press **Ctrl+C** once to stop gracefully — subprocesses are terminated and GPU models unloaded. If a GPU step is still running, the process exits within ~2s; press **Ctrl+C** again to force quit immediately. A `.batch.lock` file in `output/` prevents overlapping batch runs.

`transcript.txt` will use `[Speaker 1]` / `[Speaker 2]` by default. The pipeline always tries to infer real names from speech (`[Sergio]` when found); labels stay `Speaker N` when not inferable. Optional `diarization.roster` in `config.yaml` helps match known attendees.

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

| Stage                                    | Status                |
| ---------------------------------------- | --------------------- |
| Transcript (ffmpeg + faster-whisper)     | **Real**              |
| Speaker diarization (pyannote, optional) | **Real** when enabled |
| Visual Capture (ffmpeg + Ollama VL)      | **Real**              |
| Structured Extraction + Summary          | **Real** (Ollama)     |

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

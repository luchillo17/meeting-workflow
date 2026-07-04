# Setup

Meeting Workflow runs on **Windows, Linux, and macOS**. GPU transcription uses NVIDIA CUDA on Windows/Linux; macOS uses CPU for Whisper (set `whisper.device: cpu` in `config.yaml`).

## Prerequisites

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Python 3.12 + dependencies |
| [ffmpeg](https://ffmpeg.org/) | Audio extraction and frame capture |
| [Ollama](https://ollama.com/) | Vision + structured extraction (stages #4–#5) |

**GPU (optional but recommended):** NVIDIA GPU with CUDA 12 for faster-whisper on Windows/Linux. CUDA runtime libraries are installed automatically via `pyproject.toml` (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`).

### Install prerequisites by platform

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

**Linux (examples)**

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# ffmpeg — use your distro package manager, e.g.:
sudo apt install ffmpeg   # Debian/Ubuntu
# ollama — see https://ollama.com/download
```

## Quick start

From the project root:

```bash
# Option A: install helper (runs uv sync + setup)
python3 scripts/install.py

# Option B: manual steps
uv sync
uv run meeting-workflow setup
```

Edit `.env` (copy is created by `setup` if missing):

```env
RECORDING_PATH=/path/to/recording.mp4
OUTPUT_DIR=output
```

Pull Ollama models (optional now; required when Visual Capture + Extraction land):

```bash
uv run meeting-workflow setup --pull-models
```

Verify everything:

```bash
uv run meeting-workflow check
```

Run a Workflow Run:

```bash
uv run meeting-workflow process
uv run meeting-workflow process --force
uv run meeting-workflow process --file "/path/to/recording.mp4"
```

Equivalent via `run.py`:

```bash
uv run python run.py process --force
```

## v1 status

| Stage | Status |
|-------|--------|
| Transcript (ffmpeg + faster-whisper) | **Real** |
| Visual Capture | Fixture (issue #4) |
| Structured Extraction + Summary | Fixture (issue #5) |

`transcript.txt` / `transcript.json` are real; `extraction.json` is placeholder until #5.

## macOS notes

- No NVIDIA CUDA — set in `config.yaml`:

  ```yaml
  whisper:
    device: cpu
    compute_type: int8
  ```

- First run downloads the Whisper model (~3 GB for `large-v3`).

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

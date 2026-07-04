# Meeting Workflow

Local-first workflow that turns Teams **Meeting Recordings** (`.mp4`) into **Extractions**: Spanish Transcript, Visual Capture, Structured Extraction (`extraction.json`), and Summary (`summary.md`).

Runs on Windows with faster-whisper + Ollama. ~$0 API cost.

## Status

**v1 in development.** Domain model and PRD are defined; implementation follows the [PRD issue](https://github.com/luchillo17/meeting-workflow/issues/1).

## Prerequisites

- Windows 10/11, Python 3.12, ffmpeg, Ollama
- NVIDIA GPU with CUDA (tested target: RTX 4080, 16 GB VRAM)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: RECORDING_PATH, OUTPUT_DIR
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

## Usage

Set `RECORDING_PATH` in `.env`, then:

```powershell
python run.py process              # uses RECORDING_PATH from .env
python run.py process --force      # reprocess even if Extraction exists
```

Or pass a file explicitly (overrides `.env`):

```powershell
python run.py process --file "C:\path\to\recording.mp4"
```

## Output layout

```
output/<slug>/
  transcript.txt
  transcript.json
  frames/
  visual_content.json
  extraction.json
  summary.md
```

## Docs

- [CONTEXT.md](CONTEXT.md) — ubiquitous language
- [docs/adr/](docs/adr/) — architectural decisions

## License

MIT (pending — add LICENSE file with implementation)

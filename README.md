# Meeting Workflow

Local-first workflow that turns Teams **Meeting Recordings** (`.mp4`) into **Extractions**: Spanish Transcript, Visual Capture, Structured Extraction (`extraction.json`), and Summary (`summary.md`).

Runs locally with faster-whisper + Ollama. ~$0 API cost. **Windows, Linux, and macOS** supported.

## Status

**v1 in development** — Transcript, Visual Capture, and Structured Extraction are real ([#5](https://github.com/luchillo17/meeting-workflow/issues/5) complete; GPU pilot tuning in [#6](https://github.com/luchillo17/meeting-workflow/issues/6)).

## Quick start

```bash
# Install deps + create .env
python3 scripts/install.py

# Edit .env: RECORDING_PATH, OUTPUT_DIR
uv run meeting-workflow check
uv run meeting-workflow process --force
```

Platform-specific prerequisites and GPU notes: **[docs/SETUP.md](docs/SETUP.md)**

## Commands

| Command                                       | Description                                      |
| --------------------------------------------- | ------------------------------------------------ |
| `uv run meeting-workflow setup`               | Create `.env`, ensure output dir                 |
| `uv run meeting-workflow setup --pull-models` | Also `ollama pull` configured models             |
| `uv run meeting-workflow check`               | Preflight: ffmpeg, CUDA, Ollama, paths           |
| `uv run meeting-workflow process`             | Workflow Run (uses `RECORDING_PATH` from `.env`) |
| `uv run meeting-workflow process --file PATH` | Process a specific recording                     |
| `uv run meeting-workflow process --force`     | Reprocess even if Extraction exists              |
| `uv run meeting-workflow frames`              | Frame extraction only (uses `transcript.json`)   |
| `uv run meeting-workflow frames --force`      | Re-extract frames without re-transcribing        |

Legacy entrypoint: `uv run python run.py process`

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

## Development

Auto-format runs on **pre-commit** (staged files only):

| Language                    | Tool                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| Python                      | [Ruff](https://docs.astral.sh/ruff/) (`format` + lint fix)                                        |
| Markdown, YAML, JSON, JS/TS | [Prettier](https://prettier.io/)                                                                  |
| PowerShell (`.ps1`)         | Prettier + [prettier-plugin-powershell](https://github.com/Nick2bad4u/Prettier-Plugin-Powershell) |
| Shell (`.sh`)               | [shfmt](https://github.com/mvdan/sh)                                                              |

```bash
uv sync
uv run pre-commit install          # once per clone
uv run pre-commit run --all-files  # format entire repo
```

## Docs

- [docs/SETUP.md](docs/SETUP.md) — install (all platforms)
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common errors
- [CONTEXT.md](CONTEXT.md) — ubiquitous language
- [docs/adr/](docs/adr/) — architectural decisions

## License

MIT (pending — add LICENSE file with implementation)

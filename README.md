# Meeting Workflow

Local-first pipeline: **Meeting Recordings** (`.mp4`, etc.) to **Extractions** — Spanish Transcript, Visual Capture, Structured Extraction (`extraction.json`), Summary (`summary.md`).

faster-whisper + Ollama. ~$0 API cost. **Windows, Linux, macOS**.

## Status

**v1 feature-complete** on `main`. Whisper anti-hallucination: [PR #10](https://github.com/luchillo17/meeting-workflow/pull/10). Pilot sign-off: [#6](https://github.com/luchillo17/meeting-workflow/issues/6).

## Pilot

After [SETUP.md](docs/SETUP.md) + `uv run meeting-workflow check`:

```bash
# .env: RECORDING_PATH, OUTPUT_DIR
uv run meeting-workflow setup --pull-models
uv run meeting-workflow process --force
```

Outputs: `OUTPUT_DIR/<slug>/`. Re-run without `--force` skips if `extraction.json` exists. Acceptance + gaps: [docs/issues/](docs/issues/).

## Quick start

```bash
python3 scripts/install.py   # deps + .env
# edit .env: RECORDING_PATH, OUTPUT_DIR
uv run meeting-workflow check
uv run meeting-workflow process --force
```

Platform + GPU: **[docs/SETUP.md](docs/SETUP.md)**

## Commands

| Command                                       | Description                                 |
| --------------------------------------------- | ------------------------------------------- |
| `uv run meeting-workflow setup`               | Create `.env`, ensure output dir            |
| `uv run meeting-workflow setup --pull-models` | Also `ollama pull` configured models        |
| `uv run meeting-workflow check`               | Preflight: ffmpeg, CUDA, Ollama, paths      |
| `uv run meeting-workflow process`             | Workflow Run (`RECORDING_PATH` from `.env`) |
| `uv run meeting-workflow process --file PATH` | Process specific recording                  |
| `uv run meeting-workflow process --force`     | Reprocess even if Extraction exists         |
| `uv run meeting-workflow frames`              | Frame extraction only (`transcript.json`)   |
| `uv run meeting-workflow frames --force`      | Re-extract frames, no re-transcribe         |

Legacy: `uv run python run.py process`

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

### Git hooks, not CI

Static checks via **pre-commit**, not GitHub Actions. Saves Actions minutes. See [CONTEXT.md](CONTEXT.md#quality-checks-git-hooks-over-ci).

**Pre-commit** (staged files; deptry scans whole project):

| Scope                       | Tool                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| Python                      | [Ruff](https://docs.astral.sh/ruff/) (`format` + lint fix)                                        |
| Python deps                 | [deptry](https://deptry.com/)                                                                     |
| Markdown, YAML, JSON, JS/TS | [Prettier](https://prettier.io/)                                                                  |
| PowerShell (`.ps1`)         | Prettier + [prettier-plugin-powershell](https://github.com/Nick2bad4u/Prettier-Plugin-Powershell) |
| Shell (`.sh`)               | [shfmt](https://github.com/mvdan/sh)                                                              |

**On demand:** `uvx pyscn@latest check --max-complexity 25 workflow tests`

```bash
uv sync
uv run pre-commit install          # once per clone
uv run pre-commit run --all-files
```

## Docs

- [docs/SETUP.md](docs/SETUP.md) — install
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — errors
- [docs/issues/](docs/issues/) — v1 checklist
- [docs/PRD.md](docs/PRD.md) — requirements
- [CONTEXT.md](CONTEXT.md) — ubiquitous language
- [docs/adr/](docs/adr/) — ADRs

## License

MIT (pending — add LICENSE file)

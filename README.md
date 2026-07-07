# Meeting Workflow

Local-first pipeline: **Meeting Recordings** (`.mp4`, etc.) to **Extractions** — Spanish Transcript, Visual Capture, Structured Extraction (`extraction.json`), Summary (`summary.md`).

faster-whisper + Ollama. ~$0 API cost. **Windows, Linux, macOS**.

## Status

**v1 feature-complete** on `main`. Batch + Qwen 3.5 ([#13](https://github.com/luchillo17/meeting-workflow/pull/13)), map-reduce extraction ([#14](https://github.com/luchillo17/meeting-workflow/pull/14)), pilot validation ([#15](https://github.com/luchillo17/meeting-workflow/pull/15)) — all merged. Whisper anti-hallucination: [PR #10](https://github.com/luchillo17/meeting-workflow/pull/10).

## Pilot

After [SETUP.md](docs/SETUP.md) + `uv run meeting-workflow check`:

```bash
# .env: RECORDING_PATH, OUTPUT_DIR
uv run meeting-workflow setup --pull-models
uv run meeting-workflow process --force

# Batch (stage-batched GPU): repeat --file per recording
uv run meeting-workflow process --file meeting-a.mp4 --file meeting-b.mp4

# Re-run extraction only (reuse transcript + vision)
uv run meeting-workflow process --extract-only --file meeting-a.mp4

# Regression spot-checks on pilot output folders
uv run meeting-workflow eval

# Scan watch folders for pending recordings (set WATCH_FOLDERS or ONEDRIVE_ROOT in .env)
uv run meeting-workflow scan
uv run meeting-workflow scan --process   # process pending only

# Full operator loop: process pending + publish new briefs
uv run meeting-workflow inbox

# Publish agent briefs to docs/meetings/ (for Cursor @ context)
uv run meeting-workflow publish --new
uv run meeting-workflow publish --all
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

| Command                                          | Description                                   |
| ------------------------------------------------ | --------------------------------------------- |
| `uv run meeting-workflow setup`                  | Create `.env`, ensure output dir              |
| `uv run meeting-workflow setup --pull-models`    | Also `ollama pull` configured models          |
| `uv run meeting-workflow check`                  | Preflight: ffmpeg, CUDA, Ollama, paths        |
| `uv run meeting-workflow process`                | Workflow Run (`RECORDING_PATH` from `.env`)   |
| `uv run meeting-workflow process --file PATH`    | Process specific recording (repeat for batch) |
| `uv run meeting-workflow process --force`        | Reprocess even if Extraction exists           |
| `uv run meeting-workflow process --extract-only` | Re-run extraction; reuse transcript + vision  |
| `uv run meeting-workflow eval`                   | Pilot regression spot-checks on output dirs   |
| `uv run meeting-workflow publish --all`          | Publish briefs to `docs/meetings/`            |
| `uv run meeting-workflow publish --new`          | Publish only extractions not yet in docs      |
| `uv run meeting-workflow publish --dir PATH`     | Publish one extraction folder                 |
| `uv run meeting-workflow inbox`                  | Process pending + publish new briefs          |
| `uv run meeting-workflow inbox --dry-run`        | Preview inbox plan without changes            |
| `uv run meeting-workflow scan`                   | List recordings in watch folders + status     |
| `uv run meeting-workflow scan --process`         | Process pending recordings from watch folders |
| `uv run meeting-workflow process --folder PATH`  | Process pending recordings in one folder      |
| `uv run meeting-workflow frames`                 | Frame extraction only (`transcript.json`)     |
| `uv run meeting-workflow frames --force`         | Re-extract frames, no re-transcribe           |

Legacy: `uv run python run.py process`

## Output layout

```
output/<slug>/
  transcript.txt
  transcript.json
  chapters/              # map-reduce mode (long meetings)
  chapters.json
  frames/
  visual_content.json
  extraction/            # per-chapter partials (map-reduce)
  extraction.json
  summary.md

docs/meetings/           # published agent briefs (no full transcripts)
  index.md
  <date>-<slug>.md
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

## Parent

#1

## Build

**WorkflowRunner** seam + thin CLI. Full **Workflow Run** on one **Meeting Recording**. Injectable adapters (mocked in tests). Complete **Extraction** from fixture data — no GPU, ffmpeg, Ollama in tests.

Output: `transcript.json`, `transcript.txt`, `visual_content.json`, `extraction.json`, `summary.md` under `OUTPUT_DIR/<slug>/`.

Config: `.env` + `config.yaml`. `RECORDING_PATH`, `process --file`. Skip if `extraction.json` exists unless `--force`. pytest at WorkflowRunner boundary.

## Acceptance

- [x] `python run.py process --file <path>` writes complete Extraction
- [x] Re-run without `--force` skips when `extraction.json` exists (exit 0, logged)
- [x] `--force` overwrites Extraction
- [x] WorkflowRunner accepts injected fakes; tests pass without GPU/network
- [x] pytest in dev deps

## Blocked by

None

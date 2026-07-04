## Parent

#1

## What to build

Establish the **WorkflowRunner** seam and thin CLI for a full **Workflow Run** on one **Meeting Recording**, using injectable adapters (all mocked in tests). A run produces a complete **Extraction** bundle (`transcript.json`, `transcript.txt`, `visual_content.json`, `extraction.json`, `summary.md`) under `OUTPUT_DIR/<slug>/` from fixture data — no GPU, ffmpeg, or Ollama required in CI.

Load configuration from `.env` + `config.yaml`. Support `RECORDING_PATH` and `process --file`. Skip when `extraction.json` exists unless `--force`. Add pytest and tests at the WorkflowRunner boundary.

## Acceptance criteria

- [ ] `python run.py process --file <path>` runs a Workflow Run and writes a complete Extraction directory
- [ ] Re-run without `--force` skips when `extraction.json` exists (exit 0, message logged)
- [ ] `--force` reprocesses and overwrites the Extraction
- [ ] WorkflowRunner accepts injected adapter fakes; tests pass without GPU/network
- [ ] `pytest` runs in CI-local fashion (add pytest to requirements)

## Blocked by

None — can start immediately

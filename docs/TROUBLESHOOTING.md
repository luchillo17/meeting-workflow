# Troubleshooting

## `meeting-workflow check`

Run before long Workflow Run:

```bash
uv run meeting-workflow check
```

Fix **FAIL** first. **WARN** may be OK (e.g. Ollama models not needed for Transcript-only).

## Common errors

### `Library cublas64_12.dll is not found` (Windows)

CUDA libs bundled in Python deps:

```bash
uv sync
uv run meeting-workflow check
```

Use `uv run meeting-workflow process`, not old venv from before repo rename.

### Stale venv / `meeting-pipeline\.venv` in errors

Repo renamed from `meeting-pipeline`. **Delete** `.venv`, recreate — do not rename to `.venv.broken`.

**Windows:** quit Cursor (File → Exit), external terminal:

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

**Linux/macOS:**

```bash
rm -rf .venv
uv sync
```

Cursor locks workspace files; delete `.venv` inside IDE often fails ("Access denied").

`scripts/install.py` detects broken venv after `uv sync` and prints these steps.

### `ffmpeg` not found

Install ffmpeg, add to `PATH`:

```bash
ffmpeg -version
```

### Hugging Face symlink warning (Windows)

Harmless. Silence:

```env
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

Or enable Windows Developer Mode.

### `Skipping - Extraction already exists`

Idempotency works. Re-run:

```bash
uv run meeting-workflow process --force
```

### `extraction.json` wrong or empty

Check Ollama running + text model pulled:

```bash
uv run meeting-workflow setup --pull-models
uv run meeting-workflow process --force
```

Very long transcript: model sees first ~24k characters only.

### `transcript.txt` repeated "Gracias." (or similar)

Whisper hallucinates on silence (waiting room, muted intro, long pauses). `config.yaml`:

```yaml
whisper:
  vad_filter: true
  condition_on_previous_text: false
  hallucination_silence_threshold: 2.0
  filter_hallucination_phrases: true
```

Re-transcribe:

```bash
uv run meeting-workflow process --force
```

### Diarization fails (`HF_TOKEN`, `pyannote.audio`)

When `diarization.enabled: true` in `config.yaml`:

- Accept [pyannote model terms](https://huggingface.co/pyannote/speaker-diarization-3.1) and set `HF_TOKEN` in `.env`
- Install: `uv sync --extra diarization`
- `process --extract-only` does **not** add speakers — use `process --force` to re-transcribe

To disable without uninstalling: `diarization.enabled: false` in `config.yaml`.

### `meeting-workflow eval` failed

Pilot regression checks (`workflow/extraction_eval.py`) guard four known output folders — counts, chapters, vision frames, domain terms, and action-item shape. Common causes:

- **Stale output** — re-run extraction or full pipeline: `uv run meeting-workflow process --extract-only --file …` or `--force`
- **Missing vision frames** — vision filter dropped tile-only descriptions; re-run vision stage or full process after filter updates
- **Transcript term missing in extraction** — term appears in `transcript.txt` but not in `extraction.json` blob; improve prompts or relax `transcript_terms` for that pilot slug
- **Grounding ratio below minimum** — bullets paraphrased too far from transcript or stale extraction; re-run `--extract-only` or lower `extraction.grounding_min_overlap` in `config.yaml` ([ADR 0008](adr/0008-deterministic-token-grounding.md))
- **Short topic or empty action `task`** — extraction model drift; re-run with current Ollama text model

```bash
uv run meeting-workflow eval
uv run meeting-workflow check
```

### Ollama not reachable

```bash
ollama serve   # if not a service
uv run meeting-workflow setup --pull-models
```

### macOS: CUDA / GPU errors

CPU mode in `config.yaml`:

```yaml
whisper:
  device: cpu
  compute_type: int8
```

### Linux: CUDA libraries not found

After `uv sync`, libs live in venv. Always:

```bash
uv run meeting-workflow process
```

`workflow/cuda_paths.py` sets `LD_LIBRARY_PATH` at runtime if needed.

## Help

1. `uv run meeting-workflow check`
2. [SETUP.md](SETUP.md)
3. [GitHub issues](https://github.com/luchillo17/meeting-workflow/issues)

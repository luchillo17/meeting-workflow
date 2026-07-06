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

# Troubleshooting

## `meeting-workflow check`

Run before a long Workflow Run:

```bash
uv run meeting-workflow check
```

Fix any **FAIL** lines first. **WARN** lines may be OK depending on what you are testing (e.g. Ollama models not needed for Transcript-only runs).

## Common errors

### `Library cublas64_12.dll is not found` (Windows)

CUDA libraries are bundled in Python deps. Fix:

```bash
uv sync
uv run meeting-workflow check
```

Use `uv run meeting-workflow process`, not an old virtualenv from before the repo rename.

### Stale venv / `meeting-pipeline\.venv` in errors

The project was renamed from `meeting-pipeline`. Recreate the environment:

```bash
rm -rf .venv          # Linux/macOS
# Remove-Item -Recurse -Force .venv   # Windows
uv sync
```

### `ffmpeg` not found

Install ffmpeg and ensure it is on your `PATH`. Verify with:

```bash
ffmpeg -version
```

### Hugging Face symlink warning (Windows)

Harmless. To silence:

```env
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

Or enable Windows Developer Mode.

### `Skipping - Extraction already exists`

Idempotency is working. Re-run with:

```bash
uv run meeting-workflow process --force
```

### `extraction.json` looks like fixture data

Expected until issue #5. Transcript is real; Structured Extraction is still mocked.

### Ollama not reachable

Start Ollama, then:

```bash
ollama serve   # if not running as a service
uv run meeting-workflow setup --pull-models
```

### macOS: CUDA / GPU errors

Use CPU in `config.yaml`:

```yaml
whisper:
  device: cpu
  compute_type: int8
```

### Linux: CUDA libraries not found

After `uv sync`, libraries live inside the venv. Always launch via:

```bash
uv run meeting-workflow process
```

If needed, `workflow/cuda_paths.py` sets `LD_LIBRARY_PATH` automatically at runtime.

## Getting help

1. `uv run meeting-workflow check`
2. [SETUP.md](SETUP.md)
3. [GitHub issues](https://github.com/luchillo17/meeting-workflow/issues)

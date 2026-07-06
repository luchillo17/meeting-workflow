## Parent

#1

## Build

Real Ollama HTTP clients for text/vision. Merge Transcript + Visual Capture into **Structured Extraction** (`extraction.json`) + **Summary** (`summary.md`) per PRD schema. English prompt templates; output language from `output.language` (default `whisper.language`). Sequential load: transcription done before vision; vision before text extraction.

## Acceptance

- [x] `extraction.json` matches PRD schema (meeting_date, topic, action_items, …)
- [x] `summary.md` same facts for humans
- [x] `meeting_date` from filename when parseable
- [x] One heavy model in VRAM at a time
- [x] WorkflowRunner integration test with fake Ollama

## Blocked by

#4

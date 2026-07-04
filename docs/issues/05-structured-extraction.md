## Parent

#1

## What to build

Replace mocked text/vision adapters with real Ollama HTTP clients. Merge Transcript + Visual Capture into **Structured Extraction** (`extraction.json`) and render **Summary** (`summary.md`) per the PRD schema. Prompt templates in English; output language from `output.language` (defaults to `whisper.language`). Sequential model loading: transcription complete before vision; vision complete before text extraction.

## Acceptance criteria

- [x] `extraction.json` matches PRD schema fields (meeting_date, topic, action_items, etc.)
- [x] `summary.md` renders the same facts for human reading
- [x] `meeting_date` populated from filename when parseable
- [x] Only one heavy model loaded at a time during a Workflow Run
- [x] WorkflowRunner integration test passes with fake Ollama responses

## Blocked by

#4 (Visual Capture)

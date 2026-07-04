## Parent

#1

## What to build

Replace mocked text/vision adapters with real Ollama HTTP clients. Merge Transcript + Visual Capture into **Structured Extraction** (`extraction.json`) and render **Summary** (`summary.md`) per the PRD schema. Spanish prompts. Sequential model loading: transcription complete before vision; vision complete before text extraction.

## Acceptance criteria

- [ ] `extraction.json` matches PRD schema fields (meeting_date, topic, action_items, etc.)
- [ ] `summary.md` renders the same facts for human reading
- [ ] `meeting_date` populated from filename when parseable
- [ ] Only one heavy model loaded at a time during a Workflow Run
- [ ] WorkflowRunner integration test passes with fake Ollama responses

## Blocked by

#4 (Visual Capture)

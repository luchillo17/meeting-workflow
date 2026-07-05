# PRD — Meeting Workflow v1

## Problem Statement

Teams Meeting Recordings capture spoken decisions, action items, and on-screen whiteboarding — but the raw `.mp4` is unusable for search, summarization, or AI tooling. Manually rewatching meetings is slow; transcript-only tools lose diagrams and screen-shared content. The user needs a local, zero-API-cost way to turn a single Meeting Recording into a structured, AI-first Extraction that is also human-readable.

## Solution

**Meeting Workflow** runs a **Workflow Run** on one Meeting Recording at a time. It produces an **Extraction** bundle: Spanish Transcript, Visual Capture of on-screen content, **Structured Extraction** (`extraction.json`), and **Summary** (`summary.md`). Runs locally on Windows with faster-whisper and Ollama. Skips completed Extractions unless `--force` is passed.

## User Stories

1. As a meeting participant, I want to point the workflow at a single Teams `.mp4`, so that I can process one recording without setting up folder watching.
2. As a meeting participant, I want a Spanish Transcript of the recording, so that I can read what was said without rewatching the video.
3. As a meeting participant, I want Visual Capture of whiteboards and screen-shared diagrams, so that content not fully described in speech is preserved.
4. As a meeting participant, I want a Structured Extraction in JSON, so that I can feed meeting content into AI tools (NotebookLM, RAG) later.
5. As a meeting participant, I want a human-readable Summary in markdown, so that I can quickly skim decisions and action items.
6. As a meeting participant, I want key decisions captured in the Extraction, so that I know what was agreed.
7. As a meeting participant, I want action items with owner, task, and deadline fields, so that follow-ups are trackable.
8. As a meeting participant, I want blockers and risks listed, so that impediments are visible.
9. As a meeting participant, I want status updates captured, so that project progress is documented.
10. As a meeting participant, I want technical details extracted, so that architecture and implementation notes are preserved.
11. As a meeting participant, I want open questions listed, so that unresolved topics are not lost.
12. As a meeting participant, I want next steps listed, so that forward momentum is clear.
13. As a meeting participant, I want visual content entries with timestamp, type, and description, so that diagrams are linked to when they appeared.
14. As a meeting participant, I want the meeting date parsed from the filename when possible, so that Extractions are chronologically identifiable.
15. As a meeting participant, I want the topic inferred and stored, so that I can find meetings by subject.
16. As a meeting participant, I want re-running the workflow on the same recording to be skipped by default, so that I do not waste GPU time on completed work.
17. As a meeting participant, I want a `--force` flag to reprocess a recording, so that I can regenerate an Extraction after prompt or model changes.
18. As a meeting participant, I want all outputs in a single Extraction folder per recording, so that artifacts stay together.
19. As a meeting participant, I want paths configured via environment variables, so that no org-specific paths are hardcoded in source.
20. As a meeting participant, I want model settings in a config file, so that I can tune whisper/vision parameters without code changes.
21. As a developer, I want a single WorkflowRunner seam, so that the full Workflow Run is testable with mocked external dependencies.
22. As a developer, I want ffmpeg, faster-whisper, and Ollama behind injectable adapters, so that tests never require GPU or network.
23. As a developer, I want sequential GPU model loading, so that 16 GB VRAM is not exceeded.
24. As a developer, I want the CLI to be a thin wrapper over WorkflowRunner, so that orchestration logic is not duplicated.
25. As a developer, I want CONTEXT.md defining ubiquitous language, so that domain terms stay consistent.
26. As a developer, I want ADRs for hard-to-reverse decisions, so that future readers understand why choices were made.
27. As a developer, I want a public GitHub repo with no secrets or meeting media, so that the project can be shared safely.
28. As a developer, I want `.env.example` documenting required variables, so that setup is reproducible.
29. As a meeting participant, I want frame capture guided by scene changes, so that slide transitions are sampled efficiently.
30. As a meeting participant, I want frame capture boosted when the Transcript mentions visual keywords (tablero, diagrama, etc.), so that whiteboard segments are not missed.
31. As a meeting participant, I want a cap on frames per recording, so that long meetings do not explode processing time.
32. As a meeting participant, I want frames resized before vision analysis, so that Ollama inference stays fast.
33. As a meeting participant, I want near-duplicate frames deduplicated, so that the same whiteboard shot is not described multiple times.
34. As a developer, I want rich CLI output for progress, so that long Workflow Runs show what stage is running.
35. As a meeting participant, I want the workflow to run on Windows natively, so that OneDrive-synced recordings are read at full speed.

## Implementation Decisions

- **System rename:** Project is **Meeting Workflow**; repo `meeting-workflow`; Python package `workflow/`.
- **Single seam:** `WorkflowRunner` orchestrates a full Workflow Run. CLI delegates to it. External I/O behind injectable adapters (ffmpeg runner, transcriber, vision client, text extractor).
- **Input (v1):** One Meeting Recording per Workflow Run via `RECORDING_PATH` env var or `--file` CLI flag. No folder scan, no inbox watch, no SQLite manifest.
- **Idempotency:** Filesystem-based. If `extraction.json` exists in the Extraction output directory and `--force` is not set, exit successfully without reprocessing.
- **Output slug:** Derived from recording filename via slugify helper; output written to `OUTPUT_DIR/<slug>/`.
- **Stages (sequential):** audio extract → transcribe → frame select → vision describe → structured extract + summary render. Unload whisper before vision; one model in VRAM at a time.
- **Transcription:** faster-whisper `large-v3`, CUDA, Spanish, configured via yaml. Silero VAD, decoding thresholds, and optional post-filter for known silence hallucinations (see `config.yaml` `whisper.*`).
- **Visual Capture:** ffmpeg scene detection + interval fallback + transcript keyword boost; max frames, resize, optional perceptual-hash dedup — all configured via yaml.
- **Vision:** Ollama Qwen2.5-VL HTTP API; English prompt templates; output text in `output.language` (default `whisper.language`).
- **Structured Extraction:** Ollama Qwen2.5 text model; JSON schema with meeting_date, topic, key_decisions, action_items, blockers_risks, status_updates, technical_details, open_questions, next_steps, visual_content. English prompts; string values in `output.language`.
- **Summary:** Markdown rendered from Structured Extraction JSON.
- **Config split:** `.env` for paths and Ollama URL/models; `config.yaml` for whisper, frame, and cue settings.
- **Domain docs:** CONTEXT.md glossary + ADRs for local-first GPU, hybrid vision v1, filesystem idempotency.

## Testing Decisions

- **Principle:** Test external behavior at the highest seam, not implementation internals.
- **Primary seam:** `WorkflowRunner.run(recording_path, force=False)` with all heavy dependencies mocked.
- **Runner tests:** skip when Extraction complete; reprocess with force; correct output directory layout; valid extraction.json schema shape; summary.md contains key fields from JSON.
- **Adapter contract tests:** Each adapter interface has a minimal fake implementation used by runner tests.
- **Pure helper unit tests:** slugify, parse_meeting_date, visual-cue segment matching, extraction-to-markdown rendering.
- **Prior art:** None (greenfield). Establish pytest as test runner.
- **Out of automated CI:** Real GPU Workflow Run on a Meeting Recording (manual pilot only).

## Out of Scope

- Folder scanning and inbox watch
- Cross-file hash dedup and SQLite manifest
- NotebookLM upload, RAG indexing, or downstream consumption of Extractions
- Web UI, Streamlit, Tauri tray app
- n8n VPS orchestration, Docker, Microsoft Graph API
- Windows Task Scheduler automation
- Multi-recording batch queue
- Cloud transcription or vision APIs
- English or multilingual support (Spanish only for v1)
- Committing meeting media, transcripts, or org-specific paths to the public repo

## Further Notes

- Pilot with any available Teams `.mp4` via `RECORDING_PATH`. See README **Pilot** section.
- **v1 delivery (2026-07):** Issues #2–#5 and PR #10 shipped on `main`. GPU pilot sign-off tracked in `docs/issues/06-gpu-pilot.md` ([#6](https://github.com/luchillo17/meeting-workflow/issues/6)). Epic [#1](https://github.com/luchillo17/meeting-workflow/issues/1) closes with #6.
- Deferred post-pilot: folder scan, manifest dedup, Task Scheduler, optional UI, extraction fidelity hardening.
- Future inspiration (knowledge delivery → Cursor, second brain, RAG): [inspiration/knowledge-delivery.md](inspiration/knowledge-delivery.md).

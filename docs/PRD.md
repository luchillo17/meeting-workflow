# PRD — Meeting Workflow v1

## Problem

Meeting recordings hold decisions, action items, whiteboarding — but raw video useless for search, summarization, AI. Rewatching slow. Transcript-only loses diagrams. Need local, zero-API-cost path: one Meeting Recording to structured AI-first Extraction + human Summary.

## Solution

**Meeting Workflow** runs **Workflow Run** on one Meeting Recording. **Extraction** bundle: Spanish Transcript, Visual Capture, **Structured Extraction** (`extraction.json`), **Summary** (`summary.md`). Local Windows + faster-whisper + Ollama. Skip completed Extractions unless `--force`.

## User Stories

**Participant**

1. Single meeting recording — no folder watch
2. Spanish Transcript — no rewatch
3. Visual Capture — whiteboards, diagrams speech missed
4. Structured Extraction JSON — NotebookLM, RAG later
5. Summary markdown — skim decisions, action items
6. Key decisions in Extraction
7. Action items: owner, task, deadline
8. Blockers and risks
9. Status updates
10. Technical details
11. Open questions
12. Next steps
13. Visual content: timestamp, type, description
14. `meeting_date` from filename when parseable
15. Topic inferred and stored
16. Re-run skips by default
17. `--force` after prompt/model change
18. One Extraction folder per recording
19. Paths via env vars
20. Model settings in `config.yaml`
21. Frame capture on scene changes
22. Frame boost on Transcript visual keywords (tablero, diagrama, …)
23. Cap frames per recording
24. Resize frames before vision
25. Dedup near-duplicate frames
26. Run locally on Windows — read recordings from local or cloud-synced paths at full disk speed

**Developer**

21. Single `WorkflowRunner` seam — testable with mocks
22. ffmpeg, faster-whisper, Ollama injectable — tests no GPU/network
23. Sequential GPU load — under 16 GB VRAM
24. Thin CLI over WorkflowRunner
25. CONTEXT.md ubiquitous language
26. ADRs for hard-to-reverse decisions
27. Public repo — no secrets, no meeting media
28. `.env.example` reproducible setup
29. Rich CLI progress on long runs

## Implementation Decisions

- **Rename:** Meeting Workflow; repo `meeting-workflow`; package `workflow/`
- **Seam:** `WorkflowRunner` orchestrates run. CLI delegates. I/O via injectable adapters
- **Input (v1):** One recording per run — `RECORDING_PATH` or `--file`. No folder scan, inbox watch, SQLite manifest
- **Idempotency:** `extraction.json` exists + no `--force` = skip success
- **Slug:** filename via slugify; output `OUTPUT_DIR/<slug>/`
- **Stages (sequential):** audio extract, transcribe, frame select, vision describe, structured extract + summary. Unload whisper before vision; one model in VRAM
- **Transcription:** faster-whisper `large-v3`, CUDA, Spanish, yaml config. Silero VAD, decoding thresholds, optional silence hallucination filter (`config.yaml` `whisper.*`)
- **Visual Capture:** ffmpeg scene + interval fallback + transcript keyword boost; max frames, resize, perceptual-hash dedup — yaml
- **Vision:** Ollama Qwen2.5-VL HTTP; English prompts; output text in `output.language` (default `whisper.language`)
- **Structured Extraction:** Ollama Qwen2.5 text; JSON schema: meeting_date, topic, key_decisions, action_items, blockers_risks, status_updates, technical_details, open_questions, next_steps, visual_content. English prompts; string values in `output.language`
- **Summary:** markdown from Structured Extraction JSON
- **Config:** `.env` paths + Ollama URL/models; `config.yaml` whisper, frame, cue settings
- **Docs:** CONTEXT.md + ADRs (local-first GPU, hybrid vision v1, filesystem idempotency, optional owners, no speaker attribution, pilot eval guards, map-reduce extraction, deterministic grounding)

## Testing Decisions

- **Principle:** Test behavior at highest seam, not internals
- **Primary seam:** `WorkflowRunner.run(recording_path, force=False)` with mocked heavy deps
- **Runner tests:** skip when complete; `--force` reprocess; output layout; valid `extraction.json`; `summary.md` fields from JSON
- **Adapter tests:** minimal fake per interface
- **Unit tests:** slugify, parse_meeting_date, visual-cue matching, extraction-to-markdown
- **Runner:** pytest (greenfield)
- **Automation:** pre-commit hooks over GitHub Actions (CONTEXT.md). No CI for lint/format/deptry
- **Not gated:** real GPU Workflow Run on real recording (manual pilot)

## Out of Scope

- Folder scan, inbox watch
- Cross-file hash dedup, SQLite manifest
- NotebookLM upload, RAG indexing, downstream consumption
- Web UI, Streamlit, Tauri tray
- n8n VPS, Docker, Microsoft Graph
- Windows Task Scheduler automation
- Multi-recording batch queue
- Cloud transcription or vision APIs
- English/multilingual (Spanish only v1)
- Commit meeting media, transcripts, org paths to public repo

## Notes

- Pilot: any meeting recording via `RECORDING_PATH`. README **Pilot** section
- **v1 (2026-07):** #2–#5 + PR #10 on `main`. GPU pilot: `docs/issues/06-gpu-pilot.md` ([#6](https://github.com/luchillo17/meeting-workflow/issues/6)). Epic [#1](https://github.com/luchillo17/meeting-workflow/issues/1) closes with #6
- Post-pilot: folder scan, manifest dedup, Task Scheduler, optional UI, extraction fidelity
- Future: [inspiration/knowledge-delivery.md](inspiration/knowledge-delivery.md)

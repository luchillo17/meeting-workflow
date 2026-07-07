# Map-reduce extraction with LLM grounding pass for long meetings

Transcripts over ~12k chars use chapter split → per-chapter extraction → LLM merge →
optional LLM grounding pass against chapter text. Single pass capped at `max_transcript_chars`
with head_tail sampling.

**Decision:** Prefer recall on long Spanish meetings over one-shot JSON within context limits.
Grounding is LLM-mediated (soft); see ADR 0008 for deterministic token-overlap filter.

**Consequences:** Higher Ollama cost/time on long meetings; partials in `extraction/chapters/`.
Re-run with `--extract-only` after prompt changes without re-transcribing.

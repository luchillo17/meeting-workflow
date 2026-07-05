## Parent

#1

## What to build

Manual **GPU pilot** on a real Teams Meeting Recording. Verify rich CLI progress output, document pilot steps in README, and confirm skip/`--force` on real hardware. Out of automated CI.

## Acceptance criteria

- [x] One real `.mp4` processed end-to-end on Windows + CUDA (Vital Link pilot, 2026-07-04)
- [x] Spanish transcript quality is acceptable (after #10 VAD + phrase filter; 0 silence hallucinations on pilot)
- [x] Visual Capture runs when screen content is present (Teams participant tiles / layout changes)
- [x] Re-run skips; `--force` regenerates Extraction
- [x] README documents pilot command and prerequisites

## Known pilot gaps (not blocking v1)

- Structured Extraction can still mis-state figures or treat UI labels (e.g. read.ai) as people — prompt/quality follow-up.
- CLI shows stage names only; no per-stage progress bars (PRD story #34 partial).

## Blocked by

#5 (Structured Extraction with real Ollama) — done. #10 (Whisper anti-hallucination) — done.

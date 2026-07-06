## Parent

#1

## Build

Manual **GPU pilot** on real meeting recording. Verify CLI progress, document pilot in README, confirm skip/`--force` on real hardware. Not in automated CI.

## Acceptance

- [x] Real `.mp4` end-to-end Windows + CUDA (Vital Link pilot, 2026-07-04)
- [x] Spanish transcript acceptable (#10 VAD + phrase filter; 0 silence hallucinations on pilot)
- [x] Visual Capture when screen content present
- [x] Re-run skips; `--force` regenerates
- [x] README documents pilot + prerequisites

## Pilot gaps (not blocking v1)

- Structured Extraction may mis-state figures or treat UI labels (read.ai) as people — prompt follow-up
- CLI: stage names only; no per-stage progress bars (PRD #34 partial)

## Blocked by

#5 done. #10 done.

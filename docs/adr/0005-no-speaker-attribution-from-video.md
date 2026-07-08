# No speaker attribution from video tiles in v1

Owner/deadline attribution from "who was highlighted in Teams" would need dense frame
sampling, active-speaker detection, and OCR on roster tiles — brittle across layouts
(gallery vs content-only, screen share hides roster). Our pipeline deliberately works
against that signal:

- Vision prompt skips participant-only tiles; extraction filters tile-only descriptions.
- Frame dedup blurs active-speaker highlight rings (`dedup_layout_blur`, `dedup_global`).
- We do **not** infer who spoke from video tiles or roster OCR.

**Decision:** v1 is post-recording `.mp4` only. No meeting bot, no **video-based**
speaker-to-name mapping.

**Audio diarization (separate):** optional pyannote pass labels transcript segments
as `Speaker 1`, `Speaker 2`, … — see [ADR 0009](0009-audio-speaker-diarization.md).
That is not the same as Teams highlight attribution.

**Future (if needed):** calendar roster hints → map `Speaker N` to names → live
bot/API speaking events — in that order, not Teams highlight chasing first.
